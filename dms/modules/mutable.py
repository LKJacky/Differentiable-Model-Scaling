# Copyright (c) OpenMMLab. All rights reserved.
from mmrazor.models.mutables.derived_mutable import DerivedMutable
import torch
import torch.nn as nn
from mmengine.dist import all_reduce

from mmrazor.models.mutables import (
    BaseMutable,
    DerivedMutable,
    MutableChannelContainer,
    SimpleMutableChannel,
)
from torch.nn.functional import gumbel_softmax


#############################################################################
# for ablation


@torch.jit.script
def dtp_get_importance_estimate(
    v: torch.Tensor,
    e: torch.Tensor,
):
    e = e.clamp(0, 1)
    n = v.numel()
    k = int(n * e)

    topk = v.topk(k, largest=True)[1]
    topk_mask = torch.zeros_like(v)
    topk_mask[topk] = 1
    topk_mask = topk_mask.detach() - e.detach() + e
    return topk_mask


#############################################################################
BlockThreshold = 0.5

MaskThreshold = 0.5
# dtp with taylor importance base dtp with adaptive importance


@torch.jit.script
def dtopk(x: torch.Tensor, e: torch.Tensor, lamda: float = 1.0):
    # x is smaller is more important
    # add min or max
    # e = soft_clip(e, 1 / x.numel(), 1.0)

    y: torch.Tensor = -(x - e) * x.numel() * lamda
    s = y.sigmoid()
    return s


@torch.jit.script
def dtp_get_importance(
    v: torch.Tensor,
    e: torch.Tensor,
    lamda: float = 1.0,
    space_min: float = 0,
    space_max: float = 1.0,
    normalize: bool = True,
):
    if v.numel() == 1:
        return (-(0.5 - e) * lamda).sigmoid()
    if normalize:
        vm = v.unsqueeze(-1) - v.unsqueeze(-2)
        vm = (vm >= 0).float() - vm.detach() + vm
        v_union = vm.mean(dim=-1)  # big to small
        v_union = 1 - v_union
    else:
        v = v - v.min()
        v = v / v.max() * 0.99
        v_union = 1 - v

    if space_max != 1.0 or space_min != 0:
        v_union = v_union * (space_max - space_min) + space_min
    imp = dtopk(v_union, e, lamda=lamda)  # [0,1]
    return imp


def taylor_backward_hook_wrapper(module: "DTPTMutableChannelImp", input):
    def taylor_backward_hook(grad):
        with torch.no_grad():
            module.update_taylor(input, grad)

    return taylor_backward_hook


#############################################################################


class DrivedDTPMutableChannelImp(DerivedMutable):
    def __init__(
        self,
        choice_fn,
        mask_fn,
        expand_ratio,
        source_mutables=None,
        alias=None,
        init_cfg=None,
    ) -> None:
        super().__init__(choice_fn, mask_fn, source_mutables, alias, init_cfg)
        self.expand_ratio = expand_ratio

    @property
    def current_imp(self):
        mutable = list(self.source_mutables)[0]
        mask = mutable.current_imp
        mask = (
            torch.unsqueeze(mask, -1)
            .expand(list(mask.shape) + [int(self.expand_ratio)])
            .flatten(-2)
        )
        return mask

    @property
    def current_imp_flop(self):
        mutable = list(self.source_mutables)[0]
        mask = mutable.current_imp_flop
        mask = (
            torch.unsqueeze(mask, -1)
            .expand(list(mask.shape) + [int(self.expand_ratio)])
            .flatten(-2)
        )
        return mask


class DMSMutableMixIn:
    def _dms_mutable_mixin_init(self, num_elem, min=0, max=1.0):
        self.mask: torch.Tensor
        self.use_tayler = True

        self.e = nn.parameter.Parameter(torch.tensor([max]), requires_grad=False)
        self.grad_from_task = 0
        self.grad_from_flop = 0

        taylor = torch.zeros([num_elem])
        self.register_buffer("taylor", taylor)
        self.taylor: torch.Tensor

        self.decay = 0.99
        self.lda = 1.0
        self.requires_grad_(False)

        self.grad_scaler = 1

        self.min = min
        self.max = max

        self.taylor_type = "taylor"

        self.normlize = True
        self.estimate = False

        self.imp_cache = None
        self.imp_flop_cache = None

        # self.e_max = 0
        # self.e_min = 1
        # self.tayler_max = 0
        # self.taylor_min = 1

    def reset_saved_grad(self):
        self.grad_from_task = 0
        self.grad_from_flop = 0

    def _current_imp(self, e):
        if self.estimate:
            return dtp_get_importance_estimate(self.taylor, e)
        else:
            if self.taylor.numel() > 1 and self.taylor.max() == self.taylor.min():
                e_imp = torch.ones_like(self.taylor)
                if e.requires_grad:
                    e_imp = e_imp + e * 0
                    e_imp.requires_grad_()
            else:
                e_imp = dtp_get_importance(
                    self.taylor, e, lamda=self.lda, normalize=self.normlize
                )
            return e_imp

    def _current_imp_flop(self, e):
        return self._current_imp(e)

    @property
    def current_imp(self):
        if self.imp_cache is not None:
            return self.imp_cache
        e = self.e + 0

        @torch.no_grad()
        def hook(grad):
            # grad = (grad * 1e-5).abs()
            # self.grad_from_task += grad.detach()
            # self.e_max = max(self.e_max, grad)
            # self.e_min = min(self.e_min, grad)
            # tayler_sort = self.taylor.sort()[0]
            # tayler_step = tayler_sort[1:] - tayler_sort[:-1]
            # self.tayler_max = max(self.tayler_max, tayler_step.max())
            # tayler_step[tayler_step == 0] = 1
            # self.taylor_min = min(self.taylor_min, tayler_step.min())

            # print(
            #     f"g_max: {self.e_max}, g_min: {self.e_min}, t_max: {self.tayler_max}, t_min: {self.taylor_min}"
            # )
            # print(self.taylor.std())
            self.grad_from_task += grad.detach()

            # return torch.clip(grad, -1e-10, 1e-10)

        if e.requires_grad and self.training:
            e.register_hook(hook)
        e_imp = self._current_imp(e)

        if self.training and e.requires_grad and self.use_tayler:
            assert e_imp.requires_grad is True
            e_imp.register_hook(taylor_backward_hook_wrapper(self, e_imp.detach()))
        if self.training:
            with torch.no_grad():
                self.mask.data = (e_imp >= MaskThreshold).float()
        return e_imp

    @property
    def current_imp_flop(self):
        if self.imp_flop_cache is not None:
            return self.imp_flop_cache
        return self._current_imp_flop(self.e_for_flop)

    @property
    def e_for_flop(self):
        e = self.e + 0

        @torch.no_grad()
        def hook(grad):
            self.grad_from_flop += grad.detach()

        if e.requires_grad and self.training:
            e.register_hook(hook)
        return e

    def _limit_value(self, min=0, max=1):
        self.e.data = torch.clamp(self.e, min, max)

    @torch.no_grad()
    def limit_value(self):
        self._limit_value(max(self.min, 1 / self.taylor.numel() / 2), self.max)

    @torch.no_grad()
    def update_taylor(self, input, grad):
        if self.grad_scaler != 0:
            grad = grad.float() / self.grad_scaler

        if self.taylor_type == "taylor":
            new_taylor = (input * grad) ** 2
        elif self.taylor_type == "snip":
            new_taylor = (input * grad).abs()
        elif self.taylor_type == "fisher":
            new_taylor = grad**2
        else:
            raise NotImplementedError()
        all_reduce(new_taylor)
        if (not new_taylor.isnan().any()) and (not new_taylor.isinf().any()):
            if new_taylor.max() != new_taylor.min():
                self.taylor = self.taylor * self.decay + (1 - self.decay) * new_taylor

    @torch.no_grad()
    def sync_mask(self, fix_bug=False):
        e_imp = self._current_imp(self.e)
        idx = torch.topk(
            e_imp, k=int(self.e.item() * self.mask.shape[-1]), largest=True, dim=-1
        )[1]
        self.mask.fill_(0)
        self.mask.data.scatter_(-1, idx, 1.0)
        if self.mask.sum() == 0:
            self.mask.data[self.mask.argmin()] = 1
        if not fix_bug:
            self.mask.data = (e_imp >= MaskThreshold).float()

    def dump_chosen(self):
        pass

    def fix_chosen(self, chosen=None):
        pass

    @property
    def activated_channels(self):
        return self.mask.bool().sum().item()

    def info(self):
        return (
            f"taylor: {self.taylor.min().item():.3f}\t"
            f"{self.taylor.max().item():.3f}\t"
            f"{self.taylor.min()==self.taylor.max()}\t"
            f"e: {self.e.item():.3f}"
        )  # noqa

    def expand_mutable_channel(self, expand_ratio):
        def _expand_mask():
            mask = self.current_mask
            mask = (
                torch.unsqueeze(mask, -1)
                .expand(list(mask.shape) + [expand_ratio])
                .flatten(-2)
            )
            return mask

        return DrivedDTPMutableChannelImp(
            _expand_mask, _expand_mask, expand_ratio, [self]
        )

    @torch.no_grad()
    def to_index_importance(self):
        self.use_tayler = False
        self.taylor.data = 1 - torch.linspace(
            0, 1, self.taylor.numel(), device=self.taylor.device
        )


class DTPTMutableChannelImp(SimpleMutableChannel, DMSMutableMixIn):
    def __init__(self, num_channels: int, min=0, max=1, **kwargs) -> None:
        super().__init__(num_channels, **kwargs)
        self._dms_mutable_mixin_init(self.num_channels, min=min, max=max)

    def fix_chosen(self, chosen=None):
        return super().fix_chosen(chosen)

    def expand_mutable_channel(self, expand_ratio) -> DerivedMutable:
        return DMSMutableMixIn.expand_mutable_channel(self, expand_ratio)

    @torch.no_grad()
    def make_divisible(self, divisor):
        num = self.mask.sum()
        from mmrazor.models.utils import make_divisible

        num = min(make_divisible(num, divisor), self.num_channels)
        _, topk = self.taylor.topk(num, largest=False)
        self.mask.fill_(0)
        self.mask[topk] = 1
        self.e.data.fill_(num / self.num_channels)


class ImpMutableChannelContainer(MutableChannelContainer):
    def __init__(self, num_channels: int, **kwargs):
        super().__init__(num_channels, **kwargs)
        self.register_buffer(
            "_tmp_imp", torch.ones([self.num_channels]), persistent=False
        )
        self._tmp_imp: torch.Tensor

    @property
    def current_imp(self):
        """Get current choices."""
        if len(self.mutable_channels) == 0:
            return self._tmp_imp
        else:
            self._fill_unregistered_range()
            self._assert_mutables_valid()
            mutable_channels = list(self.mutable_channels.values())
            imps = [mutable.current_imp for mutable in mutable_channels]
            if len(imps) == 1:
                return imps[0]
            else:
                imp = torch.cat(imps)
                return imp

    @property
    def current_imp_flop(self):
        """Get current choices."""
        if len(self.mutable_channels) == 0:
            return self._tmp_imp
        else:
            self._fill_unregistered_range()
            self._assert_mutables_valid()
            mutable_channels = list(self.mutable_channels.values())
            imps = [mutable.current_imp_flop for mutable in mutable_channels]
            if len(imps) == 1:
                imp = imps[0]
            else:
                imp = torch.cat(imps)
            return imp


#############################################################################


class MutableBlocks(BaseMutable, DMSMutableMixIn):
    def __init__(self, num_blocks) -> None:
        super().__init__()

        self.num_blocks = num_blocks

        mask = torch.ones([num_blocks])
        self.register_buffer("mask", mask)
        self.mask: torch.Tensor

        self._dms_mutable_mixin_init(num_blocks)

        self.lda = 4.0
        if self.num_blocks == 1:
            self.lda = 8.0
        self.flop_scale_converter = None

    def block_scale_fun_wrapper(self, i):
        def scale():
            scale = self.current_imp[i]
            if self.flop_scale_converter is None:
                return scale
            else:
                return self.flop_scale_converter(scale)

        return scale

    def block_flop_scale_fun_wrapper(self, i):
        def scale():
            return self.current_imp_flop[i]

        return scale

    def info(self):
        def get_mask_str():
            mask_str = ""
            for i in range(self.num_blocks):
                if self.mask[i] == 1:
                    mask_str += "1"
                else:
                    mask_str += "0"
            return mask_str

        return (
            f"mutable_block: {self.num_blocks} \t e: {self.e.item():.3f}, \t"
            f"self.taylor: \t{self.taylor.min().item():.3f}\t{self.taylor.max().item():.3f}\t"  # noqa
            f"mask:\t{get_mask_str()}\t"
        )

    def limit_value(self):
        self._limit_value(self.min, self.max)  # num of blocks can be zero

    # inherit from BaseMutable

    @property
    def current_choice(self):
        return super().current_choice

    def fix_chosen(self, chosen):
        return super().fix_chosen(chosen)

    def dump_chosen(self):
        return super().dump_chosen()

    @torch.no_grad()
    def sync_mask(self, fix_bug=False):
        e_imp = self._current_imp(self.e)
        idx = torch.topk(
            e_imp, k=int(self.e.item() * self.mask.shape[-1]), largest=True, dim=-1
        )[1]
        self.mask.fill_(0)
        self.mask.data.scatter_(-1, idx, 1.0)
        if not fix_bug:
            self.mask.data = (e_imp >= MaskThreshold).float()


class MutableHead(BaseMutable, DMSMutableMixIn):
    def __init__(self, num_heads) -> None:
        super().__init__()
        self.num_heads = num_heads
        self._dms_mutable_mixin_init(num_heads)

        mask = torch.ones([num_heads])
        self.register_buffer("mask", mask)
        self.mask: torch.Tensor
        self.lda = 4.0
        self.flop_scale_converter = None

    @property
    def current_imp(self):
        if self.flop_scale_converter is None:
            return super().current_imp
        else:
            return self.flop_scale_converter(super().current_imp)

    @property
    def current_imp_flop(self):
        if self.flop_scale_converter is None:
            return super().current_imp_flop
        else:
            return self.flop_scale_converter(super().current_imp_flop)

    def dump_chosen(self):
        pass

    def fix_chosen(self, chosen=None):
        pass

    @property
    def current_choice(self):
        return None

    @torch.no_grad()
    def limit_value(self):
        self._limit_value(self.min, self.max)

    def info(self):
        def get_mask_str():
            mask_str = ""
            for i in range(self.num_heads):
                if self.mask[i] == 1:
                    mask_str += "1"
                else:
                    mask_str += "0"
            return mask_str

        return super().info() + f"\t{get_mask_str()}\t"


class MutableChannelForHead(BaseMutable, DMSMutableMixIn):
    def __init__(self, num_channels, num_heads) -> None:
        super().__init__()
        self._dms_mutable_mixin_init(num_channels)
        self.num_head = num_heads
        self.num_channels = num_channels

        self.taylor = self.taylor.reshape([num_heads, -1])

        mask = torch.ones([num_channels])
        self.register_buffer("mask", mask)
        self.mask: torch.Tensor
        self.mask = self.mask.reshape([num_heads, -1])

    def dump_chosen(self):
        pass

    def fix_chosen(self, chosen=None):
        pass

    @property
    def current_choice(self):
        return None


class MutableChannelWithHead(SimpleMutableChannel):
    def __init__(
        self, mutable_head: MutableHead, mutable_channel: MutableChannelForHead
    ) -> None:
        super().__init__(mutable_channel.num_channels)

        self.mutable_head = mutable_head
        self.mutable_channel = mutable_channel

    @property
    def current_imp(self):
        channel_imp = self.mutable_channel.current_imp
        head_imp = self.mutable_head.current_imp
        imp = head_imp.unsqueeze(-1) * channel_imp
        imp = imp.flatten()
        return imp

    @property
    def current_imp_flop(self):
        current_imp_flop = self.mutable_channel.current_imp_flop
        head_imp = self.mutable_head.current_imp_flop
        imp = head_imp.unsqueeze(-1) * current_imp_flop
        imp = imp.flatten()
        return imp

    @property
    def current_mask(self):
        channel = self.mutable_channel.mask
        head = self.mutable_head.mask.unsqueeze(-1)

        return (channel * head).bool().flatten()

    @torch.no_grad()
    def limit_value(self):
        self.mutable_head.limit_value()
        self.mutable_channel.limit_value()


#############################################################################
# for ablation


class PerElementMutableMixin(DMSMutableMixIn):
    def _dms_mutable_mixin_init(self, num_elem, min=0, max=1):
        super()._dms_mutable_mixin_init(num_elem, min=min, max=max)
        self.mask: torch.Tensor

        self.e = nn.parameter.Parameter(torch.zeros([num_elem, 2]), requires_grad=False)
        self.e[:, 0] = 0.01
        self.use_tayler = False

    def _current_imp(self, e):
        if self.training:
            e = gumbel_softmax(e, tau=0.1, hard=True, dim=-1)[:, 0]
        else:
            e = e.softmax(dim=-1)
            e = e[:, 0]
            e = (e > 0.5).float()
        return e

    def _current_imp_flop(self, e):
        e = e.softmax(dim=-1)
        e = e[:, 0]
        e_mask = (e > 0.5).float()
        e = e_mask - e.detach() + e
        return e

    @property
    def current_imp(self):
        e = self.e + 0

        @torch.no_grad()
        def hook(grad):
            self.grad_from_task += grad.detach()

        if e.requires_grad and self.training:
            e.register_hook(hook)
        e_imp = self._current_imp(e)

        if self.training and e.requires_grad and self.use_tayler:
            assert e_imp.requires_grad is True
            e_imp.register_hook(taylor_backward_hook_wrapper(self, e_imp.detach()))
        if self.training:
            with torch.no_grad():
                self.mask.data = (e.softmax(dim=-1)[:, 0] >= MaskThreshold).float()
                if self.mask.sum() == 0:
                    self.mask.data[0] = 1

        return e_imp

    @torch.no_grad()
    def limit_value(self):
        # self.e.data = torch.clamp(self.e, -1, 1)
        pass

    def info(self):
        return (
            f"e_min_max: {self.e.min().item():.3f}\t" f"{self.e.max().item():.3f}\t"
        )  # noqa


class PerElementMutableChannelImp(SimpleMutableChannel, PerElementMutableMixin):
    def __init__(self, num_channels: int, min=0, max=1, **kwargs) -> None:
        super().__init__(num_channels, **kwargs)
        self._dms_mutable_mixin_init(self.num_channels, min=min, max=max)

    def fix_chosen(self, chosen=None):
        return super().fix_chosen(chosen)

    def expand_mutable_channel(self, expand_ratio) -> DerivedMutable:
        return DMSMutableMixIn.expand_mutable_channel(self, expand_ratio)


class PerElementMutableBlocks(BaseMutable, PerElementMutableMixin):
    def __init__(self, num_blocks) -> None:
        super().__init__()

        self.num_blocks = num_blocks

        mask = torch.ones([num_blocks])
        self.register_buffer("mask", mask)
        self.mask: torch.Tensor

        self._dms_mutable_mixin_init(num_blocks)

        self.lda = 4.0
        if self.num_blocks == 1:
            self.lda = 8.0
        self.flop_scale_converter = None

    def block_scale_fun_wrapper(self, i):
        def scale():
            scale = self.current_imp[i]
            if self.flop_scale_converter is None:
                return scale
            else:
                return self.flop_scale_converter(scale)

        return scale

    def block_flop_scale_fun_wrapper(self, i):
        def scale():
            return self.current_imp_flop[i]

        return scale

    def info(self):
        return (
            f"e_min_max: {self.e.min().item():.3f}\t"
            f"{self.e.max().item():.3f}\t"
            + f"e: {self.e.tolist()} mask: {self.mask.tolist()}"
        )  # noqa

    # inherit from BaseMutable

    @property
    def current_choice(self):
        return super().current_choice

    def fix_chosen(self, chosen):
        return super().fix_chosen(chosen)

    def dump_chosen(self):
        return super().dump_chosen()

    def limit_value(self):
        # return super().limit_value()
        pass


################################################################################################
# number selection


class SelectMutableMixin(DMSMutableMixIn):
    def _dms_mutable_mixin_init(self, num_elem, min=0, max=1):
        super()._dms_mutable_mixin_init(num_elem, min=min, max=max)
        self.mask: torch.Tensor

        self.e = nn.parameter.Parameter(torch.zeros([num_elem]), requires_grad=False)
        self.e[0] = 0.1
        self.use_tayler = False

        select_mask = torch.ones([num_elem, num_elem])
        select_mask = torch.triu(select_mask, diagonal=0)
        self.register_buffer("select_mask", select_mask, persistent=False)
        self.select_mask: torch.Tensor

    def _current_imp(self, e):
        if self.training:
            e = gumbel_softmax(e, tau=0.1, hard=True, dim=-1)  # N
        else:
            e = e.softmax(dim=-1)
            max_index = e.argmax()
            e_mask = torch.zeros_like(e)
            e_mask[max_index] = 1
            e = e_mask - e.detach() + e

        mask = self.select_mask * e.unsqueeze(-1)  # N,N
        mask = mask.sum(dim=-2)  # N
        return mask

    def _current_imp_flop(self, e):
        e = e.softmax(dim=-1)
        max_index = e.argmax()
        e_mask = torch.zeros_like(e)
        e_mask[max_index] = 1
        e = e_mask - e.detach() + e
        mask = self.select_mask * e.unsqueeze(-1)  # N,N
        mask = mask.sum(dim=-2)  # N
        return mask

    @property
    def current_imp(self):
        e = self.e + 0

        @torch.no_grad()
        def hook(grad):
            self.grad_from_task += grad.detach()

        if e.requires_grad and self.training:
            e.register_hook(hook)
        e_imp = self._current_imp(e)

        if self.training and e.requires_grad and self.use_tayler:
            assert e_imp.requires_grad is True
            e_imp.register_hook(taylor_backward_hook_wrapper(self, e_imp.detach()))
        if self.training:
            with torch.no_grad():
                self.mask.data = (self._current_imp_flop(e) > MaskThreshold).float()
        return e_imp

    @torch.no_grad()
    def limit_value(self):
        pass

    def info(self):
        return (
            f"e_min_max: {self.e.min().item():.3f}\t" f"{self.e.max().item():.3f}\t"
        )  # noqa


class SelectMutableChannelImp(SimpleMutableChannel, SelectMutableMixin):
    def __init__(self, num_channels: int, min=0, max=1, **kwargs) -> None:
        super().__init__(num_channels, **kwargs)
        self._dms_mutable_mixin_init(self.num_channels, min=min, max=max)

    def fix_chosen(self, chosen=None):
        return super().fix_chosen(chosen)

    def expand_mutable_channel(self, expand_ratio) -> DerivedMutable:
        return DMSMutableMixIn.expand_mutable_channel(self, expand_ratio)


class SelectMutableBlocks(BaseMutable, SelectMutableMixin):
    def __init__(self, num_blocks) -> None:
        super().__init__()

        self.num_blocks = num_blocks

        mask = torch.ones([num_blocks])
        self.register_buffer("mask", mask)
        self.mask: torch.Tensor

        self._dms_mutable_mixin_init(num_blocks)

        self.lda = 4.0
        if self.num_blocks == 1:
            self.lda = 8.0
        self.flop_scale_converter = None

    def block_scale_fun_wrapper(self, i):
        def scale():
            scale = self.current_imp[i]
            if self.flop_scale_converter is None:
                return scale
            else:
                return self.flop_scale_converter(scale)

        return scale

    def block_flop_scale_fun_wrapper(self, i):
        def scale():
            return self.current_imp_flop[i]

        return scale

    def info(self):
        return (
            f"e_min_max: {self.e.min().item():.3f}\t"
            f"{self.e.max().item():.3f}\t"
            + f"e: {self.e.tolist()} mask: {self.mask.tolist()}"
        )  # noqa

    # inherit from BaseMutable

    @property
    def current_choice(self):
        return super().current_choice

    def fix_chosen(self, chosen):
        return super().fix_chosen(chosen)

    def dump_chosen(self):
        return super().dump_chosen()

    def limit_value(self):
        pass
