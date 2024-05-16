# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Iterator, List, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

import mmrazor.models.architectures.dynamic_ops as dynamic_ops
from mmrazor.models.architectures.dynamic_ops import DynamicChannelMixin, DynamicMixin

from .utils import CollectMixin
from .mutable import (
    BlockThreshold,
    ImpMutableChannelContainer,
    MutableBlocks,
    PerElementMutableBlocks,
    SelectMutableBlocks,
)
import copy


def ste_forward(x, mask):
    return x.detach() * mask - x.detach() + x


def soft_ceil(x):
    with torch.no_grad():
        x_ceil = torch.ceil(x.detach().data)
    return x_ceil.detach() - x.detach() + x


class QuickFlopMixin:
    def __init__(self) -> None:
        self._quick_flop_init()

    def _quick_flop_init(self) -> None:
        self.quick_flop_handlers: list = []
        self.quick_flop_recorded_out_shape: List = []
        self.quick_flop_recorded_in_shape: List = []

    def quick_flop_forward_hook_wrapper(self):
        """Wrap the hook used in forward."""

        def forward_hook(module: QuickFlopMixin, input, output):
            module.quick_flop_recorded_out_shape.append(output.shape)
            module.quick_flop_recorded_in_shape.append(input[0].shape)

        return forward_hook

    def quick_flop_start_record(self: torch.nn.Module) -> None:
        """Start recording information during forward and backward."""
        self.quick_flop_end_record()  # ensure to run start_record only once
        self.quick_flop_handlers.append(
            self.register_forward_hook(self.quick_flop_forward_hook_wrapper())
        )

    def quick_flop_end_record(self):
        """Stop recording information during forward and backward."""
        for handle in self.quick_flop_handlers:
            handle.remove()
        self.quick_flop_handlers = []

    def quick_flop_reset_recorded(self):
        """Reset the recorded information."""
        self.quick_flop_recorded_out_shape = []
        self.quick_flop_recorded_in_shape = []

    def soft_flop(self):
        raise NotImplementedError()

    @classmethod
    def get_flop(cls, model: nn.Module):
        flops = 0
        if isinstance(model, QuickFlopMixin):
            return model.soft_flop()
        for child in model.children():
            if isinstance(child, QuickFlopMixin):
                flops = flops + child.soft_flop()
            else:
                flops = flops + cls.get_flop(child)
        return flops


class ImpModuleMixin:
    def __init__(self):
        self._imp_init()

    def _imp_init(self):
        self.ste = False

    @property
    def input_imp(self: Union[DynamicChannelMixin, "ImpModuleMixin"]) -> torch.Tensor:
        mutable: ImpMutableChannelContainer = self.get_mutable_attr(  # type: ignore # noqa
            "in_channels"
        )  # type: ignore
        imp = mutable.current_imp
        return imp

    @property
    def input_imp_flop(
        self: Union[DynamicChannelMixin, "ImpModuleMixin"]
    ) -> torch.Tensor:
        mutable: ImpMutableChannelContainer = self.get_mutable_attr(  # type: ignore # noqa
            "in_channels"
        )  # type: ignore
        imp = mutable.current_imp_flop
        return imp

    @property
    def output_imp(self: nn.Module) -> torch.Tensor:
        mutable: ImpMutableChannelContainer = self.get_mutable_attr("out_channels")
        imp = mutable.current_imp
        return imp

    @property
    def output_imp_flop(self: nn.Module) -> torch.Tensor:
        mutable: ImpMutableChannelContainer = self.get_mutable_attr("out_channels")
        imp = mutable.current_imp_flop
        return imp

    def imp_forward(self, x: torch.Tensor):
        input_imp = self.input_imp
        if self.ste:
            x = ste_forward(x, input_imp)
        else:
            x = x * input_imp
        return x


@torch.jit.script
def soft_mask_sum(mask: torch.Tensor):
    soft = mask.sum()
    hard = (mask >= 0.5).float().sum()
    return hard.detach() - soft.detach() + soft


@torch.jit.script
def conv_soft_flop(
    input_imp_flop, output_imp_flop, h: int, w: int, k1: int, k2: int, groups: int
):
    in_c = soft_mask_sum(input_imp_flop)
    out_c = soft_mask_sum(output_imp_flop)
    conv_per_pos = k1 * k2 * in_c * out_c / groups
    flop = conv_per_pos * h * w
    bias_flop = out_c * h * w
    return flop + bias_flop


class ImpConv2d(
    dynamic_ops.DynamicConv2d, ImpModuleMixin, QuickFlopMixin, CollectMixin
):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._quick_flop_init()
        self._imp_init()
        self._collect_init()

    def forward(self, x: Tensor) -> Tensor:
        x = self.imp_forward(x)
        return nn.Conv2d.forward(self, x)

    def soft_flop(self):
        return conv_soft_flop(
            self.input_imp_flop,
            self.output_imp_flop,
            *self.quick_flop_recorded_out_shape[0][2:],
            self.kernel_size[0],
            self.kernel_size[1],
            self.groups,
        )

    @property
    def input_imp(self) -> Tensor:
        imp = ImpModuleMixin.input_imp.fget(self)  # type: ignore
        return imp.reshape([-1, 1, 1])


class ImpLinear(
    dynamic_ops.DynamicLinear, ImpModuleMixin, QuickFlopMixin, CollectMixin
):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._quick_flop_init()
        self._imp_init()
        self._collect_init()
        self.use_out_imp = False

    def forward(self, x: Tensor) -> Tensor:
        x = self.imp_forward(x)
        x = nn.Linear.forward(self, x)
        if self.use_out_imp:
            x = self.output_imp * x
        return x

    def soft_flop(self):
        in_c = soft_mask_sum(self.input_imp_flop)
        out_c = soft_mask_sum(self.output_imp_flop)
        num = np.prod(self.quick_flop_recorded_in_shape[0][1:-1])
        return in_c * out_c * num

    @property
    def input_imp(self) -> Tensor:
        imp = ImpModuleMixin.input_imp.fget(self)  # type: ignore
        return imp


@torch.jit.script
def bn_soft_flop(input_imp_flop, h: int, w: int):
    in_c = soft_mask_sum(input_imp_flop)
    return h * w * in_c


class ImpBatchnorm2d(dynamic_ops.DynamicBatchNorm2d, QuickFlopMixin):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._quick_flop_init()

    def forward(self, input: Tensor) -> Tensor:
        return nn.BatchNorm2d.forward(self, input)

    def soft_flop(self):
        return bn_soft_flop(
            self.input_imp_flop, *self.quick_flop_recorded_out_shape[0][2:]
        )

    @property
    def output_imp_flop(self: nn.Module) -> torch.Tensor:
        mutable: ImpMutableChannelContainer = self.get_mutable_attr("out_channels")
        imp = mutable.current_imp_flop
        return imp

    @property
    def input_imp_flop(
        self: Union[DynamicChannelMixin, "ImpModuleMixin"]
    ) -> torch.Tensor:
        mutable: ImpMutableChannelContainer = self.get_mutable_attr(  # type: ignore # noqa
            "in_channels"
        )  # type: ignore
        imp = mutable.current_imp_flop
        return imp


@torch.jit.script
def ln_soft_flop(input_imp_flop: torch.Tensor, num: int):
    in_c = soft_mask_sum(input_imp_flop)
    return num * in_c


class ImpLayerNorm(dynamic_ops.DynamicLayerNorm, QuickFlopMixin, ImpModuleMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._quick_flop_init()
        self._imp_init()

    def forward(self, input: Tensor) -> Tensor:
        shape = input.shape
        input = input.reshape([-1, np.prod(self.normalized_shape)])

        input = self.imp_forward(input)
        mutable: ImpMutableChannelContainer = self.get_mutable_attr(  # type: ignore # noqa
            "in_channels"
        )  # type: ignore
        activated = mutable.activated_channels
        means = input.sum(dim=-1, keepdim=True) / activated
        c = input - means
        std = c.pow(2).sum(dim=-1, keepdim=True) / activated
        x = c / torch.sqrt(std + self.eps)
        if self.elementwise_affine:
            x = x * self.weight + self.bias
        x = x.reshape(shape)
        return self.imp_forward(x)

    def soft_flop(self):
        return ln_soft_flop(
            self.input_imp_flop, math.prod(self.quick_flop_recorded_out_shape[1:-1])
        )

    @property
    def input_imp_flop(
        self: Union[DynamicChannelMixin, "ImpModuleMixin"]
    ) -> torch.Tensor:
        mutable: ImpMutableChannelContainer = self.get_mutable_attr(  # type: ignore # noqa
            "in_channels"
        )  # type: ignore
        imp = mutable.current_imp_flop
        return imp


#############################################################################


class DynamicBlockMixin(DynamicMixin, QuickFlopMixin):
    def __init__(self) -> None:
        self._dynamic_block_init()
        self.init_args: list = []
        self.init_kwargs: dict = {}

    def _dynamic_block_init(self):
        self._scale_func = None
        self._flop_scale_func = None
        self._quick_flop_init()

    @property
    def scale(self):
        if self._scale_func is None:
            return 1.0
        else:
            scale: torch.Tensor = self._scale_func()
            if scale.numel() == 1:
                return scale
            else:
                return scale.view([1, -1, 1, 1])

    @property
    def flop_scale(self):
        if self._flop_scale_func is None:
            return 1.0
        else:
            scale: torch.Tensor = self._flop_scale_func()
            return scale

    @property
    def is_removable(self):
        return True

    # inherit from DynamicMixin

    @classmethod
    def convert_from(cls, module):
        pass

    def to_static_op(self) -> nn.Module:
        from mmrazor.structures.subnet.fix_subnet import _dynamic_to_static

        module = self.static_op_factory(*self.init_args, **self.init_kwargs)
        for name, m in self.named_children():  # type: ignore
            assert hasattr(module, name)
            setattr(module, name, _dynamic_to_static(m))
        return module

    def register_mutable_attr(self, attr: str, mutable):
        raise NotImplementedError()

    @property
    def static_op_factory(self):
        raise NotImplementedError()

    def soft_flop(self: nn.Module):
        flops = 0
        for child in self.children():
            flops = flops + QuickFlopMixin.get_flop(child)
        scale = self.flop_scale
        return scale * flops

    @property
    def out_channel(self):
        raise NotImplementedError("unuseful property")

    def __repr__(self) -> str:
        return f"::rm_{self.is_removable}"


class DynamicStage(nn.Sequential, DynamicMixin):
    def __init__(self, *args, mutable_type="dms"):
        super().__init__(*args)
        self.mutable_attrs = {}
        if mutable_type == "dms":
            mutable = MutableBlocks(len(list(self.removable_block)))
        elif mutable_type == "per_elem":
            mutable = PerElementMutableBlocks(len(list(self.removable_block)))
        elif mutable_type == "select":
            mutable = SelectMutableBlocks(len(list(self.removable_block)))
        else:
            raise NotImplementedError()
        self.register_mutable_attr("mutable_blocks", mutable)
        self.mutable_blocks: MutableBlocks

        self.prepare_blocks()

    def prepare_blocks(self):
        for i, block in enumerate(self.removable_block):
            block._scale_func = self.mutable_blocks.block_scale_fun_wrapper(i)
            block._flop_scale_func = self.mutable_blocks.block_flop_scale_fun_wrapper(
                i
            )  # noqa

    @property
    def removable_block(self) -> Iterator[DynamicBlockMixin]:
        for block in self:
            if isinstance(block, DynamicBlockMixin):
                if block.is_removable:
                    yield block

    @property
    def mutable_blocks(self):
        assert "mutable_blocks" in self.mutable_attrs
        return self.mutable_attrs["mutable_blocks"]

    # inherit from DynamicMixin

    @classmethod
    def convert_from(cls, module):
        new_module = cls(module._modules)
        if len(list(new_module.removable_block)) == 0:
            return module
        else:
            return new_module

    def to_static_op(self) -> nn.Module:
        from mmrazor.structures.subnet.fix_subnet import _dynamic_to_static

        modules = []
        i = 0
        for module in self:
            if isinstance(module, DynamicBlockMixin) and module.is_removable:
                if self.mutable_blocks.mask[i] < BlockThreshold:
                    pass
                else:
                    modules.append(module)
                i += 1
            else:
                modules.append(module)

        module = nn.Sequential(*modules)
        return _dynamic_to_static(module)

    def static_op_factory(self):
        return super().static_op_factory

    def register_mutable_attr(self, attr: str, mutable):
        self.mutable_attrs[attr] = mutable

    def expand_module(self, ratio=1.0):
        if ratio > 1:
            num = len(list(self.removable_block))
            new_num = int((ratio - 1) * num)
            new_num = max(1, new_num)
            last_module = self._modules[list(self._modules.keys())[-1]]
            for _ in range(new_num):
                self.append(copy.deepcopy(last_module))
            mask = self.mutable_blocks.mask
            self.mutable_blocks.mask = torch.cat(
                [
                    self.mutable_blocks.mask,
                    torch.ones([new_num], dtype=mask.dtype, device=mask.device),
                ]
            )


class MutableAttn:
    def __init__(self) -> None:
        self.attn_mutables = {"head": None, "qk": None, "v": None}

    def init_mutables(self):
        raise NotImplementedError()
