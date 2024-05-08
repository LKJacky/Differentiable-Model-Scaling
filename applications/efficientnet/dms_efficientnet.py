import torch
from torch import Tensor
from timm.models.efficientnet import efficientnet_b0, EfficientNet
from timm.models._efficientnet_blocks import InvertedResidual, DepthwiseSeparableConv
from timm.layers import BatchNormAct2d
from torch import nn, nn as nn
from torch.nn.modules import Module
from torch.nn.modules.batchnorm import _BatchNorm
from dms.modules.algorithm import DmsGeneralAlgorithm
from dms.modules.op import DynamicBlockMixin
from dms.modules.mutator import DMSMutator
from mmrazor.models.architectures.dynamic_ops import (
    DynamicChannelMixin,
    DynamicBatchNormMixin,
    DynamicLinear,
)
from mmrazor.models.utils.expandable_utils.ops import ExpandableMixin
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath
from timm.scheduler.scheduler import Scheduler
from mmrazor.utils import print_log
import numpy as np
import timm
import copy
import math
from mmrazor.registry import MODELS
from timm.layers.adaptive_avgmax_pool import SelectAdaptivePool2d
from timm.models._efficientnet_blocks import (
    InvertedResidual,
    SqueezeExcite,
    DepthwiseSeparableConv,
)
import json
from dms.modules.op import QuickFlopMixin, ImpConv2d, ImpModuleMixin
from types import MethodType

# OPS ####################################################################################


class DynamicBatchNormAct2d(BatchNormAct2d, DynamicBatchNormMixin):
    def __init__(
        self,
        num_features,
        eps=0.00001,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        apply_act=True,
        act_layer=nn.ReLU,
        act_kwargs=None,
        inplace=True,
        drop_layer=None,
        device=None,
        dtype=None,
    ):
        super().__init__(
            num_features,
            eps,
            momentum,
            affine,
            track_running_stats,
            apply_act,
            act_layer,
            act_kwargs,
            inplace,
            drop_layer,
            device,
            dtype,
        )
        self.mutable_attrs = nn.ModuleDict()

    @property
    def static_op_factory(self):
        """Corresponding Pytorch OP."""
        return BatchNormAct2d

    def to_static_op(self: _BatchNorm) -> BatchNormAct2d:
        running_mean, running_var, weight, bias = self.get_dynamic_params()
        if "num_features" in self.mutable_attrs:
            num_features = self.mutable_attrs["num_features"].current_mask.sum().item()
        else:
            num_features = self.num_features

        static_bn = BatchNormAct2d(
            num_features=num_features,
            eps=self.eps,
            momentum=self.momentum,
            affine=self.affine,
            track_running_stats=self.track_running_stats,
        )
        static_bn.drop = self.drop
        static_bn.act = self.act

        if running_mean is not None:
            static_bn.running_mean.copy_(running_mean)
            static_bn.running_mean = static_bn.running_mean.to(running_mean.device)
        if running_var is not None:
            static_bn.running_var.copy_(running_var)
            static_bn.running_var = static_bn.running_var.to(running_var.device)
        if weight is not None:
            static_bn.weight = nn.Parameter(weight)
        if bias is not None:
            static_bn.bias = nn.Parameter(bias)

        return static_bn

    @classmethod
    def convert_from(cls, module: BatchNormAct2d):
        new_module = cls(
            num_features=module.num_features,
            eps=module.eps,
            momentum=module.momentum,
            affine=module.affine,
            track_running_stats=module.track_running_stats,
        )
        new_module.act = module.act
        new_module.drop = module.drop
        new_module.load_state_dict(module.state_dict())
        return new_module

    def forward(self, x):
        running_mean, running_var, weight, bias = self.get_dynamic_params()

        # cut & paste of torch.nn.BatchNorm2d.forward impl to avoid issues with torchscript and tracing
        # _assert(x.ndim == 4, f'expected 4D input (got {x.ndim}D input)')

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked.add_(1)  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum
        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)
        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        x = F.batch_norm(
            x,
            # If buffers are not to be tracked, ensure that they won't be updated
            running_mean if not self.training or self.track_running_stats else None,
            running_var if not self.training or self.track_running_stats else None,
            weight,
            bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        )
        x = self.drop(x)
        x = self.act(x)
        return x


class ImpBatchNormAct2d(DynamicBatchNormAct2d):
    def forward(self, x):
        y = BatchNormAct2d.forward(self, x)
        return y


class ExpandableBatchNormAct2d(DynamicBatchNormAct2d, ExpandableMixin):
    @property
    def _original_in_channel(self):
        return self.num_features

    @property
    def _original_out_channel(self):
        return self.num_features

    def get_expand_op(self, in_c, out_c, zero=False):
        assert in_c == out_c
        module = BatchNormAct2d(
            in_c,
            eps=self.eps,
            momentum=self.momentum,
            affine=self.affine,
            track_running_stats=self.track_running_stats,
        )
        module.drop = self.drop
        module.act = self.act
        if zero:
            ExpandableMixin.zero_weight_(module)

        if module.running_mean is not None:
            module.running_mean.data = self.expand_bias(
                module.running_mean, self.running_mean
            )

        if module.running_var is not None:
            module.running_var.data = self.expand_bias(
                module.running_var, self.running_var
            )
        module.weight.data = self.expand_bias(module.weight, self.weight)
        module.bias.data = self.expand_bias(module.bias, self.bias)
        return module


# blocks ####################################################################################


class DynamicInvertedResidual(InvertedResidual, DynamicBlockMixin):
    def __init__(self, *args, **kwargs):
        InvertedResidual.__init__(self, *args, **kwargs)
        DynamicBlockMixin.__init__(self)
        self.init_args = args
        self.init_kwargs = kwargs

    @property
    def is_removable(self):
        return self.has_skip

    @classmethod
    def convert_from(cls, module: InvertedResidual):
        static = cls(
            in_chs=module.conv_pw.in_channels,
            out_chs=module.conv_pwl.out_channels,
            dw_kernel_size=module.conv_dw.kernel_size[0],
            stride=module.conv_dw.stride,
            dilation=module.conv_dw.dilation,
            pad_type=module.conv_dw.padding,
        )
        static.has_skip = module.has_skip
        static.conv_pw = module.conv_pw
        static.bn1 = module.bn1
        static.conv_dw = module.conv_dw
        static.bn2 = module.bn2
        static.se = module.se
        static.conv_pwl = module.conv_pwl
        static.bn3 = module.bn3
        static.drop_path = module.drop_path
        static.load_state_dict(module.state_dict())
        return static

    @property
    def static_op_factory(self):
        return InvertedResidual

    def forward(self, x):
        shortcut = x
        x = self.conv_pw(x)
        x = self.bn1(x)
        x = self.conv_dw(x)
        x = self.bn2(x)
        x = self.se(x)
        x = self.conv_pwl(x)
        x = self.bn3(x)
        if self.has_skip:
            if hasattr(self, "add"):
                x = self.add(self.drop_path(x) * self.scale, shortcut)
            else:
                x = self.drop_path(x) * self.scale + shortcut
        return x

    def to_static_op(self):
        static = InvertedResidual(
            in_chs=self.conv_pw.in_channels,
            out_chs=self.conv_pwl.out_channels,
            dw_kernel_size=self.conv_dw.kernel_size[0],
            stride=self.conv_dw.stride,
            dilation=self.conv_dw.dilation,
            pad_type=self.conv_dw.padding,
        )
        static.has_skip = self.has_skip
        static.conv_pw = self.conv_pw
        static.bn1 = self.bn1
        static.conv_dw = self.conv_dw
        static.bn2 = self.bn2
        static.se = self.se
        static.conv_pwl = self.conv_pwl
        static.bn3 = self.bn3
        static.drop_path = self.drop_path
        static.load_state_dict(self.state_dict())
        from mmrazor.structures.subnet.fix_subnet import _dynamic_to_static

        module = static
        for name, m in self.named_children():  # type: ignore
            assert hasattr(module, name)
            setattr(module, name, _dynamic_to_static(m))
        return module

    def set_shape(self):
        self.bn1.act.in_shape = self.conv_pw.out_shape
        self.bn1.act.out_shape = self.conv_pw.out_shape

        self.bn2.act.in_shape = self.conv_dw.out_shape
        self.bn2.act.out_shape = self.conv_dw.out_shape

        self.se.act1.in_shape = self.se.conv_reduce.out_shape
        self.se.act1.out_shape = self.se.conv_reduce.out_shape

        # sigmoid
        self.se.gate.in_shape = self.se.conv_expand.out_shape
        self.se.gate.out_shape = self.se.conv_expand.out_shape

        # add

        if self.has_skip:
            self.add.in_shape = self.in_shape
            self.add.out_shape = self.in_shape

        # mul
        self.se.mutl.in_shape = self.conv_dw.out_shape
        self.se.mutl.out_shape = self.conv_dw.out_shape


class EffStage(nn.Sequential):
    @classmethod
    def convert_from(cls, module: nn.Sequential):
        return cls(module._modules)

    # Latency ####################################################################################


def soft_latency_wrapper_for_block(predictor: "Predictor"):

    def soft_latency(self: InvertedResidual):
        flops = 0
        for child in self.children():
            flops = flops + QuickFlopMixin.get_flop(child)
        # silu
        numc = self.conv_pw.output_imp_flop.sum()
        flops = flops + predictor.get_net(self.bn1.act)(numc, numc)
        numc = self.conv_dw.output_imp_flop.sum()
        flops = flops + predictor.get_net(self.bn2.act)(numc, numc)
        numc = self.se.conv_reduce.output_imp_flop.sum()
        flops = flops + predictor.get_net(self.se.act1)(numc, numc)
        # sigmoid
        numc = self.se.conv_expand.output_imp_flop.sum()
        flops = flops + predictor.get_net(self.se.gate)(numc, numc)
        # add

        if self.has_skip:
            numc = self.conv_dw.output_imp_flop.sum()
            flops = flops + predictor.get_net(self.add)(numc, numc)
        # mult
        numc = self.conv_dw.output_imp_flop.sum()
        flops = flops + predictor.get_net(self.se.mutl)(numc, numc)

        scale = self.flop_scale
        return scale * flops

    return soft_latency


def soft_latency_wrapper(predictor: "Predictor"):

    def soft_latency(self: nn.Module):
        flops = 0
        if isBasicModule(self) and hasattr(self, "quick_flop_recorded_in_shape"):
            return predictor.compute_latency(self)

        else:
            for child in self.children():
                flops = flops + QuickFlopMixin.get_flop(child)

            if isinstance(self, DynamicBlockMixin):
                return flops * self.flop_scale
            else:
                return flops

    return soft_latency


def isBasicModule(m: nn.Module):
    basic_modules = [
        nn.Conv2d,
        nn.Linear,
        nn.modules.activation.SiLU,
        nn.modules.activation.Sigmoid,
        SelectAdaptivePool2d,
        AddModule,
        MultModule,
    ]
    for t in basic_modules:
        if isinstance(m, t):
            return True
    return False


class Predictor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.table = json.load(open("l.json"))
        self.build_net()
        self.requires_grad_(False)

    @classmethod
    def parse_m(self, m):
        args = self.parse_m_name(m)
        inc, outc = m.in_shape[1], m.out_shape[1]
        return args, inc, outc

    @classmethod
    def parse_m_name(self, m):
        if isinstance(m, nn.Conv2d):
            is_depth_wise = 1 if m.groups == m.in_channels else 0
            has_bias = 1 if m.bias is not None else 0
            k = m.kernel_size[0]

            args = f"conv2d_k{m.kernel_size[0]}_s{m.stride[0]}_p{m.padding[0]}_r{m.in_shape[-1]}_b{has_bias}_dw{is_depth_wise}"
            if k == 1:
                args += f"_ex{1 if m.in_channels<m.out_channels else 0}"
        elif isinstance(m, nn.Linear):
            args = f"linear"
        else:
            args_map = {
                nn.modules.activation.SiLU: "silu",
                nn.modules.activation.Sigmoid: "sigmoid",
                SelectAdaptivePool2d: "pool",
                AddModule: "add",
                MultModule: "mult",
            }
            r = m.in_shape[-1]
            args = f"{args_map[type(m)]}_r{r}"
        return args

    def build_net(self):
        self.nets = nn.ModuleDict()
        for args in self.table:
            config = self.table[args]
            records = config["records"]
            if (
                args.endswith("dw1")
                or args.startswith("mult")
                or args.startswith("add")
                or args.startswith("silu")
                or args.startswith("pool")
                or args.startswith("sigmoid")
            ):
                records = [(a, 1, c) for a, b, c in records]
            net = NoTrainNet.build_net(records)
            self.nets[args] = net

    @torch.no_grad()
    def predict(self, model):
        ls = []
        for m in model.modules():
            if isBasicModule(m) and hasattr(m, "in_shape"):
                ls.append(self.predict_m(m))
        return sum(ls)

    @torch.no_grad()
    def predict_m(self, m):
        args, inc, outc = self.parse_m(m)
        net = self.nets[args]
        if (
            args.endswith("dw1")
            or args.startswith("mult")
            or args.startswith("add")
            or args.startswith("silu")
            or args.startswith("pool")
            or args.startswith("sigmoid")
        ):
            outc = 1
        return net(torch.tensor(inc), torch.tensor(outc))

    def compute_latency(self, m: ImpModuleMixin):
        name = self.parse_m_name(m)
        inc = m.input_imp_flop.sum()
        outc = m.output_imp_flop.sum()
        return self.nets[name](inc, outc)

    def get_net(self, m):
        name = self.parse_m_name(m)
        return self.nets[name]


class NoTrainNet(nn.Module):
    def __init__(self, grid, max_x, max_y) -> None:
        super().__init__()
        self.register_buffer("grid", grid)
        self.grid: Tensor
        self.max_x = (max_x).item()
        self.max_y = (max_y).item()

        self.n_x = self.grid.shape[-2]
        self.n_y = self.grid.shape[-1]

        self.x_step = self.max_x // self.n_x
        self.y_step = self.max_y // self.n_y

        assert self.x_step * self.n_x == self.max_x
        assert self.y_step * self.n_y == self.max_y

    @classmethod
    def get_range(self, x, step):
        with torch.no_grad():
            x0 = torch.floor(x).long().clamp_min_(0)
            x1 = torch.ceil(x).long().clamp_min_(0)
            if x0 == x1:
                if x0 == 0:
                    x1 += 1
                else:
                    x0 -= 1
        return x0, x1

    def forward_1d(self, x, y):
        with torch.no_grad():
            if self.n_y == 1:
                x = x
                step = self.x_step
            else:
                x = y
                step = self.y_step
            grid = self.grid.flatten()
        x = x / step - 1
        with torch.no_grad():
            x0, x1 = self.get_range(x, step)

            v1, v2 = (
                grid[x0],
                grid[x1],
            )

        i1 = (x1 - x) * v1 + (x - x0) * v2
        return i1

    def forward_2d(self, x, y):

        x = x / self.x_step - 1
        y = y / self.y_step - 1

        with torch.no_grad():
            x0, x1 = self.get_range(x, self.x_step)
            y0, y1 = self.get_range(y, self.y_step)

            v1, v2, v3, v4 = (
                self.grid[x0, y0],
                self.grid[x0, y1],
                self.grid[x1, y0],
                self.grid[x1, y1],
            )

        i1 = (x - x0) * v3 + (x1 - x) * v1
        i2 = (x - x0) * v4 + (x1 - x) * v2
        i3 = (y - y0) * i2 + (y1 - y) * i1
        return i3

    def forward(self, x, y):
        if self.n_x == 1 or self.n_y == 1:
            return self.forward_1d(x, y)
        else:
            return self.forward_2d(x, y)

    @classmethod
    def build_net(cls, records):
        data = torch.tensor(records)
        inc, outc, l = data.T

        # build
        if l.std() < 1e-3:
            return ConstantNet(l.mean())
        else:
            inc, outc = inc.long(), outc.long()
            num_in = len(set(inc.tolist()))
            num_out = len(set(outc.tolist()))
            grid = torch.zeros([num_in, num_out])

            step_in = inc.max() // num_in
            step_out = outc.max() // num_out

            for in_, out_, l_ in zip(inc, outc, l):
                grid[in_ // step_in - 1, out_ // step_out - 1] = l_
            return NoTrainNet(grid, inc.max(), outc.max())


class ConstantNet(nn.Module):
    def __init__(self, c) -> None:
        super().__init__()
        self.register_buffer("c", c)
        self.c: Tensor

    def forward(self, x, y):
        return self.c


class AddModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x1, x2):
        return x1 + x2


def replaece_InvertedResidual_forward():
    def forward(self, x):
        shortcut = x
        x = self.conv_pw(x)
        x = self.bn1(x)
        x = self.conv_dw(x)
        x = self.bn2(x)
        x = self.se(x)
        x = self.conv_pwl(x)
        x = self.bn3(x)
        if self.has_skip:
            x = self.add(self.drop_path(x), shortcut)
        return x

    def init_wrapper(origin):
        def init(self, *args, **kwargs):
            origin(self, *args, **kwargs)
            self.add = AddModule()

        return init

    InvertedResidual.forward = forward
    InvertedResidual.__init__ = init_wrapper(InvertedResidual.__init__)

    def forward2(self, x):
        shortcut = x
        x = self.conv_dw(x)
        x = self.bn1(x)
        x = self.se(x)
        x = self.conv_pw(x)
        x = self.bn2(x)
        if self.has_skip:
            x = self.add(self.drop_path(x), shortcut)
        return x

    DepthwiseSeparableConv.forward = forward2
    DepthwiseSeparableConv.__init__ = init_wrapper(DepthwiseSeparableConv.__init__)


class MultModule(nn.Module):

    def forward(self, x1, x2):
        return x1 * x2


def replaece_Mult_forward():
    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return self.mutl(x, self.gate(x_se))

    def init_wrapper(origin):
        def init(self, *args, **kwargs):
            origin(self, *args, **kwargs)
            self.mutl = MultModule()

        return init

    SqueezeExcite.forward = forward
    SqueezeExcite.__init__ = init_wrapper(SqueezeExcite.__init__)


# Algo ####################################################################################


@MODELS.register_module()
class EffLatencyMutator(DMSMutator):
    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.latency_model = Predictor()
        self.latency_model.requires_grad_(False)

    @torch.no_grad()
    def init_quick_flop(self, model: Module):
        super().init_quick_flop(model)
        self.set_shape(model)
        for m in model.modules():
            if isinstance(m, DynamicInvertedResidual):
                m.soft_flop = MethodType(
                    soft_latency_wrapper_for_block(self.latency_model), m
                )
            elif isinstance(m, QuickFlopMixin):
                m.soft_flop = MethodType(soft_latency_wrapper(self.latency_model), m)

    def set_shape(self, supernet: EfficientNet):
        for m in supernet.modules():
            if hasattr(m, "quick_flop_recorded_in_shape"):
                m.in_shape = list(m.quick_flop_recorded_in_shape[0])
                m.out_shape = list(m.quick_flop_recorded_out_shape[0])
        for m in supernet.modules():
            if isinstance(m, InvertedResidual):
                m.set_shape()
        supernet.bn2.act.in_shape = supernet.conv_head.out_shape
        supernet.bn2.act.out_shape = supernet.conv_head.out_shape

        supernet.bn1.act.in_shape = supernet.conv_stem.out_shape
        supernet.bn1.act.out_shape = supernet.conv_stem.out_shape

    def get_soft_flop(self, model):
        flops = super().get_soft_flop(model)
        numc = model.conv_stem.output_imp_flop.sum()
        flops = flops + self.latency_model.get_net(model.bn1.act)(numc, numc)
        numc = model.conv_head.output_imp_flop.sum()
        flops = flops + self.latency_model.get_net(model.bn2.act)(numc, numc)
        self.last_soft_flop = flops.detach()
        return flops


class EffDmsAlgorithm(DmsGeneralAlgorithm):
    default_mutator_kwargs = dict(
        prune_qkv=False,
        prune_block=True,
        dtp_mutator_cfg=dict(
            type="DTPAMutator",
            channel_unit_cfg=dict(
                type="DTPTUnit",
                default_args=dict(extra_mapping={BatchNormAct2d: ImpBatchNormAct2d}),
            ),
            parse_cfg=dict(
                _scope_="mmrazor",
                type="ChannelAnalyzer",
                demo_input=dict(
                    type="DefaultDemoInput",
                    input_shape=(1, 3, 224, 224),
                ),
                tracer_type="FxTracer",
                extra_mapping={BatchNormAct2d: DynamicBatchNormAct2d},
            ),
        ),
        extra_module_mapping={},
        block_initilizer_kwargs=dict(
            stage_mixin_layers=[EffStage],
            dynamic_block_mapping={InvertedResidual: DynamicInvertedResidual},
        ),
    )
    default_scheduler_kargs = dict(
        flops_target=0.8,
        decay_ratio=0.8,
        refine_ratio=0.2,
        flop_loss_weight=1000,
        structure_log_interval=1000,
        by_epoch=True,
        target_scheduler="cos",
    )

    expand_unit_config = dict(
        type="ExpandableUnit",
        default_args=dict(extra_mapping={BatchNormAct2d: ExpandableBatchNormAct2d}),
    )

    def __init__(
        self, model: EfficientNet, mutator_kwargs={}, scheduler_kargs={}
    ) -> None:
        # nn.Sequential -> EffStage
        new_seq = nn.Sequential()
        for name, block in model.blocks.named_children():
            new_seq.add_module(name, EffStage.convert_from(block))
        model.blocks = new_seq

        super().__init__(
            model,
            mutator_kwargs=mutator_kwargs,
            scheduler_kargs=scheduler_kargs,
        )

    def to_static_model(self, drop_path=-1, drop=-1, scale=False, divisor=1):
        model: EfficientNet = super().to_static_model(scale=scale, divisor=divisor)
        if drop_path != -1:
            num_blocks = sum([len(stage) for stage in model.blocks])
            i = 0
            for stage in model.blocks:
                for block in stage:
                    drop_path_rate = drop_path * i / num_blocks
                    block.drop_path = (
                        DropPath(drop_path_rate)
                        if drop_path_rate != 0
                        else nn.Identity()
                    )
                    i += 1
            assert i == num_blocks
        if drop != -1:
            model.drop_rate = drop
        return model

    def get_fold_config(self, max_divisor=8):
        def make_divisor(n):
            import math

            for d in [8, 4, 2]:
                if d < math.sqrt(n):
                    if n % d == 0:
                        return n // d
            return n

        config = self.mutator.dtp_mutator.config_template(
            with_unit_init_args=True, with_channels=True
        )["channel_unit_cfg"]["units"]
        for value in config.values():
            value["init_args"]["num_channels"] = make_divisor(
                value["init_args"]["num_channels"]
            )
            value.pop("choice")
        return config

    @classmethod
    def init_fold_algo(cls, model, mutator_kwargs={}, scheduler_kargs={}):
        mutator_kwargs = copy.deepcopy(mutator_kwargs)
        save_model = copy.deepcopy(model)
        algo = cls(
            model,
        )
        unit_config = algo.get_fold_config()
        mutator_kwargs.update(
            dict(
                dtp_mutator_cfg=dict(
                    channel_unit_cfg=dict(
                        type="DTPTUnit",
                        units=unit_config,
                    ),
                    parse_cfg=dict(
                        _scope_="mmrazor",
                        type="Config",
                    ),
                ),
            )
        )
        new_algo = cls(
            save_model, mutator_kwargs=mutator_kwargs, scheduler_kargs=scheduler_kargs
        )
        return new_algo


class MyScheduler(Scheduler):
    def __init__(
        self,
        optimizer,
        num_epoch=10,
        cycle_epoch=5,
        decay=0.5,
        warmup_t=0,
        warmup_lr_init=0,
        t_in_epochs: bool = True,
        noise_range_t=None,
        noise_type="normal",
        noise_pct=0.67,
        noise_std=1,
        noise_seed=None,
        initialize: bool = True,
    ) -> None:
        super().__init__(
            optimizer,
            "lr",
            t_in_epochs,
            noise_range_t,
            noise_type,
            noise_pct,
            noise_std,
            noise_seed,
            initialize,
        )

        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init

        if self.warmup_t:
            self.warmup_steps = [
                (v - warmup_lr_init) / self.warmup_t for v in self.base_values
            ]
            init_lr = [self.warmup_lr_init] * len(self.base_values)
            init_lr[-1] = self.base_values[-1]
            super().update_groups(init_lr)
        else:
            self.warmup_steps = [1 for _ in self.base_values]

        self.num_epoch = num_epoch
        self.cycle_epoch = cycle_epoch
        self.decay = decay

    def _get_lr(self, t: int) -> float:
        if t < self.warmup_t:
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            t = t - self.warmup_t
            t = t % self.cycle_epoch
            lrs = [lr * (self.decay**t) for lr in self.base_values]
        lrs[-1] = self.base_values[-1]
        return lrs


# zico distill


def extract_stage_features_and_logit(self: EfficientNet, x, target_downsample_ratio=16):
    image_size = x.shape[2]

    stage_features_list = []
    x = self.conv_stem(x)
    x = self.bn1(x)
    for block in self.blocks:
        x = block(x)
        dowsample_ratio = round(image_size / x.shape[2])
        if dowsample_ratio == target_downsample_ratio:
            stage_features_list.append(x)
            target_downsample_ratio *= 2

    x = self.conv_head(x)
    x = self.bn2(x)

    x = self.global_pool(x)
    if self.drop_rate > 0.0:
        x = F.dropout(x, p=self.drop_rate, training=self.training)
    x = self.classifier(x)

    return stage_features_list, x


EfficientNet.extract_stage_features_and_logit = extract_stage_features_and_logit


def network_weight_zero_init(net: nn.Module):
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                device = m.weight.device
                in_channels, out_channels, k1, k2 = m.weight.shape
                m.weight[:] = (
                    torch.randn(m.weight.shape, device=device)
                    / np.sqrt(k1 * k2 * in_channels)
                    * 1e-4
                )
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                device = m.weight.device
                in_channels, out_channels = m.weight.shape
                m.weight[:] = (
                    torch.randn(m.weight.shape, device=device)
                    / np.sqrt(in_channels)
                    * 1e-4
                )
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)
            else:
                continue

    return net


class TeacherStudentDistillNet(nn.Module):
    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        teacher_input_image_size=224,
        input_image_size=224,
        ts_proj_no_bn=True,
        ts_proj_no_relu=True,
        ts_clip=None,
        target_downsample_ratio=16,
    ):
        self.teacher_input_image_size = teacher_input_image_size
        self.input_image_size = input_image_size
        self.ts_clip = ts_clip

        super(TeacherStudentDistillNet, self).__init__()
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.target_downsample_ratio = target_downsample_ratio

        assert hasattr(self.teacher_model, "extract_stage_features_and_logit")
        assert hasattr(self.student_model, "extract_stage_features_and_logit")

        self.teacher_model.eval()
        self.teacher_model.requires_grad_(False)
        self.student_model.eval()

        # create project layer
        test_img = torch.randn(
            (1, 3, teacher_input_image_size, teacher_input_image_size)
        )
        (
            teacher_stage_features_list,
            teacher_logit,
        ) = self.teacher_model.extract_stage_features_and_logit(
            test_img, target_downsample_ratio=target_downsample_ratio
        )
        test_img = torch.randn((1, 3, input_image_size, input_image_size))
        (
            student_stage_features_list,
            student_logit,
        ) = self.student_model.extract_stage_features_and_logit(
            test_img, target_downsample_ratio=target_downsample_ratio
        )

        assert len(teacher_stage_features_list) == len(
            student_stage_features_list
        ), f"{len(teacher_stage_features_list)} != {len(student_stage_features_list)}"

        self.proj_conv_list = nn.ModuleList()
        for tf, sf in zip(teacher_stage_features_list, student_stage_features_list):
            proj_conv_seq_blocks = [
                nn.Conv2d(sf.shape[1], tf.shape[1], kernel_size=1, stride=1)
            ]

            if not ts_proj_no_bn:
                proj_conv_seq_blocks.append(nn.BatchNorm2d(tf.shape[1]))
            else:
                print("--- use ts_proj_no_bn")

            if not ts_proj_no_relu:
                proj_conv_seq_blocks.append(nn.ReLU(tf.shape[1]))
            else:
                print("--- use ts_proj_no_relu")

            proj_conv = nn.Sequential(*proj_conv_seq_blocks)
            self.proj_conv_list.append(proj_conv)

        self.teacher_stage_features_list = None
        self.teacher_logit = None
        self.student_stage_features_list = None
        self.student_logit = None

        # default initialize
        self.init_parameters()

        # # bn eps
        # for layer in self.student_model.modules():
        #     if isinstance(layer, nn.BatchNorm2d):
        #         layer.eps = 1e-3
        # for block in self.proj_conv_list:
        #     for layer in block.modules():
        #         if isinstance(layer, nn.BatchNorm2d):
        #             layer.eps = 1e-3

    def train(self, mode=True):
        self.training = True
        self.student_model.train(mode)
        self.teacher_model.eval()

    def eval(self):
        self.training = False
        self.student_model.eval()
        self.teacher_model.eval()

    def forward(self, x):
        if self.training:
            if x.shape[2] != self.teacher_input_image_size:
                teacher_x = F.interpolate(
                    x, self.teacher_input_image_size, mode="bilinear"
                )
            else:
                teacher_x = x
            (
                self.teacher_stage_features_list,
                self.teacher_logit,
            ) = self.teacher_model.extract_stage_features_and_logit(
                teacher_x, target_downsample_ratio=self.target_downsample_ratio
            )

        if x.shape[2] != self.input_image_size:
            student_x = F.interpolate(x, self.input_image_size, mode="bilinear")
        else:
            student_x = x
        (
            self.student_stage_features_list,
            self.student_logit,
        ) = self.student_model.extract_stage_features_and_logit(
            student_x, target_downsample_ratio=self.target_downsample_ratio
        )

        return self.student_logit

    def compute_ts_distill_loss(self):
        def ts_feature_map_loss(x, y):
            return torch.nn.functional.smooth_l1_loss(x, y)

        feature_loss = 0.0
        for tf, sf, proj_conv in zip(
            self.teacher_stage_features_list,
            self.student_stage_features_list,
            self.proj_conv_list,
        ):
            if self.ts_clip is not None:
                tf = torch.clamp(tf, -1 * self.ts_clip, self.ts_clip)

            if tf.shape[2] != sf.shape[2]:
                tf = F.interpolate(tf, sf.shape[2], mode="bilinear")

            proj_sf = proj_conv(sf)
            # proj_tf = proj_conv(tf)

            # feature_loss = feature_loss + hierarchical_loss(proj_sf, tf, alpha=0.5, k=2)
            feature_loss = feature_loss + ts_feature_map_loss(proj_sf, tf)
            # feature_loss = feature_loss + hierarchical_loss(sf, proj_tf, alpha=alpha, k=k)
        pass

        prob_logit = F.log_softmax(self.student_logit, dim=1)
        target = F.softmax(self.teacher_logit, dim=1)
        logit_loss = -(target * prob_logit).sum(dim=1).mean()

        return feature_loss, logit_loss

    def init_parameters(self):
        for block in self.proj_conv_list:
            network_weight_zero_init(block)


# DIST distill
# Copyright (c) OpenMMLab. All rights reserved.


def cosine_similarity(a, b, eps=1e-8):
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + eps)


def pearson_correlation(a, b, eps=1e-8):
    return cosine_similarity(
        a - a.mean(1, keepdim=True), b - b.mean(1, keepdim=True), eps
    )


def inter_class_relation(y_s, y_t):
    return 1 - pearson_correlation(y_s, y_t).mean()


def intra_class_relation(y_s, y_t):
    return inter_class_relation(y_s.transpose(0, 1), y_t.transpose(0, 1))


class DISTLoss(nn.Module):
    def __init__(
        self,
        inter_loss_weight=1.0,
        intra_loss_weight=1.0,
        tau=1.0,
        loss_weight: float = 2.0,
        teacher_detach: bool = True,
    ):
        super(DISTLoss, self).__init__()
        self.inter_loss_weight = inter_loss_weight
        self.intra_loss_weight = intra_loss_weight
        self.tau = tau

        self.loss_weight = loss_weight
        self.teacher_detach = teacher_detach

    def forward(self, logits_S, logits_T: torch.Tensor):
        if self.teacher_detach:
            logits_T = logits_T.detach()
        y_s = (logits_S / self.tau).softmax(dim=1)
        y_t = (logits_T / self.tau).softmax(dim=1)
        inter_loss = self.tau**2 * inter_class_relation(y_s, y_t)
        intra_loss = self.tau**2 * intra_class_relation(y_s, y_t)
        kd_loss = (
            self.inter_loss_weight * inter_loss + self.intra_loss_weight * intra_loss
        )  # noqa
        return kd_loss * self.loss_weight


class TeacherStudentDistillNetDIST(TeacherStudentDistillNet):
    def __init__(
        self,
        teacher_model: Module,
        student_model: Module,
        teacher_input_image_size=224,
        input_image_size=224,
        loss_weight=2.0,
    ):
        super().__init__(
            teacher_model,
            student_model,
            teacher_input_image_size,
            input_image_size,
            ts_proj_no_bn=True,
            ts_proj_no_relu=True,
            ts_clip=None,
            target_downsample_ratio=48,
        )
        self.dist_loss = DISTLoss(loss_weight=loss_weight)

    def compute_ts_distill_loss(self):
        return self.dist_loss(self.student_logit, self.teacher_logit)


if __name__ == "__main__":
    model = efficientnet_b0(drop_path_rate=0.3, drop_rate=0.2)
    print(model)

    algo = EffDmsAlgorithm(
        model,
        mutator_kwargs=dict(
            dtp_mutator_cfg=dict(
                parse_cfg=dict(
                    demo_input=dict(
                        input_shape=(1, 3, 240, 240),
                    ),
                )
            ),
        ),
    )
    print(algo.mutator.info())

    model = algo.to_static_model(drop_path=0.5, drop=0.4)
    EffDmsAlgorithm.show_structure(model)
    model = algo.expand_model(model, channel_ratio=1.5, block_ratio=2.0)
    print(model)
    EffDmsAlgorithm.show_structure(model)

    a = torch.rand(2, 3, 224, 224)
    model.drop_rate = 0
    res = model.extract_stage_features_and_logit(a)
    for t in res[0]:
        print(t.shape)

    logits = model(a)
    print(logits)
    print(logits.shape)
    print((logits - res[-1]).abs().max())
    assert (logits == res[-1]).all()

    teacher = timm.create_model("efficientnet_b0", pretrained=True)

    distill_model = TeacherStudentDistillNet(
        teacher, model, teacher_input_image_size=288, input_image_size=224
    )
    distill_model(a)
    distill_model.compute_ts_distill_loss()

    distill_model = TeacherStudentDistillNetDIST(
        teacher, model, teacher_input_image_size=288, input_image_size=224
    )
    distill_model(a)
    distill_model.compute_ts_distill_loss()
