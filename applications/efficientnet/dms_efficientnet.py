import torch
from timm.models.efficientnet import efficientnet_b0, EfficientNet
from timm.models._efficientnet_blocks import InvertedResidual
from timm.layers import BatchNormAct2d
from torch import nn, nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
from dms.modules.algorithm import DmsGeneralAlgorithm
from dms.modules.op import DynamicBlockMixin
from mmrazor.models.architectures.dynamic_ops import (
    DynamicBatchNormMixin,
)
from mmrazor.models.utils.expandable_utils.ops import ExpandableMixin
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath
from timm.scheduler.scheduler import Scheduler
import numpy as np
import timm
import copy
from timm.models._efficientnet_blocks import (
    InvertedResidual,
)

# Operations ####################################################################################


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
            num_features = self.mutable_attrs["num_features"].current_mask.sum(
            ).item()
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
            static_bn.running_mean = static_bn.running_mean.to(
                running_mean.device)
        if running_var is not None:
            static_bn.running_var.copy_(running_var)
            static_bn.running_var = static_bn.running_var.to(
                running_var.device)
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
                    exponential_average_factor = 1.0 / \
                        float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum
        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (
                self.running_var is None)
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


# Algo ####################################################################################


class EffDmsAlgorithm(DmsGeneralAlgorithm):
    default_mutator_kwargs = dict(
        prune_qkv=False,
        prune_block=True,
        dtp_mutator_cfg=dict(
            type="DTPAMutator",
            channel_unit_cfg=dict(
                type="DTPTUnit",
                default_args=dict(
                    extra_mapping={BatchNormAct2d: ImpBatchNormAct2d}),
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
    )

    expand_unit_config = dict(
        type="ExpandableUnit",
        default_args=dict(
            extra_mapping={BatchNormAct2d: ExpandableBatchNormAct2d}),
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

    def get_fold_config(self):
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

# Distillation ####################################################################################


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


class TeacherStudentDistillNetDIST(nn.Module):
    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        teacher_input_image_size=224,
        input_image_size=224,
        loss_weight=2.0,
    ):

        super().__init__()
        self.teacher_input_image_size = teacher_input_image_size
        self.input_image_size = input_image_size

        self.student_model = student_model
        self.teacher_model = teacher_model

        self.teacher_model.eval()
        self.teacher_model.requires_grad_(False)
        self.student_model.eval()

        self.teacher_logit = None
        self.student_logit = None
        self.dist_loss = DISTLoss(loss_weight=loss_weight)

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
            self.teacher_logit = self.teacher_model(teacher_x)

        if x.shape[2] != self.input_image_size:
            student_x = F.interpolate(
                x, self.input_image_size, mode="bilinear")
        else:
            student_x = x
        self.student_logit = self.student_model(student_x)
        return self.student_logit

    def compute_ts_distill_loss(self):
        return self.dist_loss(self.student_logit, self.teacher_logit)
