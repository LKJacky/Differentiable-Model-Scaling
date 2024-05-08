# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Union
from mmengine.optim import OptimWrapper

from mmengine.optim.optimizer.optimizer_wrapper import OptimWrapper
from mmengine.model.wrappers import MMDistributedDataParallel
from mmengine.registry import MODEL_WRAPPERS

from mmengine.registry import OPTIM_WRAPPERS
from torch import Tensor
import torch
import torch.nn as nn
from mmengine.model import BaseModel
from mmengine.structures import BaseDataElement

from mmrazor.models import BaseAlgorithm
from mmrazor.registry import MODELS, TASK_UTILS
from mmrazor.utils import RuntimeInfo
from .mutator import DMSMutator
from .scheduler import DMSScheduler
import copy
from mmrazor.models.utils.expandable_utils import expand_expandable_dynamic_model
from .op import DynamicStage
from mmrazor.structures.subnet.fix_subnet import export_fix_subnet, load_fix_subnet
from mmengine.model import revert_sync_batchnorm
from mmrazor.utils import print_log
import types
from mmengine.model import BaseModel


def to_static_model_x(
    model,
    reset_params=False,
    **kargs,
):

    # to static model
    fix_mutable = export_fix_subnet(model)[0]
    load_fix_subnet(model, fix_mutable)
    model = model

    if reset_params:
        print_log("reset parameters")
        for module in model.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

    return model


def convert_float_to_tenosr(res: dict, device):
    for k in res:
        if not isinstance(res[k], torch.Tensor):
            res[k] = torch.tensor(res[k], device=device)
    return res


def update_dict_reverse(config1: dict, config2: dict):
    for key in config2:
        if (
            key in config1
            and isinstance(config2[key], dict)
            and isinstance(config1[key], dict)
        ):
            update_dict_reverse(config1[key], config2[key])
        else:
            config1[key] = config2[key]
    return config1


def to_static_model(
    algorithm,
):
    fix_mutable = export_fix_subnet(algorithm.model)[0]
    load_fix_subnet(algorithm.model, fix_mutable)
    model = algorithm.model
    return model


def check_sync_bn(model: nn.Module):
    for m in model.modules():
        if isinstance(m, nn.SyncBatchNorm):
            return True
    return False


def hacky_init_weights_wrapper():
    """This init weight method is used to prevent the model init again after
    build.

    Besides, It also save fix_subnet.json after RuntimeInfo is ready.
    """

    def hacky_init_weights(model):
        pass

    return hacky_init_weights


def clean_params_init_info(model: nn.Module):
    """Clean param init info."""
    if hasattr(model, "_params_init_info"):
        delattr(model, "_params_init_info")
    for module in model.modules():
        if hasattr(module, "_params_init_info"):
            delattr(module, "_params_init_info")


# Algorithm #############################################################################################


def make_mutator_divisible(mutator: DMSMutator, divisor=8):
    for unit in mutator.dtp_mutator.mutable_units:
        unit.mutable_channel.make_divisible(divisor)


class DmsAlgorithmMixin:
    default_mutator_kwargs = dict(
        prune_qkv=False,
        prune_block=False,
        dtp_mutator_cfg=dict(
            type="DTPAMutator",
            channel_unit_cfg=dict(type="DTPTUnit", default_args=dict(extra_mapping={})),
            parse_cfg=dict(
                _scope_="mmrazor",
                type="ChannelAnalyzer",
                demo_input=dict(
                    type="DefaultDemoInput",
                    input_shape=(1, 3, 224, 224),
                ),
                tracer_type="FxTracer",
            ),
        ),
    )
    default_scheduler_kargs = dict(
        flops_target=0.8,
        decay_ratio=0.8,
        refine_ratio=0.2,
        flop_loss_weight=1000,
        structure_log_interval=10,
        by_epoch=False,
        target_scheduler="cos",
    )
    expand_unit_config = (
        dict(type="ExpandableUnit", default_args=dict(extra_mapping={})),
    )

    def __init__(self, model: nn.Module, mutator_kwargs={}, scheduler_kargs={}) -> None:
        mutator_kwargs = update_dict_reverse(
            copy.deepcopy(self.default_mutator_kwargs), mutator_kwargs
        )
        scheduler_kargs = update_dict_reverse(
            copy.deepcopy(self.default_scheduler_kargs), scheduler_kargs
        )

        origin_model = copy.deepcopy(model)
        self.architecture = model

        self.use_sync_bn = check_sync_bn(self.architecture)
        if self.use_sync_bn:
            revert_sync_batchnorm(self.architecture)

        if "type" not in mutator_kwargs:
            self.mutator: DMSMutator = DMSMutator(**mutator_kwargs)
        else:
            mutator_type = mutator_kwargs.pop("type")
            self.mutator: DMSMutator = MODELS.module_dict[mutator_type](
                **mutator_kwargs
            )

        self.scheduler = DMSScheduler(
            self.architecture,
            self.mutator,
            **scheduler_kargs,
        )
        self.mutator.channel_depth_train()
        self.architecture.load_state_dict(origin_model.state_dict(), strict=False)

        self.runtime_info = None

        self.extra_out = None

    @torch.no_grad()
    def to_static_model(self, scale=False, reset_params=False, divisor=1):
        print_log(self.mutator.info())
        if scale:
            self.mutator.scale_flop_to(
                self.architecture,
                target=self.scheduler.init_flop * self.scheduler.flops_target,
            )
        if divisor != 1:
            make_mutator_divisible(self.mutator, divisor)

        self.model = self.architecture
        model = to_static_model(self)
        if reset_params:
            print_log("reset parameters")
            for module in model.modules():
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()
        if self.use_sync_bn:
            print_log("convert sync bn")
            nn.SyncBatchNorm.convert_sync_batchnorm(model)
        return model

    @classmethod
    def expand_model(
        cls, model: nn.Module, channel_ratio: float = 2.0, block_ratio=2.0
    ):
        mutator_kargs = copy.deepcopy(cls.default_mutator_kwargs)
        mutator_kargs["dtp_mutator_cfg"]["channel_unit_cfg"] = cls.expand_unit_config
        mutator = DMSMutator(**mutator_kargs)
        mutator.prepare_from_supernet(model)
        structure = mutator.dtp_mutator.choice_template
        for key, num in structure.items():
            unit = mutator.dtp_mutator._name2unit[key]
            unit.expand_to(int(num * channel_ratio))
        model = expand_expandable_dynamic_model(model, zero=False)

        algo = cls(model)
        for module in algo.architecture.modules():
            if isinstance(module, DynamicStage):
                module.expand_module(block_ratio)
        return to_static_model_x(algo.architecture)

    @classmethod
    def show_structure(cls, model: nn.Module):
        model = copy.deepcopy(model)
        algo = cls(model)

        print(algo.mutator.info())


@MODELS.register_module()
class BaseDTPAlgorithm(BaseAlgorithm, DmsAlgorithmMixin):

    def __init__(
        self,
        architecture: Union[BaseModel, Dict],
        mutator_cfg=dict(
            prune_qkv=False,
            prune_block=False,
            dtp_mutator_cfg=dict(
                type="DTPAMutator",
                channel_unit_cfg=dict(
                    type="DTPTUnit", default_args=dict(extra_mapping={})
                ),
                parse_cfg=dict(
                    _scope_="mmrazor",
                    type="ChannelAnalyzer",
                    demo_input=dict(
                        type="DefaultDemoInput",
                        input_shape=(1, 3, 224, 224),
                    ),
                    tracer_type="FxTracer",
                ),
            ),
        ),
        scheduler=dict(
            flops_target=0.5,
            decay_ratio=0.6,
            refine_ratio=0.2,
            flop_loss_weight=1,
        ),
        #
        data_preprocessor: Optional[Union[Dict, nn.Module]] = None,
        init_cfg: Optional[Dict] = None,
    ) -> None:
        BaseAlgorithm.__init__(self, architecture, data_preprocessor, init_cfg)
        self.prepare_state_dict_for_init()
        DmsAlgorithmMixin.__init__(
            self,
            self.architecture,
            mutator_kwargs=mutator_cfg,
            scheduler_kargs=scheduler,
        )

    def prepare_state_dict_for_init(self):
        model = copy.deepcopy(self.architecture)
        model.init_weights()
        self._old_state = model.state_dict()

    def forward(
        self,
        inputs: torch.Tensor,
        data_samples: Optional[List[BaseDataElement]] = None,
        mode: str = "tensor",
    ):
        if self.training and mode == "loss":
            self.scheduler.before_train_forward(
                RuntimeInfo.iter(),
                RuntimeInfo.epoch(),
                RuntimeInfo().max_iters(),
                RuntimeInfo().max_epochs(),
            )
        res: dict = super().forward(inputs, data_samples, mode)  # type: ignore
        if self.training and mode == "loss":
            extra_dict = self.scheduler.after_train_forward(
                RuntimeInfo.iter(),
                RuntimeInfo.epoch(),
                RuntimeInfo().max_iters(),
                RuntimeInfo().max_epochs(),
            )
            extra_dict = convert_float_to_tenosr(extra_dict, inputs.device)
            res.update(extra_dict)
        return res

    def to_static_model(self, scale=False, reset_params=False):
        model = super().to_static_model(scale, reset_params=False)
        model.data_preprocessor = self.data_preprocessor
        if reset_params is False:
            if isinstance(model, BaseModel):
                model.init_cfg = None
                model.init_weights = types.MethodType(
                    hacky_init_weights_wrapper(), model
                )
        return model

    def init_weights(self):
        self.architecture.load_state_dict(self._old_state)
        delattr(self, "_old_state")
        print_log("load init weights")

    def train_step(self, data, optim_wrapper: OptimWrapper):
        optim_wrapper.algo = [self]
        return super().train_step(data, optim_wrapper)


class DmsGeneralAlgorithm(nn.Module, DmsAlgorithmMixin):

    def __init__(
        self,
        model: nn.Module,
        mutator_kwargs=dict(
            prune_qkv=False,
            prune_block=False,
            dtp_mutator_cfg=dict(
                type="DTPAMutator",
                channel_unit_cfg=dict(
                    type="DTPTUnit", default_args=dict(extra_mapping={})
                ),
                parse_cfg=dict(
                    _scope_="mmrazor",
                    type="ChannelAnalyzer",
                    demo_input=dict(
                        type="DefaultDemoInput",
                        input_shape=(1, 3, 224, 224),
                    ),
                    tracer_type="FxTracer",
                ),
            ),
            extra_module_mapping={},
        ),
        scheduler_kargs=dict(
            flops_target=0.8,
            decay_ratio=0.8,
            refine_ratio=0.2,
            flop_loss_weight=1000,
            structure_log_interval=10,
            by_epoch=False,
            target_scheduler="cos",
        ),
    ) -> None:
        nn.Module.__init__(self)
        DmsAlgorithmMixin.__init__(self, model, mutator_kwargs, scheduler_kargs)

    def forward(self, x):
        if self.training:
            self.scheduler.before_train_forward(*self.runtime_info)
        self.mutator.init_mutable_cache()
        out = self.architecture(x)
        if self.training:
            if self.runtime_info is not None:
                extra_dict = self.scheduler.after_train_forward(*self.runtime_info)
            res = out, extra_dict["flops_loss"]
        else:
            res = out
        self.mutator.reset_mutable_cache()
        return res


# Sub model #############################################################################################


@MODELS.register_module()
def DmsSubModel(
    algorithm: BaseDTPAlgorithm,
    pruned="",
    reset_params=False,
    **kargs,
):
    """Convert a algorithm(with an architecture) to a static pruned
    architecture.

    Args:
        algorithm (Union[BaseAlgorithm, dict]): The pruning algorithm to
            finetune.
        divisor (int): The divisor to make the channel number
            divisible. Defaults to 1.

    Returns:
        nn.Module: a static model.
    """
    # # init algorithm
    if isinstance(algorithm, dict):
        algorithm = MODELS.build(algorithm)  # type: ignore
    assert isinstance(algorithm, BaseDTPAlgorithm)
    if pruned != "":
        state = torch.load(pruned, map_location="cpu")
        if "ema_state_dict" in state:
            state = state["ema_state_dict"]
            new_state = {}
            for key in copy.copy(state):
                new_state[key.replace("module.", "", 1)] = state[key]
            state = new_state
            print_log("load ema checkpoint")
        else:
            state = state["state_dict"]
            print_log("load checkpoint")
        algorithm.load_state_dict(state, strict=False)
    print_log(f"{algorithm.mutator.info()}")
    model = algorithm.to_static_model(reset_params=reset_params)
    return model


# for norm #############################################################################################


@MODEL_WRAPPERS.register_module()
class DmsDDPWrapper(MMDistributedDataParallel):

    def train_step(self, data, optim_wrapper: OptimWrapper) -> Dict[str, Tensor]:
        optim_wrapper.algo = [self]
        return super().train_step(data, optim_wrapper)
