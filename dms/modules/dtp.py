# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import List

import torch
from torch import nn

from mmrazor.models.mutators import ChannelMutator
from mmrazor.models.task_modules.demo_inputs import DefaultDemoInput
from mmrazor.registry import MODELS, TASK_UTILS
from .unit import DTPTUnit

from .op import QuickFlopMixin
from .utils import CollectMutatorMixin


class BaseDTPMutator(ChannelMutator, CollectMutatorMixin):
    def __init__(
        self,
        channel_unit_cfg=dict(type="ImpUnit", default_args=dict()),
        parse_cfg=dict(
            _scope_="mmrazor",
            type="ChannelAnalyzer",
            demo_input=dict(
                type="DefaultDemoInput",
                input_shape=[1, 3, 224, 224],
            ),
            tracer_type="BackwardTracer",
        ),
    ) -> None:
        super().__init__(channel_unit_cfg, parse_cfg)
        self.demo_input = parse_cfg["demo_input"]

    def prepare_from_supernet(self, supernet) -> None:
        res = super().prepare_from_supernet(supernet)
        for unit in self.mutable_units:
            unit.requires_grad_(True)
        return res

    def _prepare_from_unit_cfg(self, model, config: dict):
        """Initialize units using config dict."""
        assert isinstance(config, dict)
        units = []
        for unit_key in config:
            init_args = copy.deepcopy(self.unit_default_args)
            if "init_args" in config[unit_key]:
                init_args.update(config[unit_key]["init_args"])
            config[unit_key]["init_args"] = init_args
            unit = self.unit_class.init_from_cfg(model, config[unit_key])
            units.append(unit)
        return units

    @torch.no_grad()
    def init_quick_flop(self, model: nn.Module):
        for module in model.modules():
            if isinstance(module, QuickFlopMixin):
                module.quick_flop_start_record()
        demo_input: DefaultDemoInput = TASK_UTILS.build(self.demo_input)
        model.eval()
        input = demo_input.get_data(model, training=False)
        if isinstance(input, dict):
            input["mode"] = "tensor"
            model(**input)
        else:
            model(input)
        for module in model.modules():
            if isinstance(module, QuickFlopMixin):
                module.quick_flop_end_record()

    def get_soft_flop(self, model):
        flop = 0
        for _, module in model.named_modules():
            if isinstance(module, QuickFlopMixin):
                flop += module.soft_flop()
        assert isinstance(flop, torch.Tensor)
        return flop

    def info(self):
        import json

        res = ""
        structure = self.current_choices
        res += (json.dumps(structure, indent=4)) + "\n"
        for unit in self.mutable_units:
            res += (f"{unit.name}:\t{unit.info()}") + "\n"
        return res


@MODELS.register_module()
class DTPAMutator(BaseDTPMutator):
    def __init__(
        self,
        channel_unit_cfg=dict(type="DTPAUnit", default_args=dict()),
        parse_cfg=dict(
            _scope_="mmrazor",
            type="ChannelAnalyzer",
            demo_input=dict(
                type="DefaultDemoInput",
                input_shape=[1, 3, 224, 224],
            ),
            tracer_type="FxTracer",
        ),
    ) -> None:
        super().__init__(channel_unit_cfg, parse_cfg)
        self.mutable_units: List[DTPTUnit]

    @torch.no_grad()
    def limit_value(self):
        for unit in self.mutable_units:
            unit.mutable_channel.limit_value()

    def ratio_train(self):
        for unit in self.mutable_units:
            unit.requires_grad_(True)
