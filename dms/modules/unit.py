# Copyright (c) OpenMMLab. All rights reserved.

from typing import Dict
import torch
import torch.nn as nn

from mmrazor.models.architectures import dynamic_ops
from mmrazor.models.mutables import L1MutableChannelUnit
from mmrazor.registry import MODELS
from .utils import CollectMutatorMixin
from .mutable import (
    DTPTMutableChannelImp,
    ImpMutableChannelContainer,
    PerElementMutableChannelImp,
    SelectMutableChannelImp,
)
from .op import ImpBatchnorm2d, ImpConv2d, ImpLinear
from mmengine.model.utils import _BatchNormXd
from .utils import CollectUnitMixin

DTPMutableChannelImp = DTPTMutableChannelImp


class BaseDTPUnit(L1MutableChannelUnit, CollectUnitMixin):
    def __init__(
        self,
        num_channels: int,
        extra_mapping={},
    ) -> None:
        super().__init__(
            num_channels, choice_mode="number", extra_mapping=extra_mapping
        )
        self.mutable_channel: DTPMutableChannelImp = DTPMutableChannelImp(
            self.num_channels
        )
        self.requires_grad_(False)
        from .op import ImpLayerNorm

        self.module_mapping = {
            nn.Conv2d: ImpConv2d,
            nn.BatchNorm2d: ImpBatchnorm2d,
            _BatchNormXd: ImpBatchnorm2d,
            nn.Linear: ImpLinear,
            nn.SyncBatchNorm: dynamic_ops.DynamicSyncBatchNorm,
            nn.LayerNorm: ImpLayerNorm,
        }
        self.module_mapping.update(extra_mapping)

    def prepare_for_pruning(self, model: nn.Module):
        self._replace_with_dynamic_ops(model, self.module_mapping)
        self._register_channel_container(model, ImpMutableChannelContainer)
        self._register_mutable_channel(self.mutable_channel)

    def info(self):
        raise NotImplementedError()


@MODELS.register_module()
class DTPTUnit(BaseDTPUnit):
    def __init__(
        self,
        num_channels: int,
        min=0,
        max=1.0,
        fold=1,
        mutable_type="dms",
        extra_mapping={},
    ) -> None:
        self.fold = fold
        super().__init__(num_channels, extra_mapping=extra_mapping)
        if mutable_type == "dms":
            self.mutable_channel: DTPTMutableChannelImp = DTPTMutableChannelImp(
                self.num_channels, min=min, max=max
            )
        elif mutable_type == "per_elem":
            self.mutable_channel: PerElementMutableChannelImp = (
                PerElementMutableChannelImp(self.num_channels, min=min, max=max)
            )
        elif mutable_type == "select":
            self.mutable_channel: SelectMutableChannelImp = SelectMutableChannelImp(
                self.num_channels, min=min, max=max
            )
        else:
            raise NotImplementedError()

        self.requires_grad_(False)
        self.module_mapping.update(extra_mapping)

    @torch.no_grad()
    def info(self) -> str:
        return f"taylor: {self.mutable_channel.info()}\t"  # noqa

    def config_template(self, with_init_args=False, with_channels=False) -> Dict:
        """Template of config."""
        config = super().config_template(with_init_args, with_channels)
        if with_init_args:
            config["init_args"] = {
                "num_channels": self.num_channels,
                "min": self.mutable_channel.min,
                "max": self.mutable_channel.max,
                "fold": self.fold,
            }
        return config
