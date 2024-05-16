# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmengine.model.utils import _BatchNormXd

from mmrazor.models.mutables import (L1MutableChannelUnit,
                                     MutableChannelContainer)
from .ops import ExpandableBatchNorm2d, ExpandableConv2d, ExpandLinear
from mmrazor.registry import MODELS


@MODELS.register_module()
class ExpandableUnit(L1MutableChannelUnit):
    """The units to inplace modules with expandable dynamic ops."""

    def __init__(
        self,
        num_channels: int,
        choice_mode='number',
        divisor=1,
        min_value=1,
        min_ratio=0.9,
        extra_mapping={},
    ) -> None:
        super().__init__(num_channels, choice_mode, divisor, min_value,
                         min_ratio, extra_mapping)

        self.module_mapping = {
            nn.Conv2d: ExpandableConv2d,
            nn.BatchNorm2d: ExpandableBatchNorm2d,
            nn.Linear: ExpandLinear,
        }
        self.module_mapping.update(extra_mapping)

    def expand(self, num):
        expand_mask = self.mutable_channel.mask.new_zeros([num])
        mask = torch.cat([self.mutable_channel.mask, expand_mask])
        self.mutable_channel.mask = mask

    def expand_to(self, num):
        self.expand(num - self.num_channels)
