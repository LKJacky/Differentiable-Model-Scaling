# Copyright (c) OpenMMLab. All rights reserved.

import numpy as np
import torch
import torch.nn as nn

from mmrazor.registry import TASK_UTILS
from mmrazor.utils import print_log
from .dtp import BaseDTPMutator
from .mutator import DMSMutator
import os
TRY = os.environ.get('TRY', 'false') == 'true'


@TASK_UTILS.register_module()
class DMSScheduler():

    # init

    def __init__(self,
                 model: nn.Module,
                 mutator: BaseDTPMutator,
                 flops_target=0.5,
                 decay_ratio=0.6,
                 refine_ratio=0.2,
                 flop_loss_weight=1,
                 by_epoch=False,
                 step=1,
                 target_scheduler='linear',
                 loss_type='l2',
                 structure_log_interval=100,
                 grad_scale=-1.0,
                 train_model=True,
                 pretrain_step=0,
                 negative_flop_loss=False) -> None:

        self.grad_scale = grad_scale

        self.model = model
        self.mutator: DMSMutator = mutator
        self._init()
        self.model.requires_grad_(train_model)

        self.decay_ratio = decay_ratio
        self.refine_ratio = refine_ratio
        self.flops_target = flops_target
        self.flop_loss_weight = flop_loss_weight

        self.structure_log_interval = structure_log_interval

        self.by_epoch = by_epoch

        self.target_scheduler = target_scheduler
        self.loss_type = loss_type

        if isinstance(by_epoch, bool):
            self.by_epoch = by_epoch
        elif isinstance(by_epoch, int):
            self.by_epoch = True
        else:
            raise NotImplementedError()
        self.train_model = train_model

        self.step = step
        self.pretrain_step = pretrain_step
        self.negative_flop_loss = negative_flop_loss

    def _init(self):
        self.mutator.prepare_from_supernet(self.model)
        self.mutator.init_quick_flop(self.model)
        self.init_flop = self.mutator.get_soft_flop(self.model).item()
        print_log(f'Get initial soft flops: {self.init_flop/1e6}')

    # hook

    def before_train_forward(self, iter, epoch, max_iters, max_epochs):
        self.mutator.reset_saved_grad()
        self.mutator.limit_value()
        ratio = self.get_ratio(iter, epoch, max_iters, max_epochs)
        if ratio <= 1:
            self.mutator.channel_depth_train()
        else:
            self.mutator.channel_train()
        self.model.requires_grad_(self.train_model)
        if self.train_model is False:
            self.model.eval()
            for module in self.model.modules():
                if isinstance(module,nn.BatchNorm2d):
                    module.train()
        else:
            self.model.train()

        # else:
        #     self.mutator.requires_grad_(False)

    def after_train_forward(self, iter, epoch, max_iters, max_epochs):
        if iter < (self.decay_ratio + self.refine_ratio) * max_iters:
            res = {
                'flops_loss':
                self.flop_loss(iter, epoch, max_iters, max_epochs) *
                self.flop_loss_weight,
                'soft_flop':
                self.mutator.last_soft_flop.detach() / 1e6,
                'target':
                self.current_target(iter, epoch, max_iters, max_epochs),
                "e_norm":
                self.mutator.e_norm,
                "e_flop_norm":
                self.mutator.e_flop_norm,
            }
            return res
        else:
            return {}

    # flops
    def get_ratio(self, iter, epoch, max_iters, max_epochs):
        total_steps = (max_epochs
                       if self.by_epoch else max_iters) - self.pretrain_step
        cur_step = (epoch if self.by_epoch else iter) - self.pretrain_step
        total_steps = int(total_steps * self.decay_ratio)

        ratio = cur_step / (total_steps - 1)
        if ratio > 0:
            cur_step = cur_step // self.step
            total_steps = total_steps // self.step
            ratio = cur_step / (total_steps - 1)
        return ratio

    def current_target(self, iter, epoch, max_iters, max_epochs):
        if TRY:
            epoch = epoch + 1

        def get_target(train_ratio, final_target=self.flops_target):
            if self.target_scheduler == 'no':
                return final_target
            elif self.target_scheduler == 'root':
                t = final_target**train_ratio
                return t
            else:
                raise NotImplementedError(f'{self.target_scheduler}')

        ratio = self.get_ratio(iter, epoch, max_iters, max_epochs)
        if ratio <= 0:
            return 1.0
        elif ratio >= 1:
            return self.flops_target
        else:
            return get_target(ratio, self.flops_target)

    def flop_loss(self, iter, epoch, max_iters, max_epochs):
        target = self.current_target(iter, epoch, max_iters, max_epochs)
        soft_flop = self.mutator.get_soft_flop(self.model) / self.init_flop

        loss_type = self.loss_type
        if loss_type == 'l2':
            loss = (soft_flop - target)**2 * (1 if soft_flop > target else 0)
        elif loss_type == 'l2+':
            loss = (soft_flop - target)**2 + (soft_flop - target) * (
                1 if soft_flop > target else 0)
        elif loss_type == 'log':
            loss = torch.log(
                soft_flop / target) * (1 if soft_flop > target else (-1 if self.negative_flop_loss else 0))
        else:
            raise NotImplementedError()

        return loss

    def flop_loss_by_target(self, target):
        return (max(
            self.mutator.get_soft_flop(self.model) / self.init_flop, target) -
            target)**2

    def norm_grad(self):
        self.mutator.norm_gradient(self.grad_scale)
