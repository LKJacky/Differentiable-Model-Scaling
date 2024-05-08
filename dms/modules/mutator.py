# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch
import torch.nn as nn

from mmrazor.models.mutators.base_mutator import BaseMutator
from mmrazor.models.task_modules.demo_inputs import DefaultDemoInput
from mmrazor.registry import MODELS, TASK_UTILS
from .dtp import DTPAMutator, QuickFlopMixin
from .mutable import BlockThreshold, DMSMutableMixIn, MutableBlocks, MutableHead
from .op import DynamicStage, MutableAttn
import math
from mmrazor.utils import print_log


def replace_modules(model: nn.Module, module_map={}):
    def replace_op(model: nn.Module, name: str, module: nn.Module):
        names = name.split(".")
        for sub_name in names[:-1]:
            model = getattr(model, sub_name)

        setattr(model, names[-1], module)

    for name, module in model.named_modules():
        if name != "" and type(module) in module_map:
            new_module = module_map[type(module)].convert_from(module)
            replace_op(model, name, new_module)


class BlockInitialer:
    def __init__(
        self,
        dynamic_statge_module=DynamicStage,
        stage_mixin_layers=[],
        dynamic_block_mapping={},
        mutable_type="dms",
    ) -> None:
        class my_dynamic_statge(dynamic_statge_module):
            def __init__(
                self,
                *args,
            ):
                super().__init__(*args, mutable_type=mutable_type)

        self.dyanmic_stage_module = my_dynamic_statge
        self.block_mixin_layers = stage_mixin_layers
        self.stages: List[DynamicStage] = []
        self.dynamic_block_mapping = dynamic_block_mapping

    def prepare_from_supernet(self, supernet: nn.Module) -> List:
        map_dict = {}
        for block in self.block_mixin_layers:
            map_dict[block] = self.dyanmic_stage_module
        replace_modules(supernet, self.dynamic_block_mapping)
        replace_modules(
            supernet,
            module_map=map_dict,
        )
        mutables = []
        for module in supernet.modules():
            if isinstance(module, self.dyanmic_stage_module):
                mutables.append(module.mutable_blocks)
                self.stages.append(module)
                module.mutable_blocks.requires_grad_(True)
        return mutables


class AttnInitialer:
    def __init__(self) -> None:
        pass

    def prepare_from_supernet(self, supernet: nn.Module) -> List:

        attn_mutables = nn.ModuleList()

        for module in supernet.modules():
            if isinstance(module, MutableAttn):
                module.init_mutables()
                map_dict = nn.ModuleDict(module.attn_mutables)
                attn_mutables.append(map_dict)
        attn_mutables.requires_grad_(True)
        return attn_mutables


def to_hard(scale):
    hard = (scale >= BlockThreshold).float()
    return hard.detach() - scale.detach() + scale


@MODELS.register_module()
class DMSMutator(BaseMutator):
    def __init__(
        self,
        prune_qkv=True,
        prune_block=True,
        use_tayler=True,
        dtp_mutator_cfg=dict(
            type="DTPAMutator",
            channel_unit_cfg=dict(type="DTPTUnit", default_args=dict()),
            parse_cfg=dict(
                _scope_="mmrazor",
                type="ChannelAnalyzer",
                demo_input=dict(
                    type="DefaultDemoInput",
                    input_shape=(1, 3, 32, 32),
                ),
                tracer_type="FxTracer",
            ),
        ),
        extra_module_mapping={},
        block_initilizer_kwargs={},
        train_model=True,
        decay=0.99,
        tayler_type="taylor",
        normalize=True,
        init_cfg=None,
        estimate=False,
    ) -> None:
        super().__init__(init_cfg)

        self.dtp_mutator: DTPAMutator = MODELS.build(dtp_mutator_cfg)

        self.block_initializer = BlockInitialer(**block_initilizer_kwargs)
        self.block_mutables: List[MutableBlocks] = nn.ModuleList()

        self.attn_initialzer = AttnInitialer()
        self.attn_mutables = nn.ModuleList()

        self.prune_qkv = prune_qkv
        self.prune_block = prune_block

        self.use_tayler = use_tayler

        self.module_mapping = {}
        self.module_mapping.update(extra_module_mapping)

        self.train_model = train_model

        self.last_soft_flop = -1

        self.register_buffer("e_norm", torch.tensor(-1.0))
        self.register_buffer("e_flop_norm", torch.tensor(-1.0))
        self.e_norm: torch.Tensor
        self.e_flop_norm: torch.Tensor

        self.decay = decay

        self.taylor_type = tayler_type
        self.normalize = normalize
        self.estimate = estimate

    def set_decay(self, decay):
        for mutable in self.mutables():
            mutable.decay = decay

    def prepare_from_supernet(self, supernet) -> None:
        replace_modules(supernet, module_map=self.module_mapping)

        self.saved_model = [supernet]
        self.dtp_mutator.prepare_from_supernet(supernet)
        if self.prune_block:
            self.block_mutables = nn.ModuleList(
                self.block_initializer.prepare_from_supernet(supernet)
            )
        else:
            self.block_mutables = nn.ModuleList()
        self.attn_mutables = self.attn_initialzer.prepare_from_supernet(supernet)
        # {'head': None, 'qk': None, 'v': None}
        if not self.prune_qkv:
            self.fix_qkv()

        if not self.use_tayler:
            self.to_index_importance()
        self.set_decay(self.decay)

        for mutable in self.mutables():
            mutable.taylor_type = self.taylor_type
            mutable.normlize = self.normalize
            mutable.estimate = self.estimate

    def to_index_importance(self):
        for unit in self.dtp_mutator.mutable_units:
            unit.mutable_channel.to_index_importance()
        for mutable in self.block_mutables:
            mutable.to_index_importance()
        for mutable in self.attn_mutables:
            mutable["qk"].to_index_importance()
            mutable["v"].to_index_importance()
            mutable["head"].to_index_importance()

    def info(self):
        mutable_info = ""

        @torch.no_grad()
        def stage_info(stage: DynamicStage):
            scales = ""
            for block in stage.removable_block:
                scales += f"{block.flop_scale:.2f} "
            return scales

        for mut, stage in zip(self.block_mutables, self.block_initializer.stages):
            mutable_info += mut.info() + "\tratio:\t" + stage_info(stage) + "\n"

        flop_info = f"soft_flop: {self.get_soft_flop(self.saved_model[0])/1e6}"

        attn_info = ""
        for mutables in self.attn_mutables:
            attn_info += f"head: {mutables['head'].info()}\tqk: {mutables['qk'].info()}\tv: {mutables['v'].info()}\n"  # noqa

        out = (
            self.dtp_mutator.info()
            + "\n"
            + attn_info
            + "\n"
            + mutable_info
            + "\n"
            + flop_info
        )
        out += "\n" + f"E Gradient: {self.e_norm} / {self.e_flop_norm}"
        return out

    @torch.no_grad()
    def init_quick_flop(self, model: nn.Module):
        for module in model.modules():
            if isinstance(module, QuickFlopMixin):
                module.quick_flop_start_record()
        demo_input: DefaultDemoInput = TASK_UTILS.build(self.dtp_mutator.demo_input)
        model.eval()
        input = demo_input.get_data(model, training=False)
        if isinstance(input, dict):
            if "mode" in input:
                input["mode"] = "tensor"
            model(**input)
        else:
            model(input)
        for module in model.modules():
            if isinstance(module, QuickFlopMixin):
                module.quick_flop_end_record()

    @torch.no_grad()
    def limit_value(self):
        self.dtp_mutator.limit_value()
        for mul in self.block_mutables:
            mul.limit_value()
        for mutables in self.attn_mutables.modules():
            if isinstance(mutables, DMSMutableMixIn):
                mutables.limit_value()

    def get_soft_flop(self, model):
        res = QuickFlopMixin.get_flop(model)
        self.last_soft_flop = res.detach()
        return res

    @torch.no_grad()
    def scale_flop_to(self, model, target):
        def scale_mutable(mutable: DMSMutableMixIn, scale):
            mutable.e.data = mutable.e * scale
            mutable.limit_value()
            mutable.sync_mask()

        def scale_model(scale):
            scale_block = math.sqrt(scale)
            scale = math.sqrt(math.sqrt(scale))
            for unit in self.dtp_mutator.mutable_units:
                scale_mutable(unit.mutable_channel, scale)
            # {'head': None, 'qk': None, 'v': None}:
            for mutable_attn in self.attn_mutables:
                m_head, m_qk, m_v = (
                    mutable_attn["head"],
                    mutable_attn["qk"],
                    mutable_attn["v"],
                )
                if self.prune_qkv:
                    scale_mutable(m_qk, math.sqrt(scale))
                    scale_mutable(m_v, math.sqrt(scale))
                    scale_mutable(m_head, math.sqrt(scale))
                else:
                    scale_mutable(m_head, scale)
            for mutable_block in self.block_mutables:
                scale_mutable(mutable_block, scale=scale_block)

        current_flop = self.get_soft_flop(model)
        while not (target * 0.99 <= current_flop <= target * 1.01):
            print_log(f"target:{target/1e6},current: {current_flop/1e6}")
            scale = target / current_flop
            scale_model(scale)
            current_flop = self.get_soft_flop(model)
        print_log(f"last {current_flop/1e6}")

    @torch.no_grad()
    def init_random_tayler(self):
        for mutable in self.mutables():
            mutable.taylor = torch.rand_like(mutable.taylor)

    def fix_qkv(self):
        for mutables in self.attn_mutables:
            mutables["qk"].requires_grad_(False)
            mutables["v"].requires_grad_(False)

    def channel_depth_train(self):
        self.dtp_mutator.ratio_train()
        for mul in self.block_mutables:
            mul.requires_grad_(True)
        self.attn_mutables.requires_grad_(True)
        self.set_soft_flop_scale_converter(None)
        if not self.prune_qkv:
            self.fix_qkv()

    def channel_train(self):
        self.dtp_mutator.ratio_train()
        for mul in self.block_mutables:
            mul.requires_grad_(False)
        self.attn_mutables.requires_grad_(True)
        for modele in self.attn_mutables.modules():
            if isinstance(modele, MutableHead):
                modele.requires_grad_(False)
        self.set_soft_flop_scale_converter(to_hard)
        if not self.prune_qkv:
            self.fix_qkv()

    @property
    def choice_template(self):
        return self.dtp_mutator.choice_template

    # inherit from BaseMutator
    def search_groups(self):
        return super().search_groups

    def set_soft_flop_scale_converter(self, fun):
        for mut in self.block_mutables:
            mut.flop_scale_converter = fun
        for mut in self.attn_mutables.modules():
            if isinstance(mut, MutableHead):
                mut.flop_scale_converter = fun

    def mutables(self):
        for m in self.modules():
            if isinstance(m, DMSMutableMixIn):
                yield m

    def require_grad_mutables(self):
        for m in self.modules():
            if isinstance(m, DMSMutableMixIn) and m.e.requires_grad:
                yield m

    def named_require_grad_mutables(self):
        for n, m in self.named_modules():
            if isinstance(m, DMSMutableMixIn) and m.e.requires_grad:
                yield n, m

    def structure(self, toItem=True):
        res = {}
        for n, m in self.named_require_grad_mutables():
            if isinstance(m, MutableBlocks):
                for i in range(m.mask.numel()):
                    imp_flop = m.current_imp_flop
                    res[n + f"_{i}"] = imp_flop[i]
            else:
                m: DMSMutableMixIn
                res[n] = m.current_imp_flop.sum() / m.mask.numel()
        if toItem:
            for k in res:
                res[k] = res[k].item()

        return res

    def reset_saved_grad(self):
        for mutable in self.mutables():
            mutable.reset_saved_grad()

    @torch.no_grad()
    def norm_gradient(self, scale=1.0):
        e_grads = [m.grad_from_task for m in self.require_grad_mutables()]
        e_flop_grads = [m.grad_from_flop for m in self.require_grad_mutables()]

        e_grads = torch.cat(e_grads).detach()
        e_flop_grads = torch.cat(e_flop_grads).detach()

        e_norm = torch.norm(e_grads)
        e_flop_norm = torch.norm(e_flop_grads)

        if not (e_norm.isnan() or e_flop_norm.isnan()):
            if self.e_norm == -1:
                self.e_norm.data = e_norm
            else:
                self.e_norm = e_norm * (1 - 0.99) + self.e_norm * 0.99
            self.e_flop_norm.data = e_flop_norm

            if scale >= 0 and e_norm != 0 and e_flop_norm != 0:
                e_grads = e_grads
                e_flop_grads = e_flop_grads / self.e_flop_norm * self.e_norm
                new_grad = e_grads + e_flop_grads * scale

                for i, m in enumerate(self.require_grad_mutables()):
                    m.e.grad.data.fill_(new_grad[i])

        self.reset_saved_grad()

    def init_mutable_cache(self):
        for mutable in self.mutables():
            mutable.imp_cache = mutable.current_imp
            mutable.imp_flop_cache = mutable.current_imp_flop

    def reset_mutable_cache(self):
        for mutable in self.mutables():
            mutable.imp_cache = None
            mutable.imp_flop_cache = None
