@OPTIM_WRAPPERS.register_module()
class DmsOptimWrapper(OptimWrapper):

    def backward(self, loss: Tensor, **kwargs) -> None:
        res = super().backward(loss, **kwargs)
        algo: BaseDTPAlgorithm = self.algo[0]
        if isinstance(algo, DmsDDPWrapper):
            algo = algo.module
        algo.scheduler.norm_grad()
        return res


# optimizers #############################################################################################

from mmengine.optim.scheduler import LinearLR, CosineAnnealingLR
from mmengine.registry import PARAM_SCHEDULERS


@PARAM_SCHEDULERS.register_module()
class MyLinearLR(LinearLR):

    def __init__(self, optimizer, *args, mutator_lr=0.1, **kwargs):
        self.mutator_lr = mutator_lr
        super().__init__(optimizer, *args, **kwargs)

    def _get_value(self):
        """Compute value using chainable form of the scheduler."""
        res = super()._get_value()
        mutator_lr = self.mutator_lr
        for i in range(len(self.base_values)):
            if self.base_values[i] == mutator_lr:
                res[i] = mutator_lr
        return res


@PARAM_SCHEDULERS.register_module()
class MyCosineAnnealingLR(CosineAnnealingLR):

    def __init__(self, optimizer, *args, mutator_lr=0.1, **kwargs):
        self.mutator_lr = mutator_lr
        super().__init__(optimizer, *args, **kwargs)

    def _get_value(self):
        """Compute value using chainable form of the scheduler."""
        res = super()._get_value()
        mutator_lr = self.mutator_lr
        for i in range(len(self.base_values)):
            if self.base_values[i] == mutator_lr:
                res[i] = mutator_lr
        return res
