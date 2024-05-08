import torch
from typing import List
class CollectMixin:
    """The mixin class for GroupFisher ops."""

    def _collect_init(self) -> None:
        self.handlers: list = []
        self.recorded_input: List = []
        self.recorded_grad: List = []
        self.recorded_out_shape: List = []

    def forward_hook_wrapper(self):
        """Wrap the hook used in forward."""

        def forward_hook(module: CollectMixin, input, output):
            module.recorded_out_shape.append(output.shape)
            module.recorded_input.append(input[0])

        return forward_hook

    def backward_hook_wrapper(self):
        """Wrap the hook used in backward."""

        def backward_hook(module: CollectMixin, grad_in, grad_out):
            module.recorded_grad.insert(0, grad_in[0])

        return backward_hook

    def start_record(self: torch.nn.Module) -> None:
        """Start recording information during forward and backward."""
        self.end_record()  # ensure to run start_record only once
        self.handlers.append(self.register_forward_hook(self.forward_hook_wrapper()))
        self.handlers.append(self.register_backward_hook(self.backward_hook_wrapper()))

    def end_record(self):
        """Stop recording information during forward and backward."""
        for handle in self.handlers:
            handle.remove()
        self.handlers = []

    def reset_recorded(self):
        """Reset the recorded information."""
        self.recorded_input = []
        self.recorded_grad = []
        self.recorded_out_shape = []


class CollectMutatorMixin:

    def start_record_info(self) -> None:
        """Start recording the related information."""
        for unit in self.mutable_units:  # type: ignore
            unit.start_record_fisher_info()

    def end_record_info(self) -> None:
        """Stop recording the related information."""

        for unit in self.mutable_units:  # type: ignore
            unit.end_record_fisher_info()

    def reset_recorded_info(self) -> None:
        """Reset the related information."""
        for unit in self.mutable_units:  # type: ignore
            unit.reset_recorded()
class CollectUnitMixin:

    @property
    def input_related_collect_ops(self):
        for channel in self.input_related:
            if isinstance(channel.module, CollectMixin):
                yield channel.module

    @property
    def output_related_collect_ops(self):
        for channel in self.output_related:
            if isinstance(channel.module, CollectMixin):
                yield channel.module

    @property
    def collect_ops(self):
        for module in self.input_related_collect_ops:
            yield module
        for module in self.output_related_collect_ops:
            yield module
