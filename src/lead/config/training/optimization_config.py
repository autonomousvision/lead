"""Optimization, precision, and checkpointing configuration."""

from typing import TYPE_CHECKING

from lead.config.node import ConfigNode, overridable_property

if TYPE_CHECKING:
    import torch


class OptimizationConfig(ConfigNode):
    """Learning rate, precision, gradient scaling, and checkpoint knobs."""

    # Base learning rate for the model.
    lr: float = 3e-4
    # Weight decay for regularization.
    weight_decay: float = 0.01

    @overridable_property
    def epochs(self) -> int:
        """Total number of training epochs."""
        return 31

    @overridable_property
    def batch_size(self) -> int:
        """Batch size for training."""
        if self._root.debug_mode:
            return 2
        return 64

    # If true use bfloat16 mixed precision training. This can speed up training and reduce memory usage on compatible hardware.
    use_mixed_precision_training: bool = True

    @property
    def torch_float_type(self) -> "torch.dtype":
        """PyTorch float precision type for training."""
        import torch

        if self.use_mixed_precision_training:
            return torch.bfloat16
        return torch.float32

    # If true synchronize batch normalization across distributed processes.
    sync_batchnorm: bool = False
    # If true compile the model for optimization.
    compile: bool = True
    # If true use channel last memory format for input tensors.
    channel_last: bool = True

    # If true save model checkpoints during training.
    save_model_checkpoint: bool = True
    # Epoch numbers whose checkpoints are kept during training.
    epoch_checkpoints_keep: list[int] = []
