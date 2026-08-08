"""Optimization, precision, and checkpointing configuration."""

from typing import TYPE_CHECKING

from lead.config.node import ConfigNode, overridable_property

if TYPE_CHECKING:
    import torch


class OptimizationConfig(ConfigNode):
    """Learning rate, precision, gradient scaling, and checkpoint knobs."""

    # Base learning rate for the model.
    learning_rate: float = 3e-4
    # Weight decay for regularization.
    weight_decay: float = 0.01

    @overridable_property
    def num_epochs(self) -> int:
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
    def torch_dtype(self) -> "torch.dtype":
        """PyTorch float precision type for training."""
        import torch

        if self.use_mixed_precision_training:
            return torch.bfloat16
        return torch.float32

    # If true synchronize batch normalization across distributed processes.
    sync_batchnorm: bool = False
    # If true run the model through ``torch.compile``.
    use_torch_compile: bool = True
    # What ``torch.compile`` optimizes for: "default", "reduce-overhead" (CUDA
    # graphs), "max-autotune", or "max-autotune-no-cudagraphs". The tuning modes
    # spend minutes searching before the first step.
    torch_compile_mode: str = "default"
    # If true require one graph, turning any graph break into an error instead of
    # a silent fallback to eager for that frame.
    torch_compile_fullgraph: bool = False
    # If true use channel last memory format for input tensors.
    use_channels_last_memory_format: bool = True

    # If true save model checkpoints during training.
    save_model_checkpoint: bool = True
    # Epoch numbers whose checkpoints are kept during training. Empty = keep only the last.
    epochs_to_keep_checkpoints_for: list[int] = []
