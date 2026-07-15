"""Force FP32 precision for layers and functions that are unstable under autocast."""

import functools
import inspect
from collections.abc import Callable

import torch
import torch.nn as nn
from torch.amp.autocast_mode import autocast


def _fp32_forward_wrapper(original_forward):
    """Wrap a normalization layer's forward method to force FP32 operations."""

    def forward(self, x):
        # Convert input to FP32, apply normalization, return in original dtype
        input_dtype = x.dtype
        x_fp32 = x.float()
        out_fp32 = original_forward(x_fp32)
        return out_fp32.to(input_dtype)

    return forward


def patch_norm_fp32(module: torch.nn.Module) -> torch.nn.Module:
    """Patch normalization layers to use FP32 operations while preserving module structure.

    Args:
        module: The module to patch.

    Returns:
        The patched module with FP32 normalization operations.
    """
    for child in module.modules():
        if isinstance(
            child,
            nn.modules.batchnorm._BatchNorm | nn.GroupNorm | nn.LayerNorm,
        ):
            # Ensure parameters are in FP32
            child.float()
            # Patch the forward method to handle input/output dtype conversion
            child.forward = _fp32_forward_wrapper(child.forward).__get__(
                child,
                type(child),
            )
    return module


def force_fp32(apply_to: tuple[str, ...] | None = None):
    """
    Decorator to force a function to run in fp32 precision.

    Args:
        apply_to: Tuple of argument names to convert to fp32. If None, converts all tensor arguments.

    Returns:
        Decorated function that runs in fp32 precision.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Disable autocast to prevent fp16 operations
            with autocast(device_type="cuda", enabled=False):
                # Convert specified arguments to fp32
                if apply_to is not None:
                    sig = inspect.signature(func)
                    param_names = list(sig.parameters.keys())

                    # Convert positional arguments
                    new_args = list(args)
                    for i, arg in enumerate(args):
                        if i < len(param_names) and param_names[i] in apply_to:
                            if isinstance(arg, torch.Tensor):
                                new_args[i] = arg.float()

                    # Convert keyword arguments
                    new_kwargs = {}
                    for key, value in kwargs.items():
                        if key in apply_to and isinstance(value, torch.Tensor):
                            new_kwargs[key] = value.float()
                        else:
                            new_kwargs[key] = value

                    return func(*new_args, **new_kwargs)
                else:
                    # Convert all tensor arguments to fp32
                    new_args = []
                    for arg in args:
                        if isinstance(arg, torch.Tensor):
                            new_args.append(arg.float())
                        else:
                            new_args.append(arg)

                    new_kwargs = {}
                    for key, value in kwargs.items():
                        if isinstance(value, torch.Tensor):
                            new_kwargs[key] = value.float()
                        else:
                            new_kwargs[key] = value

                    return func(*new_args, **new_kwargs)

        return wrapper

    return decorator
