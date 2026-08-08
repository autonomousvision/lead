"""Throughput variant: GroupNorm in place of every encoder BatchNorm.

GroupNorm keeps no running statistics and normalizes within the sample, so it
needs no fp32 patching under bf16 autocast and no cross-batch state. The swap
happens after construction, so pretrained conv weights (and the BN affine
parameters, which GroupNorm shares shapes with) are preserved.
"""

import torch
from torch import nn

from lead.config import LeadConfig
from lead.policy.transfuser.encoder.transfuser_backbone import TransfuserBackbone


def _group_count(channels: int, preferred: int = 32) -> int:
    """The largest group count at most ``preferred`` that divides the channels.

    Args:
        channels: The layer's channel count.
        preferred: Upper bound on the group count.

    Returns:
        The group count.
    """
    groups = min(preferred, channels)
    while channels % groups:
        groups -= 1
    return groups


def _swap_batchnorm_for_groupnorm(module: nn.Module) -> None:
    """Replace every BatchNorm in a module tree with an affine-preserving GroupNorm.

    Args:
        module: The tree to rewrite in place.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.modules.batchnorm._BatchNorm):
            channels = child.num_features
            group_norm = nn.GroupNorm(
                _group_count(channels),
                channels,
                eps=child.eps,
                affine=child.affine,
            )
            if child.affine:
                with torch.no_grad():
                    group_norm.weight.copy_(child.weight)
                    group_norm.bias.copy_(child.bias)
            setattr(module, name, group_norm)
        else:
            _swap_batchnorm_for_groupnorm(child)


class GroupNormBackbone(TransfuserBackbone):
    """TransfuserBackbone whose encoder BatchNorms are GroupNorms."""

    def __init__(self, lead_config: LeadConfig) -> None:
        """Build the base backbone, then swap its norms.

        Args:
            lead_config: Root config tree.
        """
        super().__init__(lead_config)
        _swap_batchnorm_for_groupnorm(self.image_encoder)
        _swap_batchnorm_for_groupnorm(self.lidar_encoder)
