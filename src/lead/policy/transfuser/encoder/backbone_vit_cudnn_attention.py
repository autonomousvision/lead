"""Throughput variant: the ViT image branch under cuDNN's SDPA backend.

At this model's sequence lengths (a few hundred tokens) cuDNN attention
benchmarks within a few percent of FlashAttention-3 and well above torch's
default FA2 backend, without any dependency outside torch. The backend
priority applies to every scaled_dot_product_attention under the forward:
the ViT blocks and the fusion transformers alike, with math as the fallback
so CPU runs keep working.
"""

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from lead.policy.transfuser.encoder.backbone_vit_patch32 import VitPatch32Backbone

_BACKEND_PRIORITY = [
    SDPBackend.CUDNN_ATTENTION,
    SDPBackend.FLASH_ATTENTION,
    SDPBackend.EFFICIENT_ATTENTION,
    SDPBackend.MATH,
]


class VitPatch32CudnnBackbone(VitPatch32Backbone):
    """VitPatch32Backbone with cuDNN preferred for every attention call."""

    def _forward(
        self,
        image: torch.Tensor,
        lidar: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Inherited, see superclass; runs under the cuDNN-first SDPA priority.
        """
        with sdpa_kernel(_BACKEND_PRIORITY, set_priority=True):
            return super()._forward(image, lidar)
