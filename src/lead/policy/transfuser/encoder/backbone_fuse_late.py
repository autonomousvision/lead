"""Throughput variant: cross-modal fusion at the last two stages only.

The first two fusion blocks (transformer, pools, channel maps, interpolations)
are dropped entirely; both branches run their early stages unfused. Halves the
fusion work on the bet that the deep fusions carry the value.
"""

import torch
from torch import nn

from lead.config import LeadConfig
from lead.policy.transfuser.encoder.transfuser_backbone import TransfuserBackbone
from lead.policy.transfuser.utils import ops

# Stages that fuse; the earlier ones run branch-only.
_FUSION_STAGES = (2, 3)


class FuseLateBackbone(TransfuserBackbone):
    """TransfuserBackbone that fuses only at the deep stages."""

    def __init__(self, lead_config: LeadConfig) -> None:
        """Build the base backbone, then drop the early fusion blocks.

        Identity placeholders keep the per-stage indexing while freeing the
        parameters; DDP requires every remaining parameter to receive gradient.

        Args:
            lead_config: Root config tree.
        """
        super().__init__(lead_config)
        for stage in range(4):
            if stage not in _FUSION_STAGES:
                self.transformers[stage] = nn.Identity()
                self.lidar_channel_to_img[stage] = nn.Identity()
                self.img_channel_to_lidar[stage] = nn.Identity()

    def _forward(
        self,
        image: torch.Tensor,
        lidar: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Image + LiDAR feature fusion at the deep stages only.
        """
        image_features = ops.normalize_imagenet(image)
        lidar_features = lidar

        if self.lead_config.training.optimization.use_channels_last_memory_format:
            image_features = image_features.to(memory_format=torch.channels_last)
            lidar_features = lidar_features.to(memory_format=torch.channels_last)

        image_layers = iter(self.image_encoder.items())
        lidar_layers = iter(self.lidar_encoder.items())

        # In some architectures the stem is not a return layer, so we need to skip it.
        if len(self.image_encoder.return_layers) > 4:
            image_features = self.forward_layer_block(
                image_layers,
                self.image_encoder.return_layers,
                image_features,
            )
        if len(self.lidar_encoder.return_layers) > 4:
            lidar_features = self.forward_layer_block(
                lidar_layers,
                self.lidar_encoder.return_layers,
                lidar_features,
            )

        for i in range(4):
            image_features = self.forward_layer_block(
                image_layers,
                self.image_encoder.return_layers,
                image_features,
            )
            lidar_features = self.forward_layer_block(
                lidar_layers,
                self.lidar_encoder.return_layers,
                lidar_features,
            )
            if i in _FUSION_STAGES:
                image_features, lidar_features = self.fuse_features(
                    image_features,
                    lidar_features,
                    i,
                )

        return lidar_features, image_features
