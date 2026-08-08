"""Throughput variant: fusion residuals stay at the anchor grid until the end.

Each stage pools to the anchor grid and runs its fusion transformer as usual,
but the fused reads are not interpolated back into the full-resolution branch
features per stage. They accumulate on the anchor grid through small per-stage
carry projections and are added back once, after the last stage — removing six
of the eight per-stage interpolations. The branch stages therefore run unfused,
which is a real semantic change, not only a cheaper operator.
"""

import torch
import torch.nn.functional as F
from torch import nn

from lead.config import LeadConfig
from lead.policy.transfuser.encoder.transfuser_backbone import TransfuserBackbone
from lead.policy.transfuser.utils import ops


class FuseAnchorBackbone(TransfuserBackbone):
    """TransfuserBackbone whose fusion residual lives on the anchor grid."""

    def __init__(self, lead_config: LeadConfig) -> None:
        """Build the base backbone, then add the anchor-grid carry projections.

        Args:
            lead_config: Root config tree.
        """
        super().__init__(lead_config)
        image_start = 1 if len(self.image_encoder.return_layers) > 4 else 0
        lidar_start = 1 if len(self.lidar_encoder.return_layers) > 4 else 0
        image_channels = [
            self.image_encoder.feature_info.info[image_start + i]["num_chs"]
            for i in range(4)
        ]
        lidar_channels = [
            self.lidar_encoder.feature_info.info[lidar_start + i]["num_chs"]
            for i in range(4)
        ]
        # Stage i's fused read, carried into stage i+1's width.
        self.image_carry = nn.ModuleList(
            [
                nn.Conv2d(image_channels[i], image_channels[i + 1], kernel_size=1)
                for i in range(3)
            ],
        )
        self.lidar_carry = nn.ModuleList(
            [
                nn.Conv2d(lidar_channels[i], lidar_channels[i + 1], kernel_size=1)
                for i in range(3)
            ],
        )

    def _forward(
        self,
        image: torch.Tensor,
        lidar: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Image + LiDAR feature fusion accumulated on the anchor grid.
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

        image_read = None
        lidar_read = None
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

            image_embd = self.avgpool_img(image_features)
            lidar_embd = self.avgpool_lidar(lidar_features)
            if image_read is not None:
                image_embd = image_embd + image_read
                lidar_embd = lidar_embd + lidar_read
            lidar_embd = self.lidar_channel_to_img[i](lidar_embd)

            image_fused, lidar_fused = self.transformers[i](image_embd, lidar_embd)
            lidar_fused = self.img_channel_to_lidar[i](lidar_fused)

            if i < 3:
                image_read = self.image_carry[i](image_fused)
                lidar_read = self.lidar_carry[i](lidar_fused)
            else:
                # The one full-resolution add, at the last stage.
                image_features = image_features + F.interpolate(
                    image_fused,
                    size=(image_features.shape[2], image_features.shape[3]),
                    mode="bilinear",
                    align_corners=False,
                )
                lidar_features = lidar_features + F.interpolate(
                    lidar_fused,
                    size=(lidar_features.shape[2], lidar_features.shape[3]),
                    mode="bilinear",
                    align_corners=False,
                )

        return lidar_features, image_features
