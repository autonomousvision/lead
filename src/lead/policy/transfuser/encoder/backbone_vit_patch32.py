"""Throughput variant: a patch-32 ViT image branch.

A ViT tapped at four depths replaces the image CNN pyramid. Every tap shares
one resolution (the patch grid) and one width (the embedding dim), so the
per-stage anchor pooling is nearly free and the fusion transformers all run at
the embedding width. The taps come from a single ViT pass, so unlike the CNN
branch the cross-modal residuals do not feed later image stages; each stage
fuses its own tap and the last fused tap is the image output.

The image architecture stays ``config.image_architecture``; point it at a
patch-32 ViT (e.g. ``vit_small_patch32_384``) when selecting this backbone.
"""

import timm
import torch
from torch import nn

from lead.config import LeadConfig
from lead.policy.transfuser.encoder.transfuser_backbone import (
    GPT,
    TransfuserBackbone,
)
from lead.policy.transfuser.utils import ops


class VitPatch32Backbone(TransfuserBackbone):
    """TransfuserBackbone with a single-pass ViT image branch."""

    def __init__(self, lead_config: LeadConfig) -> None:
        """Duplicate of the base constructor with a ViT image branch.

        Args:
            lead_config: Root config tree.
        """
        nn.Module.__init__(self)
        self.lead_config = lead_config
        config = lead_config.policy.transfuser
        self.config = config

        # Image branch: four taps of one ViT, all at the patch-grid resolution.
        self.image_encoder = timm.create_model(
            config.image_architecture,
            pretrained=True,
            features_only=True,
            out_indices=(-4, -3, -2, -1),
            dynamic_img_size=True,
        )
        image_infos = self.image_encoder.feature_info.info[-4:]
        self.num_image_features = image_infos[3]["num_chs"]
        self.avgpool_img = nn.AdaptiveAvgPool2d(
            (self.config.img_vert_anchors, self.config.img_horz_anchors),
        )

        # LiDAR branch, exactly as the base builds it.
        self.lidar_encoder = timm.create_model(
            config.lidar_architecture,
            pretrained=False,
            in_chans=2 if config.LTF else 1,
            features_only=True,
        )
        lidar_start_index = 0
        if len(self.lidar_encoder.return_layers) > 4:
            lidar_start_index += 1
        self._lidar_start_index = lidar_start_index
        self.num_lidar_features = self.lidar_encoder.feature_info.info[
            lidar_start_index + 3
        ]["num_chs"]
        self.lidar_channel_to_img = nn.ModuleList(
            [
                nn.Conv2d(
                    self.lidar_encoder.feature_info.info[lidar_start_index + i][
                        "num_chs"
                    ],
                    image_infos[i]["num_chs"],
                    kernel_size=1,
                )
                for i in range(4)
            ],
        )
        self.img_channel_to_lidar = nn.ModuleList(
            [
                nn.Conv2d(
                    image_infos[i]["num_chs"],
                    self.lidar_encoder.feature_info.info[lidar_start_index + i][
                        "num_chs"
                    ],
                    kernel_size=1,
                )
                for i in range(4)
            ],
        )
        self.avgpool_lidar = nn.AdaptiveAvgPool2d(
            (self.config.lidar_bev_grid_rows, self.config.lidar_bev_grid_cols),
        )

        self.transformers = nn.ModuleList(
            [
                GPT(n_embd=image_infos[i]["num_chs"], lead_config=lead_config)
                for i in range(4)
            ],
        )

        self.perspective_upsample_factor = image_infos[3]["reduction"]

        # The top-down pyramid feeds the box and BEV semantic heads only, so
        # with both off it would train on no gradient.
        self.builds_bev_feature_grid = config.detect_boxes or config.use_bev_semantic
        if self.builds_bev_feature_grid:
            self.upsample = nn.Upsample(
                scale_factor=self.config.bev_upsample_factor,
                mode="bilinear",
                align_corners=False,
            )
            self.upsample2 = nn.Upsample(
                size=(
                    self.config.lidar_height_pixel // self.config.bev_downsample_factor,
                    self.config.lidar_width_pixel // self.config.bev_downsample_factor,
                ),
                mode="bilinear",
                align_corners=False,
            )
            self.up_conv5 = nn.Conv2d(
                self.config.bev_feature_channels,
                self.config.bev_feature_channels,
                (3, 3),
                padding=1,
            )
            self.up_conv4 = nn.Conv2d(
                self.config.bev_feature_channels,
                self.config.bev_feature_channels,
                (3, 3),
                padding=1,
            )
            self.c5_conv = nn.Conv2d(
                self.num_lidar_features,
                self.config.bev_feature_channels,
                (1, 1),
            )

    def _forward(
        self,
        image: torch.Tensor,
        lidar: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Image + LiDAR feature fusion with single-pass ViT taps.
        """
        image = ops.normalize_imagenet(image)
        lidar_features = lidar

        if self.lead_config.training.optimization.use_channels_last_memory_format:
            image = image.to(memory_format=torch.channels_last)
            lidar_features = lidar_features.to(memory_format=torch.channels_last)

        image_taps = self.image_encoder(image)

        lidar_layers = iter(self.lidar_encoder.items())
        if self._lidar_start_index:
            lidar_features = self.forward_layer_block(
                lidar_layers,
                self.lidar_encoder.return_layers,
                lidar_features,
            )

        image_features = image_taps[0]
        for i in range(4):
            lidar_features = self.forward_layer_block(
                lidar_layers,
                self.lidar_encoder.return_layers,
                lidar_features,
            )
            image_features, lidar_features = self.fuse_features(
                image_taps[i],
                lidar_features,
                i,
            )

        return lidar_features, image_features
