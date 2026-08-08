"""Throughput variant: static pooling and nearest upsampling.

The input resolution is fixed, so each fusion stage's adaptive pool resolves to
one kernel that can be chosen at construction, and the bilinear reads in the
fusion blocks and the top-down pyramid become nearest lookups.
"""

import jaxtyping as jt
import torch
import torch.nn.functional as F
from torch import nn

from lead.config import LeadConfig
from lead.policy.transfuser.encoder.transfuser_backbone import TransfuserBackbone


def _static_pool(feature_hw: tuple[int, int], grid_hw: tuple[int, int]) -> nn.Module:
    """A fixed avg pool from one stage's feature size onto the anchor grid.

    Args:
        feature_hw: The stage's feature-map height and width.
        grid_hw: The anchor grid's height and width.

    Returns:
        The pooling module; identity when the sizes already match.
    """
    (feature_h, feature_w), (grid_h, grid_w) = feature_hw, grid_hw
    if feature_h % grid_h or feature_w % grid_w:
        raise ValueError(f"feature map {feature_hw} not divisible by grid {grid_hw}")
    kernel = (feature_h // grid_h, feature_w // grid_w)
    if kernel == (1, 1):
        return nn.Identity()
    return nn.AvgPool2d(kernel_size=kernel, stride=kernel)


class StaticResizeBackbone(TransfuserBackbone):
    """TransfuserBackbone with construction-time pool kernels and nearest upsampling."""

    def __init__(self, lead_config: LeadConfig) -> None:
        """Build the base backbone, then replace its resize machinery.

        Args:
            lead_config: Root config tree.
        """
        super().__init__(lead_config)
        config = self.config
        image_start = 1 if len(self.image_encoder.return_layers) > 4 else 0
        lidar_start = 1 if len(self.lidar_encoder.return_layers) > 4 else 0
        image_grid = (config.img_vert_anchors, config.img_horz_anchors)
        lidar_grid = (config.lidar_bev_grid_rows, config.lidar_bev_grid_cols)
        self.static_pool_img = nn.ModuleList(
            [
                _static_pool(
                    self._stage_hw(
                        (config.final_image_height, config.final_image_width),
                        self.image_encoder.feature_info.info[image_start + i][
                            "reduction"
                        ],
                    ),
                    image_grid,
                )
                for i in range(4)
            ],
        )
        self.static_pool_lidar = nn.ModuleList(
            [
                _static_pool(
                    self._stage_hw(
                        (config.lidar_height_pixel, config.lidar_width_pixel),
                        self.lidar_encoder.feature_info.info[lidar_start + i][
                            "reduction"
                        ],
                    ),
                    lidar_grid,
                )
                for i in range(4)
            ],
        )
        if self.builds_bev_feature_grid:
            self.upsample = nn.Upsample(
                scale_factor=self.config.bev_upsample_factor,
                mode="nearest",
            )
            self.upsample2 = nn.Upsample(
                size=(
                    self.config.lidar_height_pixel // self.config.bev_downsample_factor,
                    self.config.lidar_width_pixel // self.config.bev_downsample_factor,
                ),
                mode="nearest",
            )

    @staticmethod
    def _stage_hw(input_hw: tuple[int, int], reduction: int) -> tuple[int, int]:
        """One stage's feature-map size from the input size and its stride.

        Args:
            input_hw: The branch's input height and width.
            reduction: The stage's total stride.

        Returns:
            The stage's feature-map height and width.
        """
        return (input_hw[0] // reduction, input_hw[1] // reduction)

    def fuse_features(
        self,
        image_features: jt.Float[torch.Tensor, "B C H W"],
        lidar_features: jt.Float[torch.Tensor, "B C2 H2 W2"],
        layer_idx: int,
    ) -> tuple[jt.Float[torch.Tensor, "B C H W"], jt.Float[torch.Tensor, "B C2 H2 W2"]]:
        """Inherited, see superclass; static pools in, nearest reads out.

        Args:
            image_features: Features from the image branch.
            lidar_features: Features from the LiDAR branch.
            layer_idx: Transformer layer index.

        Returns:
            image_features and lidar_features with added features from the other branch.
        """
        image_embd_layer = self.static_pool_img[layer_idx](image_features)
        lidar_embd_layer = self.static_pool_lidar[layer_idx](lidar_features)
        lidar_embd_layer = self.lidar_channel_to_img[layer_idx](lidar_embd_layer)

        image_features_layer, lidar_features_layer = self.transformers[layer_idx](
            image_embd_layer,
            lidar_embd_layer,
        )

        lidar_features_layer = self.img_channel_to_lidar[layer_idx](
            lidar_features_layer,
        )
        image_features_layer = F.interpolate(
            image_features_layer,
            size=(image_features.shape[2], image_features.shape[3]),
            mode="nearest",
        )
        lidar_features_layer = F.interpolate(
            lidar_features_layer,
            size=(lidar_features.shape[2], lidar_features.shape[3]),
            mode="nearest",
        )
        return (
            image_features + image_features_layer,
            lidar_features + lidar_features_layer,
        )
