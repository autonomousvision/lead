"""Throughput variant: learned strided-conv pooling and PixelShuffle upsampling.

Depthwise strided convs replace the adaptive pools (initialized as averaging,
so the variant starts numerically at the baseline), and conv+PixelShuffle
replaces every bilinear upsample (initialized as nearest replication).
"""

import jaxtyping as jt
import torch
import torch.nn.functional as F
from torch import nn

from lead.config import LeadConfig
from lead.policy.transfuser.encoder.transfuser_backbone import TransfuserBackbone


def _conv_pool(channels: int, factor_hw: tuple[int, int]) -> nn.Module:
    """A depthwise strided conv pooling one stage onto the anchor grid.

    Args:
        channels: The stage's channel count.
        factor_hw: Downsampling factor per axis.

    Returns:
        The pooling module, initialized to plain averaging; identity at factor 1.
    """
    if factor_hw == (1, 1):
        return nn.Identity()
    conv = nn.Conv2d(
        channels,
        channels,
        kernel_size=factor_hw,
        stride=factor_hw,
        groups=channels,
        bias=False,
    )
    nn.init.constant_(conv.weight, 1.0 / (factor_hw[0] * factor_hw[1]))
    return conv


def _shuffle_up(channels: int, factor: int) -> nn.Module:
    """A conv + PixelShuffle upsampler, initialized as nearest replication.

    Args:
        channels: The feature channel count, unchanged by the module.
        factor: Integer upsampling factor per axis.

    Returns:
        The upsampling module; identity at factor 1.
    """
    if factor == 1:
        return nn.Identity()
    conv = nn.Conv2d(channels, channels * factor * factor, kernel_size=1, bias=False)
    nn.init.zeros_(conv.weight)
    for channel in range(channels):
        conv.weight.data[
            channel * factor * factor : (channel + 1) * factor * factor,
            channel,
        ] = 1.0
    return nn.Sequential(conv, nn.PixelShuffle(factor))


class ConvResizeBackbone(TransfuserBackbone):
    """TransfuserBackbone with learned pooling and PixelShuffle upsampling."""

    def __init__(self, lead_config: LeadConfig) -> None:
        """Build the base backbone, then replace its resize machinery.

        Args:
            lead_config: Root config tree.
        """
        super().__init__(lead_config)
        config = self.config
        image_start = 1 if len(self.image_encoder.return_layers) > 4 else 0
        lidar_start = 1 if len(self.lidar_encoder.return_layers) > 4 else 0

        image_factors = self._stage_factors(
            (config.final_image_height, config.final_image_width),
            (config.img_vert_anchors, config.img_horz_anchors),
            [
                self.image_encoder.feature_info.info[image_start + i]["reduction"]
                for i in range(4)
            ],
        )
        lidar_factors = self._stage_factors(
            (config.lidar_height_pixel, config.lidar_width_pixel),
            (config.lidar_bev_grid_rows, config.lidar_bev_grid_cols),
            [
                self.lidar_encoder.feature_info.info[lidar_start + i]["reduction"]
                for i in range(4)
            ],
        )
        image_channels = [
            self.image_encoder.feature_info.info[image_start + i]["num_chs"]
            for i in range(4)
        ]
        lidar_channels = [
            self.lidar_encoder.feature_info.info[lidar_start + i]["num_chs"]
            for i in range(4)
        ]
        self.conv_pool_img = nn.ModuleList(
            [_conv_pool(image_channels[i], image_factors[i]) for i in range(4)],
        )
        self.conv_pool_lidar = nn.ModuleList(
            [_conv_pool(lidar_channels[i], lidar_factors[i]) for i in range(4)],
        )
        # The fused reads travel back at the image/lidar channel widths.
        self.shuffle_up_img = nn.ModuleList(
            [_shuffle_up(image_channels[i], image_factors[i][0]) for i in range(4)],
        )
        self.shuffle_up_lidar = nn.ModuleList(
            [_shuffle_up(lidar_channels[i], lidar_factors[i][0]) for i in range(4)],
        )
        if self.builds_bev_feature_grid:
            bev_channels = config.bev_feature_channels
            grid_rows = config.lidar_bev_grid_rows
            factor1 = int(config.bev_upsample_factor)
            target_h = config.lidar_height_pixel // config.bev_downsample_factor
            factor2, remainder = divmod(target_h, grid_rows * factor1)
            if remainder:
                raise ValueError(
                    f"top-down target height {target_h} is not an integer multiple "
                    f"of the pyramid height {grid_rows * factor1}",
                )
            self.shuffle_up_bev1 = _shuffle_up(bev_channels, factor1)
            self.shuffle_up_bev2 = _shuffle_up(bev_channels, factor2)

    @staticmethod
    def _stage_factors(
        input_hw: tuple[int, int],
        grid_hw: tuple[int, int],
        reductions: list[int],
    ) -> list[tuple[int, int]]:
        """Per-stage integer factors between feature maps and the anchor grid.

        Args:
            input_hw: The branch's input height and width.
            grid_hw: The anchor grid's height and width.
            reductions: Each stage's total stride.

        Returns:
            One (height, width) factor per stage.

        Raises:
            ValueError: If a stage does not map onto the grid by integer factors.
        """
        factors = []
        for reduction in reductions:
            feature_hw = (input_hw[0] // reduction, input_hw[1] // reduction)
            if feature_hw[0] % grid_hw[0] or feature_hw[1] % grid_hw[1]:
                raise ValueError(
                    f"feature map {feature_hw} not divisible by grid {grid_hw}",
                )
            factor = (feature_hw[0] // grid_hw[0], feature_hw[1] // grid_hw[1])
            if factor[0] != factor[1]:
                raise ValueError(
                    f"PixelShuffle needs a square factor, got {factor}",
                )
            factors.append(factor)
        return factors

    def top_down(
        self,
        x: jt.Float[torch.Tensor, "B C H W"],
    ) -> jt.Float[torch.Tensor, "B C2 H2 W2"]:
        """Inherited, see superclass; PixelShuffle instead of bilinear.

        Args:
            x: Input BEV feature tensor from the LiDAR encoder.

        Returns:
            Upsampled and refined BEV feature tensor at target resolution.
        """
        p5 = F.relu(self.c5_conv(x), inplace=True)
        p4 = F.relu(self.up_conv5(self.shuffle_up_bev1(p5)), inplace=True)
        return F.relu(self.up_conv4(self.shuffle_up_bev2(p4)), inplace=True)

    def fuse_features(
        self,
        image_features: jt.Float[torch.Tensor, "B C H W"],
        lidar_features: jt.Float[torch.Tensor, "B C2 H2 W2"],
        layer_idx: int,
    ) -> tuple[jt.Float[torch.Tensor, "B C H W"], jt.Float[torch.Tensor, "B C2 H2 W2"]]:
        """Inherited, see superclass; learned pools in, PixelShuffle out.

        Args:
            image_features: Features from the image branch.
            lidar_features: Features from the LiDAR branch.
            layer_idx: Transformer layer index.

        Returns:
            image_features and lidar_features with added features from the other branch.
        """
        image_embd_layer = self.conv_pool_img[layer_idx](image_features)
        lidar_embd_layer = self.conv_pool_lidar[layer_idx](lidar_features)
        lidar_embd_layer = self.lidar_channel_to_img[layer_idx](lidar_embd_layer)

        image_features_layer, lidar_features_layer = self.transformers[layer_idx](
            image_embd_layer,
            lidar_embd_layer,
        )

        lidar_features_layer = self.img_channel_to_lidar[layer_idx](
            lidar_features_layer,
        )
        image_features_layer = self.shuffle_up_img[layer_idx](image_features_layer)
        lidar_features_layer = self.shuffle_up_lidar[layer_idx](lidar_features_layer)
        return (
            image_features + image_features_layer,
            lidar_features + lidar_features_layer,
        )
