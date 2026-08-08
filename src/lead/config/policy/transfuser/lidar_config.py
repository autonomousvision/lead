"""LiDAR input processing of the TransFuser model."""

from lead.config.node import ConfigNode, overridable_property

# Total downsampling of the backbone's BEV branch: the token grid is the raster
# divided by this, so it follows the architecture rather than the configuration.
_BACKBONE_BEV_STRIDE = 32


class TransfuserLidarConfig(ConfigNode):
    """Point filtering, and the BEV raster and feature-map geometry.

    The sweep window itself is declared per modality on
    :class:`~lead.config.policy.abstract_policy_config.AbstractPolicyConfig`.
    """

    # --- BEV raster the network sees ---
    # How many pixels make up 1 meter of the BEV raster; with the extents below
    # it sets the raster resolution, and through that the BEV token count.
    bev_pixels_per_meter: float = 4.0

    # Back boundary of the BEV crop in meters.
    bev_min_x_meter: int = -32
    # Front boundary of the BEV crop in meters.
    bev_max_x_meter: int = 64
    # Left boundary of the BEV crop in meters.
    bev_min_y_meter: int = -40
    # Right boundary of the BEV crop in meters.
    bev_max_y_meter: int = 40

    @property
    def lidar_width_pixel(self) -> int:
        """Width resolution of LiDAR BEV representation in pixels."""
        return int(
            (self.bev_max_x_meter - self.bev_min_x_meter) * self.bev_pixels_per_meter,
        )

    @property
    def lidar_height_pixel(self) -> int:
        """Height resolution of LiDAR BEV representation in pixels."""
        return int(
            (self.bev_max_y_meter - self.bev_min_y_meter) * self.bev_pixels_per_meter,
        )

    @overridable_property
    def lidar_bev_grid_rows(self) -> int:
        """Number of vertical anchors for LiDAR feature maps."""
        return self.lidar_height_pixel // _BACKBONE_BEV_STRIDE

    @overridable_property
    def lidar_bev_grid_cols(self) -> int:
        """Number of horizontal anchors for LiDAR feature maps."""
        return self.lidar_width_pixel // _BACKBONE_BEV_STRIDE

    # Max number of LiDAR points per pixel in voxelized LiDAR.
    max_lidar_points_per_bev_pixel: int = 5
    # Maximum height threshold for LiDAR points (meters, points above are discarded).
    lidar_max_height_meter: float = 10.0
    # Minimum height threshold for LiDAR points (meters, points below are discarded).
    lidar_min_height_meter: float = -4.0
    # If true stack the past LiDAR sweeps into the model input.
    accumulate_lidar_sweeps: bool = True
    # If true remove ground points from the LiDAR input via RANSAC.
    remove_lidar_ground_points: bool = True
    # If true merge the radar returns into the LiDAR input.
    merge_radar_into_lidar: bool = True
    # If true duplicate radar points near the ego to weight them like LiDAR points.
    duplicate_radar_near_ego: bool = True
    # Radius around the ego within which radar points are duplicated (meters).
    duplicate_radar_radius_meter: float = 32
    # Multiplication factor for the duplicated radar points.
    duplicate_radar_repeat_count: int = 5
