"""Data storage configuration: LiDAR compression, depth encoding and temporal buffers."""

from lead.config.node import ConfigNode


class StorageConfig(ConfigNode):
    """How sensor and temporal data is compressed and stored during data collection."""

    # --- LiDAR Compression ---
    # LARS point format used for storing LiDAR data
    point_format: int = 0
    # Precision up to which LiDAR points are stored (x, y, z coordinates)
    point_precision_x: float = 0.1
    point_precision_y: float = 0.1
    point_precision_z: float = 0.1
    # Maximum height threshold for LiDAR points (meters, points above are discarded)
    max_height_lidar: float = 10.0
    # Minimum height threshold for LiDAR points (meters, points below are discarded)
    min_height_lidar: float = -4.0

    # --- Data Storage ---
    # If true save depth images at lower resolution
    save_depth_lower_resolution: bool = True
    # Resolution reduction ratio for depth image storage.
    save_depth_resolution_ratio: int = 4
    # Number of bits used for saving depth images
    save_depth_bits: int = 8
    # If true save only non-ground LiDAR points
    save_only_non_ground_lidar: bool = True
    # If true save semantic segmentation in grouped format
    save_grouped_semantic: bool = True

    # --- Temporal Data ---
    # Number of temporal data points saved for ego vehicle
    ego_num_temporal_data_points_saved: int = 60
    # Number of temporal data points saved for other vehicles
    other_vehicles_num_temporal_data_points_saved: int = 40
