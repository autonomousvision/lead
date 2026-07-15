"""Data storage configuration: LiDAR compression, depth encoding and temporal buffers."""

from lead.config.node import ConfigNode


class StorageConfig(ConfigNode):
    """How sensor and temporal data is compressed and stored during data collection."""

    # --- LiDAR Compression ---
    # LAS point format used for storing LiDAR data. Format 0 holds xyz and the base attributes;
    # the richer formats reserve unused room for GPS time and RGB.
    point_format: int = 0
    # Precision up to which LiDAR points are stored (x, y, z coordinates)
    point_precision_x: float = 0.1
    point_precision_y: float = 0.1
    point_precision_z: float = 0.1

    # --- Data Storage ---
    # Number of bits used for saving depth images
    save_depth_bits: int = 8
    # Far plane (meters) of the linear depth quantization; depth at or beyond this saturates.
    save_depth_max_meters: float = 50.0
