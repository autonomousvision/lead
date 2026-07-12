"""Occlusion-check thresholds of the expert."""

from lead.config.node import ConfigNode


class ExpertOcclusionConfig(ConfigNode):
    """Minimum LiDAR-point / visible-pixel counts for actors to be considered valid."""

    # Minimum number of LiDAR points for a vehicle to be considered valid.
    vehicle_min_num_lidar_points: int = 3
    # Minimum number of visible pixels for a vehicle to be considered valid.
    vehicle_min_num_visible_pixels: int = 5
    # Minimum number of LiDAR points for a pedestrian to be considered valid.
    pedestrian_min_num_lidar_points: int = 5
    # Minimum number of visible pixels for a pedestrian to be considered valid.
    pedestrian_min_num_visible_pixels: int = 15
    # Minimum number of LiDAR points for a static prop (barrier, cone, sign) to be considered valid.
    static_prop_min_num_lidar_points: int = 1
    # Minimum number of visible pixels for a static prop (barrier, cone, sign) to be considered valid.
    static_prop_min_num_visible_pixels: int = 1

    # Minimum visible pixels for pedestrian occlusion check
    pedestrian_occlusion_check_min_visible_pixels: int = 10
    # If true enable bikers occlusion check
    bikers_occlusion_check: bool = True
    # Minimum visible pixels for bikers occlusion check
    bikers_occlusion_check_min_visible_pixels: int = 10
    # If true enable vehicle occlusion check
    vehicle_occlusion_check: bool = True
    # Minimum visible pixels for vehicle occlusion check
    vehicle_occlusion_check_min_visible_pixels: int = 1
    # Minimum number of points for vehicle occlusion check
    vehicle_occlusion_check_min_num_points: int = 1
