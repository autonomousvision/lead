"""Data-collection configuration of the expert (datagen, py123d output, planning area)."""

from collections import defaultdict
from functools import cached_property

from lead.common import weather
from lead.config.node import ConfigNode


class DataCollectionConfig(ConfigNode):
    """Datagen mode, py123d output, storage options and the BEV planning area."""

    # 123D dataset name; determines the maps directory and split prefix.
    py123d_dataset: str = "carla"
    # 123D split receiving the normal (unperturbated) sensor views; logs are written to ``<logs_root>/<split>``.
    py123d_split: str = "normal_view"
    # 123D split receiving the perturbated sensor views.
    py123d_perturbated_split: str = "perturbated_view"

    # ---- Performance settings ---
    # If true, also run speed benchmarking during expert data collection
    profile_expert: bool = False
    # How often (in steps) to print live profiler stats to the log. 0 disables live printing.
    profile_expert_freq: int = 0
    # How often we log in the main loop
    log_info_freq: int = 1

    # --- Data saving configuration for LiDAR ---
    # How many steps to stack LiDAR point clouds
    lidar_stack_size: int = 5
    # Frequency (in steps) at which data is saved during data collection
    data_save_freq: int = 5

    # --- Planning Area ---
    # How many pixels make up 1 meter in BEV grids.
    pixels_per_meter: float = 4.0
    # Pixels per meter used in the semantic segmentation map during data collection
    # On Town 13 2.0 is the highest that opencv can handle
    pixels_per_meter_collection: float = 2.0
    # Back boundary of the planning area in meters.
    min_x_meter: int = -32
    # Front boundary of the planning area in meters.
    max_x_meter: int = 64
    # Left boundary of the planning area in meters.
    min_y_meter: int = -40
    # Right boundary of the planning area in meters.
    max_y_meter: int = 40

    @property
    def lidar_width_pixel(self) -> int:
        """Width resolution of LiDAR BEV representation in pixels."""
        return int((self.max_x_meter - self.min_x_meter) * self.pixels_per_meter)

    @property
    def lidar_height_pixel(self) -> int:
        """Height resolution of LiDAR BEV representation in pixels."""
        return int((self.max_y_meter - self.min_y_meter) * self.pixels_per_meter)

    @property
    def lidar_width_meter(self) -> int:
        """Width of LiDAR coverage area in meters."""
        return int(self.max_x_meter - self.min_x_meter)

    @property
    def lidar_height_meter(self) -> int:
        """Height of LiDAR coverage area in meters."""
        return int(self.max_y_meter - self.min_y_meter)

    # --- Dataset and Timing Configuration ---
    # If true shuffle weather conditions during data collection.
    shuffle_weather: bool = True
    # If true use only nice weather conditions.
    nice_weather: bool = False
    # If true enable JPEG compression for image storage.
    jpeg_compression: bool = True
    # If true enable data generation mode.
    datagen: bool = True

    # Default bounding box size for traffic warning signs.
    traffic_warning_bb_size: list[float] = [1.186714768409729, 1.4352929592132568]
    # Default bounding box size for construction cones.
    construction_cone_bb_size: list[float] = [0.1720348298549652, 0.1720348298549652]

    # If true save instance segmentation images.
    save_instance_segmentation: bool = False
    # If true run expert evaluation. This will minimize sensor production and
    # other overheads to maximize inference speed.
    eval_expert: bool = False

    # --- Data Storage Configuration ---
    save_sensors: bool = True

    @property
    def save_depth(self) -> bool:
        """If true depth images should be saved."""
        return not self.eval_expert

    # PNG compression level for storing images
    png_storage_compression_level: int = 6

    # --- Weather Configuration ---
    @cached_property
    def weather_jpeg_compression_quality(self) -> dict[str, dict[int, float]]:
        """JPEG compression quality distribution per weather condition."""

        # Use default high level compression for all datasets.
        return defaultdict(lambda: {30: 1.0})

    @property
    def weather_settings(self) -> dict[str, dict[str, float]]:
        """Weather presets used for data-collection weather shuffling."""
        return weather.WEATHER_SETTINGS
