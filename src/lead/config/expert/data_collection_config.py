"""Data-collection configuration of the expert (datagen, py123d output, planning area)."""

from functools import cached_property

from lead.config.node import ConfigNode, overridable_property


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

    # Period (in simulator steps) of the save ticks: the render-heavy camera
    # streams are rendered and stored at this cadence (5 = 4 Hz at 20 fps).
    data_save_freq: int = 5

    # --- Per-modality storage periods (in simulator steps) ---
    # Every-tick state modalities; storage stays aligned to the anchor save
    # ticks, so any period that divides ``data_save_freq`` is exact and other
    # periods are re-phased at each anchor.
    lidar_store_freq: int = 1
    radar_store_freq: int = 1
    box_detections_store_freq: int = 1
    traffic_light_store_freq: int = 1

    # Anchor-locked modalities can only be stored on save ticks: their period
    # must be a multiple of ``data_save_freq``.
    @overridable_property
    def camera_store_freq(self) -> int:
        """Storage period of the RGB cameras; defaults to every save tick."""
        return self.data_save_freq

    @overridable_property
    def depth_store_freq(self) -> int:
        """Storage period of the depth cameras; defaults to every save tick."""
        return self.data_save_freq

    @overridable_property
    def instance_store_freq(self) -> int:
        """Storage period of the instance segmentation; defaults to every save tick."""
        return self.data_save_freq

    # --- Planning Area ---
    # The region around the ego the expert plans in and logs labels for; a
    # policy rasterizing this area picks its own resolution and crop.
    # Back boundary of the planning area in meters.
    min_x_meter: int = -32
    # Front boundary of the planning area in meters.
    max_x_meter: int = 64
    # Left boundary of the planning area in meters.
    min_y_meter: int = -40
    # Right boundary of the planning area in meters.
    max_y_meter: int = 40

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

        def normalize(d):
            return {k: v / sum(d.values()) for k, v in d.items()}

        LOW_COMPRESSION = normalize({80: 0.30, 85: 0.5, 90: 0.2})
        MILD_COMPRESSION = normalize({70: 0.15, 75: 0.25, 80: 0.4, 90: 0.15})
        MEDIUM_COMPRESSION = normalize({70: 0.25, 75: 0.2, 80: 0.15, 90: 0.15})
        HIGH_COMPRESION = normalize(
            {60: 0.10, 65: 0.1, 70: 0.1, 75: 0.2, 80: 0.1, 90: 0.15},
        )
        VERY_HIGH_COMPRESION = normalize({55: 0.20, 65: 0.4, 75: 0.15, 90: 0.15})
        EXTREME_COMPRESSION = normalize({30: 0.25, 40: 0.35, 50: 0.15, 90: 0.15})

        return {
            "ClearNight": MEDIUM_COMPRESSION,
            "ClearNoon": EXTREME_COMPRESSION,
            "ClearSunset": EXTREME_COMPRESSION,
            "ClearSunrise": EXTREME_COMPRESSION,
            # Cloudy weather
            "CloudyNight": MILD_COMPRESSION,
            "CloudyNoon": EXTREME_COMPRESSION,
            "CloudySunset": EXTREME_COMPRESSION,
            "CloudySunrise": EXTREME_COMPRESSION,
            # Dust storm
            "DustStorm": VERY_HIGH_COMPRESION,
            # Hard rain
            "HardRainNight": LOW_COMPRESSION,
            "HardRainNoon": MEDIUM_COMPRESSION,
            "HardRainSunset": MEDIUM_COMPRESSION,
            "HardRainSunrise": MEDIUM_COMPRESSION,
            # Mid rain
            "MidRainyNight": LOW_COMPRESSION,
            "MidRainyNoon": HIGH_COMPRESION,
            "MidRainSunset": HIGH_COMPRESION,
            "MidRainSunrise": HIGH_COMPRESION,
            # Soft rain
            "SoftRainNight": MILD_COMPRESSION,
            "SoftRainNoon": VERY_HIGH_COMPRESION,
            "SoftRainSunset": VERY_HIGH_COMPRESION,
            "SoftRainSunrise": VERY_HIGH_COMPRESION,
            # Wet cloudy
            "WetCloudyNight": LOW_COMPRESSION,
            "WetCloudyNoon": VERY_HIGH_COMPRESION,
            "WetCloudySunset": VERY_HIGH_COMPRESION,
            "WetCloudySunrise": VERY_HIGH_COMPRESION,
            # Wet
            "WetNight": MILD_COMPRESSION,
            "WetNoon": VERY_HIGH_COMPRESION,
            # Foggy cloudy
            "FoggyCloudyNight": MEDIUM_COMPRESSION,
            "FoggyCloudyNoon": MEDIUM_COMPRESSION,
            "FoggyCloudySunset": MEDIUM_COMPRESSION,
            "FoggyCloudySunrise": MEDIUM_COMPRESSION,
            # Foggy Wet cloudy
            "FoggyWetCloudyNight": MEDIUM_COMPRESSION,
            "FoggyWetCloudyNoon": MEDIUM_COMPRESSION,
            "FoggyWetCloudySunset": MEDIUM_COMPRESSION,
            "FoggyWetCloudySunrise": MEDIUM_COMPRESSION,
            # Foggy Wet
            "FoggyWetNoon": MEDIUM_COMPRESSION,
            # Foggy Soft Rain
            "FoggySoftRainNight": MEDIUM_COMPRESSION,
            "FoggySoftRainNoon": MEDIUM_COMPRESSION,
            "FoggySoftRainSunset": MEDIUM_COMPRESSION,
            "FoggySoftRainSunrise": MEDIUM_COMPRESSION,
            # Foggy Hard Rain
            "FoggyHardRainNight": LOW_COMPRESSION,
            # Custom weather
            "Custom0": HIGH_COMPRESION,
            "Custom9": HIGH_COMPRESION,
            "Custom10": LOW_COMPRESSION,
            "Custom11": HIGH_COMPRESION,
            "Custom12": HIGH_COMPRESION,
            "Custom13": LOW_COMPRESSION,
            "Custom14": HIGH_COMPRESION,
            "Custom15": HIGH_COMPRESION,
            "Custom19": LOW_COMPRESSION,
            "Custom20": LOW_COMPRESSION,
            "Custom21": LOW_COMPRESSION,
        }
