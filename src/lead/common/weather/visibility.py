"""Weather-to-visibility classification."""

from enum import IntEnum


class WeatherVisibility(IntEnum):
    """Visibility conditions in CARLA simulator, classified by TransFuser."""

    CLEAR = 0
    OK = 1
    LIMITED = 2
    VERY_LIMITED = 3


WEATHER_VISIBILITY_MAPPING = {
    "ClearNight": WeatherVisibility.VERY_LIMITED,
    "ClearNoon": WeatherVisibility.CLEAR,
    "ClearSunset": WeatherVisibility.CLEAR,
    "ClearSunrise": WeatherVisibility.CLEAR,
    # Cloudy weather
    "CloudyNight": WeatherVisibility.VERY_LIMITED,
    "CloudyNoon": WeatherVisibility.CLEAR,
    "CloudySunset": WeatherVisibility.CLEAR,
    "CloudySunrise": WeatherVisibility.CLEAR,
    # Dust storm
    "DustStorm": WeatherVisibility.CLEAR,
    # Hard rain
    "HardRainNight": WeatherVisibility.VERY_LIMITED,
    "HardRainNoon": WeatherVisibility.LIMITED,
    "HardRainSunset": WeatherVisibility.LIMITED,
    "HardRainSunrise": WeatherVisibility.LIMITED,
    # Mid rain
    "MidRainyNight": WeatherVisibility.VERY_LIMITED,
    "MidRainyNoon": WeatherVisibility.OK,
    "MidRainSunset": WeatherVisibility.OK,
    "MidRainSunrise": WeatherVisibility.OK,
    # Soft rain
    "SoftRainNight": WeatherVisibility.VERY_LIMITED,
    "SoftRainNoon": WeatherVisibility.CLEAR,
    "SoftRainSunset": WeatherVisibility.CLEAR,
    "SoftRainSunrise": WeatherVisibility.CLEAR,
    # Wet cloudy
    "WetCloudyNight": WeatherVisibility.VERY_LIMITED,
    "WetCloudyNoon": WeatherVisibility.OK,
    "WetCloudySunset": WeatherVisibility.OK,
    "WetCloudySunrise": WeatherVisibility.OK,
    # Wet
    "WetNight": WeatherVisibility.VERY_LIMITED,
    "WetNoon": WeatherVisibility.OK,
    # Foggy cloudy
    "FoggyCloudyNight": WeatherVisibility.VERY_LIMITED,
    "FoggyCloudyNoon": WeatherVisibility.OK,
    "FoggyCloudySunset": WeatherVisibility.OK,
    "FoggyCloudySunrise": WeatherVisibility.OK,
    # Foggy Wet cloudy
    "FoggyWetCloudyNight": WeatherVisibility.VERY_LIMITED,
    "FoggyWetCloudyNoon": WeatherVisibility.LIMITED,
    "FoggyWetCloudySunset": WeatherVisibility.LIMITED,
    "FoggyWetCloudySunrise": WeatherVisibility.LIMITED,
    # Foggy Wet
    "FoggyWetNoon": WeatherVisibility.OK,
    # Foggy Soft Rain
    "FoggySoftRainNight": WeatherVisibility.VERY_LIMITED,
    "FoggySoftRainNoon": WeatherVisibility.LIMITED,
    "FoggySoftRainSunset": WeatherVisibility.LIMITED,
    "FoggySoftRainSunrise": WeatherVisibility.LIMITED,
    # Foggy Hard Rain
    "FoggyHardRainNight": WeatherVisibility.VERY_LIMITED,
    # Custom weather
    "Custom0": WeatherVisibility.LIMITED,
    "Custom9": WeatherVisibility.LIMITED,
    "Custom10": WeatherVisibility.VERY_LIMITED,
    "Custom11": WeatherVisibility.LIMITED,
    "Custom12": WeatherVisibility.LIMITED,
    "Custom13": WeatherVisibility.VERY_LIMITED,
    "Custom14": WeatherVisibility.LIMITED,
    "Custom15": WeatherVisibility.LIMITED,
    "Custom19": WeatherVisibility.VERY_LIMITED,
    "Custom20": WeatherVisibility.VERY_LIMITED,
    "Custom21": WeatherVisibility.VERY_LIMITED,
}
