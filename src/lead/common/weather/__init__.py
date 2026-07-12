"""Weather presets and weather-to-visibility classification."""

from lead.common.weather.presets import WEATHER_SETTINGS
from lead.common.weather.visibility import (
    WEATHER_VISIBILITY_MAPPING,
    WeatherVisibility,
)

__all__ = [
    "WEATHER_SETTINGS",
    "WEATHER_VISIBILITY_MAPPING",
    "WeatherVisibility",
]
