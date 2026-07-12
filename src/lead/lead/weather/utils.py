"""Weather helper functions for the expert agent."""

import logging
import math
import typing

import carla

if typing.TYPE_CHECKING:
    from lead.config import ExpertConfig

LOG = logging.getLogger(__name__)


def get_night_mode(weather: carla.WeatherParameters) -> bool:
    """Check whether or not the street lights need to be turned on"""
    SUN_ALTITUDE_THRESHOLD_1 = 15
    SUN_ALTITUDE_THRESHOLD_2 = 165

    # For higher fog and cloudness values, the amount of light in scene starts to rapidly decrease
    CLOUDINESS_THRESHOLD = 80
    FOG_THRESHOLD = 40

    # In cases where more than one weather conditition is active, decrease the thresholds
    COMBINED_THRESHOLD = 10

    altitude_dist = weather.sun_altitude_angle - SUN_ALTITUDE_THRESHOLD_1
    altitude_dist = min(
        altitude_dist,
        SUN_ALTITUDE_THRESHOLD_2 - weather.sun_altitude_angle,
    )
    cloudiness_dist = CLOUDINESS_THRESHOLD - weather.cloudiness
    fog_density_dist = FOG_THRESHOLD - weather.fog_density

    # Check each parameter independently
    if altitude_dist < 0 or cloudiness_dist < 0 or fog_density_dist < 0:
        return True

    # Check if two or more values are close to their threshold
    joined_threshold = int(altitude_dist < COMBINED_THRESHOLD)
    joined_threshold += int(cloudiness_dist < COMBINED_THRESHOLD)
    joined_threshold += int(fog_density_dist < COMBINED_THRESHOLD)

    if joined_threshold >= 2:
        return True

    return False


def get_weather_name(
    weather_parameter: carla.WeatherParameters,
    config: "ExpertConfig",
) -> str:
    """
    Return the name of the weather preset matching the given CARLA WeatherParameters.
    Args:
        weather_parameter: The CARLA WeatherParameters object describing current weather.
        config: An expert config containing a dictionary of known weather presets.

    Returns:
        str: The name of the matched preset if found, otherwise the name of the nearest preset.
    """
    best_name, best_dist = None, float("inf")
    NEAREST_NEIGHBOR_KEYS = [
        "cloudiness",
        "dust_storm",
        "fog_density",
        "precipitation",
        "precipitation_deposits",
        "sun_altitude_angle",
        "wetness",
        "wind_intensity",
    ]

    for name, preset in config.data_collection.weather_settings.items():
        # Check exact match
        if all(
            math.isclose(getattr(weather_parameter, key), val, abs_tol=1e-2)
            for key, val in preset.items()
        ):
            return name

        # Compute distance for fallback (restricted keys)
        dist = 0.0
        for key in NEAREST_NEIGHBOR_KEYS:
            diff = getattr(weather_parameter, key) - preset[key]
            dist += diff * diff
        if dist < best_dist:
            best_name, best_dist = name, dist

    LOG.warning(f"Weather preset not found, using nearest {best_name}")
    return best_name


def weather_parameter_to_dict(weather_parameter: carla.WeatherParameters) -> dict:
    """
    Convert a CARLA WeatherParameters object into a plain Python dictionary.

    This extracts the subset of weather-related attributes that are relevant
    for comparison or storage (e.g., matching against preset configurations).

    Args:
        weather_parameter: A CARLA WeatherParameters object.

    Returns:
        dict: A dictionary mapping attribute names to their corresponding float values.
    """
    keys = [
        "cloudiness",
        "dust_storm",
        "fog_density",
        "fog_distance",
        "fog_falloff",
        "mie_scattering_scale",
        "precipitation",
        "precipitation_deposits",
        "rayleigh_scattering_scale",
        "scattering_intensity",
        "sun_altitude_angle",
        "sun_azimuth_angle",
        "wetness",
        "wind_intensity",
    ]
    return {k: getattr(weather_parameter, k) for k in keys}
