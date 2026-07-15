"""Weather shuffling for visual diversity, and matching CARLA weather back to a preset."""

import dataclasses
import logging
import math
import random

import carla
import numpy as np

from lead.config import ExpertConfig
from lead.expert.utils.weather_presets import WEATHER_PRESETS

LOG = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class WeatherState:
    """The weather a run is driving in, as the expert records it."""

    setting: str
    parameters: dict[str, float]
    visibility: int


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


def get_weather_name(weather_parameter: carla.WeatherParameters) -> str:
    """
    Return the name of the weather preset matching the given CARLA WeatherParameters.
    Args:
        weather_parameter: The CARLA WeatherParameters object describing current weather.

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

    for name, preset in WEATHER_PRESETS.items():
        # Check exact match
        if all(
            math.isclose(getattr(weather_parameter, key), val, abs_tol=1e-2)
            for key, val in preset.parameters.items()
        ):
            return name

        # Compute distance for fallback (restricted keys)
        dist = 0.0
        for key in NEAREST_NEIGHBOR_KEYS:
            diff = getattr(weather_parameter, key) - preset.parameters[key]
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


def observe(weather: carla.WeatherParameters) -> WeatherState:
    """Describe the weather a world is already in, without changing it.

    Args:
        weather: The world's current weather.

    Returns:
        The matching preset name, its parameters and its visibility.
    """
    setting = get_weather_name(weather)
    return WeatherState(
        setting=setting,
        parameters=weather_parameter_to_dict(weather),
        visibility=int(WEATHER_PRESETS[setting].visibility),
    )


def sample_parameters(config: ExpertConfig) -> tuple[str, dict[str, float]]:
    """Draw a weather preset and jitter its parameters for visual diversity.

    Args:
        config: Expert config selecting between nice and randomized weather.

    Returns:
        The chosen preset name and its jittered parameters.
    """
    if config.data_collection.nice_weather:
        setting = "ClearNoon"
        LOG.info(f"Chose nice weather {setting}")
    else:
        setting = random.choice(list(WEATHER_PRESETS.keys()))
        LOG.info(f"Chose random weather {setting}")

    parameters: dict[str, float] = dict(WEATHER_PRESETS[setting].parameters)

    if "Noon" in setting:
        parameters["sun_altitude_angle"] += np.random.uniform(-45.0, 45.0)
    elif "Custom" not in setting:
        parameters["sun_altitude_angle"] += np.random.uniform(-15.0, 15.0)

    for key in ["wind_intensity", "fog_density", "wetness"]:
        if parameters[key] < 30:
            parameters[key] += np.random.uniform(-5.0, 5.0)
        elif parameters[key] < 80:
            parameters[key] += np.random.uniform(-10.0, 10.0)
        else:
            parameters[key] += np.random.uniform(-5.0, 5.0)
        parameters[key] = np.clip(parameters[key], 0.0, 100.0)

    return setting, parameters


def shuffle(carla_world: carla.World, config: ExpertConfig) -> WeatherState:
    """Randomize the world's weather for visual diversity.

    Applies the sampled weather and matching vehicle lights; when shuffling is
    disabled, the world is left alone and its current weather is reported as-is.

    Args:
        carla_world: The world whose weather to set.
        config: Expert config selecting between nice, randomized and unchanged weather.

    Returns:
        The weather the world is in once this returns.
    """
    LOG.info("Shuffling weather settings")

    if not (
        config.data_collection.shuffle_weather or config.data_collection.nice_weather
    ):
        state = observe(carla_world.get_weather())
        LOG.info(f"Current weather setting: {state.setting}")
        return state

    setting, parameters = sample_parameters(config)

    weather = carla.WeatherParameters(**parameters)
    carla_world.set_weather(weather)

    lights = (
        carla.VehicleLightState(
            carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam,
        )
        if get_night_mode(weather)
        else carla.VehicleLightState.NONE
    )
    for vehicle in carla_world.get_actors().filter("*vehicle*"):
        vehicle.set_light_state(lights)

    LOG.info(f"Current weather setting: {setting}")
    return WeatherState(
        setting=setting,
        parameters=parameters,
        visibility=int(WEATHER_PRESETS[setting].visibility),
    )
