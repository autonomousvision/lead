"""Mixin handling weather shuffling and weather-derived state of the expert."""

import logging
import random

import carla
import numpy as np

from lead.common import weather as common_weather
from lead.lead.weather import utils as weather_utils

LOG = logging.getLogger(__name__)


class WeatherMixin:
    """Shuffles the CARLA weather for visual diversity and tracks its state.

    Maintains ``weather_setting``, ``weather_parameters`` and
    ``visual_visibility`` on the agent.
    """

    def shuffle_weather(self) -> None:
        LOG.info("Shuffling weather settings")
        # change weather for visual diversity
        weather = self.carla_world.get_weather()

        if (
            self.config_expert.data_collection.shuffle_weather
            or self.config_expert.data_collection.nice_weather
        ):
            if self.config_expert.data_collection.nice_weather:
                self.weather_setting = "ClearNoon"
                LOG.info(f"Chose nice weather {self.weather_setting}")
            else:
                self.weather_setting = random.choice(
                    list(common_weather.WEATHER_SETTINGS.keys()),
                )
                LOG.info(f"Chose random weather {self.weather_setting}")
            LOG.info(f"Chose weather {self.weather_setting}")
            self.weather_parameters: dict[str, float] = common_weather.WEATHER_SETTINGS[
                self.weather_setting
            ]

            if "Noon" in self.weather_setting:
                self.weather_parameters["sun_altitude_angle"] += np.random.uniform(
                    -45.0,
                    45.0,
                )
            elif "Custom" not in self.weather_setting:
                self.weather_parameters["sun_altitude_angle"] += np.random.uniform(
                    -15.0,
                    15.0,
                )

            for randomizing_parameter in ["wind_intensity", "fog_density", "wetness"]:
                if self.weather_parameters[randomizing_parameter] < 30:
                    self.weather_parameters[randomizing_parameter] += np.random.uniform(
                        -5.0,
                        5.0,
                    )
                elif self.weather_parameters[randomizing_parameter] < 80:
                    self.weather_parameters[randomizing_parameter] += np.random.uniform(
                        -10.0,
                        10.0,
                    )
                else:
                    self.weather_parameters[randomizing_parameter] += np.random.uniform(
                        -5.0,
                        5.0,
                    )
                self.weather_parameters[randomizing_parameter] = np.clip(
                    self.weather_parameters[randomizing_parameter],
                    0.0,
                    100.0,
                )

            weather = carla.WeatherParameters(**self.weather_parameters)

            self.carla_world.set_weather(weather)

            # night mode
            vehicles = self.carla_world.get_actors().filter("*vehicle*")
            if weather_utils.get_night_mode(weather):
                for vehicle in vehicles:
                    vehicle.set_light_state(
                        carla.VehicleLightState(
                            carla.VehicleLightState.Position
                            | carla.VehicleLightState.LowBeam,
                        ),
                    )
            else:
                for vehicle in vehicles:
                    vehicle.set_light_state(carla.VehicleLightState.NONE)
        else:
            self.weather_setting = weather_utils.get_weather_name(
                weather,
                self.config_expert,
            )
            self.weather_parameters = weather_utils.weather_parameter_to_dict(weather)

        LOG.info(f"Current weather setting: {self.weather_setting}")
        self.visual_visibility = int(
            common_weather.WEATHER_VISIBILITY_MAPPING[self.weather_setting],
        )
