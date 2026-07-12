"""Abstract CARLA leaderboard agent wrapping a driving policy for evaluation."""

import abc
import json
import logging
import os
import shutil
import typing

import carla
import numpy as np
import torch
import yaml
from agents.navigation.local_planner import RoadOption
from leaderboard.autoagents import autonomous_agent
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from lead.common.base_agent import BaseAgent
from lead.config import load_lead_config
from lead.evaluation.recorder.infraction_recorder import InfractionRecorder
from lead.evaluation.recorder.video_recorder import VideoRecorder
from lead.lead import weather as expert_weather
from lead.lead.sensor_rig import av_sensor_setup

LOG = logging.getLogger(__name__)


class AbstractDrivingAgent(BaseAgent, autonomous_agent.AutonomousAgent, abc.ABC):
    """CARLA leaderboard protocol adapter around a driving policy.

    Owns everything policy-agnostic about driving one evaluation route:
    rebuilding the config tree from the checkpoint, the sensor rig, weather,
    infraction and video recording, and the :meth:`run_step` skeleton.
    Subclasses wrap a concrete policy by implementing :meth:`setup_policy` and
    :meth:`compute_control`, and may override :meth:`save_step_visualizations`.
    """

    def setup(
        self,
        path_to_conf_file: str,
        route_index: str | None = None,
        traffic_manager: carla.TrafficManager | None = None,
    ) -> None:
        self.config_path = path_to_conf_file
        # Bench2Drive appends "+..." to the checkpoint path.
        checkpoint_dir = path_to_conf_file.split("+")[0]

        # Rebuild the config tree saved during training; env/CLI overrides
        # apply on top, and configs written before knob renames stay loadable.
        with open(
            os.path.join(checkpoint_dir, "config.yaml"),
            encoding="utf-8",
        ) as f:
            stored_config = yaml.safe_load(f)
        lead_config = load_lead_config(
            loaded_config=stored_config,
            raise_error_on_missing_key=False,
        )

        super().setup(lead_config, sensor_agent=True)
        self.step = -1
        self.initialized = False
        self.device = torch.device("cuda:0")

        self.setup_policy(checkpoint_dir)

        self.metric_info = {}
        self.meters_travelled = 0.0

        # Infraction tracking
        self.infraction_recorder = InfractionRecorder(
            config_evaluation=lead_config.evaluation,
            agent_name=type(self).__name__,
        )

        self.track = autonomous_agent.Track.SENSORS

        if not shutil.which("ffmpeg"):
            raise RuntimeError(
                "ffmpeg is not installed or not found in PATH. Please install ffmpeg to use video compression.",
            )

    @abc.abstractmethod
    def setup_policy(self, checkpoint_dir: str) -> None:
        """Build the wrapped policy and everything needed to run it.

        Called once during :meth:`setup`, after the config tree
        (``self.lead_config``) and ``self.device`` are available.

        Args:
            checkpoint_dir: Directory holding the trained model checkpoint(s).
        """

    @abc.abstractmethod
    def compute_control(self, input_data: dict) -> carla.VehicleControl:
        """Run the policy on one pre-processed tick and compute the control.

        Args:
            input_data: Sensor data processed by :meth:`tick`.

        Returns:
            The vehicle control to apply this step.
        """

    def save_step_visualizations(self, input_data: dict) -> None:
        """Save per-step visualizations of the last control computation; no-op by default.

        Args:
            input_data: Sensor data processed by :meth:`tick`.
        """

    def set_global_plan(
        self,
        global_plan_gps: typing.Any,
        privileged_org_dense_route_world_coord: list[
            tuple[carla.Transform, RoadOption]
        ],
    ) -> None:
        """Store the global plan for privileged logging .
        The dense world-coordinate route is stored as privileged information for
        offline logging and metric computation and must not be
        used by the driving policy. This is expected to be called by the
        leaderboard/runner before the scenario starts and before `_init`
        initialises components that consume the stored plan.

        Args:
            global_plan_gps: Global route waypoints in GPS space as provided by
                the leaderboard.
            privileged_org_dense_route_world_coord: Dense global route in world coordinates.
        """
        self.privileged_org_dense_route_world_coord = (
            privileged_org_dense_route_world_coord
        )
        LOG.info(
            "Set global plan with %d waypoints.",
            len(self.privileged_org_dense_route_world_coord),
        )
        super().set_global_plan(global_plan_gps, privileged_org_dense_route_world_coord)

    def set_scenario(self, scenario) -> None:
        """Set the scenario reference to track infractions.

        This should be called by the leaderboard after loading the scenario.
        """
        self.infraction_recorder.set_scenario(scenario)
        LOG.info(
            "[%s] Scenario reference set for infraction tracking",
            type(self).__name__,
        )

    def _init(self) -> None:
        # Get the hero vehicle and the CARLA world
        self._vehicle: carla.Actor = CarlaDataProvider.get_hero_actor()
        self._world: carla.World = self._vehicle.get_world()

        # Set up video recorder
        self.video_recorder = VideoRecorder(
            config_evaluation=self.lead_config.evaluation,
            vehicle=self._vehicle,
            world=self._world,
            step_counter=self.step,
            lead_config=self.lead_config,
        )

        self.set_weather()

        self.initialized = True

    def set_weather(self) -> None:
        weather_name = None

        if self.lead_config.evaluation.random_weather:
            weathers = self.lead_config.expert.data_collection.weather_settings.keys()
            weather_name = np.random.choice(list(weathers))

        if self.lead_config.evaluation.custom_weather is not None:
            weather_name = self.lead_config.evaluation.custom_weather

        if weather_name is not None:
            weather = carla.WeatherParameters(
                **self.lead_config.expert.data_collection.weather_settings[
                    weather_name
                ],
            )
            self._world.set_weather(weather)
            LOG.info(f"Set weather to: {weather_name}")
            # night mode
            vehicles = self._world.get_actors().filter("*vehicle*")
            if expert_weather.get_night_mode(weather):
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

    def sensors(self) -> list[dict]:
        return av_sensor_setup(
            config=self.lead_config.expert,
            lidar=True,
            radar=True,
            sensor_agent=True,
            perturbate=False,
            perturbation_rotation=0.0,
            perturbation_translation=0.0,
        )

    def check_infractions(self) -> None:
        """Check and record infractions for the current rollout step."""
        self.infraction_recorder.check_infractions(
            step=self.step,
            meters_travelled=self.meters_travelled,
        )

    @torch.inference_mode()
    def run_step(self, input_data: dict, _, __=None) -> carla.VehicleControl:
        """Drive one simulation step: tick, run the policy, record the outcome.

        Args:
            input_data: Raw sensor data provided by the leaderboard.

        Returns:
            The vehicle control to apply this step.
        """
        self.step += 1

        if not self.initialized:
            self._init()
            self.control = carla.VehicleControl(steer=0.0, throttle=0.0, brake=1.0)
            input_data = self.tick(input_data)
            return self.control

        # Update video recorder step and demo cameras
        if hasattr(self, "video_recorder"):
            self.video_recorder.update_step(self.step)
            self.video_recorder.move_demo_cameras_with_ego()

        # Need to run this every step for GPS filtering
        input_data = self.tick(input_data)

        self.control = self.compute_control(input_data)

        self.meters_travelled += (
            input_data["speed"].item()
            * self.lead_config.expert.simulation.carla_frame_rate
        )
        input_data["meters_travelled"] = self.meters_travelled

        # CARLA will not let the car drive in the initial frames. This help the filter not get confused.
        if self.step < self.lead_config.expert.simulation.inital_frames_delay:
            self.control = carla.VehicleControl(0.0, 0.0, 1.0)

        # Check for infractions at this step
        self.check_infractions()

        self.save_step_visualizations(input_data)

        # Save metric info if in Bench2Drive mode
        if self.lead_config.evaluation.is_bench2drive and hasattr(
            self,
            "get_metric_info",
        ):
            metric = self.get_metric_info()
            self.metric_info[self.step] = metric
            with open(
                f"{self.lead_config.evaluation.save_path}/metric_info.json",
                "w",
            ) as outfile:
                json.dump(self.metric_info, outfile, indent=4)
        return self.control

    def destroy(self, results=None) -> None:
        LOG.info(results)

        # Clean up video recorder
        if hasattr(self, "video_recorder"):
            self.video_recorder.cleanup()
