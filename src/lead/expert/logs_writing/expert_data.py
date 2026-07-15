"""Log writing of the expert agent, composed from category mixins.

``ExpertData`` glues together sensor processing, bounding-box extraction,
Py123D logging and debug visualization on top of the world state.
"""

import logging
import os
import pathlib
from collections import abc

import carla
import numpy as np
from agents.navigation.local_planner import LocalPlanner

from lead.common.pid import LongitudinalController
from lead.expert.driving import forecast_kernels
from lead.expert.driving.kinematic_bicycle_model import KinematicBicycleModel
from lead.expert.logs_writing.bounding_boxes import BoundingBoxesMixin
from lead.expert.logs_writing.py123d_logging import Py123dLoggingMixin
from lead.expert.logs_writing.sensor_processing import SensorProcessingMixin
from lead.expert.logs_writing.visualization import DataVisualizationMixin
from lead.expert.utils import weather
from lead.expert.world_state import WorldStateMixin

LOG = logging.getLogger(__name__)


class NegativeIdCounter:
    """ID generator that maps keys to negative IDs, creating new ones only when needed.

    The mapping is memoized so that a key keeps its ID across frames, which is
    what lets per-box futures be matched over time.
    """

    def __init__(self, start_value: int = -100000) -> None:
        """
        Initialize the ID generator

        Args:
            start_value: Starting value for new IDs (default: -100000)
        """
        self._id_map: dict[abc.Hashable, int] = {}
        self._current = start_value

    def __call__(self, key: abc.Hashable) -> int:
        """
        Get the ID for a key, creating a new negative ID on first use.

        Args:
            key: The key to get an ID for. Must identify the box uniquely, so
                that distinct boxes never collapse onto one ID.

        Returns:
            The ID associated with the key
        """
        if key in self._id_map:
            return self._id_map[key]

        # Create new ID
        new_id = self._current
        self._id_map[key] = new_id
        self._current -= 1
        return new_id


class ExpertData(
    SensorProcessingMixin,
    Py123dLoggingMixin,
    BoundingBoxesMixin,
    DataVisualizationMixin,
    WorldStateMixin,
):
    """Data collection of the expert: composes recorders into 123D Arrow logs."""

    def setup(
        self,
        path_to_conf_file: str,
        route_index: str | None = None,
        traffic_manager: carla.TrafficManager | None = None,
    ):
        """
        Set up the autonomous agent for the CARLA simulation.

        Args:
            path_to_conf_file: Path to the configuration file.
            route_index: Index of the route to follow.
            traffic_manager: The traffic manager object.
        """
        super().setup(path_to_conf_file, route_index, traffic_manager)

        # Dynamics models
        self.vehicle_model = KinematicBicycleModel(self.config_expert)

        # To avoid failing the ActorBlockedTest, the agent has to move at least 0.1 m/s every 179 ticks
        self.waiting_ticks_at_stop_sign = 0
        self.ego_blocked_for_ticks = 0

        # Controllers
        self.perturbation_translation = 0
        self.perturbation_rotation = 0

        # Save path is only a datagen-enabled marker now: all data lives in the
        # 123D logs, so no directory is created here.
        if os.environ.get("SAVE_PATH", None) is not None:
            self.save_path = pathlib.Path(os.environ["SAVE_PATH"]) / self.log_name

        self.negative_id_counter = NegativeIdCounter()
        self.traffic_manager = traffic_manager

        self.weather_setting = "ClearNoon"

        # Py123D writers: the expert emits Arrow logs directly
        if self._writes_py123d:
            self._setup_py123d_writers()

    def _init(self) -> None:
        """
        Initialize the agent by setting up the route planner, longitudinal controller,
        command planner, and other necessary components.
        """
        super()._init()

        # Set up the longitudinal controller
        self._longitudinal_controller = LongitudinalController(self.config_expert)

        if self.config_expert.data_collection.datagen:
            weather_state = weather.shuffle(self.carla_world, self.config_expert)
            self.weather_setting = weather_state.setting
            self.weather_parameters = weather_state.parameters
            self.visual_visibility = weather_state.visibility
        jpeg_storage_quality_distribution = (
            self.config_expert.data_collection.weather_jpeg_compression_quality[
                self.weather_setting
            ]
        )  # key value: quality maps to probability
        if self.config_expert.data_collection.jpeg_compression:
            self.jpeg_storage_quality = int(
                np.random.choice(
                    list(jpeg_storage_quality_distribution.keys()),
                    p=list(jpeg_storage_quality_distribution.values()),
                ),
            )
        else:
            self.jpeg_storage_quality = 90
        LOG.info(f"[DataAgent] Chose JPEG storage quality {self.jpeg_storage_quality}")

        if self._writes_py123d:
            self._init_py123d()

        self._local_planner = LocalPlanner(
            self.ego_vehicle,
            opt_dict={},
            map_inst=self.carla_world_map,
        )
        forecast_kernels.warmup(self.config_expert)  # Pre-compile numba code

        self.initialized = True
