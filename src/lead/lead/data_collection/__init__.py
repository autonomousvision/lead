"""Data collection of the expert agent, composed from category mixins.

``ExpertData`` glues together sensor processing, bounding-box extraction,
Py123D logging, debug visualization and weather shuffling on top of the
world state.
"""

import logging
import os
import pathlib
from collections import deque

import carla
import numpy as np
from agents.navigation.local_planner import LocalPlanner

import lead.lead.utils.expert_utils as expert_utils
from lead.common import constants
from lead.common.control.pid_controller import ExpertLongitudinalController
from lead.common.planning import RoutePlanner
from lead.common.sensors import ransac_fast
from lead.lead.data_collection.bounding_boxes import BoundingBoxesMixin
from lead.lead.data_collection.py123d_logging import Py123dLoggingMixin
from lead.lead.data_collection.sensor_processing import SensorProcessingMixin
from lead.lead.data_collection.visualization import DataVisualizationMixin
from lead.lead.dynamics.kinematic_bicycle_model import KinematicBicycleModel
from lead.lead.planning import forecast_kernels
from lead.lead.weather import WeatherMixin
from lead.lead.world_state import WorldStateMixin

LOG = logging.getLogger(__name__)


class ExpertData(
    SensorProcessingMixin,
    Py123dLoggingMixin,
    BoundingBoxesMixin,
    DataVisualizationMixin,
    WeatherMixin,
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
        self.ego_model = KinematicBicycleModel(self.config_expert)
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
            self.save_path = (
                pathlib.Path(os.environ["SAVE_PATH"])
                / f"{self.town}_Rep{self.rep}_{self.route_index}"
            )

        self.transform_queue = deque(
            maxlen=self.config_expert.storage.ego_num_temporal_data_points_saved + 1,
        )
        # Ego world positions of the same ticks as ``transform_queue``
        self.ego_position_queue = deque(
            maxlen=self.config_expert.storage.ego_num_temporal_data_points_saved + 1,
        )
        self.negative_id_counter = expert_utils.NegativeIdCounter()
        self.traffic_manager = traffic_manager

        self.weather_setting = "ClearNoon"
        self.semantics_converter = np.uint8(
            list(constants.SEMANTIC_SEGMENTATION_CONVERTER.values()),
        )

        # Py123D writers: the expert emits Arrow logs directly
        if self._writes_py123d:
            self._setup_py123d_writers()

    def _init(self) -> None:
        """
        Initialize the agent by setting up the route planner, longitudinal controller,
        command planner, and other necessary components.
        """
        super()._init()

        # Set up the longitudinal controller and target-point planners
        self._longitudinal_controller = ExpertLongitudinalController(self.config_expert)
        self._command_planner = RoutePlanner(
            self.config_expert.simulation.route_planner_min_distance,
            self.config_expert.simulation.route_planner_max_distance,
        )
        self._command_planner.set_route(self._global_plan_world_coord)

        self._command_planners_dict = {}
        for dist in self.config_expert.simulation.tp_distances:
            planner = RoutePlanner(
                dist,
                self.config_expert.simulation.route_planner_max_distance,
            )
            planner.set_route(self._global_plan_world_coord)
            self._command_planners_dict[dist] = planner

        if self.config_expert.data_collection.datagen:
            self.shuffle_weather()
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
        ransac_fast.remove_ground_fast(
            np.random.rand(1000, 3),
            self.config_expert,
        )  # Pre-compile numba code
        forecast_kernels.warmup(self.config_expert)  # Pre-compile numba code

        self.initialized = True


__all__ = [
    "BoundingBoxesMixin",
    "DataVisualizationMixin",
    "ExpertData",
    "Py123dLoggingMixin",
    "SensorProcessingMixin",
]
