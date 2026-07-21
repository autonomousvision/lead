import logging
from collections import deque
from typing import Any

import carla
import numpy as np
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from lead.common.localization import gps
from lead.common.localization.kalman_filter import KalmanFilter
from lead.common.planning import RoutePlanner
from lead.common.runtime_property_caching import step_cached_property
from lead.common.sensors import point_clouds
from lead.config import LeadConfig

LOG = logging.getLogger(__name__)


class BaseAgent:
    """Agent class handle basic sensor processing that both expert and student need."""

    # Set by the composed `AutonomousAgent` base (CARLA leaderboard) via
    # `set_global_plan`, not by this class itself.
    _global_plan_world_coord: list[tuple[carla.Transform, Any]]
    _global_plan: list[tuple[dict[str, float], Any]]

    def setup(self, lead_config: LeadConfig, sensor_agent: bool = False) -> None:
        self.noisy_lat_ref, self.noisy_lon_ref = gps.find_gps_ref(
            self._global_plan_world_coord,
            self._global_plan,
        )
        LOG.info(
            "Noisy lat ref: %s, Noisy lon ref: %s",
            self.noisy_lat_ref,
            self.noisy_lon_ref,
        )
        self.lead_config = lead_config
        self.config_expert = lead_config.expert
        self.kalman_filter = KalmanFilter(self.config_expert)
        self.gnss_uses_transverse_mercator = gps.gnss_uses_transverse_mercator(
            CarlaDataProvider.get_client(),
        )

        self.yaws_queue = deque(
            maxlen=self.config_expert.data_collection.ego_history_length,
        )

        self.control = carla.VehicleControl(steer=0.0, throttle=0.0, brake=1.0)
        self.previous_compass: float | None = None
        self.compass: float | None = None

        # --- Route planner ---
        self.sensor_agent = sensor_agent
        route_planner_max_distance = (
            lead_config.evaluation.controller.route_planner_max_distance
            if sensor_agent
            else self.config_expert.simulation.route_planner_max_distance
        )

        self.gps_waypoint_planners_dict: dict[float, RoutePlanner] = {}
        for dist in self.config_expert.simulation.tp_distances:
            planner = RoutePlanner(
                dist,
                route_planner_max_distance,
            )
            planner.set_route(
                self._global_plan,
                is_gps=True,
                lat_ref=self.noisy_lat_ref,
                lon_ref=self.noisy_lon_ref,
            )
            self.gps_waypoint_planners_dict[dist] = planner

    def tick(self, input_data: dict) -> dict:
        # Get the vehicle's speed from sensor
        speed = input_data["speed"][1]["speed"]

        # Preprocess the compass data from the IMU
        self.previous_compass = self.compass
        compass = gps.preprocess_compass(
            input_data["imu"][1][-1],
        )  # Range [-pi,pi]
        if self.previous_compass is not None:
            compass = float(
                np.unwrap([self.previous_compass, compass])[1],
            )  # Unbounded range
        self.compass = compass
        self.yaws_queue.append(self.compass)

        # Filter the GPS position with Kalman filter
        noisy_gps_pos = gps.convert_gnss_to_carla(
            input_data["gps"][1],
            self.noisy_lat_ref,
            self.noisy_lon_ref,
            self.gnss_uses_transverse_mercator,
        )
        self.filtered_state = self.kalman_filter.step(
            noisy_position=noisy_gps_pos,
            compass=self.compass,
            speed=speed,
            control=self.control,
        )
        self.filtered_history = np.array([self.kalman_filter.history_x]).reshape(-1, 4)[
            :,
            :2,
        ]

        # The ego localizes itself, never on the ground truth pose: with the
        # filter, or on the raw noisy GPS the filter takes as its measurement.
        self.localized_position = (
            self.filtered_state[:2]
            if self.config_expert.simulation.use_kalman_filter
            else noisy_gps_pos[:2]
        )
        for planner in self.gps_waypoint_planners_dict.values():
            planner.run_step(np.append(self.localized_position, noisy_gps_pos[2]))

        # Create a dictionary containing the vehicle's state
        input_data.update(
            {
                "theta": self.compass,
                "localized_position": self.localized_position,
                "speed": speed,
            },
        )

        # --- LiDAR (the rig always has two) ---
        input_data["lidar"] = np.concatenate(
            (
                point_clouds.lidar_to_ego_coordinate(
                    self.config_expert.sensor_rig.lidar_rot_1,
                    self.config_expert.sensor_rig.lidar_pos_1,
                    input_data["lidar1"],
                ),
                point_clouds.lidar_to_ego_coordinate(
                    self.config_expert.sensor_rig.lidar_rot_2,
                    self.config_expert.sensor_rig.lidar_pos_2,
                    input_data["lidar2"],
                ),
            ),
            axis=0,
        )
        lidar_x, lidar_y = input_data["lidar"][:, 0], input_data["lidar"][:, 1]
        # Remove lidar points inside ego bounding boxes. We already need LiDAR for expert.
        input_data["lidar"] = input_data["lidar"][
            (np.abs(lidar_x) > self.config_expert.simulation.ego_extent_x)
            & (np.abs(lidar_y) > self.config_expert.simulation.ego_extent_y)
        ]
        # The raw sweep is what gets stored and what the expert's box labels
        # count hits on; model-specific processing (ground removal, radar
        # merging) is the driving agent's and the data loader's business.
        input_data["lidar_sweep"] = input_data["lidar"]

        # --- Radar ---
        if self.config_expert.sensor_rig.use_radars:
            for i, radar_calibration in enumerate(
                self.config_expert.sensor_rig.radars,
                start=1,
            ):
                input_data[f"radar{i}"] = point_clouds.radar_points_to_ego(
                    input_data[f"radar{i}"][1],
                    sensor_pos=radar_calibration["pos"],
                    sensor_rot=radar_calibration["rot"],
                )

        # --- Process camera images ---
        # CARLA cameras
        for camera_idx in range(1, self.config_expert.sensor_rig.num_cameras + 1):
            # The expert's RGB cameras only produce data on save ticks
            if f"rgb_{camera_idx}" not in input_data:
                assert not self.sensor_agent, f"rgb_{camera_idx} missing"
                continue
            input_data[f"rgb_{camera_idx}"] = input_data[f"rgb_{camera_idx}"][1][
                :,
                :,
                :3,
            ]
        return input_data

    @step_cached_property
    def ego_past_positions(self) -> tuple[tuple[float, float], ...]:
        """Return ego past position in current ego frame. Goes from the oldest (i = 0) to current step (i = -1)."""
        past_states = self.filtered_history[:, :2]
        if len(past_states) == 0 or self.compass is None:
            return ()
        current_pos = past_states[-1]
        current_yaw = self.compass
        R_world_to_current = np.array(
            [
                [np.cos(-current_yaw), -np.sin(-current_yaw)],
                [np.sin(-current_yaw), np.cos(-current_yaw)],
            ],
        )
        return tuple(
            (x, y)
            for x, y in ((past_states - current_pos) @ R_world_to_current.T).tolist()
        )

    @step_cached_property
    def ego_past_yaws(self) -> tuple[float, ...]:
        """Return ego past yaws in current ego frame. Goes from the oldest (i = 0) to current step (i = -1)."""
        if len(self.yaws_queue) == 0:
            return ()
        past_yaws = []
        yaw_now = self.yaws_queue[-1]
        for yaw in self.yaws_queue:
            dyaw = (yaw - yaw_now + np.pi) % (2 * np.pi) - np.pi
            past_yaws.append(dyaw)
        return tuple(past_yaws)
