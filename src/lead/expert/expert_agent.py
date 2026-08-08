"""The LEAD expert: orchestrates path planning, speed planning and data collection."""

import cProfile
import io
import logging
import os
import pathlib
import pstats
import time
import typing

import carla
import matplotlib
import numpy as np

from lead.api import py123d_log_api
from lead.common.logging_setup import setup_logging
from lead.common.pid import LateralPIDController
from lead.expert.driving.path_planning import PathPlanningMixin
from lead.expert.driving.speed_planning import SpeedPlanningMixin
from lead.expert.logs_writing import ExpertData
from lead.expert.utils import roadgraph
from lead.expert.utils.scenario_sorter import ScenarioSorter

matplotlib.use("Agg")  # non-GUI backend for headless servers


setup_logging()
LOG = logging.getLogger(__name__)


def get_entry_point() -> str:
    return "Expert"


class Expert(PathPlanningMixin, SpeedPlanningMixin, ExpertData):
    """Privileged expert agent driving in CARLA simulations.

    Each step it reshapes the route around obstacles (path planning), selects a
    target speed (speed planning), converts both into vehicle controls, and
    hands the tick over to the data-collection recorders.
    """

    def setup(
        self,
        path_to_conf_file: str,
        route_index: str | None = None,
        traffic_manager: carla.TrafficManager | None = None,
    ):
        super().setup(path_to_conf_file, route_index, traffic_manager)

        self._turn_controller = LateralPIDController(self.config_expert)
        self.cleared_stop_sign = False

        # Initialize scenario sorter (will be configured with route in _init)
        self.scenario_sorter = ScenarioSorter()

        if self.config_expert.data_collection.profile_expert:
            # The profiler is reset each snapshot for per-interval logs; each
            # interval is merged into episode_stats for the full-run dump.
            self.profiler = cProfile.Profile()
            self.profiler.enable()
            self.episode_stats: pstats.Stats | None = None

    def destroy(self, results=None) -> None:
        if self.config_expert.data_collection.profile_expert:
            self._accumulate_profile_interval(restart=False)

            # Save to file for visualization
            try:
                profile_dir = pathlib.Path(os.environ["PY123D_DATA_ROOT"]) / "profiles"
                profile_dir.mkdir(parents=True, exist_ok=True)
                profile_path = profile_dir / f"{self.log_name}.prof"
                self.episode_stats.dump_stats(str(profile_path))
                LOG.info(
                    f"Profile stats saved to '{profile_path}'. Run 'pip install snakeviz && snakeviz {profile_path}' to view.",
                )
            except Exception as e:
                LOG.error(f"Failed to dump profile stats: {e}")

            s = io.StringIO()
            self.episode_stats.stream = s
            self.episode_stats.sort_stats("cumulative").print_stats("src/lead/")
            LOG.info(s.getvalue())
        super().destroy(results)

    def run_step(
        self,
        input_data: dict,
        _: float,
        __: list[list[str | typing.Any]] | None,
    ) -> carla.VehicleControl:
        """
        Run a single step of the agent's control loop.

        Each step runs six phases: basic path planning, sensor collection,
        detections collection, path refining, target speed planning and data
        writing.

        Args:
            input_data: Input data for the current step.

        Returns:
            The control commands.
        """
        start_time = time.time()

        # Initialize the agent if not done yet
        if not self.initialized:
            self._init()
        self.step += 1

        # --- Basic path planning ---
        self._plan_basic_path()

        # --- Sensor collection ---
        input_data = self.tick(input_data)

        # --- Detections collection ---
        self._collect_detections()

        # --- Path refining: reshape the route around obstacle scenarios ---
        self._plan_target_speed_limit()
        target_speed_route_obstacle, obstacle_override, speed_reduced_by_obj = (
            self._solve_obstacle_scenarios(self.target_speed_limit)
        )

        # --- Target speed planning ---
        target_speed, brake, speed_reduced_by_obj = self._plan_target_speed(
            target_speed_route_obstacle,
            obstacle_override,
            speed_reduced_by_obj,
        )
        control = self.get_control(brake, target_speed)

        # --- Write data ---
        self._write_data(
            control,
            target_speed,
            input_data,
            speed_reduced_by_obj,
            start_time,
        )
        return control

    def _plan_basic_path(self) -> None:
        """Advance the privileged route planner along the ego's route."""
        self.privileged_route_planner.run_step(self.ego_location_array)

    def _collect_detections(self) -> None:
        """Sort active scenarios by distance and reset the per-step hazard flags."""
        self.scenario_sorter.sort_scenarios(self.ego_vehicle.get_location())

        self.stop_sign_close = False
        self.walker_close = False
        self.walker_close_id = None
        self.vehicle_hazard = False
        self.vehicle_affecting_id = None
        self.walker_hazard = False
        self.walker_affecting_id = None
        self.traffic_light_hazard = False
        self.stop_sign_hazard = False
        self.europe_traffic_light = self.over_head_traffic_light = False

    def _write_data(
        self,
        control: carla.VehicleControl,
        target_speed: float,
        input_data: dict,
        speed_reduced_by_obj: list | None,
        start_time: float,
    ) -> None:
        """Record driving meta, sensor data and debug visualizations for this step.

        Args:
            control: The control commands for the current frame.
            target_speed: The target speed for the current frame.
            input_data: Processed sensor data of the current step.
            speed_reduced_by_obj: Information about the object that caused speed reduction.
            start_time: Wall-clock time at the start of the step, for timing logs.
        """
        if self.config_expert.visualization.visualize_route:
            self._visualize_route()
        if self.config_expert.visualization.visualize_original_route:
            self._visualize_original_route()

        if not self.config_expert.data_collection.eval_expert:
            self.driving_meta = self.save_meta(
                control,
                target_speed,
                input_data,
                speed_reduced_by_obj,
            )

        if self.save_path is not None and self.config_expert.data_collection.datagen:
            self.save_sensors(input_data)
        if self.step % self.config_expert.data_collection.log_info_freq == 0:
            elapsed_time = time.time() - start_time
            LOG.info(f"Step: {self.step}, Time per step: {elapsed_time * 1000:.2f} ms")

        if (
            self.config_expert.data_collection.profile_expert
            and self.config_expert.data_collection.profile_expert_freq
            and self.step % self.config_expert.data_collection.profile_expert_freq == 0
        ):
            self._log_profile_snapshot()

    def _log_profile_snapshot(self) -> None:
        """Print profiler stats for this interval, then merge and reset for the next one."""
        self.profiler.disable()
        s = io.StringIO()
        ps = pstats.Stats(self.profiler, stream=s).sort_stats("cumulative")
        ps.print_stats("src/lead/", 30)
        LOG.info(f"Profile snapshot at step {self.step}:\n{s.getvalue()}")

        self._accumulate_profile_interval(restart=True)

    def _accumulate_profile_interval(self, restart: bool) -> None:
        """Merge the current interval into the episode stats, optionally starting a fresh interval.

        Args:
            restart: Whether to disable the current profiler and start a fresh one. Snapshots
                restart to keep collecting; teardown does not.
        """
        self.profiler.disable()
        if self.episode_stats is None:
            self.episode_stats = pstats.Stats(self.profiler)
        else:
            self.episode_stats.add(self.profiler)

        if restart:
            self.profiler = cProfile.Profile()
            self.profiler.enable()

    def save_meta(
        self,
        control: carla.VehicleControl,
        target_speed: float,
        tick_data: dict,
        speed_reduced_by_obj: list | None,
    ) -> dict:
        """
        Save the driving data for the current frame.

        Args:
            control: The control commands for the current frame.
            target_speed: The target speed for the current frame.
            tick_data: Dictionary containing the current state of the vehicle.
            speed_reduced_by_obj: List containing information about the object that caused speed reduction.

        Returns:
            A dictionary containing the driving data for the current frame.
        """
        # Get the remaining route points in the global frame (like the target
        # points), so consumers can convert to the ego frame of their view
        # (normal or perturbed) at load time.
        remaining_route = self.remaining_route[
            : self.config_expert.simulation.num_route_points_saved
        ]
        remaining_route_original = self.remaining_route_original[
            : self.config_expert.simulation.num_route_points_saved
        ]
        dense_route = [checkpoint[:2].tolist() for checkpoint in remaining_route]
        dense_route_original = [
            checkpoint[:2].tolist() for checkpoint in remaining_route_original
        ]

        changed_route = bool(
            (
                self.privileged_route_planner.route_points[
                    self.privileged_route_planner.route_index
                ]
                != self.privileged_route_planner.original_route_points[
                    self.privileged_route_planner.route_index
                ]
            ).any(),
        )

        # Extract speed reduction object information
        (
            speed_reduced_by_obj_type,
            speed_reduced_by_obj_id,
            speed_reduced_by_obj_distance,
        ) = None, None, None
        if speed_reduced_by_obj is not None:
            (
                speed_reduced_by_obj_type,
                speed_reduced_by_obj_id,
                speed_reduced_by_obj_distance,
            ) = speed_reduced_by_obj[1:]
            # Convert numpy to float so that it can be saved to json.
            if speed_reduced_by_obj_distance is not None:
                speed_reduced_by_obj_distance = float(speed_reduced_by_obj_distance)

        ego_wp: carla.Waypoint = self.carla_world_map.get_waypoint(
            self.ego_vehicle.get_location(),
            project_to_road=True,
            lane_type=carla.libcarla.LaneType.Any,
        )
        # how far is next junction
        next_wps = roadgraph.wps_next_until_lane_end(ego_wp)
        try:
            next_lane_wps_ego = next_wps[-1].next(1)
            if len(next_lane_wps_ego) == 0:
                next_lane_wps_ego = [next_wps[-1]]
        except:
            next_lane_wps_ego = []
        if ego_wp.is_junction:
            distance_to_junction_ego = 0.0
            # get distance to ego vehicle
        elif len(next_lane_wps_ego) > 0 and next_lane_wps_ego[0].is_junction:
            distance_to_junction_ego = next_lane_wps_ego[0].transform.location.distance(
                ego_wp.transform.location,
            )
        else:
            distance_to_junction_ego = None

        next_road_ids_ego = []
        next_next_road_ids_ego = []
        for wp in next_lane_wps_ego:
            next_road_ids_ego.append(wp.road_id)
            next_next_wps = roadgraph.wps_next_until_lane_end(wp)
            try:
                next_next_lane_wps_ego = next_next_wps[-1].next(1)
                if len(next_next_lane_wps_ego) == 0:
                    next_next_lane_wps_ego = [next_next_wps[-1]]
            except:
                next_next_lane_wps_ego = []
            for wp2 in next_next_lane_wps_ego:
                if wp2.road_id not in next_next_road_ids_ego:
                    next_next_road_ids_ego.append(wp2.road_id)

        (
            dangerous_adversarial_actors_ids,
            safe_adversarial_actors_ids,
            ignored_adversarial_actors_ids,
        ) = self.adversarial_actors_ids
        data = {
            "num_dangerous_adversarial": len(dangerous_adversarial_actors_ids),
            "num_safe_adversarial": len(safe_adversarial_actors_ids),
            "num_ignored_adversarial": len(ignored_adversarial_actors_ids),
            "rear_adversarial_id": -1
            if self.rear_adversarial_actor is None
            else self.rear_adversarial_actor.id,
            "localized_ego_state_se3": py123d_log_api.localized_pose_to_se3_matrix(
                tick_data["localized_position"],
                self.ego_orientation_rad,
            ),
            "target_speed": target_speed,
            "speed_limit": self.speed_limit,
            "last_encountered_speed_limit_sign": self.last_encountered_speed_limit_sign,
            "route": np.array(dense_route, dtype=np.float32),
            "route_original": np.array(dense_route_original, dtype=np.float32),
            "changed_route": changed_route,
            "speed_reduced_by_obj_type": speed_reduced_by_obj_type,
            "speed_reduced_by_obj_id": speed_reduced_by_obj_id,
            "speed_reduced_by_obj_distance": speed_reduced_by_obj_distance,
            "steer": control.steer,
            "throttle": control.throttle,
            "brake": bool(control.brake),
            "scenario": self.scenario_name,
            "distance_to_next_junction": self.distance_to_next_junction,
            "ego_lane_id": self.ego_lane_id,
            "road_id": ego_wp.road_id,
            "lane_id": ego_wp.lane_id,
            "is_junction": ego_wp.is_junction,
            "junction_id": ego_wp.junction_id,
            "next_road_ids": next_road_ids_ego,
            "next_next_road_ids_ego": next_next_road_ids_ego,
            "lane_change_str": str(ego_wp.lane_change),
            "lane_type_str": str(ego_wp.lane_type),
            "left_lane_marking_color_str": str(ego_wp.left_lane_marking.color),
            "left_lane_marking_type_str": str(ego_wp.left_lane_marking.type),
            "right_lane_marking_color_str": str(ego_wp.right_lane_marking.color),
            "right_lane_marking_type_str": str(ego_wp.right_lane_marking.type),
            "dist_to_construction_site": self.distance_to_construction_site,
            "dist_to_accident_site": self.distance_to_accident_site,
            "dist_to_parked_obstacle": self.distance_to_parked_obstacle,
            "dist_to_vehicle_opens_door": self.distance_to_vehicle_opens_door,
            "dist_to_cutin_vehicle": self.distance_to_cutin_vehicle,
            "dist_to_pedestrian": self.distance_to_pedestrian,
            "dist_to_biker": self.distance_to_biker,
            "dist_to_junction": distance_to_junction_ego,
            "current_active_scenario_type": self.current_active_scenario_type,
            "previous_active_scenario_type": self.previous_active_scenario_type,
            "scenario_obstacles_ids": self.scenario_obstacles_ids,
            "scenario_actors_ids": self.scenario_actors_ids,
            "vehicle_opened_door": self.vehicle_opened_door,
            "vehicle_door_side": self.vehicle_door_side,
            "scenario_obstacles_convex_hull": self.scenario_obstacles_convex_hull,
            "cut_in_actors_ids": self.cut_in_actors_ids,
            "distance_to_intersection_index_ego": self.distance_to_intersection_index_ego,
            "ego_lane_width": self.ego_lane_width,
            "target_lane_width": self.target_lane_width,
            "weather_setting": self.weather_setting,
            "jpeg_storage_quality": self.jpeg_storage_quality,
            "route_left_length": self.route_left_length,
            "distance_ego_to_route": self.distance_ego_to_route,
            "weather_parameters": self.weather_parameters,
            "signed_dist_to_lane_change": self.signed_dist_to_lane_change,
            "visual_visibility": int(self.visual_visibility),
            "num_parking_vehicles_in_proximity": self.num_parking_vehicles_in_proximity,
            "europe_traffic_light": self.europe_traffic_light,
            "over_head_traffic_light": self.over_head_traffic_light,
            "emergency_brake_for_special_vehicle": self.emergency_brake_for_special_vehicle,
            "vehicle_hazard": bool(self.vehicle_hazard),
            "vehicle_affecting_id": self.vehicle_affecting_id,
            "light_hazard": bool(self.traffic_light_hazard),
            "walker_hazard": bool(self.walker_hazard),
            "walker_affecting_id": self.walker_affecting_id,
            "stop_sign_hazard": bool(self.stop_sign_hazard),
            "stop_sign_close": bool(self.stop_sign_close),
            "walker_close": bool(self.walker_close),
            "walker_close_id": self.walker_close_id,
            "target_speed_limit": self.target_speed_limit,
            "slower_bad_visibility": self.slower_bad_visibility,
            "slower_clutterness": self.slower_clutterness,
            "does_emergency_brake_for_pedestrians": self.does_emergency_brake_for_pedestrians,
            "construction_obstacle_two_ways_stuck": self.construction_obstacle_two_ways_stuck,
            "accident_two_ways_stuck": self.accident_two_ways_stuck,
            "parked_obstacle_two_ways_stuck": self.parked_obstacle_two_ways_stuck,
            "vehicle_opens_door_two_ways_stuck": self.vehicle_opens_door_two_ways_stuck,
            "rear_danger_8": self.rear_danger_8,
            "rear_danger_16": self.rear_danger_16,
            "brake_cutin": self.brake_cutin,
        }

        # The target points themselves are static per log and live in the
        # driving-meta modality metadata; a tick only pins which one is current.
        data["target_point_indices"] = {
            str(dist): planner.target_point_index
            for dist, planner in self.gps_waypoint_planners_dict.items()
        }

        return data

    def _visualize_route(self) -> None:
        """Visualize the current route using CARLA debug drawing."""
        route_points = self.route_waypoints_np
        if len(route_points) == 0:
            return

        sampled_points = route_points[:1000:20]
        for point in sampled_points:
            location = carla.Location(
                x=float(point[0]),
                y=float(point[1]),
                z=float(point[2]) + 1.0,
            )
            self.carla_world.debug.draw_point(
                location,
                size=0.05,  # Smaller size
                color=carla.Color(
                    *self.config_expert.visualization.future_route_color,
                ),  # Green
                life_time=self.config_expert.visualization.draw_life_time,  # Short life time for continuous updates
            )

        # Draw lines between consecutive sampled route points
        for i in range(len(sampled_points) - 1):
            start = carla.Location(
                x=float(sampled_points[i][0]),
                y=float(sampled_points[i][1]),
                z=float(sampled_points[i][2]) + 1.0,
            )
            end = carla.Location(
                x=float(sampled_points[i + 1][0]),
                y=float(sampled_points[i + 1][1]),
                z=float(sampled_points[i + 1][2]) + 1.0,
            )
            self.carla_world.debug.draw_line(
                start,
                end,
                thickness=0.05,  # Thinner lines
                color=carla.Color(
                    *self.config_expert.visualization.future_route_color,
                ),  # Green
                life_time=self.config_expert.visualization.draw_life_time,
            )

    def _visualize_original_route(self) -> None:
        """Visualize the original route using CARLA debug drawing."""
        original_route_points = self.original_route_waypoints_np
        if len(original_route_points) == 0:
            return

        sampled_points = original_route_points[:1000:20]
        for point in sampled_points:
            location = carla.Location(
                x=float(point[0]),
                y=float(point[1]),
                z=float(point[2]) + 1.5,
            )
            self.carla_world.debug.draw_point(
                location,
                size=0.05,
                color=carla.Color(0, 0, 255),  # Blue
                life_time=self.config_expert.visualization.draw_life_time,  # Short life time for continuous updates
            )

        # Draw lines between consecutive sampled original route points
        for i in range(len(sampled_points) - 1):
            start = carla.Location(
                x=float(sampled_points[i][0]),
                y=float(sampled_points[i][1]),
                z=float(sampled_points[i][2]) + 1.5,
            )
            end = carla.Location(
                x=float(sampled_points[i + 1][0]),
                y=float(sampled_points[i + 1][1]),
                z=float(sampled_points[i + 1][2]) + 1.5,
            )
            self.carla_world.debug.draw_line(
                start,
                end,
                thickness=0.04,  # Thinner lines
                color=carla.Color(0, 0, 255),  # Blue
                life_time=self.config_expert.visualization.draw_life_time,
            )
