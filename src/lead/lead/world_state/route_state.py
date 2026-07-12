"""Cached per-step view of the route ahead of the ego vehicle."""

from functools import cached_property

import carla
import jaxtyping as jt
import numpy as np
import numpy.typing as npt
from agents.navigation.local_planner import RoadOption
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from lead.lead.planning.privileged_route_planner import PrivilegedRoutePlanner
from lead.lead.utils import expert_utils
from lead.lead.utils.expert_utils import cached_property_by, step_cached_property


class RouteStateMixin:
    """Cached per-step route waypoints, lane changes and junction distances."""

    @step_cached_property
    def distance_to_next_traffic_light(self):
        return self.privileged_route_planner.distances_to_next_traffic_lights[
            self.privileged_route_planner.route_index
        ]

    @step_cached_property
    def next_traffic_light(self):
        return self.privileged_route_planner.next_traffic_lights[
            self.privileged_route_planner.route_index
        ]

    @step_cached_property
    def distance_to_next_stop_sign(self):
        return self.privileged_route_planner.distances_to_next_stop_signs[
            self.privileged_route_planner.route_index
        ]

    @step_cached_property
    def next_stop_sign(self):
        return self.privileged_route_planner.next_stop_signs[
            self.privileged_route_planner.route_index
        ]

    @step_cached_property
    def remaining_route(self):
        return self.route_waypoints_np[
            self.config_expert.driving.tf_first_checkpoint_distance :
        ][:: self.config_expert.simulation.points_per_meter]

    @step_cached_property
    def remaining_route_original(self):
        return self.original_route_waypoints_np[
            self.config_expert.driving.tf_first_checkpoint_distance :
        ][:: self.config_expert.simulation.points_per_meter]

    @step_cached_property
    def near_lane_change(self) -> bool:
        route_points = self.route_waypoints_np
        # Calculate the braking distance based on the ego velocity
        braking_distance = (
            ((self.ego_speed * 3.6) / 10.0) ** 2 / 2.0
        ) + self.config_expert.driving.braking_distance_calculation_safety_distance

        # Determine the number of waypoints to look ahead based on the braking distance
        look_ahead_points = max(
            self.config_expert.driving.minimum_lookahead_distance_to_compute_near_lane_change,
            min(
                route_points.shape[0],
                self.config_expert.simulation.points_per_meter * int(braking_distance),
            ),
        )
        current_route_index = self.privileged_route_planner.route_index
        max_route_length = len(self.privileged_route_planner.commands)

        from_index = max(
            0,
            current_route_index
            - self.config_expert.driving.check_previous_distance_for_lane_change,
        )
        to_index = min(max_route_length - 1, current_route_index + look_ahead_points)
        # Iterate over the points around the current position, checking for lane change commands
        for i in range(from_index, to_index, 1):
            if self.privileged_route_planner.commands[i] in (
                RoadOption.CHANGELANELEFT,
                RoadOption.CHANGELANERIGHT,
            ):
                return True

        return False

    @cached_property
    def min_lane_width_route(self) -> float:
        self.ego_vehicle = CarlaDataProvider.get_hero_actor()
        self.carla_world = self.ego_vehicle.get_world()

        global_plan = self.org_dense_route_world_coord
        carla_map = self.carla_world.get_map()
        route_waypoints = [transform.location for transform, _ in global_plan]
        route_waypoints = [carla_map.get_waypoint(loc) for loc in route_waypoints]
        widths = []
        for waypoint in route_waypoints:
            if waypoint is not None and not waypoint.is_junction:
                widths.append(waypoint.lane_width)
        return max(min(widths), 2.75)

    @cached_property
    def max_speed_limit_route(self) -> float:
        self.ego_vehicle = CarlaDataProvider.get_hero_actor()
        self.carla_world = self.ego_vehicle.get_world()
        # Check if the vehicle starts from a parking spot
        distance_to_road = self.org_dense_route_world_coord[0][0].location.distance(
            self.ego_vehicle.get_location(),
        )
        # The first waypoint starts at the lane center, hence it's more than 2 m away from the center of the
        # ego vehicle at the beginning.
        starts_with_parking_exit = distance_to_road > 2
        waypoint_planner = PrivilegedRoutePlanner(self.config_expert)
        waypoint_planner.setup_route(
            self.org_dense_route_world_coord,
            self.carla_world,
            self.carla_world_map,
            starts_with_parking_exit,
            self.ego_vehicle.get_location(),
        )
        way_point_planner = waypoint_planner
        waypoints = way_point_planner.route_waypoints
        if len(waypoints) == 0:
            return 30 / 3.6
        speed_limits = []
        for wp in waypoints:
            if wp is not None:
                # Get speed limit landmarks within a reasonable distance ahead
                landmarks = wp.get_landmarks(200.0, True)
                for landmark in landmarks:
                    # Check if the landmark is a MaximumSpeed sign
                    if landmark.type == carla.LandmarkType.MaximumSpeed:
                        # Extract speed limit value from landmark
                        try:
                            speed_limit = float(landmark.value)
                            if speed_limit > 0:
                                speed_limits.append(speed_limit)
                        except (ValueError, AttributeError):
                            pass

        # Return the maximum speed limit found, or default to 30 km/h
        if len(speed_limits) > 0:
            return max(speed_limits) / 3.6  # Convert km/h to m/s
        return 30 / 3.6

    @cached_property_by(lambda self: self.privileged_route_planner.route_index)
    def route_waypoints(self) -> list[carla.Waypoint]:
        return self.privileged_route_planner.route_waypoints[
            self.privileged_route_planner.route_index :
        ]

    @cached_property_by(lambda self: self.privileged_route_planner.route_index)
    def route_waypoints_np(self) -> jt.Float[npt.NDArray, "N 3"]:
        return self.privileged_route_planner.route_points[
            self.privileged_route_planner.route_index :
        ]

    @cached_property_by(lambda self: self.privileged_route_planner.route_index)
    def original_route_waypoints_np(self) -> jt.Float[npt.NDArray, "N 3"]:
        return self.privileged_route_planner.original_route_points[
            self.privileged_route_planner.route_index :
        ]

    @step_cached_property
    def signed_dist_to_lane_change(self) -> float:
        """
        Compute the signed distance to the next or previous lane change command in the route.

        Returns:
            float: Signed distance to the next lane change command. Positive if ahead, negative if behind.
            Inf if no lane change command found in proximity.
        """
        route_points = self.privileged_route_planner.route_points
        current_index = self.privileged_route_planner.route_index
        from_index = max(0, current_index - 250)
        to_index = min(len(route_points) - 1, current_index + 250)
        # Iterate over the points around the current position, checking for lane change commands

        def dist(index_a, index_b):
            index_min = min(index_a, index_b)
            index_max = max(index_a, index_b)
            d = 0
            for i in range(index_min, index_max):
                p1 = route_points[i]
                p2 = route_points[i + 1]
                d += np.linalg.norm(p2 - p1)
            if index_a < index_b:
                return d
            return -d

        min_dist = np.inf
        for i in range(from_index, to_index, 1):
            if self.privileged_route_planner.commands[i] in (
                RoadOption.CHANGELANELEFT,
                RoadOption.CHANGELANERIGHT,
            ):
                considered_dist = dist(current_index, i)
                if abs(considered_dist) < abs(min_dist):
                    min_dist = considered_dist

        return min_dist / self.config_expert.simulation.points_per_meter

    @property
    def ego_wp(self) -> carla.Waypoint:
        return self.route_waypoints[0]

    @step_cached_property
    def route_left_length(self):
        route_points = self.route_waypoints_np
        dist_diff = np.diff(route_points[:, :2], axis=0)
        segment_lengths = np.linalg.norm(dist_diff, axis=1)
        return np.sum(segment_lengths)

    @step_cached_property
    def distance_ego_to_route(self):
        ego_wp = self.ego_wp
        route_wp = self.route_waypoints[0]
        return ego_wp.transform.location.distance(route_wp.transform.location)

    @step_cached_property
    def route_curvature(self):
        route_points = self.route_waypoints_np
        total_curvature = 0.0
        for i in range(
            1,
            min(
                len(route_points) - 1,
                self.config_expert.driving.high_road_curvature_max_future_points,
            )
            - 1,
        ):
            loc_prev = route_points[i - 1]
            loc_current = route_points[i]
            loc_next = route_points[i + 1]
            curv = (
                (loc_current[0] - loc_prev[0]) * (loc_next[1] - loc_current[1])
                - (loc_current[1] - loc_prev[1]) * (loc_next[0] - loc_current[0])
            ) / (
                (
                    (loc_current[0] - loc_prev[0]) ** 2
                    + (loc_current[1] - loc_prev[1]) ** 2
                )
                ** 0.5
                * (
                    (loc_next[0] - loc_current[0]) ** 2
                    + (loc_next[1] - loc_current[1]) ** 2
                )
                ** 0.5
            )
            total_curvature += abs(curv)
        return total_curvature

    @step_cached_property
    def distance_to_next_junction(self) -> float:
        ego_wp = self.carla_world_map.get_waypoint(
            self.ego_vehicle.get_location(),
            project_to_road=True,
            lane_type=carla.libcarla.LaneType.Any,
        )
        next_wps = expert_utils.wps_next_until_lane_end(ego_wp)
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
            distance_to_junction_ego = np.inf

        return float(distance_to_junction_ego)
