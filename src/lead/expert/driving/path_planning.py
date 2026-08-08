"""Lateral planning for the expert: shifts the privileged route around
obstacles and decides when the opposite lane is free for overtaking."""

import logging
import numbers

import carla
import numpy as np
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from lead.common import geometry
from lead.expert.driving.speed_planning import (
    compute_min_time_for_distance,
    compute_target_speed_idm,
)
from lead.expert.utils import roadgraph
from lead.expert.utils.weather_presets import WeatherVisibility

LOG = logging.getLogger(__name__)


class PathPlanningMixin:
    """Route reshaping, mixed into :class:`lead.expert.expert_agent.Expert`."""

    def is_two_ways_overtaking_path_clear(
        self,
        from_index: int,
        to_index: int,
        target_speed: numbers.Real,
        previous_lane_ids: list,
        min_speed: numbers.Real = 50.0 / 3.6,
    ) -> bool:
        """
        Checks if the path between two route indices is clear for the ego vehicle to overtake in two ways scenarios.

        Args:
            from_index: The starting route index.
            to_index: The ending route index.
            target_speed: The target speed of the ego vehicle.
            previous_lane_ids: A list of tuples containing previous road IDs and lane IDs.
            min_speed: The minimum speed to consider for overtaking. Defaults to 50/3.6 km/h.

        Returns:
            True if the path is clear for overtaking, False otherwise.
        """
        # 10 m safety distance, overtake with max. 50 km/h
        to_location = self.privileged_route_planner.route_points[to_index]
        to_location = carla.Location(to_location[0], to_location[1], to_location[2])

        from_location = self.privileged_route_planner.route_points[from_index]
        from_location = carla.Location(
            from_location[0],
            from_location[1],
            from_location[2],
        )

        # Compute the distance and time needed for the ego vehicle to overtake
        ego_distance = (
            to_location.distance(self.ego_location)
            + self.ego_vehicle.bounding_box.extent.x * 2
            + self.config_expert.driving.check_path_free_safety_distance
        )
        ego_time = compute_min_time_for_distance(
            self.config_expert,
            ego_distance,
            min(min_speed, target_speed),
            self.ego_speed,
        )

        path_clear = True

        if self.config_expert.visualization.visualize_internal_data:
            for vehicle in self.vehicles_inside_bev:
                # Sort out ego vehicle
                if vehicle.id == self.ego_vehicle.id:
                    continue

                vehicle_location = vehicle.get_location()
                vehicle_waypoint = self.carla_world_map.get_waypoint(vehicle_location)

                diff_vector = vehicle_location - self.ego_location
                dot_product = (
                    self.ego_vehicle.get_transform()
                    .get_forward_vector()
                    .dot(diff_vector)
                )
                # Draw dot_product above vehicle
                self.carla_world.debug.draw_string(
                    vehicle_location + carla.Location(x=1, y=0, z=2.5),
                    f"dot1={dot_product:.1f}",
                    color=carla.Color(255, 255, 0),
                    life_time=self.config_expert.visualization.draw_life_time,
                )

                # The overtaking path is blocked by vehicle
                diff_vector_2 = to_location - vehicle_location
                dot_product_2 = (
                    vehicle.get_transform().get_forward_vector().dot(diff_vector_2)
                )
                self.carla_world.debug.draw_string(
                    vehicle_location + carla.Location(x=2, y=0, z=3.5),
                    f"dot2={dot_product_2:.1f}",
                    color=carla.Color(0, 255, 255),
                    life_time=self.config_expert.visualization.draw_life_time,
                )

        for vehicle in self.vehicles_inside_bev:
            # Only consider visible vehicles in the BEV to make TransFuser learn easier
            if not self.is_actor_inside_bev(vehicle):
                continue

            # Sort out ego vehicle
            if vehicle.id == self.ego_vehicle.id:
                continue

            vehicle_location = vehicle.get_location()
            vehicle_waypoint = self.carla_world_map.get_waypoint(vehicle_location)

            # Check if the vehicle is on the previous lane IDs
            if (
                vehicle_waypoint.road_id,
                vehicle_waypoint.lane_id,
            ) in previous_lane_ids:
                diff_vector = vehicle_location - self.ego_location
                dot_product = (
                    self.ego_vehicle.get_transform()
                    .get_forward_vector()
                    .dot(diff_vector)
                )

                # One TwoWay scenarios we can skip this vehicle since it's not on the overtaking path and behind
                # the ego vehicle. Otherwise in other scenarios it's coming from behind and is relevant
                if dot_product < 0 and self.current_active_scenario_type in [
                    "ConstructionObstacleTwoWays",
                    "AccidentTwoWays",
                    "ParkedObstacleTwoWays",
                    "VehicleOpensDoorTwoWays",
                    "HazardAtSideLaneTwoWays",
                ]:
                    continue

                # Allow earlier acceleration only from a near stop (ego_speed < 1.0),
                # ignoring fast vehicles that are already almost out of the way.
                if self.ego_speed < 1.0 and vehicle.get_velocity().length() > 3.0:
                    if self.current_active_scenario_type in [
                        "ConstructionObstacleTwoWays",
                    ]:
                        threshold = 8
                        if self.ego_lane_width <= 2.76:
                            threshold = 8
                        elif self.ego_lane_width <= 3.01:
                            threshold = 9
                        elif self.ego_lane_width <= 3.51:
                            threshold = 10
                        if dot_product < threshold:
                            continue
                    elif self.current_active_scenario_type in ["AccidentTwoWays"]:
                        threshold = 7
                        if self.ego_lane_width <= 2.76:
                            threshold = 7
                        elif self.ego_lane_width <= 3.01:
                            threshold = 8
                        elif self.ego_lane_width <= 3.51:
                            threshold = 9
                        if dot_product < threshold:
                            continue
                    elif self.current_active_scenario_type in ["ParkedObstacleTwoWays"]:
                        threshold = 7
                        if self.ego_lane_width <= 2.76:
                            threshold = 7
                        elif self.ego_lane_width <= 3.01:
                            threshold = 8
                        elif self.ego_lane_width <= 3.51:
                            threshold = 9
                        if dot_product < threshold:
                            continue
                    elif self.current_active_scenario_type in [
                        "VehicleOpensDoorTwoWays",
                    ]:
                        threshold = 8
                        if self.ego_lane_width <= 2.76:
                            threshold = 8
                        elif self.ego_lane_width <= 3.01:
                            threshold = 9
                        elif self.ego_lane_width <= 3.51:
                            threshold = 10
                        if dot_product < threshold:
                            continue
                    elif self.current_active_scenario_type in [
                        "HazardAtSideLaneTwoWays",
                    ]:
                        threshold = 8
                        if self.ego_lane_width <= 2.76:
                            threshold = 8
                        elif self.ego_lane_width <= 3.01:
                            threshold = 9
                        elif self.ego_lane_width <= 3.51:
                            threshold = 10
                        if dot_product < threshold:
                            continue

                # The overtaking path is blocked by vehicle
                diff_vector_2 = to_location - vehicle_location
                dot_product_2 = (
                    vehicle.get_transform().get_forward_vector().dot(diff_vector_2)
                )

                if dot_product_2 < 0:
                    path_clear = False
                    break

                other_vehicle_distance = (
                    to_location.distance(vehicle_location)
                    - vehicle.bounding_box.extent.x
                )
                other_vehicle_time = other_vehicle_distance / max(
                    1.0,
                    vehicle.get_velocity().length(),
                )

                # Add 200 ms safety margin
                # Vehicle needs less time to arrive at to_location than the ego vehicle
                if (
                    other_vehicle_time
                    < ego_time + self.config_expert.driving.check_path_free_safety_time
                ):
                    path_clear = False
                    break

        return path_clear

    def _solve_obstacle_scenarios(
        self,
        target_speed: float,
    ) -> tuple[float, bool, list]:
        """
        Adjust the target speed and route for the currently active obstacle scenario
        and decide whether the ego vehicle keeps driving or waits.

        Args:
            target_speed: The current target speed of the ego vehicle.

        Returns:
            A tuple containing the updated target speed, a boolean indicating whether to keep driving,
                and a list containing information about a potential decreased target speed due to an object.
        """

        keep_driving = False
        speed_reduced_by_obj = [
            target_speed,
            None,
            None,
            None,
        ]  # [target_speed, type, id, distance]

        # Only continue if there are some active scenarios available
        if len(CarlaDataProvider.active_scenarios) != 0:
            ego_location = self.ego_vehicle.get_location()

            if self.current_active_scenario_type == "InvadingTurn":
                # Get the first (current) InvadingTurn memory entry
                invading_turn_memory = (
                    CarlaDataProvider.get_current_scenario_memory()
                    if CarlaDataProvider.get_current_scenario_memory()
                    else {}
                )
                first_cone, last_cone, offset = [
                    invading_turn_memory[k]
                    for k in ["first_cone", "last_cone", "offset"]
                ]
                closest_distance = first_cone.get_location().distance(ego_location)
                if (
                    closest_distance
                    < self.config_expert.driving.default_max_distance_to_process_scenario
                ):
                    LOG.info(
                        "[Scenario Action] InvadingTurn: Shifting route for invading turn maneuver",
                    )
                    self.privileged_route_planner.shift_route_for_invading_turn(
                        first_cone,
                        last_cone,
                        offset,
                    )
                    LOG.info(
                        "[Scenario Exit] InvadingTurn: Completed, cleaning up scenario",
                    )
                    CarlaDataProvider.clean_current_active_scenario()

            elif self.current_active_scenario_type in [
                "Accident",
                "ConstructionObstacle",
                "ParkedObstacle",
            ]:
                first_actor, last_actor, direction = [
                    CarlaDataProvider.get_current_scenario_memory()[k]
                    for k in ["first_actor", "last_actor", "direction"]
                ]

                horizontal_distance = geometry.get_horizontal_distance(
                    self.ego_vehicle,
                    first_actor,
                )

                LOG.info(
                    f"[{self.current_active_scenario_type}] Horizontal distance to obstacle: {horizontal_distance:.1f}m",
                )

                # Shift the route around the obstacles
                if (
                    horizontal_distance
                    < self.config_expert.driving.default_max_distance_to_process_scenario
                    and not CarlaDataProvider.get_current_scenario_memory()[
                        "changed_route"
                    ]
                ):
                    add_before_length = {
                        "Accident": self.config_expert.driving.add_before_accident,
                        "ConstructionObstacle": self.config_expert.driving.add_before_construction_obstacle,
                        "ParkedObstacle": self.config_expert.driving.add_before_parked_obstacle,
                    }[self.current_active_scenario_type]
                    add_after_length = {
                        "Accident": self.config_expert.driving.add_after_accident,
                        "ConstructionObstacle": self.config_expert.driving.add_after_construction_obstacle,
                        "ParkedObstacle": self.config_expert.driving.add_after_parked_obstacle,
                    }[self.current_active_scenario_type]
                    transition_length = {
                        "Accident": self.config_expert.driving.transition_smoothness_distance_accident,
                        "ConstructionObstacle": self.config_expert.driving.transition_smoothness_factor_construction_obstacle,
                        "ParkedObstacle": self.config_expert.driving.transition_smoothness_distance_parked_obstacle,
                    }[self.current_active_scenario_type]
                    if self.current_active_scenario_type not in [
                        "ConstructionObstacle",
                    ]:
                        if self.visual_visibility == WeatherVisibility.LIMITED:
                            add_before_length -= 5.0
                            transition_length -= int(
                                4.0 * self.config_expert.simulation.points_per_meter,
                            )
                        elif self.visual_visibility == WeatherVisibility.VERY_LIMITED:
                            add_before_length -= 7.0
                            transition_length -= int(
                                6.0 * self.config_expert.simulation.points_per_meter,
                            )
                        LOG.info("Later switching because of bad weather")
                    if self.speed_limit < 15:
                        add_after_length = {
                            "Accident": 0.5,
                            "ConstructionObstacle": self.config_expert.driving.add_after_construction_obstacle,
                            "ParkedObstacle": 0.5,
                        }[
                            self.current_active_scenario_type
                        ]  # Speed is low and we can switch earlier
                    lane_transition_factor = {
                        "Accident": 1.0,
                        "ConstructionObstacle": self.two_way_obstacle_distance_to_cones_factor,
                        "ParkedObstacle": 1.0,
                    }[self.current_active_scenario_type]
                    LOG.info(
                        f"[Route Change] {self.current_active_scenario_type}: Shifting route around obstacle, direction={direction}, distance={horizontal_distance:.1f}m",
                    )
                    from_index, _ = (
                        self.privileged_route_planner.shift_route_around_actors(
                            first_actor,
                            last_actor,
                            direction,
                            transition_length,
                            lane_transition_factor=lane_transition_factor,
                            extra_length_before=add_before_length,
                            extra_length_after=add_after_length,
                        )
                    )
                    CarlaDataProvider.get_current_scenario_memory().update(
                        {
                            "changed_route": True,
                            "from_index": from_index,
                        },
                    )

                elif CarlaDataProvider.get_current_scenario_memory()["changed_route"]:
                    first_actor_rel_pos = geometry.get_relative_transform(
                        self.ego_matrix,
                        np.array(first_actor.get_transform().get_matrix()),
                    )
                    if first_actor_rel_pos[0] < 3:
                        LOG.info(
                            f"[Scenario Exit] {self.current_active_scenario_type}: Passed obstacle, cleaning up scenario",
                        )
                        CarlaDataProvider.clean_current_active_scenario()
            elif self.current_active_scenario_type in [
                "AccidentTwoWays",
                "ConstructionObstacleTwoWays",
                "ParkedObstacleTwoWays",
                "VehicleOpensDoorTwoWays",
            ]:
                (
                    first_actor,
                    last_actor,
                    direction,
                    changed_route,
                    from_index,
                    to_index,
                    path_clear,
                ) = [
                    CarlaDataProvider.get_current_scenario_memory()[k]
                    for k in [
                        "first_actor",
                        "last_actor",
                        "direction",
                        "changed_route",
                        "from_index",
                        "to_index",
                        "path_clear",
                    ]
                ]

                # change the route if the ego is close enough to the obstacle
                horizontal_distance = geometry.get_horizontal_distance(
                    self.ego_vehicle,
                    first_actor,
                )

                # Shift the route around the obstacles
                if (
                    horizontal_distance
                    < self.config_expert.driving.default_max_distance_to_process_scenario
                    and not changed_route
                ):
                    transition_length = {
                        "AccidentTwoWays": self.config_expert.driving.transition_length_accident_two_ways,
                        "ConstructionObstacleTwoWays": self.config_expert.driving.transition_length_construction_obstacle_two_ways,
                        "ParkedObstacleTwoWays": self.config_expert.driving.transition_length_parked_obstacle_two_ways,
                        "VehicleOpensDoorTwoWays": self.config_expert.driving.transition_length_vehicle_opens_door_two_ways,
                    }[self.current_active_scenario_type]
                    add_before_length = {
                        "AccidentTwoWays": self.add_before_accident_two_ways,
                        "ConstructionObstacleTwoWays": self.add_before_construction_obstacle_two_ways,
                        "ParkedObstacleTwoWays": self.config_expert.driving.add_before_parked_obstacle_two_ways,
                        "VehicleOpensDoorTwoWays": self.config_expert.driving.add_before_vehicle_opens_door_two_ways,
                    }[self.current_active_scenario_type]
                    add_after_length = {
                        "AccidentTwoWays": self.add_after_accident_two_ways,
                        "ConstructionObstacleTwoWays": self.add_after_construction_obstacle_two_ways,
                        "ParkedObstacleTwoWays": self.config_expert.driving.add_after_parked_obstacle_two_ways,
                        "VehicleOpensDoorTwoWays": self.config_expert.driving.add_after_vehicle_opens_door_two_ways,
                    }[self.current_active_scenario_type]
                    factor = {
                        "AccidentTwoWays": self.config_expert.driving.factor_accident_two_ways,
                        "ConstructionObstacleTwoWays": self.two_way_obstacle_distance_to_cones_factor,
                        "ParkedObstacleTwoWays": self.config_expert.driving.factor_parked_obstacle_two_ways,
                        "VehicleOpensDoorTwoWays": self.two_way_vehicle_open_door_distance_to_center_line_factor,
                    }[self.current_active_scenario_type]
                    if self.current_active_scenario_type not in [
                        "ConstructionObstacleTwoWays",
                    ]:
                        if self.visual_visibility == WeatherVisibility.LIMITED:
                            add_before_length -= 0.5
                        elif self.visual_visibility == WeatherVisibility.VERY_LIMITED:
                            add_before_length -= 1.0

                    LOG.info(
                        f"[Route Change] {self.current_active_scenario_type}: Initiating two-way overtaking maneuver, direction={direction}, distance={horizontal_distance:.1f}m",
                    )
                    from_index, to_index = (
                        self.privileged_route_planner.shift_route_around_actors(
                            first_actor,
                            last_actor,
                            direction,
                            transition_length,
                            factor,
                            add_before_length,
                            add_after_length,
                        )
                    )

                    changed_route = True
                    CarlaDataProvider.get_current_scenario_memory().update(
                        {
                            "changed_route": changed_route,
                            "from_index": from_index,
                            "to_index": to_index,
                            "path_clear": path_clear,
                        },
                    )

                # Check if the ego can overtake the obstacle
                if (
                    changed_route
                    and from_index - self.privileged_route_planner.route_index
                    < self.config_expert.driving.max_distance_to_overtake_two_way_scnearios
                    and not path_clear
                ):
                    # Get previous roads and lanes of the target lane
                    target_lane = (
                        self.route_waypoints[0].get_left_lane()
                        if direction == "right"
                        else self.route_waypoints[0].get_right_lane()
                    )
                    if target_lane is None:
                        return target_speed, keep_driving, speed_reduced_by_obj
                    prev_road_lane_ids = roadgraph.get_previous_road_lane_ids(
                        self.config_expert,
                        target_lane,
                    )
                    path_clear = self.is_two_ways_overtaking_path_clear(
                        int(from_index),
                        int(to_index),
                        target_speed,
                        prev_road_lane_ids,
                        min_speed=self.two_way_overtake_speed,
                    )
                    CarlaDataProvider.get_current_scenario_memory()["path_clear"] = (
                        path_clear
                    )

                # If the overtaking path is clear, keep driving; otherwise, wait behind the obstacle
                if path_clear:
                    target_speed = self.two_way_overtake_speed
                    if (
                        self.privileged_route_planner.route_index
                        >= to_index
                        - self.config_expert.driving.distance_to_delete_scenario_in_two_ways
                    ):
                        LOG.info(
                            f"[Scenario Exit] {self.current_active_scenario_type}: Overtaking completed, cleaning up scenario",
                        )
                        CarlaDataProvider.clean_current_active_scenario()
                    keep_driving = True
                else:
                    offset = {
                        "AccidentTwoWays": 10,
                        "ConstructionObstacleTwoWays": 10,
                        "ParkedObstacleTwoWays": 10,
                        "VehicleOpensDoorTwoWays": 10,
                    }[
                        self.current_active_scenario_type
                    ]  # Stop a bit to the side rather than directly behind the obstacle
                    distance_to_merging_point = (
                        float(
                            from_index
                            + offset
                            - self.privileged_route_planner.route_index,
                        )
                        / self.config_expert.simulation.points_per_meter
                    )
                    target_speed = compute_target_speed_idm(
                        config=self.config_expert,
                        desired_speed=target_speed,
                        leading_actor_length=self.ego_vehicle.bounding_box.extent.x,
                        ego_speed=self.ego_speed,
                        leading_actor_speed=0,
                        distance_to_leading_actor=distance_to_merging_point,
                        s0=self.config_expert.driving.idm_two_way_scenarios_minimum_distance,
                        T=self.config_expert.driving.idm_two_way_scenarios_time_headway,
                    )

                    # Update the object causing the most speed reduction
                    if (
                        speed_reduced_by_obj is None
                        or speed_reduced_by_obj[0] > target_speed
                    ):
                        speed_reduced_by_obj = [
                            target_speed,
                            first_actor.type_id,
                            first_actor.id,
                            distance_to_merging_point,
                        ]

            elif self.current_active_scenario_type == "HazardAtSideLaneTwoWays":
                (
                    first_actor,
                    last_actor,
                    changed_route,
                    from_index,
                    to_index,
                    path_clear,
                ) = [
                    CarlaDataProvider.get_current_scenario_memory()[k]
                    for k in [
                        "first_actor",
                        "last_actor",
                        "changed_route",
                        "from_index",
                        "to_index",
                        "path_clear",
                    ]
                ]

                horizontal_distance = geometry.get_horizontal_distance(
                    self.ego_vehicle,
                    first_actor,
                )

                if (
                    horizontal_distance
                    < self.config_expert.driving.max_distance_to_process_hazard_at_side_lane_two_ways
                    and not changed_route
                ):
                    to_index = self.privileged_route_planner.get_closest_route_index(
                        int(self.privileged_route_planner.route_index),
                        last_actor.get_location(),
                    )

                    # Assume the bicycles don't drive too much during the overtaking process
                    to_index += 170
                    from_index = int(self.privileged_route_planner.route_index)

                    starting_wp = self.route_waypoints[0].get_left_lane()
                    prev_road_lane_ids = roadgraph.get_previous_road_lane_ids(
                        self.config_expert,
                        starting_wp,
                    )
                    path_clear = self.is_two_ways_overtaking_path_clear(
                        int(from_index),
                        int(to_index),
                        target_speed,
                        prev_road_lane_ids,
                        min_speed=self.config_expert.driving.default_overtake_speed,
                    )

                    if path_clear:
                        transition_length = (
                            self.config_expert.driving.transition_smoothness_distance
                        )
                        self.privileged_route_planner.shift_route_smoothly(
                            int(from_index),
                            int(to_index),
                            True,
                            transition_length,
                        )
                        changed_route = True
                        CarlaDataProvider.get_current_scenario_memory().update(
                            {
                                "changed_route": changed_route,
                                "from_index": int(from_index),
                                "to_index": int(to_index),
                                "path_clear": path_clear,
                            },
                        )

                # the overtaking path is clear
                if path_clear:
                    # Check if the overtaking is done
                    if self.privileged_route_planner.route_index >= to_index:
                        CarlaDataProvider.clean_current_active_scenario()
                    # Overtake with max. 50 km/h
                    target_speed, keep_driving = (
                        self.config_expert.driving.default_overtake_speed,
                        True,
                    )

            elif self.current_active_scenario_type == "HazardAtSideLane":
                (
                    first_actor,
                    last_actor,
                    changed_first_part_of_route,
                    from_index,
                    to_index,
                ) = [
                    CarlaDataProvider.get_current_scenario_memory()[k]
                    for k in [
                        "first_actor",
                        "last_actor",
                        "changed_first_part_of_route",
                        "from_index",
                        "to_index",
                    ]
                ]

                horizontal_distance = geometry.get_horizontal_distance(
                    self.ego_vehicle,
                    last_actor,
                )

                if (
                    horizontal_distance
                    < self.config_expert.driving.max_distance_to_process_hazard_at_side_lane
                    and not changed_first_part_of_route
                ):
                    transition_length = (
                        self.config_expert.driving.transition_smoothness_distance
                    )
                    LOG.info(
                        f"[Route Change] HazardAtSideLane: Shifting route right to avoid hazard, distance={horizontal_distance:.1f}m",
                    )
                    from_index, to_index = (
                        self.privileged_route_planner.shift_route_around_actors(
                            first_actor,
                            last_actor,
                            "right",
                            transition_length,
                        )
                    )

                    to_index -= transition_length
                    changed_first_part_of_route = True
                    CarlaDataProvider.get_current_scenario_memory().update(
                        {
                            "changed_first_part_of_route": changed_first_part_of_route,
                            "from_index": from_index,
                            "to_index": to_index,
                        },
                    )

                if changed_first_part_of_route:
                    to_idx_ = self.privileged_route_planner.extend_lane_shift_transition_for_hazard_at_side_lane(
                        last_actor,
                        to_index,
                    )
                    to_index = to_idx_
                    CarlaDataProvider.get_current_scenario_memory()["to_index"] = (
                        to_index
                    )
                if self.privileged_route_planner.route_index > to_index:
                    LOG.info(
                        "[Scenario Exit] HazardAtSideLane: Passed hazard, cleaning up scenario",
                    )
                    CarlaDataProvider.clean_current_active_scenario()

            elif self.current_active_scenario_type == "YieldToEmergencyVehicle":
                emergency_veh, changed_route, from_index, to_index, to_left = [
                    CarlaDataProvider.get_current_scenario_memory()[k]
                    for k in [
                        "emergency_vehicle",
                        "changed_route",
                        "from_index",
                        "to_index",
                        "to_left",
                    ]
                ]

                horizontal_distance = geometry.get_horizontal_distance(
                    self.ego_vehicle,
                    emergency_veh,
                )

                if (
                    horizontal_distance
                    < self.config_expert.driving.default_max_distance_to_process_scenario
                    and not changed_route
                ):
                    # Assume the emergency vehicle doesn't drive more than 20 m during the overtaking process
                    from_index = (
                        self.privileged_route_planner.route_index
                        + 30 * self.config_expert.simulation.points_per_meter
                    )
                    to_index = (
                        from_index
                        + int(2 * self.config_expert.simulation.points_per_meter)
                        * self.config_expert.simulation.points_per_meter
                    )

                    transition_length = (
                        self.config_expert.driving.transition_smoothness_distance
                    )
                    to_left = (
                        self.route_waypoints[from_index].lane_change
                        != carla.LaneChange.Right
                    )
                    LOG.info(
                        f"[Route Change] YieldToEmergencyVehicle: Shifting route {'left' if to_left else 'right'} to yield to emergency vehicle",
                    )
                    self.privileged_route_planner.shift_route_smoothly(
                        int(from_index),
                        int(to_index),
                        to_left,
                        transition_length,
                    )

                    changed_route = True
                    to_index -= transition_length
                    CarlaDataProvider.get_current_scenario_memory().update(
                        {
                            "changed_route": changed_route,
                            "from_index": int(from_index),
                            "to_index": int(to_index),
                            "to_left": to_left,
                        },
                    )

                if changed_route:
                    to_idx_ = self.privileged_route_planner.extend_lane_shift_transition_for_yield_to_emergency_vehicle(
                        to_left,
                        to_index,
                    )
                    to_index = to_idx_
                    CarlaDataProvider.get_current_scenario_memory()["to_index"] = (
                        to_index
                    )

                    # Check if the emergency vehicle is in front of the ego vehicle
                    diff = emergency_veh.get_location() - ego_location
                    dot_res = (
                        self.ego_vehicle.get_transform().get_forward_vector().dot(diff)
                    )
                    if dot_res > 0:
                        LOG.info(
                            "[Scenario Exit] YieldToEmergencyVehicle: Emergency vehicle passed, cleaning up scenario",
                        )
                        CarlaDataProvider.clean_current_active_scenario()
                    else:
                        LOG.info(
                            f"[Control] YieldToEmergencyVehicle: Setting overtake speed to {self.config_expert.driving.default_overtake_speed * 3.6:.1f} km/h",
                        )
                        target_speed, keep_driving = (
                            self.config_expert.driving.default_overtake_speed,
                            True,
                        )

        # Negotiate gap
        self.construction_obstacle_two_ways_stuck = False
        self.accident_two_ways_stuck = False
        self.parked_obstacle_two_ways_stuck = False
        self.vehicle_opens_door_two_ways_stuck = False
        if self.current_active_scenario_type in ["ConstructionObstacleTwoWays"]:
            speed_constant = 2
            if self.speed_limit > 15:
                speed_constant = 2.75
            max_distance = speed_constant * self.ego_speed + 7
            obstacle_detection = self._vehicle_obstacle_detected(
                max_distance=max_distance,
            )
            if obstacle_detection[0]:
                _, targ_id, dist = obstacle_detection
                other = self.carla_world.get_actor(targ_id)

                ego_yaw = self.ego_vehicle.get_transform().rotation.yaw
                other_yaw = other.get_transform().rotation.yaw
                relative_yaw = abs(
                    geometry.normalize_angle_deg(other_yaw - ego_yaw),
                )

                # Only brake if roughly opposing direction (e.g. facing within 45° of us)
                if (
                    relative_yaw > 135 and other.get_velocity().length() > 0.0
                ):  # Only brake if the other vehicle is moving
                    self.construction_obstacle_two_ways_stuck = True
                    keep_driving = False
                    target_speed = 0.0
                    speed_reduced_by_obj = [0.0, "vehicle", targ_id, dist]
                elif relative_yaw > 135 and other.get_velocity().length() == 0.0:
                    CarlaDataProvider.clean_current_active_scenario()
                    keep_driving = False
                    target_speed = 0.0
                    speed_reduced_by_obj = [0.0, "vehicle", targ_id, dist]

        elif self.current_active_scenario_type in ["AccidentTwoWays"]:
            speed_constant = 1.75
            if self.speed_limit > 15:
                speed_constant = 2.75
            max_distance = speed_constant * self.ego_speed + 7
            obstacle_detection = self._vehicle_obstacle_detected(
                max_distance=max_distance,
            )
            if obstacle_detection[0]:
                _, targ_id, dist = obstacle_detection
                other = self.carla_world.get_actor(targ_id)

                ego_yaw = self.ego_vehicle.get_transform().rotation.yaw
                other_yaw = other.get_transform().rotation.yaw
                relative_yaw = abs(
                    geometry.normalize_angle_deg(other_yaw - ego_yaw),
                )

                # Only brake if roughly opposing direction (e.g. facing within 45° of us)
                if (
                    relative_yaw > 135 and other.get_velocity().length() > 0.0
                ):  # Only brake if the other vehicle is moving
                    self.accident_two_ways_stuck = True
                    keep_driving = False
                    target_speed = 0.0
                    speed_reduced_by_obj = [0.0, "vehicle", targ_id, dist]
                elif relative_yaw > 135 and other.get_velocity().length() == 0.0:
                    CarlaDataProvider.clean_current_active_scenario()
                    keep_driving = False
                    target_speed = 0.0
                    speed_reduced_by_obj = [0.0, "vehicle", targ_id, dist]

        elif self.current_active_scenario_type in ["ParkedObstacleTwoWays"]:
            speed_constant = 1.75
            if self.speed_limit > 15:
                speed_constant = 2.75
            max_distance = speed_constant * self.ego_speed + 7
            obstacle_detection = self._vehicle_obstacle_detected(
                max_distance=max_distance,
            )
            if obstacle_detection[0]:
                _, targ_id, dist = obstacle_detection
                other = self.carla_world.get_actor(targ_id)

                ego_yaw = self.ego_vehicle.get_transform().rotation.yaw
                other_yaw = other.get_transform().rotation.yaw
                relative_yaw = abs(
                    geometry.normalize_angle_deg(other_yaw - ego_yaw),
                )

                # Only brake if roughly opposing direction (e.g. facing within 45° of us)
                if (
                    relative_yaw > 135 and other.get_velocity().length() > 0.0
                ):  # Only brake if the other vehicle is moving
                    self.parked_obstacle_two_ways_stuck = True
                    keep_driving = False
                    target_speed = 0.0
                    speed_reduced_by_obj = [0.0, "vehicle", targ_id, dist]
                elif relative_yaw > 135 and other.get_velocity().length() == 0.0:
                    CarlaDataProvider.clean_current_active_scenario()
                    keep_driving = False
                    target_speed = 0.0
                    speed_reduced_by_obj = [0.0, "vehicle", targ_id, dist]

        elif self.current_active_scenario_type in ["VehicleOpensDoorTwoWays"]:
            speed_constant = 1.75
            if self.speed_limit > 15:
                speed_constant = 2.75
            max_distance = speed_constant * self.ego_speed + 7
            obstacle_detection = self._vehicle_obstacle_detected(
                max_distance=max_distance,
            )
            if obstacle_detection[0]:
                _, targ_id, dist = obstacle_detection
                other = self.carla_world.get_actor(targ_id)

                ego_yaw = self.ego_vehicle.get_transform().rotation.yaw
                other_yaw = other.get_transform().rotation.yaw
                relative_yaw = abs(
                    geometry.normalize_angle_deg(other_yaw - ego_yaw),
                )

                # Only brake if roughly opposing direction (e.g. facing within 45° of us)
                if (
                    relative_yaw > 135 and other.get_velocity().length() > 0.0
                ):  # Only brake if the other vehicle is moving
                    self.vehicle_opens_door_two_ways_stuck = True
                    keep_driving = False
                    target_speed = 0.0
                    speed_reduced_by_obj = [0.0, "vehicle", targ_id, dist]
                elif relative_yaw > 135 and other.get_velocity().length() == 0.0:
                    CarlaDataProvider.clean_current_active_scenario()
                    keep_driving = False
                    target_speed = 0.0
                    speed_reduced_by_obj = [0.0, "vehicle", targ_id, dist]

        return target_speed, keep_driving, speed_reduced_by_obj
