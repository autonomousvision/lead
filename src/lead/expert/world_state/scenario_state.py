"""Cached per-step view of the active scenario and its actors."""

import logging

import carla
import cv2
import numpy as np
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

import lead.common.geometry as geometry
from lead.expert.utils import roadgraph
from lead.expert.utils.caching import step_cached_property

LOG = logging.getLogger(__name__)


class ScenarioStateMixin:
    """Cached per-step scenario type, adversarial actors and obstacle distances."""

    @step_cached_property
    def distance_to_intersection_index_ego(self) -> float:
        """
        Returns the index of the intersection point in the route waypoints.
        If no intersection point is found, returns None.
        """
        if self.current_active_scenario_type in [
            "SignalizedJunctionLeftTurn",
            "NonSignalizedJunctionLeftTurn",
            "SignalizedJunctionRightTurn",
            "NonSignalizedJunctionRightTurn",
            "SignalizedJunctionLeftTurnEnterFlow",
            "NonSignalizedJunctionLeftTurnEnterFlow",
            "InterurbanActorFlow",
        ]:
            intersection_index_ego = (
                CarlaDataProvider.get_current_scenario_memory().get(
                    "intersection_index_ego",
                    None,
                )
            )
            if intersection_index_ego is not None:
                return (
                    intersection_index_ego - self.privileged_route_planner.route_index
                ) / self.config_expert.simulation.points_per_meter
        return float("inf")

    @step_cached_property
    def adversarial_actors_ids(self) -> tuple[list, list, list]:
        """
        Return a tuple of:
            - dangerous adversarial actors IDs: we should be very waried of them
            - safe adversarial actors IDs: we can treat their bounding boxes a bit smaller
            - ignored adversarial actors IDs: we can ignore them completely
        """
        # Obstacle scenarios: compute source and target lane once
        if self.current_active_scenario_type in [
            "Accident",
            "ConstructionObstacle",
            "ParkedObstacle",
        ]:
            obstacle, direction = [
                CarlaDataProvider.get_current_scenario_memory()[key]
                for key in ["first_actor", "direction"]
            ]
            source_lane = self.carla_world_map.get_waypoint(
                obstacle.get_location(),
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            target_lane = (
                source_lane.get_right_lane()
                if direction == "left"
                else source_lane.get_left_lane()
            )
            if source_lane and target_lane:
                CarlaDataProvider.get_current_scenario_memory()["source_lane"] = (
                    source_lane
                )
                CarlaDataProvider.get_current_scenario_memory()["target_lane"] = (
                    target_lane
                )

        if self.current_active_scenario_type in ["HazardAtSideLane"]:
            if CarlaDataProvider.get_current_scenario_memory()["bicycle_1"] is not None:
                target_lane = CarlaDataProvider.get_current_scenario_memory()[
                    "target_lane"
                ]
                source_lane = CarlaDataProvider.get_current_scenario_memory()[
                    "source_lane"
                ]
                if target_lane is None or source_lane is None:
                    bicycle_1 = CarlaDataProvider.get_current_scenario_memory()[
                        "bicycle_1"
                    ]
                    source_lane = self.carla_world_map.get_waypoint(
                        bicycle_1.get_location(),
                        project_to_road=True,
                        lane_type=carla.LaneType.Driving,
                    )
                    target_lane = source_lane.get_left_lane()
                    CarlaDataProvider.get_current_scenario_memory()["target_lane"] = (
                        target_lane
                    )
                    CarlaDataProvider.get_current_scenario_memory()["soure_lane"] = (
                        source_lane
                    )

        # One way obstacle scenarios: adversarial actors are those on the target lane
        if (
            min(
                [
                    self.distance_to_accident_site,
                    self.distance_to_construction_site,
                    self.distance_to_parked_obstacle,
                ],
            )
            <= 40
            or self.current_active_scenario_type == "HazardAtSideLane"
        ):
            for scenario in [
                "Accident",
                "ConstructionObstacle",
                "ParkedObstacle",
                "HazardAtSideLane",
            ]:
                dangerous_adversarial_actors_ids = []
                safe_adversarial_actors_ids = []
                ignored_adversarial_actors_ids = []
                if self.current_active_scenario_type != scenario:
                    continue

                # Get memory for the current active scenario
                current_memory = CarlaDataProvider.get_current_scenario_memory()
                if current_memory is None:
                    continue

                if (
                    current_memory
                    and "source_lane" in current_memory
                    and "target_lane" in current_memory
                    and current_memory["source_lane"] is not None
                    and current_memory["target_lane"] is not None
                ):
                    target_lane = current_memory["target_lane"]
                    for actor in self.vehicles_inside_bev:
                        if actor.id == self.ego_vehicle.id:
                            continue
                        actor_lane = self.carla_world_map.get_waypoint(
                            actor.get_location(),
                            project_to_road=True,
                            lane_type=carla.LaneType.Driving,
                        )
                        if actor_lane and actor_lane.lane_id == target_lane.lane_id:
                            rel_loc = geometry.get_relative_transform(
                                self.ego_matrix,
                                np.array(actor.get_transform().get_matrix()),
                            )
                            if self.speed_limit > 25:
                                dangerous_adversarial_actors_ids.append(actor.id)
                            else:
                                if 0.5 <= rel_loc[0]:  # actor in front, is safe
                                    safe_adversarial_actors_ids.append(actor.id)
                                else:  # Normal speed, be more careful
                                    dangerous_adversarial_actors_ids.append(actor.id)
                current_memory["dangerous_adversarial_actors_ids"] = (
                    dangerous_adversarial_actors_ids
                )
                current_memory["safe_adversarial_actors_ids"] = (
                    safe_adversarial_actors_ids
                )
                current_memory["ignored_adversarial_actors_ids"] = (
                    ignored_adversarial_actors_ids
                )

        # High speed merging scenarios
        for scenario in [
            "EnterActorFlow",
            "EnterActorFlowV2",
            "InterurbanAdvancedActorFlow",
        ]:
            if self.current_active_scenario_type != scenario:
                continue
            safe_adversarial_actors_ids = []
            ignored_adversarial_actors_ids = []
            dangerous_adversarial_actors_ids = []
            for adversarial_actor in CarlaDataProvider.get_current_scenario_memory()[
                "adversarial_actors"
            ]:
                try:
                    if not self.is_actor_inside_bev(adversarial_actor):
                        continue
                    adversarial_lane = self.carla_world_map.get_waypoint(
                        adversarial_actor.get_location(),
                        project_to_road=True,
                        lane_type=carla.LaneType.Driving,
                    )
                    if self.ego_lane_id != adversarial_lane.lane_id:
                        continue
                    dangerous_adversarial_actors_ids.append(adversarial_actor.id)
                except:
                    pass

            CarlaDataProvider.get_current_scenario_memory()[
                "dangerous_adversarial_actors_ids"
            ] = dangerous_adversarial_actors_ids
            CarlaDataProvider.get_current_scenario_memory()[
                "safe_adversarial_actors_ids"
            ] = safe_adversarial_actors_ids
            CarlaDataProvider.get_current_scenario_memory()[
                "ignored_adversarial_actors_ids"
            ] = ignored_adversarial_actors_ids

        # Priority scenarios
        for scenario in [
            "OppositeVehicleRunningRedLight",
            "OppositeVehicleTakingPriority",
        ]:
            if self.current_active_scenario_type != scenario:
                continue
            safe_adversarial_actors_ids = []
            ignored_adversarial_actors_ids = []
            dangerous_adversarial_actors_ids = []
            for adversarial_actor in CarlaDataProvider.get_current_scenario_memory()[
                "adversarial_actors"
            ]:
                try:
                    if (
                        not self.is_actor_inside_bev(adversarial_actor)
                        or adversarial_actor.get_velocity().length() < 0.1
                        or (
                            self.id2bb_map[adversarial_actor.id]["visible_pixels"] < 10
                            and self.id2bb_map[adversarial_actor.id]["num_points"] < 10
                        )
                    ):
                        continue
                    dangerous_adversarial_actors_ids.append(adversarial_actor.id)
                except:
                    pass

            CarlaDataProvider.get_current_scenario_memory()[
                "dangerous_adversarial_actors_ids"
            ] = dangerous_adversarial_actors_ids
            CarlaDataProvider.get_current_scenario_memory()[
                "safe_adversarial_actors_ids"
            ] = safe_adversarial_actors_ids
            CarlaDataProvider.get_current_scenario_memory()[
                "ignored_adversarial_actors_ids"
            ] = ignored_adversarial_actors_ids

        # Unprotected left and right turns scenarios
        for scenario in [
            "SignalizedJunctionLeftTurn",
            "NonSignalizedJunctionLeftTurn",
            "SignalizedJunctionRightTurn",
            "NonSignalizedJunctionRightTurn",
            "SignalizedJunctionLeftTurnEnterFlow",
            "NonSignalizedJunctionLeftTurnEnterFlow",
            "InterurbanActorFlow",
        ]:
            if self.current_active_scenario_type != scenario:
                continue

            # Get memory for the current active scenario
            current_memory = CarlaDataProvider.get_current_scenario_memory()
            if current_memory is None:
                continue

            # Computer intersection point of ego route and adversarial route
            source_wp: carla.Waypoint = current_memory["source_wp"]
            sink_wp: carla.Waypoint = current_memory["sink_wp"]

            # Skip if waypoints are None (invalid spawn locations)
            if source_wp is None or sink_wp is None:
                continue

            opponent_traffic_route = current_memory["opponent_traffic_route"]
            if opponent_traffic_route is None:
                opponent_traffic_route = roadgraph.compute_global_route(
                    world=self.carla_world,
                    source_location=source_wp.transform.location,
                    sink_location=sink_wp.transform.location,
                )
                current_memory["opponent_traffic_route"] = opponent_traffic_route

            intersection_point = current_memory["intersection_point"]
            if opponent_traffic_route is not None and intersection_point is None:
                intersection_point, intersection_index_ego = (
                    roadgraph.intersection_of_routes(
                        points_a=self.route_waypoints_np[
                            : self.config_expert.driving.draw_future_route_till_distance
                        ],  # Don't use full route otherwise too expensive
                        points_b=opponent_traffic_route,
                    )
                )
                if intersection_index_ego is not None:
                    intersection_index_ego += self.privileged_route_planner.route_index
                current_memory["intersection_index_ego"] = intersection_index_ego
                current_memory["intersection_point"] = intersection_point

            # Filter adversarial actors for unprotected left turns
            intersection_point = current_memory["intersection_point"]
            if intersection_point is not None:
                safe_adversarial_actors_ids = current_memory[
                    "safe_adversarial_actors_ids"
                ]  # Once safe, an adversarial actor won't become dangerous again
                ignored_adversarial_actors_ids = []
                dangerous_adversarial_actors_ids = []
                for adversarial_actor in current_memory["adversarial_actors"]:
                    if adversarial_actor.id == self.ego_vehicle.id:
                        continue
                    if adversarial_actor.id in safe_adversarial_actors_ids:
                        continue
                    try:
                        if not self.is_actor_inside_bev(adversarial_actor):
                            continue
                        if (
                            self.distance_to_next_junction < 10
                            and (
                                (
                                    self.distance_to_intersection_index_ego < 13
                                    and scenario
                                    in [
                                        "SignalizedJunctionLeftTurnEnterFlow",
                                        "NonSignalizedJunctionLeftTurnEnterFlow",
                                    ]
                                )
                                or (
                                    self.distance_to_intersection_index_ego < 13
                                    and scenario
                                    in [
                                        "NonSignalizedJunctionLeftTurn",
                                        "InterurbanActorFlow",
                                    ]
                                )
                                or (
                                    self.distance_to_intersection_index_ego < 18
                                    and scenario
                                    in [
                                        "SignalizedJunctionRightTurn",
                                        "NonSignalizedJunctionRightTurn",
                                    ]
                                )
                                or (
                                    self.distance_to_intersection_index_ego < 23
                                    and scenario in ["SignalizedJunctionLeftTurn"]
                                )
                            )
                            and not self.stop_sign_hazard
                            and not self.traffic_light_hazard
                        ):  # If only we are not near enough to the junction, we ignore adversarial actors. Smoother stopping
                            if scenario in [
                                "SignalizedJunctionLeftTurn",
                                "NonSignalizedJunctionLeftTurn",
                                "InterurbanActorFlow",
                            ]:
                                safe_threshold = (
                                    self.distance_to_intersection_index_ego * 1.1
                                )  # Safe threshold, the lower, the earlier we ignore an adversarial actor
                                if (
                                    scenario in ["SignalizedJunctionLeftTurn"]
                                ):  # Urban scenarios, we need to treat them a bit differently
                                    if self.distance_to_intersection_index_ego < 13:
                                        safe_threshold = (
                                            self.distance_to_intersection_index_ego
                                            * 1.2
                                        )  # Urban, we need more time to reach intersection. Only empirical observations.
                                    else:
                                        safe_threshold = (
                                            self.distance_to_intersection_index_ego
                                            * 1.6
                                        )  # Urban, we need more time to reach intersection. Only empirical observations.
                                safe_threshold = min(
                                    safe_threshold,
                                    22,
                                )  # Don't go too far, otherwise we ignore all actors
                                if (
                                    adversarial_actor.get_location().distance(
                                        intersection_point,
                                    )
                                    < safe_threshold
                                ):  # If actor is/was near enough to the intersection point, we can safely ignore it
                                    safe_adversarial_actors_ids.append(
                                        adversarial_actor.id,
                                    )
                                else:
                                    dangerous_adversarial_actors_ids.append(
                                        adversarial_actor.id,
                                    )
                            elif scenario in [
                                "SignalizedJunctionRightTurn",
                                "NonSignalizedJunctionRightTurn",
                            ]:
                                safe_threshold = (
                                    self.distance_to_intersection_index_ego * 1.1
                                )
                                if (
                                    scenario in ["SignalizedJunctionRightTurn"]
                                ):  # Urban scenarios, we need to treat them a bit differently
                                    if self.distance_to_intersection_index_ego < 13:
                                        safe_threshold = (
                                            self.distance_to_intersection_index_ego
                                            * 1.2
                                        )  # Urban, we need more time to reach intersection. Only empirical observations.
                                    else:
                                        safe_threshold = (
                                            self.distance_to_intersection_index_ego
                                            * 1.6
                                        )  # Urban, we need more time to reach intersection. Only empirical observations.
                                safe_threshold = min(
                                    safe_threshold,
                                    22,
                                )  # Don't go too far, otherwise we ignore all actors
                                if self.distance_to_intersection_index_ego < 5:
                                    safe_adversarial_actors_ids.append(
                                        adversarial_actor.id,
                                    )  # If we are very close to the intersection, we really want to commit
                                elif (
                                    adversarial_actor.get_location().distance(
                                        intersection_point,
                                    )
                                    < safe_threshold
                                ):  # If actor is/was near enough to the intersection point, we can safely ignore it
                                    safe_adversarial_actors_ids.append(
                                        adversarial_actor.id,
                                    )
                                else:
                                    dangerous_adversarial_actors_ids.append(
                                        adversarial_actor.id,
                                    )
                            elif scenario in [
                                "SignalizedJunctionLeftTurnEnterFlow",
                                "NonSignalizedJunctionLeftTurnEnterFlow",
                            ]:
                                adversarial_actor_location = (
                                    adversarial_actor.get_location()
                                )
                                if (
                                    roadgraph.distance_location_to_route(
                                        route=current_memory["opponent_traffic_route"],
                                        location=np.array(
                                            [
                                                adversarial_actor_location.x,
                                                adversarial_actor_location.y,
                                                adversarial_actor_location.z,
                                            ],
                                        ),
                                    )
                                    > 1.0
                                ):
                                    # If actor is further than the intersection point in the route, we can safely ignore it
                                    safe_adversarial_actors_ids.append(
                                        adversarial_actor.id,
                                    )
                                    LOG.info(
                                        "Adversarial actor went out of route. ignore",
                                    )
                                else:
                                    dangerous_adversarial_actors_ids.append(
                                        adversarial_actor.id,
                                    )
                        else:
                            ignored_adversarial_actors_ids = [
                                actor.id
                                for actor in CarlaDataProvider.get_current_scenario_memory()[
                                    "adversarial_actors"
                                ]
                            ]

                    except RuntimeError as e:
                        if "trying to operate on a destroyed actor" in str(e):
                            ignored_adversarial_actors_ids.append(adversarial_actor.id)
                            continue
                        else:
                            raise e

                current_memory["dangerous_adversarial_actors_ids"] = (
                    dangerous_adversarial_actors_ids
                )
                current_memory["safe_adversarial_actors_ids"] = (
                    safe_adversarial_actors_ids
                )
                current_memory["ignored_adversarial_actors_ids"] = (
                    ignored_adversarial_actors_ids
                )

        current_scenario_memory = CarlaDataProvider.get_current_scenario_memory()
        if (
            current_scenario_memory is not None
            and "dangerous_adversarial_actors_ids" in current_scenario_memory
        ):
            return (
                current_scenario_memory["dangerous_adversarial_actors_ids"],
                current_scenario_memory["safe_adversarial_actors_ids"],
                current_scenario_memory["ignored_adversarial_actors_ids"],
            )
        return [], [], []

    @step_cached_property
    def rear_adversarial_actor(self) -> carla.Actor | None:
        rear_adversarial_vehicle = None
        if (
            self.current_active_scenario_type
            in [
                "SignalizedJunctionRightTurn",
                "NonSignalizedJunctionRightTurn",
                "SignalizedJunctionLeftTurnEnterFlow",
                "NonSignalizedJunctionLeftTurnEnterFlow",
            ]
            and self.distance_to_intersection_index_ego < 2
        ):
            min_distance = float("inf")
            rear_adversarial_vehicle = None
            (
                dangerous_adversarial_actors_ids,
                safe_adversarial_actors_ids,
                ignored_adversarial_actors_ids,
            ) = self.adversarial_actors_ids
            for vehicle in self.vehicles_inside_bev:
                if vehicle.id == self.ego_vehicle.id:
                    continue
                if (
                    vehicle.id not in dangerous_adversarial_actors_ids
                    and vehicle.id not in safe_adversarial_actors_ids
                ):
                    continue
                rel_loc = geometry.get_relative_transform(
                    self.ego_matrix,
                    np.array(vehicle.get_transform().get_matrix()),
                )
                if rel_loc[0] < -1.0:  # Vehicle is behind the ego vehicle
                    distance = np.linalg.norm(rel_loc[:2])
                    if distance < min_distance:
                        min_distance = distance
                        rear_adversarial_vehicle = vehicle
        elif self.current_active_scenario_type in [
            "EnterActorFlow",
            "EnterActorFlowV2",
            "InterurbanAdvancedActorFlow",
        ]:
            min_distance = float("inf")
            rear_adversarial_vehicle = None
            (
                dangerous_adversarial_actors_ids,
                safe_adversarial_actors_ids,
                ignored_adversarial_actors_ids,
            ) = self.adversarial_actors_ids
            for vehicle in self.vehicles_inside_bev:
                if vehicle.id == self.ego_vehicle.id:
                    continue
                if vehicle.id not in dangerous_adversarial_actors_ids:
                    continue
                rel_loc = geometry.get_relative_transform(
                    self.ego_matrix,
                    np.array(vehicle.get_transform().get_matrix()),
                )
                if rel_loc[0] < -1.0:  # Vehicle is behind the ego vehicle
                    distance = np.linalg.norm(rel_loc[:2])
                    if distance < min_distance:
                        min_distance = distance
                        rear_adversarial_vehicle = vehicle
        elif self.current_active_scenario_type in [
            "OppositeVehicleRunningRedLight",
            "OppositeVehicleTakingPriority",
        ]:
            min_distance = float("inf")
            rear_adversarial_vehicle = None
            (
                dangerous_adversarial_actors_ids,
                safe_adversarial_actors_ids,
                ignored_adversarial_actors_ids,
            ) = self.adversarial_actors_ids
            for vehicle in self.vehicles_inside_bev:
                if vehicle.id == self.ego_vehicle.id:
                    continue
                if vehicle.id not in dangerous_adversarial_actors_ids:
                    continue
                rel_loc = geometry.get_relative_transform(
                    self.ego_matrix,
                    np.array(vehicle.get_transform().get_matrix()),
                )
                if rel_loc[0] < -1.0:  # Vehicle is behind the ego vehicle
                    distance = np.linalg.norm(rel_loc[:2])
                    if distance < min_distance:
                        min_distance = distance
                        rear_adversarial_vehicle = vehicle
        return rear_adversarial_vehicle

    @step_cached_property
    def target_lane_width(self):
        if self.current_active_scenario_type in [
            "Accident",
            "ConstructionObstacle",
            "ParkedObstacle",
            "HazardAtSideLane",
        ]:
            target_lane = CarlaDataProvider.get_current_scenario_memory()["target_lane"]
            if target_lane is not None:
                return target_lane.lane_width

        if self.current_active_scenario_type in [
            "SignalizedJunctionLeftTurn",
            "NonSignalizedJunctionLeftTurn",
            "SignalizedJunctionRightTurn",
            "NonSignalizedJunctionRightTurn",
            "SignalizedJunctionLeftTurnEnterFlow",
            "NonSignalizedJunctionLeftTurnEnterFlow",
            "InterurbanActorFlow",
        ]:
            sink_wp = CarlaDataProvider.get_current_scenario_memory()["sink_wp"]
            if sink_wp is not None:
                return sink_wp.lane_width

        return None

    @property
    def current_active_scenario_type(self) -> str | None:
        if len(CarlaDataProvider.active_scenarios) == 0:
            return None
        return CarlaDataProvider.active_scenarios[0].name

    @property
    def previous_active_scenario_type(self) -> str | None:
        if CarlaDataProvider.previous_active_scenario is not None:
            return CarlaDataProvider.previous_active_scenario.name
        return None

    @step_cached_property
    def distance_to_construction_site(self) -> float:
        if self.current_active_scenario_type in [
            "ConstructionObstacle",
            "ConstructionObstacleTwoWays",
        ] or self.previous_active_scenario_type in [
            "ConstructionObstacle",
            "ConstructionObstacleTwoWays",
        ]:
            num_cones = 0
            num_warning_traffic_signs = 0
            distances = []
            for static in self.static_inside_bev:
                if static.type_id == "static.prop.constructioncone":
                    num_cones += 1
                    distances.append(static.get_location().distance(self.ego_location))
                elif static.type_id == "static.prop.trafficwarning":
                    num_warning_traffic_signs += 1
                    distances.append(static.get_location().distance(self.ego_location))
            if num_cones > 0 and num_warning_traffic_signs > 0:
                distances = np.array(distances)
                distance = distances.mean()
                return float(distance)
        return float("inf")

    @step_cached_property
    def distance_to_scenario_obstacle(self) -> float:
        return min(
            [
                self.distance_to_accident_site,
                self.distance_to_construction_site,
                self.distance_to_parked_obstacle,
                self.distance_to_vehicle_opens_door,
            ],
        )

    @step_cached_property
    def distance_to_accident_site(self) -> float:
        if self.current_active_scenario_type in [
            "Accident",
            "AccidentTwoWays",
        ] or self.previous_active_scenario_type in [
            "Accident",
            "AccidentTwoWays",
        ]:
            distances = []
            num_scenario_cars = 0
            for actor in self.scenario_obstacles:
                if (
                    "scenario" in actor.attributes["role_name"]
                    and self._get_actor_forward_speed(actor) == 0.0
                ):
                    num_scenario_cars += 1
                    distances.append(actor.get_location().distance(self.ego_location))
            if num_scenario_cars > 0:
                return float(np.mean(distances))
        return float("inf")

    @step_cached_property
    def distance_to_parked_obstacle(self) -> float:
        if self.current_active_scenario_type in [
            "ParkedObstacle",
            "ParkedObstacleTwoWays",
        ] or self.previous_active_scenario_type in [
            "ParkedObstacle",
            "ParkedObstacleTwoWays",
        ]:
            distances = []
            num_scenario_cars = 0
            for actor in self.scenario_obstacles:
                if (
                    "scenario" in actor.attributes["role_name"]
                    and self._get_actor_forward_speed(actor) == 0.0
                ):
                    num_scenario_cars += 1
                    distances.append(actor.get_location().distance(self.ego_location))
            if num_scenario_cars > 0:
                return float(np.mean(distances))
        return float("inf")

    @step_cached_property
    def distance_to_vehicle_opens_door(self) -> float:
        if self.current_active_scenario_type in [
            "VehicleOpensDoorTwoWays",
        ] or self.previous_active_scenario_type in ["VehicleOpensDoorTwoWays"]:
            distances = []
            num_scenario_cars = 0
            for actor in self.scenario_obstacles:
                if (
                    "scenario" in actor.attributes["role_name"]
                    and self._get_actor_forward_speed(actor) == 0.0
                ):
                    num_scenario_cars += 1
                    distances.append(actor.get_location().distance(self.ego_location))
            if num_scenario_cars > 0:
                return float(np.mean(distances))
        return float("inf")

    @step_cached_property
    def distance_to_cutin_vehicle(self) -> float:
        if not self.config_expert.data_collection.datagen:
            return float("inf")
        if self.current_active_scenario_type in [
            "ParkingCutIn",
            "StaticCutIn",
            "HighwayCutIn",
        ]:
            distances = []
            num_scenario_cars = 0
            for actor in self.cutin_actors:
                if self.is_actor_inside_bev(actor):
                    num_scenario_cars += 1
                    distances.append(actor.get_location().distance(self.ego_location))
            if num_scenario_cars > 0:
                return float(np.mean(distances))
        return float("inf")

    @step_cached_property
    def scenario_actors(self) -> list[carla.Actor]:
        ret = []
        for actor in (
            self.vehicles_inside_bev + self.walkers_inside_bev + self.bikers_inside_bev
        ):
            if "scenario" in actor.attributes["role_name"]:
                ret.append(actor)
        return ret

    @step_cached_property
    def scenario_actors_ids(self) -> list[int]:
        """
        Get the IDs of the scenario actors that are currently inside the BEV (Bird's Eye View) range.

        Returns:
            list: A list of IDs of the scenario actors.
        """
        return [actor.id for actor in self.scenario_actors]

    @step_cached_property
    def scenario_obstacles(self) -> list[carla.Actor]:
        ret = []
        scenarios = [
            "Accident",
            "ConstructionObstacle",
            "ParkedObstacle",
            "AccidentTwoWays",
            "ConstructionObstacleTwoWays",
            "ParkedObstacleTwoWays",
            "VehicleOpensDoorTwoWays",
            "InvadingTurn",
            "BlockedIntersection",
        ]
        if self.current_active_scenario_type in scenarios:
            ret = CarlaDataProvider.get_current_scenario_memory()["obstacles"]
        elif self.previous_active_scenario_type in scenarios:
            if CarlaDataProvider.previous_active_scenario is not None:
                obstacles = CarlaDataProvider.previous_active_scenario.meta["obstacles"]
                try:
                    obstacles = [
                        actor for actor in obstacles if self.is_actor_inside_bev(actor)
                    ]
                    ret = obstacles
                except RuntimeError as e:
                    if "trying to operate on a destroyed actor" in str(e):
                        # If the scenario obstacles were destroyed, return an empty list
                        ret = []
                    else:
                        raise e
        ret = [actor for actor in ret if self.is_actor_inside_bev(actor)]
        return ret

    @step_cached_property
    def scenario_obstacles_convex_hull(self):
        """
        Get the convex hull of the scenario obstacles' bounding box corners that are currently inside the BEV range.

        Returns:
            list: A list of (x, y) points representing the convex hull of the obstacles.
        """
        if not self.scenario_obstacles:
            return []

        points = []
        for actor in self.scenario_obstacles:
            bbox = actor.bounding_box
            actor_transform = actor.get_transform()

            # Get bounding box center in world coordinates
            bbox_center_world = actor_transform.transform(bbox.location)

            # Rotation matrix from actor yaw
            yaw = np.radians(actor_transform.rotation.yaw)
            cos_yaw = np.cos(yaw)
            sin_yaw = np.sin(yaw)

            extent = bbox.extent

            # Define corners in local coordinates
            local_corners = [
                (extent.x, extent.y),
                (-extent.x, extent.y),
                (-extent.x, -extent.y),
                (extent.x, -extent.y),
            ]

            # Transform corners to world coordinates
            for lx, ly in local_corners:
                x = bbox_center_world.x + lx * cos_yaw - ly * sin_yaw
                y = bbox_center_world.y + lx * sin_yaw + ly * cos_yaw
                points.append((x, y))

        if len(points) < 3:
            return points

        points_np = np.array(points, dtype=np.float32)
        hull = cv2.convexHull(points_np)
        return hull.squeeze().tolist()

    @step_cached_property
    def scenario_obstacles_ids(self):
        """
        Get the IDs of the scenario obstacles that are currently inside the BEV (Bird's Eye View) range.

        Returns:
            list: A list of IDs of the scenario obstacles.
        """
        return [actor.id for actor in self.scenario_obstacles]

    @step_cached_property
    def vehicle_opened_door(self):
        """
        Check if the vehicle opened its door in the current scenario.
        This is used to determine if the agent should react to a vehicle opening its door.
        """
        if self.current_active_scenario_type == "VehicleOpensDoorTwoWays":
            return CarlaDataProvider.get_current_scenario_memory()[
                "vehicle_opened_door"
            ]
        elif self.previous_active_scenario_type == "VehicleOpensDoorTwoWays":
            if CarlaDataProvider.previous_active_scenario is not None:
                try:
                    CarlaDataProvider.previous_active_scenario.meta["obstacles"][
                        0
                    ].get_location()
                    return CarlaDataProvider.previous_active_scenario.meta[
                        "vehicle_opened_door"
                    ]
                except RuntimeError as e:
                    if "trying to operate on a destroyed actor" in str(e):
                        return False
                    else:
                        raise e
        return False

    @step_cached_property
    def vehicle_door_side(self):
        """
        Get the side of the vehicle that opened its door in the current scenario.
        This is used to determine if the agent should react to a vehicle opening its door.
        """
        if self.current_active_scenario_type == "VehicleOpensDoorTwoWays":
            return CarlaDataProvider.get_current_scenario_memory()["vehicle_door_side"]
        elif self.previous_active_scenario_type == "VehicleOpensDoorTwoWays":
            if CarlaDataProvider.previous_active_scenario is not None:
                return CarlaDataProvider.previous_active_scenario.meta[
                    "vehicle_door_side"
                ]
        return None

    @step_cached_property
    def cutin_actors(self):
        if self.current_active_scenario_type in [
            "ParkingCutIn",
            "StaticCutIn",
            "HighwayCutIn",
        ]:
            return [CarlaDataProvider.get_current_scenario_memory()["cut_in_vehicle"]]
        return []

    @step_cached_property
    def cut_in_actors_ids(self):
        return [actor.id for actor in self.cutin_actors]

    @step_cached_property
    def two_way_obstacle_distance_to_cones_factor(self):
        if self.ego_lane_width <= 2.76:
            return 1.13
        elif self.ego_lane_width <= 3.01:
            return 1.12
        return 1.12

    @step_cached_property
    def two_way_vehicle_open_door_distance_to_center_line_factor(self):
        if self.ego_lane_width <= 2.76:
            return 1.0
        elif self.ego_lane_width <= 3.01:
            return 0.875
        return 0.75

    @step_cached_property
    def add_after_construction_obstacle_two_ways(self):
        if self.ego_lane_width <= 2.76:
            return (
                self.config_expert.driving.add_after_construction_obstacle_two_ways
                + 0.5
            )
        return self.config_expert.driving.add_after_construction_obstacle_two_ways

    @step_cached_property
    def add_before_construction_obstacle_two_ways(self):
        if self.ego_lane_width <= 2.76:
            return (
                self.config_expert.driving.add_before_construction_obstacle_two_ways
                + 0.5
            )
        return self.config_expert.driving.add_before_construction_obstacle_two_ways

    @step_cached_property
    def two_way_overtake_speed(self):
        return {
            "AccidentTwoWays": self.config_expert.driving.default_overtake_speed,
            "ConstructionObstacleTwoWays": self.config_expert.driving.default_overtake_speed,
            "ParkedObstacleTwoWays": self.config_expert.driving.default_overtake_speed,
            "VehicleOpensDoorTwoWays": self.config_expert.driving.default_overtake_speed
            if self.ego_lane_width > 3.01
            else self.config_expert.driving.overtake_speed_vehicle_opens_door_two_ways,
        }[self.current_active_scenario_type]

    @step_cached_property
    def add_after_accident_two_ways(self):
        if self.ego_lane_width <= 2.76:
            return self.config_expert.driving.add_after_accident_two_ways + 0.5
        return self.config_expert.driving.add_after_accident_two_ways

    @step_cached_property
    def add_before_accident_two_ways(self):
        if self.ego_lane_width <= 2.76:
            return self.config_expert.driving.add_before_accident_two_ways + 0.5
        return self.config_expert.driving.add_before_accident_two_ways
