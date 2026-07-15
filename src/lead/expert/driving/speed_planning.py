"""Longitudinal planning for the expert: IDM target-speed selection from
forecasted actor bounding-box overlaps, red lights and stop signs."""

import logging
import numbers

import carla
import jaxtyping as jt
import numpy as np
import numpy.typing as npt
from agents.tools.misc import compute_distance, is_within_distance
from scipy.integrate import RK45
from shapely.geometry import Polygon
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from lead.common import constants
from lead.config import ExpertConfig
from lead.expert.driving import forecast_kernels
from lead.expert.utils import weather
from lead.expert.utils.weather_presets import WeatherVisibility

LOG = logging.getLogger(__name__)


def compute_target_speed_idm(
    config: ExpertConfig,
    desired_speed: numbers.Real,
    leading_actor_length: numbers.Real,
    ego_speed: numbers.Real,
    leading_actor_speed: numbers.Real,
    distance_to_leading_actor: numbers.Real,
    s0: numbers.Real = 4.0,
    T: numbers.Real = 0.5,
):
    """
    Compute the target speed for the ego vehicle using the Intelligent Driver Model (IDM).

    Args:
        config: Configuration object containing IDM parameters.
        desired_speed: The desired speed of the ego vehicle.
        leading_actor_length: The length of the leading actor (vehicle or obstacle).
        ego_speed: The current speed of the ego vehicle.
        leading_actor_speed: The speed of the leading actor.
        distance_to_leading_actor: The distance to the leading actor.
        s0: The minimum desired net distance.
        T: The desired time headway.

    Returns:
        The computed target speed for the ego vehicle.
    """

    a = config.driving.idm_maximum_acceleration  # Maximum acceleration [m/s²]
    b = (
        config.driving.idm_comfortable_braking_deceleration_high_speed
        if ego_speed > config.driving.idm_comfortable_braking_deceleration_threshold
        else config.driving.idm_comfortable_braking_deceleration_low_speed
    )  # Comfortable deceleration [m/s²]
    delta = config.driving.idm_acceleration_exponent  # Acceleration exponent

    t_bound = config.driving.idm_t_bound

    def idm_equations(t: float, x: jt.Float[npt.NDArray, " 2"]) -> list[float]:
        """
        Differential equations for the Intelligent Driver Model.

        Args:
            t: Time.
            x: State variables [position, speed].

        Returns:
            Derivatives of the state variables.
        """
        ego_position, ego_speed = x

        speed_diff = ego_speed - leading_actor_speed
        s_star = s0 + ego_speed * T + ego_speed * speed_diff / 2.0 / np.sqrt(a * b)
        # The maximum is needed to avoid numerical unstabilities
        s = max(
            0.1,
            distance_to_leading_actor
            + t * leading_actor_speed
            - ego_position
            - leading_actor_length,
        )
        dvdt = a * (1.0 - (ego_speed / desired_speed) ** delta - (s_star / s) ** 2)

        return [ego_speed, dvdt]

    # Set the initial conditions
    y0 = [0.0, ego_speed]

    # Integrate the differential equations using RK45
    rk45 = RK45(fun=idm_equations, t0=0.0, y0=y0, t_bound=t_bound)
    while rk45.status == "running":
        rk45.step()

    # The target speed is the final speed obtained from the integration
    target_speed = rk45.y[1]

    # Clip the target speed to non-negative values
    return np.clip(target_speed, 0, np.inf)


def compute_min_time_for_distance(
    config: ExpertConfig,
    distance: float,
    target_speed: float,
    ego_speed: float,
) -> float:
    """
    Computes the minimum time the ego vehicle needs to travel a given distance.

    Args:
        config: Configuration object containing vehicle dynamics parameters.
        distance: The distance to be traveled.
        target_speed: The target speed of the ego vehicle.
        ego_speed: The current speed of the ego vehicle.

    Returns:
        The minimum time needed to travel the given distance.
    """
    min_time_needed = 0.0
    remaining_distance = distance
    current_speed = ego_speed

    # Iterate over time steps until the distance is covered
    while True:
        if current_speed == 0 and target_speed == 0 and remaining_distance > 0:
            return np.inf
        # Takes less than a tick to cover remaining_distance with current_speed
        if remaining_distance - current_speed * config.driving.fps_inv < 0:
            break

        remaining_distance -= current_speed * config.driving.fps_inv
        min_time_needed += config.driving.fps_inv

        # Values from kinematic bicycle model
        normalized_speed = current_speed / 120.0
        speed_change_params = config.driving.compute_min_time_to_cover_distance_params
        speed_change = np.clip(
            speed_change_params[0]
            + normalized_speed * speed_change_params[1]
            + speed_change_params[2] * normalized_speed**2
            + speed_change_params[3] * normalized_speed**3,
            0.0,
            np.inf,
        )
        current_speed = np.clip(
            120 * (normalized_speed + speed_change),
            0,
            target_speed,
        )

    # Add remaining time at the current speed
    min_time_needed += remaining_distance / current_speed

    return min_time_needed


class SpeedPlanningMixin:
    """Target-speed computation, mixed into :class:`lead.expert.expert_agent.Expert`."""

    def _solve_vulnerable_vehicles_scenarios(
        self,
        target_speed: float,
    ) -> tuple[float, bool, bool, list | None]:
        """
        Check for vulnerable road users (pedestrians, cyclists) during crossing
        scenarios. Brake and set the target speed to 0 if a visible one is moving
        in front of the ego vehicle.

        Args:
            target_speed: The current target speed of the ego vehicle.

        Returns:
            A tuple containing (adjusted target_speed, is_overridden, should_brake, details or None).
        """
        ego_matrix = self.ego_matrix
        ego_location = ego_matrix[:3, 3]

        if self.current_active_scenario_type in [
            "VehicleTurningRoute",
            "VehicleTurningRoutePedestrian",
            "DynamicObjectCrossing",
            "ParkingCrossingPedestrian",
            "PedestrianCrossing",
        ]:
            for actor in self.walkers_inside_bev + self.bikers_inside_bev:
                num_visible_pixel = (
                    self.id2bb_map[actor.id]["visible_pixels"]
                    if self.config_expert.data_collection.datagen
                    else -1
                )
                actor_height = actor.bounding_box.extent.z
                threshold = (
                    50
                    * (
                        self.config_expert.sensor_rig.image_height
                        * self.config_expert.sensor_rig.image_width
                    )
                    / ((384**2) * 3)
                    * (3 / self.config_expert.sensor_rig.num_cameras)
                )
                if 0 < num_visible_pixel / (actor_height**2) < threshold:
                    continue
                actor_velocity = actor.get_velocity().length()
                if (
                    actor_velocity < 0.25
                    and not CarlaDataProvider.get_current_scenario_memory()[
                        "pedestrian_moved"
                    ][actor.id]
                ):
                    continue  # Actor did not move yet and is not moving, we come nearer to trigger the scenario
                elif (
                    actor_velocity >= 0.25
                ):  # Actor is moving, we mark it as moving and continue with the scenario
                    CarlaDataProvider.get_current_scenario_memory()["pedestrian_moved"][
                        actor.id
                    ] = True

                bb_location = np.array(actor.get_transform().get_matrix())[:3, 3]
                rel_vector = bb_location - ego_location
                distance = np.linalg.norm(rel_vector)

                local_coords = self.inv_ego_matrix @ np.append(bb_location, 1.0)
                x, y = local_coords[0], local_coords[1]
                angle = np.abs(np.degrees(np.arctan2(y, x)))

                if x < 0:
                    continue  # Behind ego vehicle

                LOG.info(
                    f"Found vulnerable vehicle in front: {angle}°, {distance}m, emergency braking.",
                )
                return 0.0, True, True, [0, actor.type_id, actor.id, distance]

        return target_speed, False, False, None

    def _solve_general_scenarios(
        self,
        initial_target_speed: float,
        speed_reduced_by_obj: list | None,
    ) -> tuple[bool, float, list | None]:
        """
        Compute the brake command and target speed for the ego vehicle based on various factors.

        Args:
            initial_target_speed: The initial target speed for the ego vehicle.
            speed_reduced_by_obj: A list containing [reduced speed, object type, object ID, distance]
                    for the object that caused the most speed reduction, or None if no speed reduction.

        Returns:
            A tuple containing the brake command, target speed, and the updated speed_reduced_by_obj list.
        """
        target_speed = initial_target_speed

        # Calculate the global bounding box of the ego vehicle
        center_ego_bb_global = self.ego_transform.transform(
            self.ego_vehicle.bounding_box.location,
        )
        ego_bb_global = carla.BoundingBox(
            center_ego_bb_global,
            self.ego_vehicle.bounding_box.extent,
        )
        ego_bb_global.rotation = self.ego_transform.rotation

        if self.config_expert.visualization.visualize_bounding_boxes:
            self.carla_world.debug.draw_box(
                box=ego_bb_global,
                rotation=ego_bb_global.rotation,
                thickness=0.1,
                color=carla.Color(
                    *self.config_expert.visualization.ego_vehicle_bb_color,
                ),
                life_time=self.config_expert.visualization.draw_life_time,
            )

        # Compute the number of future frames to consider for collision detection
        num_future_frames = int(
            self.config_expert.bicycle_model.bicycle_frame_rate
            * (
                self.config_expert.driving.forecast_length_lane_change
                if self.near_lane_change
                else self.config_expert.driving.default_forecast_length
            ),
        )

        # Get future bounding boxes of pedestrians
        nearby_pedestrians, nearby_pedestrian_ids = self.forecast_walkers(
            num_future_frames,
        )

        # Forecast the ego vehicle's bounding boxes for the future frames
        ego_forecast = self.forecast_ego_agent(
            num_future_frames,
            initial_target_speed,
        )

        # Predict bounding boxes of other actors (vehicles, bicycles, etc.)
        predicted_bounding_boxes = self.predict_other_actors_bounding_boxes(
            num_future_frames,
        )

        # Compute the target speed with respect to the leading vehicle
        target_speed_leading, speed_reduced_by_obj = (
            self.compute_target_speed_wrt_leading_vehicle(
                initial_target_speed,
                predicted_bounding_boxes,
                speed_reduced_by_obj,
            )
        )

        # Compute the target speeds with respect to all actors (vehicles, bicycles, pedestrians)
        (
            target_speed_bicycle,
            target_speed_pedestrian,
            target_speed_vehicle,
            speed_reduced_by_obj,
        ) = self.compute_target_speeds_wrt_all_actors(
            initial_target_speed,
            ego_forecast,
            predicted_bounding_boxes,
            speed_reduced_by_obj,
            nearby_pedestrians,
            nearby_pedestrian_ids,
        )

        # Compute the target speed with respect to the red light
        target_speed_red_light = self.ego_agent_affected_by_red_light(
            initial_target_speed,
        )

        # Update the object causing the most speed reduction
        if (
            speed_reduced_by_obj is None
            or speed_reduced_by_obj[0] > target_speed_red_light
        ):
            speed_reduced_by_obj = [
                target_speed_red_light,
                None
                if self.next_traffic_light is None
                else self.next_traffic_light.type_id,
                None if self.next_traffic_light is None else self.next_traffic_light.id,
                self.distance_to_next_traffic_light,
            ]

        # Compute the target speed with respect to the stop sign
        target_speed_stop_sign = self.ego_agent_affected_by_stop_sign(
            initial_target_speed,
        )
        # Update the object causing the most speed reduction
        if (
            speed_reduced_by_obj is None
            or speed_reduced_by_obj[0] > target_speed_stop_sign
        ):
            speed_reduced_by_obj = [
                target_speed_stop_sign,
                None if self.next_stop_sign is None else self.next_stop_sign.type_id,
                None if self.next_stop_sign is None else self.next_stop_sign.id,
                self.distance_to_next_stop_sign,
            ]

        # Compute the minimum target speed considering all factors
        target_speed = min(
            target_speed_leading,
            target_speed_bicycle,
            target_speed_vehicle,
            target_speed_pedestrian,
            target_speed_red_light,
            target_speed_stop_sign,
        )

        # Set the hazard flags based on the target speed and its cause
        if (
            target_speed == target_speed_pedestrian
            and target_speed_pedestrian != initial_target_speed
        ):
            self.walker_hazard = True
            self.walker_close = True
        elif (
            target_speed == target_speed_red_light
            and target_speed_red_light != initial_target_speed
        ):
            self.traffic_light_hazard = True
        elif (
            target_speed == target_speed_stop_sign
            and target_speed_stop_sign != initial_target_speed
        ):
            self.stop_sign_hazard = True
            self.stop_sign_close = True

        # target_speed can be a numpy scalar, so coerce the comparison's np.bool_
        # back to a plain bool.
        brake = bool(target_speed == 0)
        return brake, target_speed, speed_reduced_by_obj

    def predict_other_actors_bounding_boxes(
        self,
        num_future_frames: int,
    ) -> dict[int, forecast_kernels.ActorForecast]:
        """
        Predict the future bounding boxes of actors for a given number of frames.

        Args:
            num_future_frames: The number of future frames to predict.

        Returns:
            A dictionary mapping actor IDs to their predicted bounding boxes as arrays.
        """
        predicted_bounding_boxes: dict[int, forecast_kernels.ActorForecast] = {}

        # --- Determine how wider we should make dangerous adversarial actors' bounding boxes
        if (
            self.current_active_scenario_type
            in [
                "Accident",
                "ConstructionObstacle",
                "ParkedObstacle",
                "HazardAtSideLane",
            ]
        ):  # Adversarials drive in same direction, we make their bounding boxes not too wide
            adversarial_bb_extra_width = (
                max(
                    (self.ego_lane_width - self.ego_vehicle.bounding_box.extent.y) / 2,
                    0,
                )
                * 0.6
            )
        elif (
            self.current_active_scenario_type
            in [
                "SignalizedJunctionLeftTurn",
                "NonSignalizedJunctionLeftTurn",
                "SignalizedJunctionLeftTurnEnterFlow",
                "NonSignalizedJunctionLeftTurnEnterFlow",
                "InterurbanActorFlow",
            ]
        ):  # Adversarials drive in opposite direction, we make their bounding boxes wider
            adversarial_bb_extra_width = (
                max(
                    (self.ego_lane_width - self.ego_vehicle.bounding_box.extent.y) / 2,
                    0,
                )
                * 0.8
            )
        elif (
            self.current_active_scenario_type
            in [
                "SignalizedJunctionRightTurn",
                "NonSignalizedJunctionRightTurn",
            ]
        ):  # Adversarials drive in same direction, we make their bounding boxes even wider
            adversarial_bb_extra_width = (
                max(
                    (self.ego_lane_width - self.ego_vehicle.bounding_box.extent.y) / 2,
                    0,
                )
                * 1.75
            )  # Right turn larger BB to avoid collisions bc of blinding fleck

        minimal_adversarial_bb_width = None
        if self.target_lane_width is not None:
            minimal_adversarial_bb_width = (
                adversarial_bb_extra_width + self.target_lane_width
            )

        # --- Determine which adversarial actors to consider, ignore or make dangerous
        (
            dangerous_adversarial_actors_ids,
            safe_adversarial_actors_ids,
            ignored_adversarial_actors_ids,
        ) = self.adversarial_actors_ids

        # Filter out nearby actors within the detection radius, excluding the ego vehicle
        nearby_actors = [
            actor
            for actor in self.vehicles_inside_bev
            if actor.id != self.ego_vehicle.id
        ]

        # If there are nearby actors, calculate their future bounding boxes
        if nearby_actors:
            # Get the previous control inputs (steering, throttle, brake) for the nearby actors
            previous_controls = [actor.get_control() for actor in nearby_actors]
            previous_actions = np.array(
                [
                    [control.steer, control.throttle, control.brake]
                    for control in previous_controls
                ],
            )

            # Get the current velocities, locations, and headings of the nearby actors
            velocities = []
            for actor in nearby_actors:
                actor_original_velocity = actor.get_velocity().length()
                velocities.append(actor_original_velocity)

            velocities = np.array(velocities)
            locations = np.array(
                [
                    [
                        actor.get_location().x,
                        actor.get_location().y,
                        actor.get_location().z,
                    ]
                    for actor in nearby_actors
                ],
            )
            headings = np.deg2rad(
                np.array(
                    [actor.get_transform().rotation.yaw for actor in nearby_actors],
                ),
            )

            # Forecast the future locations, headings, and velocities for the nearby actors
            future_locations, future_headings, future_velocities = (
                self.vehicle_model.forecast_other_vehicles(
                    locations,
                    headings,
                    velocities,
                    previous_actions,
                    num_future_frames,
                )
            )
            # Convert future headings to degrees
            future_headings = np.rad2deg(future_headings)

            # Adjust the bounding box size based on velocity and lane change maneuver to adjust for
            # uncertainty during forecasting
            frame_ratios = np.arange(num_future_frames) / float(num_future_frames)
            s = (
                self.config_expert.driving.high_speed_min_extent_x_other_vehicle_lane_change
                if self.near_lane_change
                else self.config_expert.driving.high_speed_min_extent_x_other_vehicle
            )
            high_speed_length_factors = np.maximum(
                s,
                self.config_expert.driving.high_speed_min_extent_x_other_vehicle
                * frame_ratios,
            )
            high_speed_width_factors = np.maximum(
                self.config_expert.driving.high_speed_min_extent_y_other_vehicle,
                self.config_expert.driving.high_speed_extent_y_factor_other_vehicle
                * frame_ratios,
            )
            slow_masks = (
                future_velocities
                < self.config_expert.driving.extent_other_vehicles_bbs_speed_threshold
            )

            # Calculate the predicted bounding boxes for each nearby actor and future frame
            for actor_idx, actor in enumerate(nearby_actors):
                extent = actor.bounding_box.extent
                length_factors = np.where(
                    slow_masks[:, actor_idx],
                    self.config_expert.driving.slow_speed_extent_factor_ego,
                    high_speed_length_factors,
                )
                width_factors = np.where(
                    slow_masks[:, actor_idx],
                    self.config_expert.driving.slow_speed_extent_factor_ego,
                    high_speed_width_factors,
                )

                if self.current_active_scenario_type in ["CrossingBicycleFlow"]:
                    if actor.type_id in constants.BIKER_MESHES:
                        length_factors[:] = 4.0
                        width_factors[:] = 10.0
                elif self.current_active_scenario_type in [
                    "NonSignalizedJunctionRightTurn",
                ]:
                    if actor.id in dangerous_adversarial_actors_ids:
                        length_factors[:] = 1.5
                        width_factors[:] = minimal_adversarial_bb_width / (extent.y * 2)
                    elif actor.id in safe_adversarial_actors_ids:
                        length_factors[:] = 1.0
                        width_factors[:] = 1.0
                    elif actor.id in ignored_adversarial_actors_ids:
                        length_factors[:] = 0.0
                        width_factors[:] = 0.0
                elif self.current_active_scenario_type in [
                    "SignalizedJunctionRightTurn",
                ]:
                    if actor.id in dangerous_adversarial_actors_ids:
                        length_factors[:] = 1.6
                        width_factors[:] = minimal_adversarial_bb_width / (extent.y * 2)
                    elif actor.id in safe_adversarial_actors_ids:
                        length_factors[:] = 1.0
                        width_factors[:] = 1.0
                    elif actor.id in ignored_adversarial_actors_ids:
                        length_factors[:] = 0.0
                elif self.current_active_scenario_type in [
                    "SignalizedJunctionLeftTurnEnterFlow",
                    "NonSignalizedJunctionLeftTurnEnterFlow",
                ]:
                    if actor.id in dangerous_adversarial_actors_ids:
                        length_factors[:] = 1.25
                        width_factors[:] = minimal_adversarial_bb_width / (extent.y * 2)
                    elif actor.id in safe_adversarial_actors_ids:
                        length_factors[:] = 0.75
                        width_factors[:] = 1.0
                    elif actor.id in ignored_adversarial_actors_ids:
                        length_factors[:] = 0.0
                        width_factors[:] = 0.0

                extents_x = extent.x * length_factors
                extents_y = extent.y * width_factors
                extents = np.stack(
                    [extents_x, extents_y, np.full(num_future_frames, extent.z)],
                    axis=1,
                )

                # Store the predicted bounding boxes for this actor in the dictionary
                predicted_bounding_boxes[actor.id] = forecast_kernels.ActorForecast(
                    actor=actor,
                    centers=future_locations[:, actor_idx, :],
                    yaws_deg=future_headings[:, actor_idx],
                    extents=extents,
                )

        self.visualize_forecasted_bounding_boxes(predicted_bounding_boxes)

        return predicted_bounding_boxes

    def compute_target_speed_wrt_leading_vehicle(
        self,
        initial_target_speed: float,
        predicted_bounding_boxes: dict,
        speed_reduced_by_obj: list | None,
    ) -> tuple[float, list | None]:
        """
        Compute the target speed for the ego vehicle considering the leading vehicle.

        Args:
            initial_target_speed: The initial target speed for the ego vehicle.
            predicted_bounding_boxes: A dictionary mapping actor IDs to lists of predicted bounding boxes.
            speed_reduced_by_obj: A list containing [reduced speed, object type, object ID, distance]
                for the object that caused the most speed reduction, or None if no speed reduction.

        Returns:
            A tuple containing the target speed considering the leading vehicle and the updated speed_reduced_by_obj.
        """
        target_speed_wrt_leading_vehicle = initial_target_speed
        # _, _, ignored_adversarial_actors_ids = self.adversarial_actors_ids

        for vehicle_id, _ in predicted_bounding_boxes.items():
            # if vehicle_id in ignored_adversarial_actors_ids:
            #     continue
            if vehicle_id in self.leading_vehicle_ids and not self.near_lane_change:
                # Vehicle is in front of the ego vehicle
                ego_speed = self.ego_vehicle.get_velocity().length()
                vehicle = self.carla_world.get_actor(vehicle_id)
                other_speed = vehicle.get_velocity().length()
                distance_to_vehicle = self.ego_location.distance(vehicle.get_location())

                # Compute the target speed using the IDM
                target_speed_wrt_leading_vehicle = min(
                    target_speed_wrt_leading_vehicle,
                    compute_target_speed_idm(
                        config=self.config_expert,
                        desired_speed=initial_target_speed,
                        leading_actor_length=vehicle.bounding_box.extent.x * 2,
                        ego_speed=ego_speed,
                        leading_actor_speed=other_speed,
                        distance_to_leading_actor=distance_to_vehicle,
                        s0=self.config_expert.driving.idm_leading_vehicle_minimum_distance,
                        T=self.config_expert.driving.idm_leading_vehicle_time_headway,
                    ),
                )

                # Update the object causing the most speed reduction
                if (
                    speed_reduced_by_obj is None
                    or speed_reduced_by_obj[0] > target_speed_wrt_leading_vehicle
                ):
                    speed_reduced_by_obj = [
                        target_speed_wrt_leading_vehicle,
                        vehicle.type_id,
                        vehicle.id,
                        distance_to_vehicle,
                    ]

            if self.config_expert.visualization.visualize_bounding_boxes:
                for vehicle_id in predicted_bounding_boxes.keys():
                    # check if vehicle is in front of the ego vehicle
                    if (
                        vehicle_id in self.leading_vehicle_ids
                        and not self.near_lane_change
                    ):
                        vehicle = self.carla_world.get_actor(vehicle_id)
                        extent = vehicle.bounding_box.extent
                        bb = carla.BoundingBox(vehicle.get_location(), extent)
                        bb.rotation = carla.Rotation(
                            pitch=0,
                            yaw=vehicle.get_transform().rotation.yaw,
                            roll=0,
                        )
                        self.carla_world.debug.draw_box(
                            box=bb,
                            rotation=bb.rotation,
                            thickness=0.5,
                            color=carla.Color(
                                *self.config_expert.visualization.leading_vehicle_color,
                            ),
                            life_time=self.config_expert.visualization.draw_life_time,
                        )
                    elif vehicle_id in self.trailing_vehicle_ids:
                        vehicle = self.carla_world.get_actor(vehicle_id)
                        extent = vehicle.bounding_box.extent
                        bb = carla.BoundingBox(vehicle.get_location(), extent)
                        bb.rotation = carla.Rotation(
                            pitch=0,
                            yaw=vehicle.get_transform().rotation.yaw,
                            roll=0,
                        )
                        self.carla_world.debug.draw_box(
                            box=bb,
                            rotation=bb.rotation,
                            thickness=0.5,
                            color=carla.Color(
                                *self.config_expert.visualization.trailing_vehicle_color,
                            ),
                            life_time=self.config_expert.visualization.draw_life_time,
                        )

        return target_speed_wrt_leading_vehicle, speed_reduced_by_obj

    def compute_target_speeds_wrt_all_actors(
        self,
        initial_target_speed: float,
        ego_forecast: forecast_kernels.EgoForecast,
        predicted_bounding_boxes: dict[int, forecast_kernels.ActorForecast],
        speed_reduced_by_obj: list | None,
        nearby_walkers: list[forecast_kernels.WalkerForecast],
        nearby_walkers_ids: list[int],
    ) -> tuple[float, float, float, list | None]:
        """
        Compute the target speeds for the ego vehicle considering all actors (vehicles, bicycles,
        and pedestrians) by checking for intersecting bounding boxes.

        All frame-wise box intersections are evaluated in one numba kernel; the
        loop below only reacts to the hits, in the original iteration order.

        Args:
            initial_target_speed: The initial target speed for the ego vehicle.
            ego_forecast: The future bounding boxes of the ego vehicle as arrays.
            predicted_bounding_boxes: A dictionary mapping actor IDs to their predicted bounding boxes.
            speed_reduced_by_obj: A list containing [reduced speed, object type,
                object ID, distance] for the object that caused the most speed reduction, or None if
                no speed reduction.
            nearby_walkers: A list with the predicted bounding boxes of each nearby pedestrian.
            nearby_walkers_ids: A list of IDs for nearby pedestrians.

        Returns:
            A tuple containing the target speeds for bicycles, pedestrians, vehicles, and the updated
                speed_reduced_by_obj list.
        """
        target_speed_bicycle = initial_target_speed
        target_speed_pedestrian = initial_target_speed
        target_speed_vehicle = initial_target_speed
        ego_vehicle_location = self.ego_vehicle.get_location()
        hazard_color = carla.Color(
            *self.config_expert.visualization.ego_vehicle_forecasted_bbs_hazard_color,
        )
        normal_color = carla.Color(
            *self.config_expert.visualization.ego_vehicle_forecasted_bbs_normal_color,
        )
        color = normal_color

        num_future_frames = ego_forecast.centers.shape[0]
        ego_rotations = np.zeros((num_future_frames, 3))
        ego_rotations[:, 1] = ego_forecast.yaws_deg
        ego_extents = np.tile(ego_forecast.extent, (num_future_frames, 1))

        # Skip leading and rear vehicles if not near a lane change
        considered_vehicles = [
            (vehicle_id, forecast)
            for vehicle_id, forecast in predicted_bounding_boxes.items()
            if not (
                vehicle_id in self.leading_vehicle_ids and not self.near_lane_change
            )
            and not (
                vehicle_id in self.trailing_vehicle_ids and not self.near_lane_change
            )
        ]
        vehicle_hits = np.zeros((len(considered_vehicles), num_future_frames), np.bool_)
        if considered_vehicles and num_future_frames > 0:
            rotations = np.zeros((len(considered_vehicles), num_future_frames, 3))
            rotations[:, :, 1] = np.stack(
                [forecast.yaws_deg for _, forecast in considered_vehicles],
            )
            vehicle_hits = forecast_kernels.batch_obb_intersection(
                ego_forecast.centers,
                ego_rotations,
                ego_extents,
                np.stack([forecast.centers for _, forecast in considered_vehicles]),
                rotations,
                np.stack([forecast.extents for _, forecast in considered_vehicles]),
            )

        walker_hits = np.zeros((len(nearby_walkers), num_future_frames), np.bool_)
        if nearby_walkers and num_future_frames > 0:
            walker_hits = forecast_kernels.batch_obb_intersection(
                ego_forecast.centers,
                ego_rotations,
                ego_extents,
                np.stack([walker.centers for walker in nearby_walkers]),
                np.stack(
                    [
                        np.tile(walker.rotation_deg, (num_future_frames, 1))
                        for walker in nearby_walkers
                    ],
                ),
                np.stack(
                    [
                        np.tile(walker.extent, (num_future_frames, 1))
                        for walker in nearby_walkers
                    ],
                ),
            )

        # The IDM inputs of an actor do not change between future frames, so
        # each actor's target speed is integrated at most once
        bicycle_idm_results: dict[int, tuple[float, float]] = {}
        walker_idm_results: dict[int, tuple[float, float]] = {}

        # Iterate over the ego vehicle's bounding boxes and predicted bounding boxes of other actors
        for i in range(num_future_frames):
            for vehicle_index, (vehicle_id, forecast) in enumerate(considered_vehicles):
                # Check if the ego bounding box intersects with the predicted bounding box of the actor
                if not vehicle_hits[vehicle_index, i]:
                    continue
                blocking_actor = forecast.actor
                # Handle the case when the blocking actor is a bicycle
                if (
                    "base_type" in blocking_actor.attributes
                    and blocking_actor.attributes["base_type"] == "bicycle"
                ):
                    if vehicle_id not in bicycle_idm_results:
                        other_speed = blocking_actor.get_velocity().length()
                        distance_to_actor = ego_vehicle_location.distance(
                            blocking_actor.get_location(),
                        )

                        # Compute the target speed for bicycles using the IDM
                        bicycle_idm_results[vehicle_id] = (
                            compute_target_speed_idm(
                                config=self.config_expert,
                                desired_speed=initial_target_speed,
                                leading_actor_length=blocking_actor.bounding_box.extent.x
                                * 2,
                                ego_speed=self.ego_speed,
                                leading_actor_speed=other_speed,
                                distance_to_leading_actor=distance_to_actor,
                                s0=self.config_expert.driving.idm_bicycle_minimum_distance,
                                T=self.config_expert.driving.idm_bicycle_desired_time_headway,
                            ),
                            distance_to_actor,
                        )
                    idm_speed, distance_to_actor = bicycle_idm_results[vehicle_id]
                    target_speed_bicycle = min(target_speed_bicycle, idm_speed)

                    # Update the object causing the most speed reduction
                    if (
                        speed_reduced_by_obj is None
                        or speed_reduced_by_obj[0] > target_speed_bicycle
                    ):
                        speed_reduced_by_obj = [
                            target_speed_bicycle,
                            blocking_actor.type_id,
                            blocking_actor.id,
                            distance_to_actor,
                        ]

                # Handle the case when the blocking actor is not a bicycle
                else:
                    self.vehicle_hazard = True  # Set the vehicle hazard flag
                    self.vehicle_affecting_id = (
                        vehicle_id  # Store the ID of the vehicle causing the hazard
                    )
                    color = hazard_color  # Change the following colors from green to red (no hazard to hazard)
                    target_speed_vehicle = (
                        0.0  # Set the target speed for vehicles to zero
                    )
                    distance_to_actor = blocking_actor.get_location().distance(
                        ego_vehicle_location,
                    )

                    # Update the object causing the most speed reduction
                    if (
                        speed_reduced_by_obj is None
                        or speed_reduced_by_obj[0] > target_speed_vehicle
                    ):
                        speed_reduced_by_obj = [
                            target_speed_vehicle,
                            blocking_actor.type_id,
                            blocking_actor.id,
                            distance_to_actor,
                        ]

            # Iterate over nearby pedestrians and check for intersections with the ego bounding box
            for walker_index, (walker_forecast, pedestrian_id) in enumerate(
                zip(nearby_walkers, nearby_walkers_ids, strict=False),
            ):
                if not walker_hits[walker_index, i]:
                    continue
                color = hazard_color
                blocking_actor = walker_forecast.actor
                if pedestrian_id not in walker_idm_results:
                    distance_to_actor = ego_vehicle_location.distance(
                        blocking_actor.get_location(),
                    )

                    # Compute the target speed for pedestrians using the IDM
                    walker_idm_results[pedestrian_id] = (
                        compute_target_speed_idm(
                            config=self.config_expert,
                            desired_speed=initial_target_speed,
                            leading_actor_length=0.5
                            + self.ego_vehicle.bounding_box.extent.x,
                            ego_speed=self.ego_speed,
                            leading_actor_speed=0.0,
                            distance_to_leading_actor=distance_to_actor,
                            s0=self.config_expert.driving.idm_pedestrian_minimum_distance,
                            T=self.config_expert.driving.idm_pedestrian_desired_time_headway,
                        ),
                        distance_to_actor,
                    )
                idm_speed, distance_to_actor = walker_idm_results[pedestrian_id]
                target_speed_pedestrian = min(target_speed_pedestrian, idm_speed)

                # Update the object causing the most speed reduction
                if (
                    speed_reduced_by_obj is None
                    or speed_reduced_by_obj[0] > target_speed_pedestrian
                ):
                    speed_reduced_by_obj = [
                        target_speed_pedestrian,
                        blocking_actor.type_id,
                        blocking_actor.id,
                        distance_to_actor,
                    ]
            if self.config_expert.visualization.visualize_bounding_boxes:
                ego_bounding_box = carla.BoundingBox(
                    carla.Location(
                        x=ego_forecast.centers[i, 0],
                        y=ego_forecast.centers[i, 1],
                        z=ego_forecast.centers[i, 2],
                    ),
                    carla.Vector3D(
                        x=ego_forecast.extent[0],
                        y=ego_forecast.extent[1],
                        z=ego_forecast.extent[2],
                    ),
                )
                ego_bounding_box.rotation = carla.Rotation(
                    pitch=0,
                    yaw=ego_forecast.yaws_deg[i],
                    roll=0,
                )
                self.carla_world.debug.draw_box(
                    box=ego_bounding_box,
                    rotation=ego_bounding_box.rotation,
                    thickness=0.1,
                    color=color,
                    life_time=self.config_expert.visualization.draw_life_time,
                )
        return (
            float(target_speed_bicycle),
            float(target_speed_pedestrian),
            float(target_speed_vehicle),
            speed_reduced_by_obj,
        )

    def forecast_ego_agent(
        self,
        num_future_frames: int,
        initial_target_speed: float,
    ) -> forecast_kernels.EgoForecast:
        """
        Forecast the future states of the ego agent with the kinematic bicycle model,
        assuming no hazard, for the subsequent collision checks.

        The PID / bicycle-model / route-planner recurrence runs in a numba
        kernel on local copies of the controller window and route index, so no
        state needs to be saved and restored around it.

        Args:
            num_future_frames: The number of future frames to forecast.
            initial_target_speed: The initial target speed for the ego vehicle.

        Returns:
            The future bounding boxes of the ego vehicle as arrays.
        """
        ego_transform = self.ego_transform
        locations, headings = forecast_kernels.run_ego_forecast(
            self.config_expert,
            self.privileged_route_planner.route_points,
            int(self.privileged_route_planner.route_index),
            ego_transform.location.x,
            ego_transform.location.y,
            ego_transform.location.z,
            float(np.deg2rad(ego_transform.rotation.yaw)),
            float(self.ego_speed),
            float(initial_target_speed),
            self._turn_controller.error_history,
            num_future_frames,
        )

        # Shrink the ego bounding box when slow, so a collision does not leave the
        # boxes permanently intersecting; when driving, enlarge it for safety.
        extent = self.ego_vehicle.bounding_box.extent
        is_slow = (
            self.ego_speed < self.config_expert.driving.extent_ego_bbs_speed_threshold
        )
        extent_x = extent.x * (
            self.config_expert.driving.slow_speed_extent_factor_ego
            if is_slow
            else self.config_expert.driving.high_speed_extent_factor_ego_x
        )
        extent_y = extent.y * (
            self.config_expert.driving.slow_speed_extent_factor_ego
            if is_slow
            else self.config_expert.driving.high_speed_extent_factor_ego_y
        )

        return forecast_kernels.EgoForecast(
            centers=locations,
            yaws_deg=np.rad2deg(headings),
            extent=np.array([extent_x, extent_y, extent.z]),
        )

    def forecast_walkers(
        self,
        number_of_future_frames: int,
    ) -> tuple[list[forecast_kernels.WalkerForecast], list[int]]:
        """
        Forecast the future locations of pedestrians in the vicinity of the ego vehicle assuming they
        keep their velocity and direction

        Args:
            number_of_future_frames: The number of future frames to forecast.

        Returns:
            A tuple containing two lists:
                1. A list with the future bounding boxes of each pedestrian as arrays.
                2. A list of IDs for the pedestrians whose locations were forecasted.
        """
        nearby_pedestrians_bbs: list[forecast_kernels.WalkerForecast] = []
        nearby_pedestrian_ids: list[int] = []

        # Filter pedestrians within the detection radius
        pedestrians = self.walkers_inside_bev

        # If no pedestrians are found, return empty lists
        if not pedestrians:
            return nearby_pedestrians_bbs, nearby_pedestrian_ids

        # Extract pedestrian locations, speeds, and directions
        pedestrian_locations = np.array(
            [
                [ped.get_location().x, ped.get_location().y, ped.get_location().z]
                for ped in pedestrians
            ],
        )
        pedestrian_speeds = np.array(
            [ped.get_velocity().length() for ped in pedestrians],
        )
        pedestrian_speeds = np.maximum(
            pedestrian_speeds,
            self.config_expert.driving.min_walker_speed,
        )
        pedestrian_directions = np.array(
            [
                [
                    ped.get_control().direction.x,
                    ped.get_control().direction.y,
                    ped.get_control().direction.z,
                ]
                for ped in pedestrians
            ],
        )

        # Calculate future pedestrian locations based on their current locations, speeds, and directions
        future_pedestrian_locations = (
            pedestrian_locations[:, None, :]
            + np.arange(1, number_of_future_frames + 1)[None, :, None]
            * pedestrian_directions[:, None, :]
            * pedestrian_speeds[:, None, None]
            / self.config_expert.bicycle_model.bicycle_frame_rate
        )

        # Iterate over pedestrians and calculate their future bounding boxes
        for i, ped in enumerate(pedestrians):
            bb, transform = ped.bounding_box, ped.get_transform()
            rotation_deg = np.array(
                [
                    bb.rotation.pitch + transform.rotation.pitch,
                    bb.rotation.yaw + transform.rotation.yaw,
                    bb.rotation.roll + transform.rotation.roll,
                ],
            )
            extent = np.array(
                [
                    # Ensure a minimum width and length
                    max(
                        self.config_expert.driving.pedestrian_minimum_extent,
                        bb.extent.x,
                    ),
                    max(
                        self.config_expert.driving.pedestrian_minimum_extent,
                        bb.extent.y,
                    ),
                    bb.extent.z,
                ],
            )

            nearby_pedestrian_ids.append(ped.id)
            nearby_pedestrians_bbs.append(
                forecast_kernels.WalkerForecast(
                    actor=ped,
                    centers=future_pedestrian_locations[i],
                    rotation_deg=rotation_deg,
                    extent=extent,
                ),
            )

        self.visualize_pedestrian_bounding_boxes(nearby_pedestrians_bbs)

        return nearby_pedestrians_bbs, nearby_pedestrian_ids

    def ego_agent_affected_by_red_light(self, target_speed: float) -> float:
        """
        Handles the behavior of the ego vehicle when approaching a traffic light.

        Args:
            target_speed: The target speed for the ego vehicle.

        Returns:
            The adjusted target speed for the ego vehicle.
        """

        self.close_traffic_lights.clear()

        for index in self.close_traffic_light_indices(
            self.config_expert.driving.light_radius,
        ):
            traffic_light, _, traffic_light_waypoints = self.list_traffic_lights[index]

            affects_ego = (
                self.next_traffic_light is not None
                and traffic_light.id == self.next_traffic_light.id
            )
            traffic_light_state = traffic_light.state
            traffic_light_id = traffic_light.id

            for wp, bounding_box in self.traffic_light_stop_boxes(
                traffic_light,
                traffic_light_waypoints,
            ):
                self.close_traffic_lights.append(
                    [
                        traffic_light,
                        bounding_box,
                        traffic_light_state,
                        traffic_light_id,
                        affects_ego,
                    ],
                )

                self.visualize_traffic_lights(traffic_light, wp, bounding_box)

        # During data generation, use the more consistent bounding-box distance to the
        # traffic light; overhead and European lights require stopping farther back
        # than normal ones.
        distance_to_traffic_light_stop_point = self.distance_to_next_traffic_light
        if (
            self.config_expert.data_collection.datagen
            and self.next_traffic_light is not None
            and self.next_traffic_light.id in self.id2bb_map
        ):
            next_traffic_light_bb = self.id2bb_map[
                self.next_traffic_light.id
            ]  # Red light bounding box of the traffic light projected to ego road, doesn't have to be on same lane as ego
            if next_traffic_light_bb is not None:
                distance_ego_to_traffic_light_bb = next_traffic_light_bb["position"][
                    0
                ]  # Longitudinal distance

                if next_traffic_light_bb[
                    "is_over_head_traffic_light"
                ]:  # Overhead traffic light, must stand much further
                    distance_to_traffic_light_stop_point = max(
                        distance_ego_to_traffic_light_bb
                        - self.config_expert.driving.idm_overhead_red_light_minimum_distance,
                        0,
                    )
                    self.over_head_traffic_light = True
                elif next_traffic_light_bb[
                    "is_europe_traffic_light"
                ]:  # Overhead traffic light, must stand much further
                    distance_to_traffic_light_stop_point = max(
                        distance_ego_to_traffic_light_bb
                        - self.config_expert.driving.idm_europe_red_light_minimum_distance,
                        0,
                    )
                    self.europe_traffic_light = True
                else:  # Traffic light is on other side of intersection, we can stop normally
                    distance_to_traffic_light_stop_point = (
                        distance_ego_to_traffic_light_bb
                    )
            else:
                LOG.info(
                    "Warning: Could not find traffic light bounding box, using default distance to stop point.",
                )

        if self.next_traffic_light is not None:
            CarlaDataProvider._global_memory["next_traffic_light"] = (
                self.next_traffic_light
            )
        # If green light, just skip
        if (
            self.next_traffic_light is None
            or self.next_traffic_light.state == carla.TrafficLightState.Green
        ):
            # No traffic light or green light, continue with the current target speed
            return target_speed

        # Compute the target speed using the IDM
        new_target_speed = compute_target_speed_idm(
            config=self.config_expert,
            desired_speed=target_speed,
            leading_actor_length=0.0,
            ego_speed=self.ego_speed,
            leading_actor_speed=0.0,
            distance_to_leading_actor=distance_to_traffic_light_stop_point,
            s0=self.config_expert.driving.idm_red_light_minimum_distance,
            T=self.config_expert.driving.idm_red_light_desired_time_headway,
        )

        if (
            abs(new_target_speed - target_speed) > 1e-2
            and distance_to_traffic_light_stop_point < 30.0
        ):
            LOG.info(
                f"""[{self.step}] Ego target speed adjusted for red light in {distance_to_traffic_light_stop_point:.2f}m
    from {target_speed:.2f} to {new_target_speed:.2f}""",
            )

        return new_target_speed

    def ego_agent_affected_by_stop_sign(self, target_speed: float) -> float:
        """
        Handles the behavior of the ego vehicle when approaching a stop sign.

        Args:
            target_speed: The target speed for the ego vehicle.

        Returns:
            The adjusted target speed for the ego vehicle.
        """
        self.close_stop_signs.clear()
        stop_signs = self.nearby_stop_signs(self.config_expert.driving.light_radius)

        for stop_sign in stop_signs:
            center_bb_stop_sign = stop_sign.get_transform().transform(
                stop_sign.trigger_volume.location,
            )
            stop_sign_extent = carla.Vector3D(1.5, 1.5, 0.5)
            bounding_box_stop_sign = carla.BoundingBox(
                center_bb_stop_sign,
                stop_sign_extent,
            )
            rotation_stop_sign = stop_sign.get_transform().rotation
            bounding_box_stop_sign.rotation = carla.Rotation(
                pitch=rotation_stop_sign.pitch,
                yaw=rotation_stop_sign.yaw,
                roll=rotation_stop_sign.roll,
            )

            affects_ego = (
                self.next_stop_sign is not None
                and self.next_stop_sign.id == stop_sign.id
                and not self.cleared_stop_sign
            )
            self.close_stop_signs.append(
                [bounding_box_stop_sign, stop_sign.id, affects_ego],
            )

            self.visualize_stop_signs(bounding_box_stop_sign, affects_ego)

        if self.next_stop_sign is None:
            # No stop sign, continue with the current target speed
            return target_speed

        # Calculate the accurate distance to the stop sign
        distance_to_stop_sign = (
            self.next_stop_sign.get_transform()
            .transform(self.next_stop_sign.trigger_volume.location)
            .distance(self.ego_location)
        )

        # Reset the stop sign flag if we are farther than 10m away
        if (
            distance_to_stop_sign
            > self.config_expert.driving.unclearing_distance_to_stop_sign
        ):
            self.cleared_stop_sign = False
            self.waiting_ticks_at_stop_sign = 0
        else:
            # Set the stop sign flag if we are closer than 3m and speed is low enough
            if (
                self.ego_speed < 0.1
                and distance_to_stop_sign
                < self.config_expert.driving.clearing_distance_to_stop_sign
            ):
                self.waiting_ticks_at_stop_sign += 1
                if self.waiting_ticks_at_stop_sign > 25:
                    self.cleared_stop_sign = True
            else:
                self.waiting_ticks_at_stop_sign = 0

        # Set the distance to stop sign as infinity if the stop sign has been cleared
        distance_to_stop_sign = (
            np.inf if self.cleared_stop_sign else distance_to_stop_sign
        )

        # If being a sensory agent, we use the more consistent distance to stop sign using bounding box
        if (
            self.config_expert.data_collection.datagen
            and distance_to_stop_sign < np.inf
            and self.next_stop_sign.id in self.id2bb_map
        ):
            next_traffic_light_bb = self.id2bb_map[self.next_stop_sign.id]
            distance_to_stop_sign = next_traffic_light_bb["distance"]

        # Compute the target speed using the IDM
        target_speed = compute_target_speed_idm(
            config=self.config_expert,
            desired_speed=target_speed,
            leading_actor_length=0,
            ego_speed=self.ego_speed,
            leading_actor_speed=0.0,
            distance_to_leading_actor=distance_to_stop_sign,
            s0=self.config_expert.driving.idm_stop_sign_minimum_distance,
            T=self.config_expert.driving.idm_stop_sign_desired_time_headway,
        )

        # Return whether the ego vehicle is affected by the stop sign and the adjusted target speed
        return target_speed

    def _vehicle_obstacle_detected(
        self,
        max_distance: float | None = None,
        up_angle_th: float = 90,
        low_angle_th: float = 0,
        lane_offset: float = 0,
    ) -> tuple[bool, int | None, float | None]:
        """
        Check if there is a vehicle in front of the agent blocking its path.

        Args:
            max_distance: Maximum freespace to check for obstacles. If None, the base threshold value is used.
            up_angle_th: Upper angle threshold in degrees.
            low_angle_th: Lower angle threshold in degrees.
            lane_offset: Lane offset for checking adjacent lanes.

        Returns:
            A tuple containing (obstacle_detected, vehicle_id, distance_to_obstacle).
        """
        self._use_bbs_detection = False
        self._offset = 0

        def get_route_polygon():
            route_bb = []
            extent_y = self.ego_vehicle.bounding_box.extent.y
            r_ext = extent_y + self._offset
            l_ext = -extent_y + self._offset
            r_vec = ego_transform.get_right_vector()
            p1 = ego_location + carla.Location(r_ext * r_vec.x, r_ext * r_vec.y)
            p2 = ego_location + carla.Location(l_ext * r_vec.x, l_ext * r_vec.y)
            route_bb.extend([[p1.x, p1.y, p1.z], [p2.x, p2.y, p2.z]])

            for wp, _ in self._local_planner.get_plan():
                if ego_location.distance(wp.transform.location) > max_distance:
                    break

                r_vec = wp.transform.get_right_vector()
                p1 = wp.transform.location + carla.Location(
                    r_ext * r_vec.x,
                    r_ext * r_vec.y,
                )
                p2 = wp.transform.location + carla.Location(
                    l_ext * r_vec.x,
                    l_ext * r_vec.y,
                )
                route_bb.extend([[p1.x, p1.y, p1.z], [p2.x, p2.y, p2.z]])

            # Two points don't create a polygon, nothing to check
            if len(route_bb) < 3:
                return None

            return Polygon(route_bb)

        ego_transform = self.ego_vehicle.get_transform()
        ego_location = ego_transform.location
        ego_wpt = self.carla_world_map.get_waypoint(
            ego_location,
            lane_type=carla.libcarla.LaneType.Any,
        )

        # Get the right offset
        if ego_wpt.lane_id < 0 and lane_offset != 0:
            lane_offset *= -1

        # Get the transform of the front of the ego
        ego_front_transform = ego_transform
        ego_front_transform.location += carla.Location(
            self.ego_vehicle.bounding_box.extent.x * ego_transform.get_forward_vector(),
        )

        opposite_invasion = (
            abs(self._offset) + self.ego_vehicle.bounding_box.extent.y
            > ego_wpt.lane_width / 2
        )
        use_bbs = self._use_bbs_detection or opposite_invasion or ego_wpt.is_junction

        # Get the route bounding box
        route_polygon = get_route_polygon()

        for target_vehicle in self.vehicles_inside_bev:
            if target_vehicle.id == self.ego_vehicle.id:
                continue

            target_transform = target_vehicle.get_transform()
            if target_transform.location.distance(ego_location) > max_distance:
                continue

            target_wpt = self.carla_world_map.get_waypoint(
                target_transform.location,
                lane_type=carla.LaneType.Any,
            )

            # General approach for junctions and vehicles invading other lanes due to the offset
            if (use_bbs or target_wpt.is_junction) and route_polygon:
                target_bb = target_vehicle.bounding_box
                target_vertices = target_bb.get_world_vertices(
                    target_vehicle.get_transform(),
                )
                target_list = [[v.x, v.y, v.z] for v in target_vertices]
                target_polygon = Polygon(target_list)

                if route_polygon.intersects(target_polygon):
                    return (
                        True,
                        target_vehicle.id,
                        float(
                            compute_distance(
                                target_vehicle.get_location(),
                                ego_location,
                            ),
                        ),
                    )

            # Simplified approach, using only the plan waypoints (similar to TM)
            else:
                if (
                    target_wpt.road_id != ego_wpt.road_id
                    or target_wpt.lane_id != ego_wpt.lane_id + lane_offset
                ):
                    next_wpt = self._local_planner.get_incoming_waypoint_and_direction(
                        steps=3,
                    )[0]
                    if not next_wpt:
                        continue
                    if (
                        target_wpt.road_id != next_wpt.road_id
                        or target_wpt.lane_id != next_wpt.lane_id + lane_offset
                    ):
                        continue

                target_forward_vector = target_transform.get_forward_vector()
                target_extent = target_vehicle.bounding_box.extent.x
                target_rear_transform = target_transform
                target_rear_transform.location -= carla.Location(
                    x=target_extent * target_forward_vector.x,
                    y=target_extent * target_forward_vector.y,
                )

                if is_within_distance(
                    target_rear_transform,
                    ego_front_transform,
                    max_distance,
                    [low_angle_th, up_angle_th],
                ):
                    return (
                        True,
                        target_vehicle.id,
                        float(
                            compute_distance(
                                target_transform.location,
                                ego_transform.location,
                            ),
                        ),
                    )

        return False, None, -1.0

    def _plan_target_speed_limit(self) -> float:
        """Compute the highest target speed currently allowed.

        Starts from the speed limit and lowers it for bad weather, night time,
        occluded junctions and clutterness in cities.

        Returns:
            The target speed limit in m/s (also stored as ``self.target_speed_limit``).
        """
        self.target_speed_limit = self.speed_limit
        self.target_speed_limit = max(
            self.target_speed_limit,
            self.second_highest_speed_limit,
        )  # We don't to drive too slow
        if (
            self.second_highest_speed > 30.0 / 3.6
        ):  # We don't want to drive too fast either
            self.target_speed_limit = min(
                self.target_speed_limit,
                self.second_highest_speed / 0.9,
            )

        # --- Drive slower with bad weather, night time, junction and clutterness in city ---
        weather_state = weather.observe(self.carla_world.get_weather())
        self.weather_setting = weather_state.setting
        self.visual_visibility = weather_state.visibility
        self.slower_bad_visibility = False
        self.slower_clutterness = False
        if (
            self.current_active_scenario_type
            not in [
                "EnterActorFlow",
                "EnterActorFlowV2",
                "MergerIntoSlowTraffic",
                "MergerIntoSlowTrafficV2",
                "HighwayExit",
                "NonSignalizedJunctionLeftTurn",
                "SignalizedJunctionLeftTurn",
                "SignalizedJunctionLeftTurnEnterFlow",
                "NonSignalizedJunctionLeftTurnEnterFlow",
                "SignalizedJunctionRightTurn",
                "NonSignalizedJunctionRightTurn",
                "InterurbanActorFlow",
                "InterurbanAdvancedActorFlow",
            ]
            and min(self.target_speed_limit, self.speed_limit)
            < 60 / 3.6  # Assume urban = 60kmh
        ):
            if self.distance_to_next_junction > 0:
                if self.visual_visibility == WeatherVisibility.VERY_LIMITED:
                    self.target_speed_limit -= 4.0
                    self.slower_bad_visibility = True
                if self.num_parking_vehicles_in_proximity >= 2:
                    self.target_speed_limit -= 2.0
                    self.slower_clutterness = True

            # Reduce target speed if there is a junction ahead
            for i in range(
                min(
                    self.config_expert.driving.max_lookahead_to_check_for_junction,
                    len(self.route_waypoints),
                ),
            ):
                if self.route_waypoints[i].is_junction:
                    if self.speed_limit < 40 / 3.6:
                        self.target_speed_limit = min(
                            self.target_speed_limit,
                            self.config_expert.driving.max_speed_in_junction_urban,
                        )

        else:
            self.target_speed_limit = max(
                self.target_speed_limit,
                40.0 / 3.6,
            )  # We don't want to drive too slow

        self.target_speed_limit = max(
            self.target_speed_limit,
            self.config_expert.driving.min_target_speed_limit,
        )
        self.target_speed_limit = min(self.target_speed_limit, 20.0)
        return self.target_speed_limit

    def _plan_target_speed(
        self,
        target_speed_route_obstacle: float,
        obstacle_override: bool,
        speed_reduced_by_obj: list | None,
    ) -> tuple[float, bool, list | None]:
        """Select the target speed for this step, starting from the target speed limit.

        Combines the route-obstacle suggestion from path refining with
        vulnerable road users, general scenarios and scenario-specific
        adjustments (rear danger, cut-ins, visible obstacles).

        Args:
            target_speed_route_obstacle: Target speed suggested while solving route obstacle scenarios.
            obstacle_override: Whether the obstacle scenario forces driving.
            speed_reduced_by_obj: Information about the object that caused speed reduction.

        Returns:
            A tuple containing the target speed, the brake flag and the
            updated speed_reduced_by_obj information.
        """
        target_speed = self.target_speed_limit

        # Manage distance to pedestrians and bikers
        (
            target_speed_vulnerable_vehicles,
            vulnerable_vehicle_override,
            brake_vulnerable_vehicle,
            speed_reduced_by_obj_vulnerable_vehicles,
        ) = self._solve_vulnerable_vehicles_scenarios(target_speed)
        self.does_emergency_brake_for_pedestrians = vulnerable_vehicle_override

        assert int(obstacle_override) + int(vulnerable_vehicle_override) <= 1, (
            "Only one override can be active at a time."
        )

        # Specific cases
        if obstacle_override:
            # In obstacle override, we force the vehicle to drive no matter what.
            LOG.info(
                f"[Control Override] Obstacle override active, forcing drive with speed {target_speed_route_obstacle * 3.6:.1f} km/h",
            )
            brake, target_speed = False, target_speed_route_obstacle
            speed_reduced_by_obj = speed_reduced_by_obj
        elif vulnerable_vehicle_override:
            LOG.info(
                f"[Control Override] Vulnerable vehicle override active, adjusting speed to {min(target_speed, target_speed_vulnerable_vehicles) * 3.6:.1f} km/h, brake={brake_vulnerable_vehicle}",
            )
            brake, target_speed = (
                brake_vulnerable_vehicle,
                min(target_speed, target_speed_vulnerable_vehicles),
            )
            speed_reduced_by_obj = speed_reduced_by_obj_vulnerable_vehicles
        else:  # Generate cases
            brake, target_speed, speed_reduced_by_obj = self._solve_general_scenarios(
                target_speed,
                speed_reduced_by_obj,
            )

        target_speed = min(target_speed, target_speed_route_obstacle)

        self.emergency_brake_for_special_vehicle = False
        if self.current_active_scenario_type in [
            "OppositeVehicleTakingPriority",
            "OppositeVehicleRunningRedLight",
        ]:
            if len(self.adversarial_actors_ids[0]) > 0:
                LOG.info(
                    f"[Control] {self.current_active_scenario_type}: Reducing target speed to 0, waiting for dangerous adversarial vehicle to pass.",
                )
                target_speed = 0.0
                brake = True
        elif self.current_active_scenario_type in [
            "Accident",
            "ConstructionObstacle",
        ]:
            if self.speed_limit > 25:
                if 25 < self.distance_to_scenario_obstacle < 50:
                    LOG.info(
                        f"[Control] {self.current_active_scenario_type}: Reducing target speed to 10.0 m/s, approaching visible obstacle at {self.distance_to_scenario_obstacle:.1f}m",
                    )
                    target_speed = min(target_speed, 10.0)
            elif self.speed_limit > 20:
                if 25 < self.distance_to_scenario_obstacle < 45:
                    LOG.info(
                        f"[Control] {self.current_active_scenario_type}: Reducing target speed to 7.5 m/s, approaching visible obstacle at {self.distance_to_scenario_obstacle:.1f}m",
                    )
                    target_speed = min(target_speed, 7.5)
            else:
                if 25 < self.distance_to_scenario_obstacle < 40:
                    LOG.info(
                        f"[Control] {self.current_active_scenario_type}: Reducing target speed to {self.config_expert.driving.min_target_speed_limit} m/s, approaching visible obstacle at {self.distance_to_scenario_obstacle:.1f}m",
                    )
                    target_speed = min(
                        target_speed,
                        self.config_expert.driving.min_target_speed_limit,
                    )
        elif self.current_active_scenario_type in [
            "ParkedObstacle",
        ]:
            if self.speed_limit > 25:
                if 25 < self.distance_to_scenario_obstacle < 45:
                    LOG.info(
                        f"[Control] ParkedObstacle: Reducing target speed to 10.0 m/s, approaching parked obstacle at {self.distance_to_scenario_obstacle:.1f}m",
                    )
                    target_speed = min(target_speed, 10.0)
            elif self.speed_limit > 20:
                if 25 < self.distance_to_scenario_obstacle < 45:
                    LOG.info(
                        f"[Control] ParkedObstacle: Reducing target speed to 7.5 m/s, approaching parked obstacle at {self.distance_to_scenario_obstacle:.1f}m",
                    )
                    target_speed = min(target_speed, 7.5)
            else:
                if 25 < self.distance_to_scenario_obstacle < 35:
                    LOG.info(
                        f"[Control] ParkedObstacle: Reducing target speed to {self.config_expert.driving.min_target_speed_limit} m/s, approaching parked obstacle at {self.distance_to_scenario_obstacle:.1f}m",
                    )
                    target_speed = min(
                        target_speed,
                        self.config_expert.driving.min_target_speed_limit,
                    )

        # Attempt to try to avoid collision in some cases
        self.rear_danger_8 = self.rear_danger_16 = False
        if self.current_active_scenario_type in [
            "NonSignalizedJunctionRightTurn",
            "SignalizedJunctionRightTurn",
            "EnterActorFlow",
            "EnterActorFlowV2",
            "InterurbanAdvancedActorFlow",
            "OppositeVehicleRunningRedLight",
            "OppositeVehicleTakingPriority",
            "OppositeVehicleTakingPriority",
            "NonSignalizedJunctionLeftTurnEnterFlow",
            "SignalizedJunctionLeftTurnEnterFlow",
        ]:
            rear_adversarial_actor = self.rear_adversarial_actor
            if rear_adversarial_actor:
                vehicle_behind_speed = rear_adversarial_actor.get_velocity().length()
                vehicle_behind_distance = self.ego_location.distance(
                    rear_adversarial_actor.get_location(),
                )
                if (
                    vehicle_behind_distance < 8
                    and vehicle_behind_speed > 5
                    and self.ego_speed > 5
                ):
                    LOG.info(
                        f"[Control] {self.current_active_scenario_type}: Rear danger (8m): Accelerating to avoid rear collision, vehicle behind speed: {vehicle_behind_speed * 3.6:.1f} km/h, adjusting target speed to {(vehicle_behind_speed + 5) * 3.6:.1f} km/h",
                    )
                    target_speed = max(target_speed, vehicle_behind_speed + 5)
                    self.rear_danger_8 = True
                    brake = False
                elif (
                    vehicle_behind_distance < 16
                    and vehicle_behind_speed > 5
                    and self.ego_speed > 5
                ):
                    LOG.info(
                        f"[Control] {self.current_active_scenario_type}: Rear danger (16m): Matching rear vehicle speed: {vehicle_behind_speed * 3.6:.1f} km/h",
                    )
                    target_speed = max(target_speed, vehicle_behind_speed)
                    self.rear_danger_16 = True
                    brake = False

        # Safety brake if some vehicles cutin from side. This make the imitation learning easier.
        self.brake_cutin = False
        if self.current_active_scenario_type in ["ParkingCutIn"]:
            assert len(self.cutin_actors) == 1
            cut_in_vehicle = self.cutin_actors[0]
            if (
                1 < cut_in_vehicle.get_velocity().length() < 4.25
                and not CarlaDataProvider.get_current_scenario_memory()["stopped"]
            ):
                LOG.info(
                    f"[Control] ParkingCutIn: Emergency braking for cut-in vehicle (speed={cut_in_vehicle.get_velocity().length() * 3.6:.1f} km/h)",
                )
                self.brake_cutin = True
                brake = True
                target_speed = 0.0
            elif (
                not CarlaDataProvider.get_current_scenario_memory()["stopped"]
                and cut_in_vehicle.get_velocity().length() >= 5
            ):
                LOG.info(
                    f"[Control] ParkingCutIn: Cut-in vehicle cleared (speed={cut_in_vehicle.get_velocity().length() * 3.6:.1f} km/h)",
                )
                CarlaDataProvider.get_current_scenario_memory()["stopped"] = True
        elif self.current_active_scenario_type in ["StaticCutIn"]:
            assert len(self.cutin_actors) == 1
            cut_in_vehicle = self.cutin_actors[0]
            if (
                2.1 < cut_in_vehicle.get_velocity().length() < 4.25
                and not CarlaDataProvider.get_current_scenario_memory()["stopped"]
            ):
                LOG.info(
                    f"[Control] StaticCutIn: Emergency braking for cut-in vehicle (speed={cut_in_vehicle.get_velocity().length() * 3.6:.1f} km/h)",
                )
                self.brake_cutin = True
                brake = True
                target_speed = 0.0
            elif (
                not CarlaDataProvider.get_current_scenario_memory()["stopped"]
                and cut_in_vehicle.get_velocity().length() >= 5
            ):
                LOG.info(
                    f"[Control] StaticCutIn: Cut-in vehicle cleared (speed={cut_in_vehicle.get_velocity().length() * 3.6:.1f} km/h)",
                )
                CarlaDataProvider.get_current_scenario_memory()["stopped"] = True

        return float(target_speed), brake, speed_reduced_by_obj

    def get_control(self, brake: bool, target_speed: float) -> carla.VehicleControl:
        """Convert the planned target speed into vehicle controls.

        Args:
            brake: Whether the planner requests braking.
            target_speed: The planned target speed in m/s.

        Returns:
            The control commands for this step.
        """
        # Compute throttle and brake control
        throttle, control_brake = self._longitudinal_controller.get_throttle_and_brake(
            brake,
            target_speed,
            self.ego_speed,
        )

        # Compute steering control
        steer = round(
            self._turn_controller.step(
                self.route_waypoints_np[:, :2],
                self.ego_speed,
                self.ego_location_array[:2],
                self.ego_orientation_rad,
            ),
            3,
        )

        # Create the control command
        self.control = carla.VehicleControl()
        self.control.steer = (
            steer + self.config_expert.driving.steer_noise * np.random.randn()
        )
        self.control.throttle = throttle
        self.control.brake = float(brake or control_brake)

        # Apply brake if the vehicle is stopped to prevent rolling back
        if (
            self.control.throttle == 0
            and self.ego_speed
            < self.config_expert.driving.minimum_speed_to_prevent_rolling_back
        ):
            self.control.brake = 1

        # Apply throttle if the vehicle is blocked for too long
        ego_velocity = CarlaDataProvider.get_velocity(self.ego_vehicle)
        if ego_velocity < 0.1:
            self.ego_blocked_for_ticks += 1
        else:
            self.ego_blocked_for_ticks = 0

        if self.ego_blocked_for_ticks >= self.config_expert.driving.max_blocked_ticks:
            self.control.throttle = 1
            self.control.brake = 0

        return self.control
