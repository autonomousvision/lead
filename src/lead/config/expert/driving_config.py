"""Driving behavior configuration of the expert.

Covers target speeds, IDM parameters, forecasting extents, scenario handling
and the privileged route planner.
"""

import jaxtyping as jt
import numpy as np
import numpy.typing as npt

from lead.config.node import ConfigNode, overridable_property


class ExpertDrivingConfig(ConfigNode):
    """Driving behavior knobs of the privileged expert."""

    @property
    def _ppm(self) -> int:
        """Route interpolation density used to convert meters to route points."""
        return self._root.expert.simulation.points_per_meter

    # --- Autopilot Configuration ---
    # Maximum speed in m/s to consider a bad view
    min_target_speed_limit: float = 6.75
    # Noise added to expert steering angle
    steer_noise: float = 1e-3
    # Distance of obstacles (in meters) in which we will check for collisions
    detection_radius: float = 50.0
    # Distance of traffic lights considered relevant (in meters)
    light_radius: float = 64.0
    # Bounding boxes in this radius around the car will be saved in the dataset
    bb_save_radius: float = 96.0
    # Ratio between the speed limit / curvature dependent speed limit and the target speed
    ratio_target_speed_limit: float = 0.72
    # Maximum number of ticks the agent doesn't take any action (max 179, speed must be >0.1)
    max_blocked_ticks: int = 170
    # Minimum walker speed in m/s
    min_walker_speed: float = 0.5

    @property
    def fps(self) -> float:
        """FPS of the simulation."""
        return float(self._root.expert.simulation.carla_fps)

    @property
    def fps_inv(self) -> float:
        """Inverse of the FPS."""
        return 1.0 / self.fps

    # Distance to the stop sign when the previous stop sign is uncleared
    unclearing_distance_to_stop_sign: float = 10
    # Distance to the stop sign when the previous stop sign is cleared
    clearing_distance_to_stop_sign: float = 3.0

    # --- Intelligent Driver Model (IDM) ---
    # IDM minimum distance for stop signs
    idm_stop_sign_minimum_distance: float = 2.0
    # IDM desired time headway for stop signs
    idm_stop_sign_desired_time_headway: float = 0.5
    # IDM desired time headway for red lights
    idm_red_light_desired_time_headway: float = 0.5
    # IDM minimum distance for red lights
    idm_red_light_minimum_distance: float = 3.0
    # IDM additional distance for overhead truck red lights in newer towns
    idm_overhead_red_light_minimum_distance: float = 9.0
    # IDM additional distance for Europe red lights in older towns
    idm_europe_red_light_minimum_distance: float = 6.0
    # IDM minimum distance for pedestrians
    idm_pedestrian_minimum_distance: float = 4.5
    # IDM desired time headway for pedestrians
    idm_pedestrian_desired_time_headway: float = 0.125
    # IDM minimum distance for bicycles
    idm_bicycle_minimum_distance: float = 6.0
    # IDM desrired time headway for bicycles
    idm_bicycle_desired_time_headway: float = 0.5
    # IDM minimum distance for leading vehicles
    idm_leading_vehicle_minimum_distance: float = 4.0
    # IDM desrired time headway for leading vehicles
    idm_leading_vehicle_time_headway: float = 0.25
    # IDM minimum distance for two way scenarios
    idm_two_way_scenarios_minimum_distance: float = 2.0
    # IDM desrired time headway for two way scenarios
    idm_two_way_scenarios_time_headway: float = 0.1
    # Boundary time - the integration won't continue beyond it.
    idm_t_bound: float = 0.05
    # IDM maximum acceleration parameter per frame
    idm_maximum_acceleration: float = 24.0
    # The following parameters were determined by measuring the vehicle's braking performance.
    # IDM maximum deceleration parameter per frame while driving slow
    idm_comfortable_braking_deceleration_low_speed: float = 8.7
    # IDM maximum deceleration parameter per frame while driving fast
    idm_comfortable_braking_deceleration_high_speed: float = 3.72
    # Threshold to determine, when to use idm_comfortable_braking_deceleration_low_speed and
    # idm_comfortable_braking_deceleration_high_speed
    idm_comfortable_braking_deceleration_threshold: float = 6.02
    # IDM acceleration exponent (default = 4.)
    idm_acceleration_exponent: float = 4.0

    # --- Forecasting ---
    # Minimum extent for pedestrian during bbs forecasting
    pedestrian_minimum_extent: float = 1.5
    # Factor to increase the ego vehicles bbs in driving direction during forecasting
    # when speed > extent_ego_bbs_speed_threshold
    high_speed_extent_factor_ego_x: float = 1.3
    # Factor to increase the ego vehicles bbs in y direction during forecasting
    # when speed > extent_ego_bbs_speed_threshold
    high_speed_extent_factor_ego_y: float = 1.2
    # Threshold to decide, when which bbs increase factor is used
    extent_ego_bbs_speed_threshold: float = 5
    # Forecast length in seconds when near a lane change
    forecast_length_lane_change: float = 1.1
    # Forecast length in seconds when not near a lane change
    default_forecast_length: float = 2.0
    # Factor to increase the ego vehicles bbs during forecasting when speed < extent_ego_bbs_speed_threshold
    slow_speed_extent_factor_ego: float = 1.0
    # Speed threshold to select which factor is used during other vehicle bbs forecasting
    extent_other_vehicles_bbs_speed_threshold: float = 1.0
    # Minimum extent of bbs, while forecasting other vehicles
    high_speed_min_extent_y_other_vehicle: float = 1.0
    # Extent factor to scale bbs during forecasting other vehicles in y direction
    high_speed_extent_y_factor_other_vehicle: float = 1.3
    # Extent factor to scale bbs during forecasting other vehicles in x direction
    high_speed_extent_x_factor_other_vehicle: float = 1.5
    # Minimum extent factor to scale bbs during forecasting other vehicles in x direction
    high_speed_min_extent_x_other_vehicle: float = 1.2
    # Minimum extent factor to scale bbs during forecasting other vehicles in x direction during lane changes to
    # account fore forecasting inaccuracies
    high_speed_min_extent_x_other_vehicle_lane_change: float = 2.0
    # Safety distance to be added to emergency braking distance
    braking_distance_calculation_safety_distance: float = 10
    # Minimum speed in m/s to prevent rolling back, when braking no throttle is applied
    minimum_speed_to_prevent_rolling_back: float = 0.5
    # Maximum seed in junctions in m/s
    max_speed_in_junction_urban: float = 25 / 3.6

    @overridable_property
    def max_lookahead_to_check_for_junction(self) -> int:
        """Lookahead distance to check, whether the ego is close to a junction."""
        return 30 * self._ppm

    @overridable_property
    def tf_first_checkpoint_distance(self) -> int:
        """Distance of the first checkpoint for TF++."""
        return int(2.5 * self._ppm)

    # Parameters to calculate how much the ego agent needs to cover a given distance. Values are taken from
    # the kinematic bicycle model
    compute_min_time_to_cover_distance_params: jt.Float[npt.NDArray, "4"] = np.array(
        [0.00904221, 0.00733342, -0.03744807, 0.0235038],
    )
    # Distance to check for road_id/lane_id for RouteObstacle scenarios
    previous_road_lane_retrieve_distance: float = 100
    # Distance to check for road_id/lane_id for RouteObstacle scenarios
    next_road_lane_retrieve_distance: float = 100
    # Safety distance during checking if the path is free for RouteObstacle scenarios
    check_path_free_safety_distance: float = 10
    # Safety time headway during checking if the path is free for RouteObstacle scenarios
    check_path_free_safety_time: float = 0.2

    @overridable_property
    def transition_smoothness_factor_construction_obstacle(self) -> float:
        """Transition length for change lane in scenario ConstructionObstacle."""
        return 10.5 * self._ppm

    @overridable_property
    def minimum_lookahead_distance_to_compute_near_lane_change(self) -> int:
        """Check in x route points if there is lane change ahead."""
        return 20 * self._ppm

    @overridable_property
    def check_previous_distance_for_lane_change(self) -> int:
        """Check if did a lane change in the previous x route points."""
        return 15 * self._ppm

    @overridable_property
    def draw_future_route_till_distance(self) -> int:
        """Draw x route points of the route during debugging."""
        return 500 * self._ppm

    # Default minimum distance to process the route obstacle scenarios
    default_max_distance_to_process_scenario: float = 50

    # --- HazardAtSideLane ---
    # If true enable smooth HazardAtSideLane merging
    hazard_at_side_lane_smooth_merging: bool = True
    # Minimum distance to process HazardAtSideLane scenarios
    max_distance_to_process_hazard_at_side_lane: float = 25
    # Minimum distance to process HazardAtSideLaneTwoWays scenarios
    max_distance_to_process_hazard_at_side_lane_two_ways: float = 10

    # --- TwoWays Scenarios with Obstacles ---
    @overridable_property
    def max_distance_to_overtake_two_way_scnearios(self) -> int:
        """Maximum distance to start the overtaking maneuver (route points)."""
        return int(8 * self._ppm)

    # Default overtaking speed in m/s for all route obstacle scenarios
    default_overtake_speed: float = 40.0 / 3.6

    @overridable_property
    def distance_to_delete_scenario_in_two_ways(self) -> int:
        """Distance in route points at which two ways scenarios are considered finished."""
        return int(2 * self._ppm)

    @overridable_property
    def transition_length_construction_obstacle_two_ways(self) -> int:
        """Transition length for scenario ConstructionObstacleTwoWays to change lanes."""
        return int(5 * self._ppm)

    @overridable_property
    def add_before_construction_obstacle_two_ways(self) -> int:
        """Increase overtaking maneuver by distance before the obstacle in ConstructionObstacleTwoWays."""
        return int(1.5 * self._ppm)

    @overridable_property
    def add_after_construction_obstacle_two_ways(self) -> int:
        """Increase overtaking maneuver by distance after the obstacle in ConstructionObstacleTwoWays."""
        return int(1.0 * self._ppm)

    # How much to drive to the center of the opposite lane while handling ConstructionObstacleTwoWays
    factor_construction_obstacle_two_ways: float = 1.08

    # --- AccidentTwoWays ---
    @overridable_property
    def add_after_accident_two_ways(self) -> int:
        """Increase overtaking maneuver by distance after the obstacle in AccidentTwoWays."""
        return int(-1.0 * self._ppm)

    @overridable_property
    def transition_length_accident_two_ways(self) -> int:
        """Transition length for scenario AccidentTwoWays to change lanes."""
        return int(4.5 * self._ppm)

    @overridable_property
    def add_before_accident_two_ways(self) -> int:
        """Increase overtaking maneuver by distance before the obstacle in AccidentTwoWays."""
        return int(-1.0 * self._ppm)

    # How much to drive to the center of the opposite lane while handling AccidentTwoWays
    factor_accident_two_ways: float = 1.0

    # --- ParkedObstacleTwoWays ---
    @overridable_property
    def transition_length_parked_obstacle_two_ways(self) -> int:
        """Transition length for scenario ParkedObstacleTwoWays to change lanes."""
        return int(4 * self._ppm)

    @overridable_property
    def add_after_parked_obstacle_two_ways(self) -> int:
        """Increase overtaking maneuver by distance after the obstacle in ParkedObstacleTwoWays."""
        return int(0.5 * self._ppm)

    # How much to drive to the center of the opposite lane while handling ParkedObstacleTwoWays
    factor_parked_obstacle_two_ways: float = 1.0

    @overridable_property
    def add_before_parked_obstacle_two_ways(self) -> int:
        """Increase overtaking maneuver by distance before the obstacle in ParkedObstacleTwoWays."""
        return int(-0.5 * self._ppm)

    # --- VehicleOpensDoorTwoWays ---
    # How much to drive to the center of the opposite lane while handling VehicleOpensDoorTwoWays
    factor_vehicle_opens_door_two_ways: float = 0.7
    # Overtaking speed in m/s for vehicle opens door two ways scenarios
    overtake_speed_vehicle_opens_door_two_ways: float = 8.75

    @overridable_property
    def add_before_vehicle_opens_door_two_ways(self) -> int:
        """Increase overtaking maneuver by distance before the obstacle in VehicleOpensDoorTwoWays."""
        return int(-1.0 * self._ppm)

    @overridable_property
    def add_after_vehicle_opens_door_two_ways(self) -> int:
        """Increase overtaking maneuver by distance after the obstacle in VehicleOpensDoorTwoWays."""
        return int(-1.0 * self._ppm)

    @overridable_property
    def transition_length_vehicle_opens_door_two_ways(self) -> int:
        """Transition length for scenario VehicleOpensDoorTwoWays to change lanes."""
        return int(5 * self._ppm)

    # --- Accident ---
    @overridable_property
    def add_before_accident(self) -> int:
        """Distance to add before accident for smooth merging."""
        return int(1.5 * self._ppm)

    @overridable_property
    def add_after_accident(self) -> int:
        """Distance to add after accident for smooth merging."""
        return int(1.2 * self._ppm)

    @overridable_property
    def transition_smoothness_distance_accident(self) -> int:
        """Transition smoothness distance for accident scenarios."""
        return int(10 * self._ppm)

    # --- ParkedObstacle ---
    @overridable_property
    def add_before_parked_obstacle(self) -> int:
        """Distance to add before parked obstacle for smooth merging."""
        return int(1.0 * self._ppm)

    @overridable_property
    def add_after_parked_obstacle(self) -> int:
        """Distance to add after parked obstacle for smooth merging."""
        return int(1.0 * self._ppm)

    @overridable_property
    def transition_smoothness_distance_parked_obstacle(self) -> int:
        """Transition smoothness distance for parked obstacle scenarios."""
        return int(10 * self._ppm)

    # --- ConstructionObstacle ---
    @overridable_property
    def add_before_construction_obstacle(self) -> int:
        """Distance to add before construction obstacle for smooth merging."""
        return int(1.5 * self._ppm)

    @overridable_property
    def add_after_construction_obstacle(self) -> int:
        """Distance to add after construction obstacle for smooth merging."""
        return int(1.5 * self._ppm)

    @overridable_property
    def transition_smoothness_distance_construction_obstacle(self) -> int:
        """Transition smoothness distance for construction obstacle scenarios."""
        return int(10 * self._ppm)

    # --- Privileged Route Planner ---
    @overridable_property
    def ego_vehicles_route_point_search_distance(self) -> int:
        """Maximum distance to search ahead for updating ego route index (route points)."""
        return 4 * self._ppm

    @overridable_property
    def lane_shift_extension_length_for_yield_to_emergency_vehicle(self) -> int:
        """Length to extend lane shift transition for YieldToEmergencyVehicle (route points)."""
        return 20 * self._ppm

    @overridable_property
    def transition_smoothness_distance(self) -> int:
        """Distance over which lane shift transition is smoothed (route points)."""
        return 8 * self._ppm

    @overridable_property
    def route_shift_start_distance_invading_turn(self) -> int:
        """Distance over which lane shift transition is smoothed for InvadingTurn (route points)."""
        return 15 * self._ppm

    @overridable_property
    def route_shift_end_distance_invading_turn(self) -> int:
        """Route shift end distance for InvadingTurn (route points)."""
        return 10 * self._ppm

    # Margin from fence when shifting route in InvadingTurn
    fence_avoidance_margin_invading_turn: float = 0.3
    # Minimum lane width to avoid early lane changes
    minimum_lane_width_threshold: float = 2.5

    @overridable_property
    def speed_limit_waypoints_spacing_check(self) -> int:
        """Spacing for checking and updating speed limits (route points)."""
        return 5 * self._ppm

    # Maximum distance on route for detecting leading vehicles
    leading_vehicles_max_route_distance: float = 2.5
    # Maximum angle difference for detecting leading vehicles (meters)
    leading_vehicles_max_route_angle_distance: float = 35.0

    @overridable_property
    def leading_vehicles_maximum_detection_radius(self) -> int:
        """Maximum radius for detecting any leading vehicles (route points)."""
        return 80 * self._ppm

    # Maximum distance on route for detecting trailing vehicles
    trailing_vehicles_max_route_distance: float = 3.0
    # Maximum route distance for trailing vehicles after lane change
    trailing_vehicles_max_route_distance_lane_change: float = 6.0

    @overridable_property
    def tailing_vehicles_maximum_detection_radius(self) -> int:
        """Maximum radius for detecting any trailing vehicles (route points)."""
        return 80 * self._ppm

    @overridable_property
    def max_distance_lane_change_trailing_vehicles(self) -> int:
        """Maximum distance to check for lane changes when detecting trailing vehicles (route points)."""
        return 15 * self._ppm

    # Distance to extend the end of the route to ensure checkpoints are available (meters)
    extra_route_length: float = 50

    # --- Discontinuous Road Configuration ---
    # Handles scenarios where road continuity is interrupted

    # Maximum distance between route points
    max_distance_between_future_route_points: float = 0.25

    @overridable_property
    def discontinuous_road_max_future_points(self) -> float:
        """Maximum future points to consider."""
        return 5 / self._ppm

    # Maximum speed in discontinuous road areas
    discontinuous_road_max_speed: float = 12.5

    @overridable_property
    def discontinuous_road_max_future_check(self) -> int:
        """Maximum distance to check ahead (route points)."""
        return 10 * self._ppm

    # --- High Road Curvature Configuration ---
    @overridable_property
    def high_road_curvature_max_future_points(self) -> int:
        """Maximum future points to consider (route points)."""
        return 20 * self._ppm

    # Maximum speed in high curvature areas
    high_road_curvature_max_speed: float = 12.5
