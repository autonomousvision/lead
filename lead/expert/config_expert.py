"""
Config class that contains all the hyperparameters needed to build any model.
"""

from collections import defaultdict
from functools import cached_property

import carla
import numpy as np

from lead.common import weathers
from lead.common.config_base import BaseConfig
from lead.common.constants import CarlaImageCroppingType, TargetDataset, WeatherVisibility

# Temporal hack for points per meter conversion - needs to be fixed
points_per_meter = 10


def rgb(r, g, b):
    """Convert RGB values to CARLA. Helper for VSCode color extension."""
    return carla.Color(r, g, b)


class ExpertConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.load_from_environment(
            loaded_config=None,
            env_key="EXPERT_CONFIG",
            raise_error_on_missing_key=True,
        )

    @property
    def target_dataset(self):
        return TargetDataset.CARLA_LEADERBOARD2_3CAMERAS

    @property
    def crop_height(self):
        """The amount of pixels cropped from the bottom of the image."""
        return self.camera_calibration[1]["height"] - self.camera_calibration[1]["cropped_height"]

    @property
    def carla_crop_height_type(self):
        """Type of cropping applied to CARLA images."""
        if self.target_dataset in [TargetDataset.CARLA_LEADERBOARD2_3CAMERAS, TargetDataset.CARLA_LEADERBOARD2_6CAMERAS]:
            return CarlaImageCroppingType.NONE
        elif self.target_dataset in [
            TargetDataset.NAVSIM_4CAMERAS,
        ]:
            return CarlaImageCroppingType.NONE
        elif self.target_dataset in [TargetDataset.WAYMO_E2E_2025_3CAMERAS]:
            return CarlaImageCroppingType.BOTTOM
        raise ValueError("Unknown target dataset")

    # --- Planning Area ---
    # Maximum planning area coordinate in x direction (meters)
    # How many pixels make up 1 meter in BEV grids.
    pixels_per_meter = 4.0

    @property
    def min_x_meter(self):
        """Back boundary of the planning area in meters."""
        return -32

    @property
    def max_x_meter(self):
        """Front boundary of the planning area in meters."""
        return 64

    @property
    def min_y_meter(self):
        """Left boundary of the planning area in meters."""
        return -40

    @property
    def max_y_meter(self):
        """Right boundary of the planning area in meters."""
        return 40

    @property
    def lidar_width_pixel(self):
        """Width resolution of LiDAR BEV representation in pixels."""
        return int((self.max_x_meter - self.min_x_meter) * self.pixels_per_meter)

    @property
    def lidar_height_pixel(self):
        """Height resolution of LiDAR BEV representation in pixels."""
        return int((self.max_y_meter - self.min_y_meter) * self.pixels_per_meter)

    @property
    def lidar_width_meter(self):
        """Width of LiDAR coverage area in meters."""
        return int(self.max_x_meter - self.min_x_meter)

    @property
    def lidar_height_meter(self):
        """Height of LiDAR coverage area in meters."""
        return int(self.max_y_meter - self.min_y_meter)

    # --- Occlusion checks ---
    # Minimum number of LiDAR points for a vehicle to be considered valid.
    vehicle_min_num_lidar_points = 3
    # Minimum number of visible pixels for a vehicle to be considered valid.
    vehicle_min_num_visible_pixels = 5
    # Minimum number of LiDAR points for a pedestrian to be considered valid.
    pedestrian_min_num_lidar_points = 5
    # Minimum number of visible pixels for a pedestrian to be considered valid.
    pedestrian_min_num_visible_pixels = 15
    # Minimum number of LiDAR points for a parking vehicle to be considered valid.
    parking_vehicle_min_num_lidar_points = 5
    # Minimum number of visible pixels for a parking vehicle to be considered valid.
    parking_vehicle_min_num_visible_pixels = 10

    # --- Debug Colors for Visualization ---
    # Color for future route visualization during debugging
    future_route_color = rgb(0, 255, 0)
    # Color for other vehicles' forecasted bounding boxes
    other_vehicles_forecasted_bbs_color = carla.Color(0, 0, 255)
    # Color for leading vehicle visualization
    leading_vehicle_color = carla.Color(255, 0, 0)
    # Color for trailing vehicle visualization
    trailing_vehicle_color = carla.Color(255, 255, 255)
    # Color for ego vehicle bounding box
    ego_vehicle_bb_color = carla.Color(0, 0, 0)
    # Color for pedestrian forecasted bounding boxes
    pedestrian_forecasted_bbs_color = carla.Color(0, 0, 255)
    # Color for red traffic lights
    red_traffic_light_color = carla.Color(0, 255, 0)
    # Color for yellow traffic lights
    yellow_traffic_light_color = carla.Color(255, 255, 0)
    # Color for green traffic lights
    green_traffic_light_color = carla.Color(255, 0, 0, 255)
    # Color for off traffic lights
    off_traffic_light_color = carla.Color(0, 0, 0)
    # Color for unknown traffic lights
    unknown_traffic_light_color = carla.Color(0, 0, 0)
    # Color for cleared stop signs
    cleared_stop_sign_color = carla.Color(0, 255, 0)
    # Color for uncleared stop signs
    uncleared_stop_sign_color = carla.Color(255, 0, 0)
    # Color for ego vehicle forecasted bounding boxes in hazard situations
    ego_vehicle_forecasted_bbs_hazard_color = carla.Color(255, 0, 0)
    # Color for ego vehicle forecasted bounding boxes in normal situations
    ego_vehicle_forecasted_bbs_normal_color = carla.Color(0, 255, 0)
    # Color for highlighted route segments
    highlight_route_segment_color = carla.Color(255, 0, 0)
    # Color for source lane visualization
    source_lane_color = carla.Color(0, 255, 0)
    # Color for target lane visualization
    target_lane_color = carla.Color(255, 0, 0)
    # Color for opponent traffic route
    opponent_traffic_route_color = carla.Color(255, 0, 0)
    # Color for intersection points
    intersection_point_color = carla.Color(0, 0, 255)
    # Color for adversarial situations
    adversarial_color = carla.Color(255, 0, 0)

    # --- Dataset and Timing Configuration ---
    # How many pixels make up 1 meter in BEV grids.
    pixels_per_meter = 4.0

    @property
    def shuffle_weather(self):
        """If true shuffle weather conditions during training."""
        if self.is_on_slurm:
            return True
        return False

    @property
    def nice_weather(self):
        """If true use only nice weather conditions."""
        if self.is_on_slurm:
            return False
        return True

    @property
    def jpeg_compression(self):
        """If true enable JPEG compression for image storage."""
        if self.is_on_slurm:
            return True
        return False

    @property
    def datagen(self):
        """If true enable data generation mode."""
        if self.is_on_slurm:
            return True
        return True

    # --- Longitudinal Linear Regression Controller ---
    # Minimum threshold for target speed (< 1 km/h) for longitudinal linear regression controller
    longitudinal_linear_regression_minimum_target_speed = 0.278
    # Maximum acceleration rate (approximately 1.9 m/tick) for the longitudinal linear regression controller
    longitudinal_linear_regression_maximum_acceleration = 1.89
    # Maximum deceleration rate (approximately -4.82 m/tick) for the longitudinal linear regression controller
    longitudinal_linear_regression_maximum_deceleration = -4.82

    # --- Lateral PID Controller ---
    # The proportional gain for the lateral PID controller
    lateral_pid_kp = 3.118357247806046
    # The derivative gain for the lateral PID controller
    lateral_pid_kd = 1.3782508892109167
    # The integral gain for the lateral PID controller
    lateral_pid_ki = 0.6406067986034124
    # The scaling factor used in the calculation of the lookahead distance based on the current speed
    lateral_pid_speed_scale = 0.9755321901954155
    # The offset used in the calculation of the lookahead distance based on the current speed
    lateral_pid_speed_offset = 1.9152884533402488
    # The default lookahead distance for the lateral PID controller
    lateral_pid_default_lookahead = 2.4 * points_per_meter
    # The speed threshold (in km/h) for switching between the default and variable lookahead distance
    lateral_pid_speed_threshold = 2.3150102938235136 * points_per_meter
    # The size of the sliding window used to store the error history for the lateral PID controller
    lateral_pid_window_size = 6
    # The minimum allowed lookahead distance for the lateral PID controller
    lateral_pid_minimum_lookahead_distance = 2.4 * points_per_meter
    # The maximum allowed lookahead distance for the lateral PID controller
    lateral_pid_maximum_lookahead_distance = 10.5 * points_per_meter
    # Longitudinal control parameters for ego vehicle dynamics.
    longitudinal_params = (
        1.1990342347353184,
        -0.8057602384167799,
        1.710818710950062,
        0.921890257450335,
        1.556497522998393,
        -0.7013479734904027,
        1.031266635497984,
    )
    # Linear regression parameters for longitudinal control as numpy array
    longitudinal_linear_regression_params = np.array(
        [
            1.1990342347353184,
            -0.8057602384167799,
            1.710818710950062,
            0.921890257450335,
            1.556497522998393,
            -0.7013479734904027,
            1.031266635497984,
        ]
    )
    # Coefficients for polynomial equation estimating speed change with throttle input for ego model
    throttle_values = np.array(
        [
            9.63873001e-01,
            4.37535692e-04,
            -3.80192912e-01,
            1.74950069e00,
            9.16787414e-02,
            -7.05461530e-02,
            -1.05996152e-03,
            6.71079346e-04,
        ]
    )
    # Coefficients for polynomial equation estimating speed change with brake input for the ego model
    brake_values = np.array(
        [
            9.31711370e-03,
            8.20967431e-02,
            -2.83832427e-03,
            5.06587474e-05,
            -4.90357228e-07,
            2.44419284e-09,
            -4.91381935e-12,
        ]
    )

    # --- Autopilot Configuration ---
    # Maximum speed in m/s to consider a bad view
    min_target_speed_limit = 6.75
    # Noise added to expert steering angle
    steer_noise = 1e-3
    # Distance of obstacles (in meters) in which we will check for collisions
    detection_radius = 50.0
    # Distance of traffic lights considered relevant (in meters)
    light_radius = 64.0
    # Bounding boxes in this radius around the car will be saved in the dataset
    bb_save_radius = 96.0
    # Ratio between the speed limit / curvature dependent speed limit and the target speed
    ratio_target_speed_limit = 0.72
    # Maximum number of ticks the agent doesn't take any action (max 179, speed must be >0.1)
    max_blocked_ticks = 170
    # Minimum walker speed in m/s
    min_walker_speed = 0.5
    # Time in seconds to draw the things during debugging
    draw_life_time = 0.051
    # FPS of the simulation
    fps = 20.0
    # Inverse of the FPS
    fps_inv = 1.0 / fps
    # Distance to the stop sign when the previous stop sign is uncleared
    unclearing_distance_to_stop_sign = 10
    # Distance to the stop sign when the previous stop sign is cleared
    clearing_distance_to_stop_sign = 3.0
    # IDM minimum distance for stop signs
    idm_stop_sign_minimum_distance = 2.0
    # IDM desired time headway for stop signs
    idm_stop_sign_desired_time_headway = 0.5
    # IDM desired time headway for red lights
    idm_red_light_desired_time_headway = 0.5
    # IDM minimum distance for red lights
    idm_red_light_minimum_distance = 3.0
    # IDM additional distance for overhead truck red lights in newer towns
    idm_overhead_red_light_minimum_distance = 9.0
    # IDM additional distance for Europe red lights in older towns
    idm_europe_red_light_minimum_distance = 6.0
    # IDM minimum distance for pedestrians
    idm_pedestrian_minimum_distance = 4.5
    # IDM desired time headway for pedestrians
    idm_pedestrian_desired_time_headway = 0.125
    # IDM minimum distance for bicycles
    idm_bicycle_minimum_distance = 6.0
    # IDM desrired time headway for bicycles
    idm_bicycle_desired_time_headway = 0.5
    # IDM minimum distance for leading vehicles
    idm_leading_vehicle_minimum_distance = 4.0
    # IDM desrired time headway for leading vehicles
    idm_leading_vehicle_time_headway = 0.25
    # IDM minimum distance for two way scenarios
    idm_two_way_scenarios_minimum_distance = 2.0
    # IDM desrired time headway for two way scenarios
    idm_two_way_scenarios_time_headway = 0.1
    # Boundary time - the integration won’t continue beyond it.
    idm_t_bound = 0.05
    # IDM maximum accelaration parameter per frame
    idm_maximum_acceleration = 24.0
    # The following parameters were determined by measuring the vehicle's braking performance.
    # IDM maximum deceleration parameter per frame while driving slow
    idm_comfortable_braking_deceleration_low_speed = 8.7
    # IDM maximum deceleration parameter per frame while driving fast
    idm_comfortable_braking_deceleration_high_speed = 3.72
    # Threshold to determine, when to use idm_comfortable_braking_deceleration_low_speed and
    # idm_comfortable_braking_deceleration_high_speed
    idm_comfortable_braking_deceleration_threshold = 6.02
    # IDM acceleration exponent (default = 4.)
    idm_acceleration_exponent = 4.0
    # Minimum extent for pedestrian during bbs forecasting
    pedestrian_minimum_extent = 1.5
    # Factor to increase the ego vehicles bbs in driving direction during forecasting
    # when speed > extent_ego_bbs_speed_threshold
    high_speed_extent_factor_ego_x = 1.3
    # Factor to increase the ego vehicles bbs in y direction during forecasting
    # when speed > extent_ego_bbs_speed_threshold
    high_speed_extent_factor_ego_y = 1.2
    # Threshold to decide, when which bbs increase factor is used
    extent_ego_bbs_speed_threshold = 5
    # Forecast length in seconds when near a lane change
    forecast_length_lane_change = 1.1
    # Forecast length in seconds when not near a lane change
    default_forecast_length = 2.0
    # Factor to increase the ego vehicles bbs during forecasting when speed < extent_ego_bbs_speed_threshold
    slow_speed_extent_factor_ego = 1.0
    # Speed threshold to select which factor is used during other vehicle bbs forecasting
    extent_other_vehicles_bbs_speed_threshold = 1.0
    # Minimum extent of bbs, while forecasting other vehicles
    high_speed_min_extent_y_other_vehicle = 1.0
    # Extent factor to scale bbs during forecasting other vehicles in y direction
    high_speed_extent_y_factor_other_vehicle = 1.3
    # Extent factor to scale bbs during forecasting other vehicles in x direction
    high_speed_extent_x_factor_other_vehicle = 1.5
    # Minimum extent factor to scale bbs during forecasting other vehicles in x direction
    high_speed_min_extent_x_other_vehicle = 1.2
    # Minimum extent factor to scale bbs during forecasting other vehicles in x direction during lane changes to
    # account fore forecasting inaccuracies
    high_speed_min_extent_x_other_vehicle_lane_change = 2.0
    # Safety distance to be added to emergency braking distance
    braking_distance_calculation_safety_distance = 10
    # Minimum speed in m/s to prevent rolling back, when braking no throttle is applied
    minimum_speed_to_prevent_rolling_back = 0.5
    # Maximum seed in junctions in m/s
    max_speed_in_junction_urban = 25 / 3.6
    # Lookahead distance to check, whether the ego is close to a junction
    max_lookahead_to_check_for_junction = 30 * points_per_meter
    # Distance of the first checkpoint for TF++
    tf_first_checkpoint_distance = int(2.5 * points_per_meter)
    # Parameters to calculate how much the ego agent needs to cover a given distance. Values are taken from
    # the kinematic bicycle model
    compute_min_time_to_cover_distance_params = np.array([0.00904221, 0.00733342, -0.03744807, 0.0235038])
    # Distance to check for road_id/lane_id for RouteObstacle scenarios
    previous_road_lane_retrieve_distance = 100
    # Distance to check for road_id/lane_id for RouteObstacle scenarios
    next_road_lane_retrieve_distance = 100
    # Safety distance during checking if the path is free for RouteObstacle scenarios
    check_path_free_safety_distance = 10
    # Safety time headway during checking if the path is free for RouteObstacle scenarios
    check_path_free_safety_time = 0.2
    # Transition length for change lane in scenario ConstructionObstacle
    transition_smoothness_factor_construction_obstacle = 10.5 * points_per_meter
    # Check in x meters if there is lane change ahead
    minimum_lookahead_distance_to_compute_near_lane_change = 20 * points_per_meter
    # Check if did a lane change in the previous x meters
    check_previous_distance_for_lane_change = 15 * points_per_meter
    # Draw x meters of the route during debugging
    draw_future_route_till_distance = 500 * points_per_meter
    # Default minimum distance to process the route obstacle scenarios
    default_max_distance_to_process_scenario = 50
    safe_crossing_bicycle_flow = True
    # --- expert V1.1 Changes: HazardAtSideLane ---
    # If true enable smooth HazardAtSideLane merging
    hazard_at_side_lane_smooth_merging = True
    # Minimum distance to process HazardAtSideLane scenarios
    max_distance_to_process_hazard_at_side_lane = 25
    # Minimum distance to process HazardAtSideLaneTwoWays scenarios
    max_distance_to_process_hazard_at_side_lane_two_ways = 10
    # If true use the safe InterurbanActorFlow behavior
    safe_interurban_actor_flow = True
    # If true avoid colliding with the actor in EnterActorFlow
    avoid_rear_collision = True

    # --- TwoWays Scenarios with Obstacles ---
    # If true use shorter acceleration way for obstacle two ways scenarios
    shorter_acceleration_way_obstacle_two_ways = True
    # Maximum distance to start the overtaking maneuver
    max_distance_to_overtake_two_way_scnearios = int(8 * points_per_meter)
    # Default overtaking speed in m/s for all route obstacle scenarios
    default_overtake_speed = 40.0 / 3.6
    # Distance in meters at which two ways scenarios are considered finished
    distance_to_delete_scenario_in_two_ways = int(2 * points_per_meter)
    # Transition length for scenario ConstructionObstacleTwoWays to change lanes
    transition_length_construction_obstacle_two_ways = int(5 * points_per_meter)
    # Increase overtaking maneuver by distance in meters before the obstacle in ConstructionObstacleTwoWays
    add_before_construction_obstacle_two_ways = int(1.5 * points_per_meter)
    # Increase overtaking maneuver by distance in meters after the obstacle in ConstructionObstacleTwoWays
    add_after_construction_obstacle_two_ways = int(1.0 * points_per_meter)
    # How much to drive to the center of the opposite lane while handling ConstructionObstacleTwoWays
    factor_construction_obstacle_two_ways = 1.08

    # --- AccidentTwoWays ---
    # If true use shorter acceleration way for accident two ways scenarios
    shorter_acceleration_way_accident_two_ways = True
    # Increase overtaking maneuver by distance in meters after the obstacle in AccidentTwoWays
    add_after_accident_two_ways = int(-1.0 * points_per_meter)
    # Transition length for scenario AccidentTwoWays to change lanes
    transition_length_accident_two_ways = int(4.5 * points_per_meter)
    # Increase overtaking maneuver by distance in meters before the obstacle in AccidentTwoWays
    add_before_accident_two_ways = int(-1.0 * points_per_meter)

    # How much to drive to the center of the opposite lane while handling AccidentTwoWays
    factor_accident_two_ways = 1.0

    # --- ParkedObstacleTwoWays ---
    # If true use secured parked obstacle two ways behavior
    secured_parked_obstacle_two_ways = True
    # Transition length for scenario ParkedObstacleTwoWays to change lanes
    transition_length_parked_obstacle_two_ways = int(4 * points_per_meter)
    # Increase overtaking maneuver by distance in meters after the obstacle in ParkedObstacleTwoWays
    add_after_parked_obstacle_two_ways = int(0.5 * points_per_meter)
    # How much to drive to the center of the opposite lane while handling ParkedObstacleTwoWays
    factor_parked_obstacle_two_ways = 1.0
    # Increase overtaking maneuver by distance in meters before the obstacle in ParkedObstacleTwoWays
    add_before_parked_obstacle_two_ways = int(-0.5 * points_per_meter)

    # --- VehicleOpensDoorTwoWays ---
    # If true use secured vehicle opens door two ways behavior
    secured_vehicle_opens_door_two_ways = True
    # How much to drive to the center of the opposite lane while handling VehicleOpensDoorTwoWays
    factor_vehicle_opens_door_two_ways = 0.7
    # Overtaking speed in m/s for vehicle opens door two ways scenarios
    overtake_speed_vehicle_opens_door_two_ways = 8.75
    # Increase overtaking maneuver by distance in meters before the obstacle in VehicleOpensDoorTwoWays
    add_before_vehicle_opens_door_two_ways = int(-1.0 * points_per_meter)
    # Increase overtaking maneuver by distance in meters after the obstacle in VehicleOpensDoorTwoWays
    add_after_vehicle_opens_door_two_ways = int(-1.0 * points_per_meter)
    # Transition length for scenario VehicleOpensDoorTwoWays to change lanes
    transition_length_vehicle_opens_door_two_ways = int(5 * points_per_meter)

    # --- OppositeVehicleTakingPriority ---
    # Whether we will brake when other vehicles come from the side
    brake_on_side_emergency_vehicle = True

    # If true apply emergency brake for pedestrians and crossing bikers
    pedestrians_and_crossing_brikers_emergency_brake = True
    # Minimum visible pixels for pedestrian occlusion check
    pedestrian_occlusion_check_min_visible_pixels = 10
    # If true enable bikers occlusion check
    bikers_occlusion_check = True
    # Minimum visible pixels for bikers occlusion check
    bikers_occlusion_check_min_visible_pixels = 10
    # If true enable vehicle occlusion check
    vehicle_occlusion_check = True
    # Minimum visible pixels for vehicle occlusion check
    vehicle_occlusion_check_min_visible_pixels = 1
    # Minimum number of points for vehicle occlusion check
    vehicle_occlusion_check_min_num_points = 1

    # --- Accident ---
    # If true enable smooth merging for accident scenarios
    accident_smooth_merging = True
    # Distance to add before accident for smooth merging
    add_before_accident = int(1.5 * points_per_meter)
    # Distance to add after accident for smooth merging
    add_after_accident = int(1.2 * points_per_meter)
    # Transition smoothness distance for accident scenarios
    transition_smoothness_distance_accident = int(10 * points_per_meter)

    # --- ParkedObstacle ---
    # If true enable smooth merging for parked obstacle scenarios
    parked_obstacle_smooth_merging = True
    # Distance to add before parked obstacle for smooth merging
    add_before_parked_obstacle = int(1.0 * points_per_meter)
    # Distance to add after parked obstacle for smooth merging
    add_after_parked_obstacle = int(1.0 * points_per_meter)
    # Transition smoothness distance for parked obstacle scenarios
    transition_smoothness_distance_parked_obstacle = int(10 * points_per_meter)

    # --- ConstructionObstacle ---
    # If true enable smooth merging for construction obstacle scenarios
    construction_obstacle_smooth_merging = True
    # Distance to add before construction obstacle for smooth merging
    add_before_construction_obstacle = int(1.5 * points_per_meter)
    # Distance to add after construction obstacle for smooth merging
    add_after_construction_obstacle = int(1.5 * points_per_meter)
    # Transition smoothness distance for construction obstacle scenarios
    transition_smoothness_distance_construction_obstacle = int(10 * points_per_meter)

    # --- Junction Scenarios ---
    # If true use secured signalized junction left turn enter flow behavior
    secured_signalized_junction_left_turn_enter_flow = True
    # If true use secured non-signalized junction left turn enter flow behavior
    secured_non_signalized_junction_left_turn_enter_flow = True
    # If true use secured non-signalized junction left turn behavior
    secured_non_signalized_junction_left_turn = True
    # If true use secured signalized junction left turn behavior
    secured_signalized_junction_left_turn = True
    # If true use secured signalized junction right turn behavior
    secured_signalized_junction_right_turn = True
    # If true use secured non-signalized junction right turn behavior
    secured_non_signalized_junction_right_turn = True

    # --- Privileged Route Planner ---
    # Maximum distance to search ahead for updating ego route index (meters)
    ego_vehicles_route_point_search_distance = 4 * points_per_meter
    # Length to extend lane shift transition for YieldToEmergencyVehicle (meters)
    lane_shift_extension_length_for_yield_to_emergency_vehicle = 20 * points_per_meter
    # Distance over which lane shift transition is smoothed (meters)
    transition_smoothness_distance = 8 * points_per_meter
    # Distance over which lane shift transition is smoothed for InvadingTurn (meters)
    route_shift_start_distance_invading_turn = 15 * points_per_meter
    # Route shift end distance for InvadingTurn (meters)
    route_shift_end_distance_invading_turn = 10 * points_per_meter
    # Margin from fence when shifting route in InvadingTurn
    fence_avoidance_margin_invading_turn = 0.3
    # Minimum lane width to avoid early lane changes
    minimum_lane_width_threshold = 2.5
    # Spacing for checking and updating speed limits (meters)
    speed_limit_waypoints_spacing_check = 5 * points_per_meter
    # Maximum distance on route for detecting leading vehicles
    leading_vehicles_max_route_distance = 2.5
    # Maximum angle difference for detecting leading vehicles (meters)
    leading_vehicles_max_route_angle_distance = 35.0
    # Maximum radius for detecting any leading vehicles (meters)
    leading_vehicles_maximum_detection_radius = 80 * points_per_meter
    # Maximum distance on route for detecting trailing vehicles
    trailing_vehicles_max_route_distance = 3.0
    # Maximum route distance for trailing vehicles after lane change
    trailing_vehicles_max_route_distance_lane_change = 6.0
    # Maximum radius for detecting any trailing vehicles (meters)
    tailing_vehicles_maximum_detection_radius = 80 * points_per_meter
    # Maximum distance to check for lane changes when detecting trailing vehicles (meters)
    max_distance_lane_change_trailing_vehicles = 15 * points_per_meter
    # Distance to extend the end of the route to ensure checkpoints are available (meters)
    extra_route_length = 50

    # --- Debug Configuration ---
    # If true allow any visualization during debugging
    visualization_allowed = False
    # If true force debug visualization regardless of environment
    forced_debug_visualization = False
    # Default bounding box size for traffic warning signs.
    traffic_warning_bb_size = [1.186714768409729, 1.4352929592132568]
    # Default bounding box size for construction cones.
    construction_cone_bb_size = [0.1720348298549652, 0.1720348298549652]
    save_camera_pc = False
    save_instance_segmentation = False
    eval_expert = False

    @property
    def image_width(self):
        """Width of image."""
        width = self.num_cameras * self.camera_calibration[1]["width"]
        return int(width)

    @property
    def image_height(self):
        """Height of images."""
        height = self.camera_calibration[1]["height"]
        return int(height)

    @property
    def camera_3rd_person_calibration(self):
        from enum import IntEnum

        class DemoCameraOptions(IntEnum):
            CINEMATIC_LEFT = 1
            BEV = 2
            HIGH_BEV = 3
            HIGH_BEV_90 = 4
            PAPER_BEV = 5

        return {
            DemoCameraOptions.CINEMATIC_LEFT: {
                "image_size_x": "1980",
                "image_size_y": "786",
                "fov": "90",
                "x": -12,
                "y": -9,
                "z": 6,
                "pitch": -22,
                "yaw": 40,
            },
            DemoCameraOptions.BEV: {
                "image_size_x": "1980",
                "image_size_y": "1980",
                "fov": "90",
                "x": 18,
                "y": 0,
                "z": 30,
                "pitch": -90,
                "yaw": 0,
            },
            DemoCameraOptions.HIGH_BEV: {
                "image_size_x": "1980",
                "image_size_y": "786",
                "fov": "90",
                "x": 18,
                "y": 0,
                "z": 50,
                "pitch": -90,
                "yaw": 0,
            },
            DemoCameraOptions.HIGH_BEV_90: {
                "image_size_x": "1980",
                "image_size_y": "786",
                "fov": "90",
                "x": 18,
                "y": 0,
                "z": 50,
                "pitch": -90,
                "yaw": 90,
            },
            DemoCameraOptions.PAPER_BEV: {
                "image_size_x": "1980",
                "image_size_y": "1980",
                "fov": "90",
                "x": 18,
                "y": 0,
                "z": 50,
                "pitch": -90,
                "yaw": 0,
            },
        }[DemoCameraOptions.PAPER_BEV]

    @property
    def visualize_bb_ids(self):
        """If true visualize bounding box IDs during debugging."""
        if not self.visualization_allowed:
            return False
        if self.forced_debug_visualization:
            return True
        if self.is_on_slurm:
            return False
        return False

    @property
    def visualize_target_points(self):
        """If true visualize target points during debugging."""
        if not self.visualization_allowed:
            return False
        if self.forced_debug_visualization:
            return True
        if self.is_on_slurm:
            return False
        return False

    @property
    def visualize_route(self):
        if not self.visualization_allowed:
            return False
        if self.forced_debug_visualization:
            return True
        if self.is_on_slurm:
            return False
        return False

    @property
    def visualize_original_route(self):
        """If true visualize the original route."""
        if not self.visualization_allowed:
            return False
        if self.forced_debug_visualization:
            return True
        if self.is_on_slurm:
            return False
        return False

    @property
    def visualize_internal_data(self):
        if not self.visualization_allowed:
            return False
        if self.forced_debug_visualization:
            return True
        if self.is_on_slurm:
            return False
        return False

    @property
    def visualize_bounding_boxes(self):
        """If true bounding boxes should be visualized."""
        if not self.visualization_allowed:
            return False
        if self.forced_debug_visualization:
            return True
        if self.is_on_slurm:
            return False
        return False

    @property
    def visualize_traffic_lights_bounding_boxes(self):
        """If true traffic light bounding boxes should be visualized."""
        if not self.visualization_allowed:
            return False
        if self.forced_debug_visualization:
            return True
        if self.is_on_slurm:
            return False
        return False

    @property
    def visualize_source_lane(self):
        """If true source lane should be visualized."""
        if not self.visualization_allowed:
            return False
        if self.forced_debug_visualization:
            return True
        if self.is_on_slurm:
            return False
        return False

    @property
    def visualize_target_lane(self):
        """If true target lane should be visualized."""
        if not self.visualization_allowed:
            return False
        if self.forced_debug_visualization:
            return True
        if self.is_on_slurm:
            return False
        return False

    @property
    def visualize_adversarial(self):
        """If true adversarial elements should be visualized."""
        if not self.visualization_allowed:
            return False
        if self.forced_debug_visualization:
            return True
        if self.is_on_slurm:
            return False
        return False

    @property
    def visualize_planning_area(self):
        """If true planning area should be visualized."""
        if not self.visualization_allowed:
            return False
        if self.forced_debug_visualization:
            return True
        if self.is_on_slurm:
            return False
        return False

    @property
    def visualize_radar(self):
        """If true planning area should be visualized."""
        if not self.visualization_allowed:
            return False
        if self.forced_debug_visualization:
            return True
        if self.is_on_slurm:
            return False
        return False

    # --- Discontinuous Road Configuration ---
    # Handles scenarios where road continuity is interrupted

    # Activate for all scenarios
    discontinuous_road_reduce_speed = False

    # Maximum distance between route points
    max_distance_between_future_route_points = 0.25

    # Maximum future points to consider
    discontinuous_road_max_future_points = 5 / points_per_meter

    # Maximum speed in discontinuous road areas
    discontinuous_road_max_speed = 12.5

    # Maximum distance to check ahead
    discontinuous_road_max_future_check = 10 * points_per_meter

    # --- High Road Curvature Configuration ---
    # Handles sharp curves found in some Leaderboard towns

    # Activate for all scenarios
    high_road_curvature_reduce_speed = False

    # Maximum future points to consider
    high_road_curvature_max_future_points = 20 * points_per_meter

    # Curvature threshold for speed reduction
    high_road_curvature_threshold = 0.8

    # Maximum speed in high curvature areas
    high_road_curvature_max_speed = 12.5

    # --- Data Storage Configuration ---
    save_sensors = True

    @property
    def save_3rd_person_camera(self):
        """If true 3rd person camera images should be saved."""
        if self.is_on_slurm:
            return False
        # If true save the 3rd person camera images
        return False

    @property
    def path_3rd_person_camera_save(self):
        """Path where 3rd person camera images are saved."""
        return "data/carla_today/3rd_person/"

    @property
    def save_depth(self):
        """If true depth images should be saved."""
        if self.is_on_slurm:
            return True
        return True

    # Storage and collision avoidance parameters

    # PNG compression level for storing images
    png_storage_compression_level = 6

    # Use instance segmentation instead of semantic
    replace_semantics_segmentation_with_instance_segmentation = True

    # Avoid rear collisions when entering actor flow
    avoid_rear_collision_enter_actor_flow = True

    # Enhanced rear collision avoidance v2
    avoid_rear_collision_enter_actor_flow_v2 = True

    # Rear collision avoidance for interurban scenarios
    avoid_rear_collision_interurban_advanced_actor_flow = True

    # Avoid collisions with red light runners
    avoid_rear_collision_opposite_vehicle_running_red_light = True

    # Avoid collisions with priority vehicles
    avoid_rear_collision_opposite_vehicle_taking_priority = True

    # JPEG quality for 3rd person camera images
    jpg_quality_3rd_person = 100

    # --- Weather Configuration ---
    # JPEG compression quality settings for different weather conditions
    @cached_property
    def weather_jpeg_compression_quality(self):
        """JPEG compression quality settings for different weather conditions.

        Returns:
            dict: Weather condition to compression quality distribution mapping.
        """

        # For non-leaderboard datasets, use default high level compression
        if self.target_dataset not in [TargetDataset.NAVSIM_4CAMERAS]:
            return defaultdict(lambda: {30: 1.0})
        elif self.target_dataset == TargetDataset.WAYMO_E2E_2025_3CAMERAS:
            return defaultdict(lambda: {30: 1.0})

        def normalize(d):
            return {k: v / sum(d.values()) for k, v in d.items()}

        LOW_COMPRESSION = normalize({80: 0.25, 85: 0.5, 90: 0.25})
        MILD_COMPRESSION = normalize({70: 0.1, 75: 0.25, 80: 0.4, 90: 0.2})
        MEDIUM_COMPRESSION = normalize({70: 0.2, 75: 0.2, 80: 0.15, 90: 0.2})
        HIGH_COMPRESION = normalize({60: 0.05, 65: 0.1, 70: 0.1, 75: 0.2, 80: 0.1, 90: 0.2})
        VERY_HIGH_COMPRESION = normalize({55: 0.15, 65: 0.4, 75: 0.15, 90: 0.2})
        EXTREME_COMPRESSION = normalize({30: 0.2, 40: 0.35, 50: 0.15, 90: 0.2})

        if self.num_cameras == 6:
            LOW_COMPRESSION = normalize({80: 0.30, 85: 0.5, 90: 0.2})
            MILD_COMPRESSION = normalize({70: 0.15, 75: 0.25, 80: 0.4, 90: 0.15})
            MEDIUM_COMPRESSION = normalize({70: 0.25, 75: 0.2, 80: 0.15, 90: 0.15})
            HIGH_COMPRESION = normalize({60: 0.10, 65: 0.1, 70: 0.1, 75: 0.2, 80: 0.1, 90: 0.15})
            VERY_HIGH_COMPRESION = normalize({55: 0.20, 65: 0.4, 75: 0.15, 90: 0.15})
            EXTREME_COMPRESSION = normalize({30: 0.25, 40: 0.35, 50: 0.15, 90: 0.15})

        return {
            "ClearNight": MEDIUM_COMPRESSION,
            "ClearNoon": EXTREME_COMPRESSION,
            "ClearSunset": EXTREME_COMPRESSION,
            "ClearSunrise": EXTREME_COMPRESSION,
            # Cloudy weather
            "CloudyNight": MILD_COMPRESSION,
            "CloudyNoon": EXTREME_COMPRESSION,
            "CloudySunset": EXTREME_COMPRESSION,
            "CloudySunrise": EXTREME_COMPRESSION,
            # Dust storm
            "DustStorm": VERY_HIGH_COMPRESION,
            # Hard rain
            "HardRainNight": LOW_COMPRESSION,
            "HardRainNoon": MEDIUM_COMPRESSION,
            "HardRainSunset": MEDIUM_COMPRESSION,
            "HardRainSunrise": MEDIUM_COMPRESSION,
            # Mid rain
            "MidRainyNight": LOW_COMPRESSION,
            "MidRainyNoon": HIGH_COMPRESION,
            "MidRainSunset": HIGH_COMPRESION,
            "MidRainSunrise": HIGH_COMPRESION,
            # Soft rain
            "SoftRainNight": MILD_COMPRESSION,
            "SoftRainNoon": VERY_HIGH_COMPRESION,
            "SoftRainSunset": VERY_HIGH_COMPRESION,
            "SoftRainSunrise": VERY_HIGH_COMPRESION,
            # Wet cloudy
            "WetCloudyNight": LOW_COMPRESSION,
            "WetCloudyNoon": VERY_HIGH_COMPRESION,
            "WetCloudySunset": VERY_HIGH_COMPRESION,
            "WetCloudySunrise": VERY_HIGH_COMPRESION,
            # Wet
            "WetNight": MILD_COMPRESSION,
            "WetNoon": VERY_HIGH_COMPRESION,
            # Foggy cloudy
            "FoggyCloudyNight": MEDIUM_COMPRESSION,
            "FoggyCloudyNoon": MEDIUM_COMPRESSION,
            "FoggyCloudySunset": MEDIUM_COMPRESSION,
            "FoggyCloudySunrise": MEDIUM_COMPRESSION,
            # Foggy Wet cloudy
            "FoggyWetCloudyNight": MEDIUM_COMPRESSION,
            "FoggyWetCloudyNoon": MEDIUM_COMPRESSION,
            "FoggyWetCloudySunset": MEDIUM_COMPRESSION,
            "FoggyWetCloudySunrise": MEDIUM_COMPRESSION,
            # Foggy Wet
            "FoggyWetNoon": MEDIUM_COMPRESSION,
            # Foggy Soft Rain
            "FoggySoftRainNight": MEDIUM_COMPRESSION,
            "FoggySoftRainNoon": MEDIUM_COMPRESSION,
            "FoggySoftRainSunset": MEDIUM_COMPRESSION,
            "FoggySoftRainSunrise": MEDIUM_COMPRESSION,
            # Foggy Hard Rain
            "FoggyHardRainNight": LOW_COMPRESSION,
            # Custom weather
            "Custom0": HIGH_COMPRESION,
            "Custom9": HIGH_COMPRESION,
            "Custom10": LOW_COMPRESSION,
            "Custom11": HIGH_COMPRESION,
            "Custom12": HIGH_COMPRESION,
            "Custom13": LOW_COMPRESSION,
            "Custom14": HIGH_COMPRESION,
            "Custom15": HIGH_COMPRESION,
            "Custom19": LOW_COMPRESSION,
            "Custom20": LOW_COMPRESSION,
            "Custom21": LOW_COMPRESSION,
        }

    @property
    def weather_settings(self):
        ret = weathers.WEATHER_SETTINGS
        if self.target_dataset == TargetDataset.NAVSIM_4CAMERAS:
            # Keep only clear weather
            return {k: v for k, v in ret.items() if weathers.WEATHER_VISIBILITY_MAPPING[k] == WeatherVisibility.CLEAR}
        elif self.target_dataset == TargetDataset.WAYMO_E2E_2025_3CAMERAS:
            # Remove foggy weather
            return {k: v for k, v in ret.items() if v["fog_density"] < 20.0}
        return ret
