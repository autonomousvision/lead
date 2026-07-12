"""
Config class that contains all the hyperparameters needed to build any model.
"""

import os
import carla
import numpy as np


class GlobalConfig:
  """
  Config class that contains all the hyperparameters needed to build any model.
  """
  # Colors used for drawing during debugging
  future_route_color = carla.Color(0, 1, 0)
  other_vehicles_forecasted_bbs_color = carla.Color(0, 0, 1, 1)
  leading_vehicle_color = carla.Color(1, 0, 0, 0)
  trailing_vehicle_color = carla.Color(1, 1, 1, 0)
  ego_vehicle_bb_color = carla.Color(0, 0, 0, 1)
  pedestrian_forecasted_bbs_color = carla.Color(0, 0, 1, 1)
  red_traffic_light_color = carla.Color(0, 1, 0, 1)
  green_traffic_light_color = carla.Color(1, 0, 0, 1)
  cleared_stop_sign_color = carla.Color(0, 1, 0, 1)
  uncleared_stop_sign_color = carla.Color(1, 0, 0, 1)
  ego_vehicle_forecasted_bbs_hazard_color = carla.Color(1, 0, 0, 0)
  ego_vehicle_forecasted_bbs_normal_color = carla.Color(0, 1, 0, 0)

  def __init__(self):
    """ base architecture configurations """
    # -----------------------------------------------------------------------------
    # Autopilot
    # -----------------------------------------------------------------------------
    # Frame rate used for the bicycle models in the autopilot
    self.bicycle_frame_rate = 20
    self.steer_noise = 1e-3  # Noise added to expert steering angle
    # Distance of obstacles (in meters) in which we will check for collisions
    self.detection_radius = 50.0
    self.num_route_points_saved = 20  # Number of future route points we save per step.
    # Distance of traffic lights considered relevant (in meters)
    self.light_radius = 64.0
    # Bounding boxes in this radius around the car will be saved in the dataset.
    self.bb_save_radius = 64.0
    # Ratio between the the speed limit / curvature dependent speed limit and the target speed.
    # By default the other vehicles drive with 70 % of the speed limit. To avoid collisions we have to be a bit faster.
    self.ratio_target_speed_limit = 0.72
    # Maximum number of ticks the agent doesn't take any action. The maximum is 179 and it's speed must be >0.1.
    # After taking 180 ticks no action the route ends with an AgentBlockTest infraction.
    self.max_blocked_ticks = 170
    # Minimum walker speed
    self.min_walker_speed = 0.5
    # Time in seconds to draw the things during debugging.
    self.draw_life_time = 0.051
    # Points sampled per meter when interpolating route.
    self.points_per_meter = 10
    # FPS of the simulation
    self.fps = 20.0
    # Inverse of the FPS
    self.fps_inv = 1.0 / self.fps
    # Distance to the stop sign, when the previous stop sign is uncleared
    self.unclearing_distance_to_stop_sign = 10
    # Distance to the stop sign, when the previous stop sign is cleared
    self.clearing_distance_to_stop_sign = 3.0
    # IDM minimum distance for stop signs
    self.idm_stop_sign_minimum_distance = 2.0
    # IDM desrired time headway for stop signs
    self.idm_stop_sign_desired_time_headway = 0.1
    # IDM minimum distance for red lights
    self.idm_red_light_minimum_distance = 6.0
    # IDM desrired time headway for red lights
    self.idm_red_light_desired_time_headway = 0.1
    # IDM minimum distance for pedestrians
    self.idm_pedestrian_minimum_distance = 4.0
    # IDM desrired time headway for pedestrians
    self.idm_pedestrian_desired_time_headway = 0.1
    # IDM minimum distance for bicycles
    self.idm_bicycle_minimum_distance = 4.0
    # IDM desrired time headway for bicycles
    self.idm_bicycle_desired_time_headway = 0.25
    # IDM minimum distance for leading vehicles
    self.idm_leading_vehicle_minimum_distance = 4.0
    # IDM desrired time headway for leading vehicles
    self.idm_leading_vehicle_time_headway = 0.25
    # IDM minimum distance for two way scenarios
    self.idm_two_way_scenarios_minimum_distance = 2.0
    # IDM desrired time headway for two way scenarios
    self.idm_two_way_scenarios_time_headway = 0.1
    # Boundary time - the integration won’t continue beyond it.
    self.idm_t_bound = 0.05
    # IDM maximum accelaration parameter per frame
    self.idm_maximum_acceleration = 24.0
    # The following parameters were determined by measuring the vehicle's braking performance.
    # IDM maximum deceleration parameter per frame while driving slow
    self.idm_comfortable_braking_deceleration_low_speed = 8.7
    # IDM maximum deceleration parameter per frame while driving fast
    self.idm_comfortable_braking_deceleration_high_speed = 3.72
    # Threshold to determine, when to use idm_comfortable_braking_deceleration_low_speed and
    # idm_comfortable_braking_deceleration_high_speed
    self.idm_comfortable_braking_deceleration_threshold = 6.02
    # IDM acceleration exponent (default = 4.)
    self.idm_acceleration_exponent = 4.0
    # Minimum extent for pedestrian during bbs forecasting
    self.pedestrian_minimum_extent = 1.5
    # Factor to increase the ego vehicles bbs in driving direction during forecasting
    # when speed > extent_ego_bbs_speed_threshold
    self.high_speed_extent_factor_ego_x = 1.3
    # Factor to increase the ego vehicles bbs in y direction during forecasting
    # when speed > extent_ego_bbs_speed_threshold
    self.high_speed_extent_factor_ego_y = 1.2
    # Threshold to decide, when which bbs increase factor is used
    self.extent_ego_bbs_speed_threshold = 5
    # Forecast length in seconds when near a lane change
    self.forecast_length_lane_change = 1.1
    # Forecast length in seconds when not near a lane change
    self.default_forecast_length = 2.0
    # Factor to increase the ego vehicles bbs during forecasting when speed < extent_ego_bbs_speed_threshold
    self.slow_speed_extent_factor_ego = 1.0
    # Speed threshold to select which factor is used during other vehicle bbs forecasting
    self.extent_other_vehicles_bbs_speed_threshold = 1.0
    # Minimum extent of bbs, while forecasting other vehicles
    self.high_speed_min_extent_y_other_vehicle = 1.0
    # Extent factor to scale bbs during forecasting other vehicles in y direction
    self.high_speed_extent_y_factor_other_vehicle = 1.3
    # Extent factor to scale bbs during forecasting other vehicles in x direction
    self.high_speed_extent_x_factor_other_vehicle = 1.5
    # Minimum extent factor to scale bbs during forecasting other vehicles in x direction
    self.high_speed_min_extent_x_other_vehicle = 1.2
    # Minimum extent factor to scale bbs during forecasting other vehicles in x direction during lane changes to
    # account fore forecasting inaccuracies
    self.high_speed_min_extent_x_other_vehicle_lane_change = 2.0
    # Safety distance to be added to emergency braking distance
    self.braking_distance_calculation_safety_distance = 10
    # Minimum speed in m/s to prevent rolling back, when braking no throttle is applied
    self.minimum_speed_to_prevent_rolling_back = 0.5
    # Maximum seed in junctions in m/s
    self.max_speed_in_junction = 64 / 3.6
    # Lookahead distance to check, whether the ego is close to a junction
    self.max_lookahead_to_check_for_junction = 30 * self.points_per_meter
    # Distance of the first checkpoint for TF++
    self.tf_first_checkpoint_distance = int(2.5 * self.points_per_meter)
    # Parameters to calculate how much the ego agent needs to cover a given distance. Values are taken from
    # the kinematic bicycle model
    self.compute_min_time_to_cover_distance_params = np.array([0.00904221, 0.00733342, -0.03744807, 0.0235038])
    # Distance to check for road_id/lane_id for RouteObstacle scenarios
    self.previous_road_lane_retrieve_distance = 100
    # Safety distance during checking if the path is free for RouteObstacle scenarios
    self.check_path_free_safety_distance = 10
    # Safety time headway during checking if the path is free for RouteObstacle scenarios
    self.check_path_free_safety_time = 0.2
    # Transition length for change lane in scenario ConstructionObstacle
    self.transition_smoothness_factor_construction_obstacle = 10.5 * self.points_per_meter
    # Check in x meters if there is lane change ahead
    self.minimum_lookahead_distance_to_compute_near_lane_change = 20 * self.points_per_meter
    # Check if did a lane change in the previous x meters
    self.check_previous_distance_for_lane_change = 15 * self.points_per_meter
    # Draw x meters of the route during debugging
    self.draw_future_route_till_distance = 50 * self.points_per_meter
    # Default minimum distance to process the route obstacle scenarios
    self.default_max_distance_to_process_scenario = 50
    # Minimum distance to process HazardAtSideLane
    self.max_distance_to_process_hazard_at_side_lane = 25
    # Minimum distance to process HazardAtSideLaneTwoWays
    self.max_distance_to_process_hazard_at_side_lane_two_ways = 10
    # Transition length for sceneario AccidentTwoWays to change lanes
    self.transition_length_accident_two_ways = int(4 * self.points_per_meter)
    # Transition length for sceneario ConstructionObstacleTwoWays to change lanes
    self.transition_length_construction_obstacle_two_ways = int(4 * self.points_per_meter)
    # Transition length for sceneario ParkedObstacleTwoWays to change lanes
    self.transition_length_parked_obstacle_two_ways = int(4 * self.points_per_meter)
    # Transition length for sceneario VehicleOpensDoorTwoWays to change lanes
    self.transition_length_vehicle_opens_door_two_ways = int(4 * self.points_per_meter)
    # Increase overtaking maneuver by distance in meters in the scenario AccidentTwoWays before the obstacle
    self.add_before_accident_two_ways = int(-1.5 * self.points_per_meter)
    # Increase overtaking maneuver by distance in meters in the scenario ConstructionObstacleTwoWays
    # before the obstacle
    self.add_before_construction_obstacle_two_ways = int(1.5 * self.points_per_meter)
    # Increase overtaking maneuver by distance in meters in the scenario ParkedObstacleTwoWays before the obstacle
    self.add_before_parked_obstacle_two_ways = int(-0.5 * self.points_per_meter)
    # Increase overtaking maneuver by distance in meters in the scenario VehicleOpensDoorTwoWays before the obstacle
    self.add_before_vehicle_opens_door_two_ways = int(-2.0 * self.points_per_meter)
    # Increase overtaking maneuver by distance in meters in the scenario AccidentTwoWays after the obstacle
    self.add_after_accident_two_ways = int(-1.5 * self.points_per_meter)
    # Increase overtaking maneuver by distance in meters in the scenario ConstructionObstacleTwoWays
    # after the obstacle
    self.add_after_construction_obstacle_two_ways = int(1.5 * self.points_per_meter)
    # Increase overtaking maneuver by distance in meters in the scenario ParkedObstacleTwoWays after the obstacle
    self.add_after_parked_obstacle_two_ways = int(-0.5 * self.points_per_meter)
    # Increase overtaking maneuver by distance in meters in the scenario VehicleOpensDoorTwoWays after the obstacle
    self.add_after_vehicle_opens_door_two_ways = int(-2.0 * self.points_per_meter)
    # How much to drive to the center of the opposite lane while handling the scenario AccidentTwoWays
    self.factor_accident_two_ways = 1.0
    # How much to drive to the center of the opposite lane while handling the scenario ConstructionObstacleTwoWays
    self.factor_construction_obstacle_two_ways = 1.0
    # How much to drive to the center of the opposite lane while handling the scenario ParkedObstacleTwoWays
    self.factor_parked_obstacle_two_ways = 0.6
    # How much to drive to the center of the opposite lane while handling the scenario VehicleOpensDoorTwoWays
    self.factor_vehicle_opens_door_two_ways = 0.475
    # Maximum distance to start the overtaking maneuver
    self.max_distance_to_overtake_two_way_scnearios = int(8 * self.points_per_meter)
    # Overtaking speed in m/s for vehicle opens door two ways scenarios
    self.overtake_speed_vehicle_opens_door_two_ways = 40. / 3.6
    # Default overtaking speed in m/s for all route obstacle scenarios
    self.default_overtake_speed = 50. / 3.6
    # Distance in meters at which two ways scenarios are considered finished
    self.distance_to_delete_scenario_in_two_ways = int(2 * self.points_per_meter)
    # -----------------------------------------------------------------------------
    # Longitudinal Linear Regression controller
    # -----------------------------------------------------------------------------
    # These parameters are tuned with Bayesian Optimization on a test track
    # Minimum threshold for target speed (< 1 km/h) for longitudinal linear regression controller.
    self.longitudinal_linear_regression_minimum_target_speed = 0.278
    # Coefficients of the linear regression model used for throttle calculation.
    self.longitudinal_linear_regression_params = np.array([
        1.1990342347353184, -0.8057602384167799, 1.710818710950062, 0.921890257450335, 1.556497522998393,
        -0.7013479734904027, 1.031266635497984
    ])
    # Maximum acceleration rate (approximately 1.9 m/tick) for the longitudinal linear regression controller.
    self.longitudinal_linear_regression_maximum_acceleration = 1.89
    # Maximum deceleration rate (approximately -4.82 m/tick) for the longitudinal linear regression controller.
    self.longitudinal_linear_regression_maximum_deceleration = -4.82
    # -----------------------------------------------------------------------------
    # Longitudinal PID controller
    # -----------------------------------------------------------------------------
    # These parameters are tuned with Bayesian Optimization on a test track
    # Gain factor for proportional control for longitudinal pid controller.
    self.longitudinal_pid_proportional_gain = 1.0016429066823955
    # Gain factor for derivative control for longitudinal pid controller.
    self.longitudinal_pid_derivative_gain = 1.5761818624794222
    # Gain factor for integral control for longitudinal pid controller.
    self.longitudinal_pid_integral_gain = 0.2941563856687906
    # Maximum length of the window for cumulative error for longitudinal pid controller.
    self.longitudinal_pid_max_window_length = 0
    # Scaling factor for speed error based on current speed for longitudinal pid controller.
    self.longitudinal_pid_speed_error_scaling = 0.0
    # Ratio to determine when to apply braking for longitudinal pid controller.
    self.longitudinal_pid_braking_ratio = 1.0324622059220139
    # Minimum threshold for target speed (< 1 km/h) for longitudinal pid controller.
    self.longitudinal_pid_minimum_target_speed = 0.278
    # -----------------------------------------------------------------------------
    # Lateral PID controller
    # -----------------------------------------------------------------------------
    # These parameters are tuned with Bayesian Optimization on a test track
    # The proportional gain for the lateral PID controller.
    self.lateral_pid_kp = 3.118357247806046
    # The derivative gain for the lateral PID controller.
    self.lateral_pid_kd = 1.3782508892109167
    # The integral gain for the lateral PID controller.
    self.lateral_pid_ki = 0.6406067986034124
    # The scaling factor used in the calculation of the lookahead distance based on the current speed.
    self.lateral_pid_speed_scale = 0.9755321901954155
    # The offset used in the calculation of the lookahead distance based on the current speed.
    self.lateral_pid_speed_offset = 1.9152884533402488
    # The default lookahead distance for the lateral PID controller.
    self.lateral_pid_default_lookahead = 2.4 * self.points_per_meter
    # The speed threshold (in km/h) for switching between the default and variable lookahead distance.
    self.lateral_pid_speed_threshold = 2.3150102938235136 * self.points_per_meter
    # The size of the sliding window used to store the error history for the lateral PID controller.
    self.lateral_pid_window_size = 6
    # The minimum allowed lookahead distance for the lateral PID controller.
    self.lateral_pid_minimum_lookahead_distance = 2.4 * self.points_per_meter
    # The maximum allowed lookahead distance for the lateral PID controller.
    self.lateral_pid_maximum_lookahead_distance = 10.5 * self.points_per_meter
    # -----------------------------------------------------------------------------
    # Kinematic Bicycle Model
    # -----------------------------------------------------------------------------
    #  Time step for the model (20 frames per second).
    self.time_step = 1. / 20.
    # Kinematic bicycle model parameters tuned from World on Rails.
    # Distance from the rear axle to the front axle of the vehicle.
    self.front_wheel_base = -0.090769015
    # Distance from the rear axle to the center of the rear wheels.
    self.rear_wheel_base = 1.4178275
    # Gain factor for steering angle to wheel angle conversion.
    self.steering_gain = 0.36848336
    # Deceleration rate when braking (m/s^2) of other vehicles.
    self.brake_acceleration = -4.952399
    # Acceleration rate when throttling (m/s^2) of other vehicles.
    self.throttle_acceleration = 0.5633837
    # Tuned parameters for the polynomial equations modeling speed changes
    # Numbers are tuned parameters for the polynomial equations below using
    # a dataset where the car drives on a straight highway, accelerates to
    # and brakes again
    # Coefficients for polynomial equation estimating speed change with throttle input for ego model.
    self.throttle_values = np.array([
        9.63873001e-01, 4.37535692e-04, -3.80192912e-01, 1.74950069e+00, 9.16787414e-02, -7.05461530e-02,
        -1.05996152e-03, 6.71079346e-04
    ])
    # Coefficients for polynomial equation estimating speed change with brake input for the ego model.
    self.brake_values = np.array([
        9.31711370e-03, 8.20967431e-02, -2.83832427e-03, 5.06587474e-05, -4.90357228e-07, 2.44419284e-09,
        -4.91381935e-12
    ])
    # Minimum throttle value that has an affect during forecasting the ego vehicle.
    self.throttle_threshold_during_forecasting = 0.3
    # -----------------------------------------------------------------------------
    # Privileged Route Planner
    # -----------------------------------------------------------------------------
    # Max distance to search ahead for updating ego route index  in meters.
    self.ego_vehicles_route_point_search_distance = 4 * self.points_per_meter
    # Length to extend lane shift transition for YieldToEmergencyVehicle  in meters.
    self.lane_shift_extension_length_for_yield_to_emergency_vehicle = 20 * self.points_per_meter
    # Distance over which lane shift transition is smoothed  in meters.
    self.transition_smoothness_distance = 8 * self.points_per_meter
    # Distance over which lane shift transition is smoothed for InvadingTurn  in meters.
    self.route_shift_start_distance_invading_turn = 15 * self.points_per_meter
    self.route_shift_end_distance_invading_turn = 10 * self.points_per_meter
    # Margin from fence when shifting route in InvadingTurn.
    self.fence_avoidance_margin_invading_turn = 0.3
    # Minimum lane width to avoid early lane changes.
    self.minimum_lane_width_threshold = 2.5
    # Spacing for checking and updating speed limits  in meters.
    self.speed_limit_waypoints_spacing_check = 5 * self.points_per_meter
    # Max distance on route for detecting leading vehicles.
    self.leading_vehicles_max_route_distance = 2.5
    # Max angle difference for detecting leading vehicles  in meters.
    self.leading_vehicles_max_route_angle_distance = 35.
    # Max radius for detecting any leading vehicles in meters.
    self.leading_vehicles_maximum_detection_radius = 80 * self.points_per_meter
    # Max distance on route for detecting trailing vehicles.
    self.trailing_vehicles_max_route_distance = 3.0
    # Max route distance for trailing vehicles after lane change.
    self.trailing_vehicles_max_route_distance_lane_change = 6.0
    # Max radius for detecting any trailing vehicles in meters.
    self.tailing_vehicles_maximum_detection_radius = 80 * self.points_per_meter
    # Max distance to check for lane changes when detecting trailing vehicles in meters.
    self.max_distance_lane_change_trailing_vehicles = 15 * self.points_per_meter
    # Distance to extend the end of the route in meters. This makes sure we always have checkpoints,
    # also at the end of the route.
    self.extra_route_length = 50
    # -----------------------------------------------------------------------------
    # DataAgent
    # -----------------------------------------------------------------------------
    # Max and min values by which the augmented camera is shifted left and right
    self.camera_translation_augmentation_min = -1.0
    self.camera_translation_augmentation_max = 1.0
    # Max and min values by which the augmented camera is rotated around the yaw
    # Numbers are in degree
    self.camera_rotation_augmentation_min = -5.0
    self.camera_rotation_augmentation_max = 5.0
    # Every data_save_freq frame the data is stored during training
    # Set to one for backwards compatibility. Released dataset was collected with 5
    self.data_save_freq = 5
    # LiDAR compression parameters
    self.point_format = 0  # LARS point format used for storing
    self.point_precision = 0.01  # Precision up to which LiDAR points are stored

    # -----------------------------------------------------------------------------
    # Sensor config
    # -----------------------------------------------------------------------------
    self.lidar_pos = [0.0, 0.0, 2.5]  # x, y, z mounting position of the LiDAR
    self.lidar_rot = [0.0, 0.0, -90.0]  # Roll Pitch Yaw of LiDAR in degree
    self.lidar_rotation_frequency = 10  # Number of Hz at which the Lidar operates
    # Number of points the LiDAR generates per second.
    # Change in proportion to the rotation frequency.
    self.lidar_points_per_second = 600000
    self.camera_pos = [-1.5, 0.0, 2.0]  # x, y, z mounting position of the camera
    self.camera_rot_0 = [0.0, 0.0, 0.0]  # Roll Pitch Yaw of camera 0 in degree

    # Therefore their size is smaller
    self.camera_width = 1024  # Camera width in pixel during data collection and eval (affects sensor agent)
    self.camera_height = 512  # Camera height in pixel during data collection and eval (affects sensor agent)
    self.camera_fov = 110

    # Crop the image during training to the values below. also affects the transformer tokens self.img_vert_anchors
    self.crop_image = True
    self.cropped_height = 384  # crops off the bottom part
    self.cropped_width = 1024  # crops off both sides symmetrically

    # -----------------------------------------------------------------------------
    # Dataloader
    # -----------------------------------------------------------------------------
    self.carla_fps = 20  # Simulator Frames per second
    self.seq_len = 1  # input timesteps
    # use different seq len for image and lidar
    self.img_seq_len = 1
    self.lidar_seq_len = 1
    # Number of initial frames to skip during data loading
    self.skip_first = int(2.5 * self.carla_fps) // self.data_save_freq
    self.pred_len = int(2.0 * self.carla_fps) // self.data_save_freq  # number of future waypoints predicted
    # Width and height of the LiDAR grid that the point cloud is voxelized into.
    self.lidar_resolution_width = 256
    self.lidar_resolution_height = 256
    # Crop the BEV semantics, bounding boxes and LiDAR range to the values above. Also affects self.lidar_vert_anchors
    self.crop_bev = False
    # If true, cuts BEV off behind the vehicle. If False, cuts off front and back symmetrically
    self.crop_bev_height_only_from_behind = False
    # Number of LiDAR hits a bounding box needs for it to be a valid label
    self.num_lidar_hits_for_detection_walker = 1
    self.num_lidar_hits_for_detection_car = 1
    # How many pixels make up 1 meter in BEV grids
    # 1 / pixels_per_meter = size of pixel in meters
    self.pixels_per_meter = 4.0
    # Pixels per meter used in the semantic segmentation map during data collection.
    # On Town 13 2.0 is the highest that opencv can handle.
    self.pixels_per_meter_collection = 2.0
    # Max number of LiDAR points per pixel in voxelized LiDAR
    self.hist_max_per_pixel = 5
    # Height at which the LiDAR points are split into the 2 channels.
    # Is relative to lidar_pos[2]
    self.lidar_split_height = 0.2
    self.realign_lidar = True
    self.use_ground_plane = False
    # Max and minimum LiDAR ranges used for voxelization
    self.min_x = -32
    self.max_x = 32
    if self.crop_bev and not self.crop_bev_height_only_from_behind:
      assert self.max_x == -self.min_x  # If we cut the bev semantics symetrically, we also need a symmetric lidar range
    self.min_y = -32
    self.max_y = 32
    self.min_z = -4
    self.max_z = 4
    self.min_z_projection = -10
    self.max_z_projection = 14

    # Angle bin thresholds
    self.angle_bins = [-0.375, -0.125, 0.125, 0.375]
    # Discrete steering angles
    self.angles = [-0.5, -0.25, 0.0, 0.25, 0.5]
    # Whether to estimate the class weights or use the default from the config.
    self.estimate_class_distributions = False
    self.estimate_semantic_distribution = False
    # Class weights applied to the cross entropy losses
    self.angle_weights = [
        204.25901201602136, 7.554315623148331, 0.21388916461734406, 5.476446162657503, 207.86684782608697
    ]
    # We don't use weighting here
    self.semantic_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    self.bev_semantic_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    # v4 target speeds (0.72*speed limits) plus extra classes for obstacle scenarios and intersections
    self.target_speeds = [0.0, 4.0, 8.0, 10, 13.88888888, 16, 17.77777777, 20]

    self.target_speed_bins = [x + 0.001 for x in self.target_speeds[1:]]  # not used with two hot encodings
    self.target_speed_weights = [1.0] * (len(self.target_speeds))

    # -----------------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------------
    self.local_rank = -999
    self.id = 'transfuser'  # Unique experiment identifier.
    self.epochs = 31  # Number of epochs to train
    self.lr = 3e-4  # Learning rate used for training
    self.batch_size = 16  # Batch size used during training
    self.logdir = ''  # Directory to log data to.
    self.load_file = None  # File to continue training from
    self.setting = 'all'  # Setting used for training
    self.root_dir = ''  # Dataset root dir
    # When to reduce the learning rate for the first and second  time
    self.schedule_reduce_epoch_01 = 30
    self.schedule_reduce_epoch_02 = 40
    self.val_every = 5  # Validation frequency in epochs
    self.sync_batch_norm = 0  # Whether batch norm was synchronized between GPUs
    # Whether zero_redundancy_optimizer was used during training
    self.zero_redundancy_optimizer = 1
    self.use_disk_cache = 0  # Whether disc cache was used during training
    self.detect_boxes = 1  # Whether to use the bounding box auxiliary task
    self.train_sampling_rate = 1  # We train on every n th sample on the route
    # Number of route points we use for prediction in TF or input in planT
    self.num_route_points = 20
    self.augment_percentage = 0.5  # Probability of the augmented sample being used.
    self.learn_origin = 1  # Whether to learn the origin of the waypoints or use 0 / 0
    self.augment = 1  # Whether to use rotation and translation augmentation
    # At which interval to save debug files to disk during training
    self.train_debug_save_freq = 1
    self.backbone = 'transFuser'  # Vision backbone architecture used
    self.use_velocity = 1  # Whether to use the velocity as input to the network
    self.image_architecture = 'regnety_032'  # Image architecture used in the backbone resnet34, regnety_032
    self.lidar_architecture = 'regnety_032'  # LiDAR architecture used in the backbone resnet34, regnety_032
    # Whether to classify target speeds and regress a path as output representation.
    self.use_controller_input_prediction = True
    # Whether to use the direct control predictions for driving
    self.inference_direct_controller = True
    # Label smoothing applied to the cross entropy losses
    self.label_smoothing_alpha = 0.1
    # Whether to use focal loss instead of cross entropy for classification
    self.use_focal_loss = False
    # Gamma hyperparameter of focal loss
    self.focal_loss_gamma = 2.0
    # Learning rate decay, applied when using multi-step scheduler
    self.multi_step_lr_decay = 0.1
    # Whether to use a cosine schedule instead of the linear one.
    self.use_cosine_schedule = True
    # Epoch of the first restart
    self.cosine_t0 = 1
    # Multiplier applied to t0 after every restart
    self.cosine_t_mult = 2
    # Weights applied to each of these losses, when combining them
    self.detailed_loss_weights = {
        'loss_wp': 1.0,
        'loss_target_speed': 1.0,
        'loss_checkpoint': 1.0,
        'loss_semantic': 1.0,
        'loss_bev_semantic': 1.0,
        'loss_depth': 1.0,
        'loss_center_heatmap': 1.0,
        'loss_wh': 1.0,
        'loss_offset': 1.0,
        'loss_yaw_class': 1.0,
        'loss_yaw_res': 1.0,
        'loss_velocity': 1.0,
        'loss_brake': 1.0,
        'loss_forcast': 0.2,
        'loss_selection': 0.0,
    }
    self.root_dir = ''
    self.val_towns = []
    # NOTE currently leads to inf gradients do not use! Whether to use automatic mixed precision during training.
    self.use_amp = 0
    self.use_grad_clip = 0  # Whether to clip the gradients
    self.grad_clip_max_norm = 1.0  # Max value for the gradients if gradient clipping is used.
    self.use_color_aug = 1  # Whether to apply image color based augmentations
    self.color_aug_prob = 0.5  # With which probability to apply the different image color augmentations.
    self.use_cutout = False  # Whether to use cutout as a data augmentation technique during training.
    self.lidar_aug_prob = 1.0  # Probability with which data augmentation is applied to the LiDAR image.
    self.freeze_backbone = False  # Whether to freeze the image backbone during training. Useful for 2 stage training.
    self.learn_multi_task_weights = False  # Whether to learn the multi-task weights
    self.use_bev_semantic = True  # Whether to use bev semantic segmentation as auxiliary loss for training.
    self.use_depth = True  # Whether to use depth prediction as auxiliary loss for training.
    self.num_repetitions = 1  # How many repetitions of the dataset we train with.
    self.continue_epoch = True  # Whether to continue the training from the loaded epoch or from 0.

    self.smooth_route = True  # Whether to smooth the route points with a spline.
    self.ignore_index = -999  # Index to ignore for future bounding box prediction task.
    self.use_speed_weights = False  # Whether to weight target speed classes
    self.use_optim_groups = False  # Whether to use optimizer groups to exclude some parameters from weight decay
    self.weight_decay = 0.01  # Weight decay coefficient used during training
    self.use_label_smoothing = False  # Whether to use label smoothing in the classification losses
    self.use_twohot_target_speeds = True  # Whether to use two hot encoding for the target speed classification
    self.compile = False  # Whether to apply torch.compile to the model.
    self.compile_mode = 'default'  # Compile mode for torch.compile

    # -----------------------------------------------------------------------------
    # PID controller
    # -----------------------------------------------------------------------------
    # We are minimizing the angle to the waypoint that is at least aim_distance
    # meters away, while driving
    self.aim_distance_fast = 3.0
    self.aim_distance_slow = 2.25
    # Meters per second threshold switching between aim_distance_fast and
    # aim_distance_slow
    self.aim_distance_threshold = 5.5
    # Controller
    self.turn_kp = 1.25
    self.turn_ki = 0.75
    self.turn_kd = 0.3
    self.turn_n = 20  # buffer size

    self.speed_kp = 1.75
    self.speed_ki = 1.0
    self.speed_kd = 2.0
    self.speed_n = 20  # buffer size

    self.brake_speed = 0.4  # desired speed below which brake is triggered
    # ratio of speed to desired speed at which brake is triggered
    self.brake_ratio = 1.1
    self.clip_delta = 1.0  # maximum change in speed input to longitudinal controller
    self.clip_throttle = 1.0  # Maximum throttle allowed by the controller

    # Numbers for the lateral PID controller
    self.lateral_k_p = 3.118357247806046
    self.lateral_k_d = 1.3782508892109167
    self.lateral_k_i = 0.6406067986034124
    self.lateral_speed_scale = 0.9755321901954155
    self.lateral_speed_offset = 1.9152884533402488
    self.lateral_default_lookahead = 24
    self.lateral_speed_threshold = 23.150102938235136
    self.lateral_n = 6

    # Parameters for longitudinal controller
    self.longitudinal_params = (1.1990342347353184, -0.8057602384167799, 1.710818710950062, 0.921890257450335,
                                1.556497522998393, -0.7013479734904027, 1.031266635497984)
    self.longitudinal_max_acceleration = 1.89  # maximum acceleration 1.9 m/tick

    # Whether the model in and outputs will be visualized and saved into SAVE_PATH
    self.debug = False

    # -----------------------------------------------------------------------------
    # Logger
    # -----------------------------------------------------------------------------
    self.logging_freq = 10  # Log every 10 th frame
    self.logger_region_of_interest = 30.0  # Meters around the car that will be logged.
    self.route_points = 10  # Number of route points to render in logger
    # Minimum distance to the next waypoint in the logger
    self.log_route_planner_min_distance = 4.0

    # -----------------------------------------------------------------------------
    # Object Detector
    # -----------------------------------------------------------------------------
    # Confidence of a bounding box that is needed for the detection to be accepted
    self.bb_confidence_threshold = 0.3
    self.max_num_bbs = 30  # Maximum number of bounding boxes our system can detect.
    # CenterNet parameters
    self.num_dir_bins = 12
    self.top_k_center_keypoints = 100
    self.center_net_max_pooling_kernel = 3
    self.bb_input_channel = 64
    self.num_bb_classes = 5  # Added emergency vehicle class

    # -----------------------------------------------------------------------------
    # TransFuser Model
    # -----------------------------------------------------------------------------
    # Waypoint GRU
    self.gru_hidden_size = 64
    self.gru_input_size = 256

    # Conv Encoder
    if self.crop_image:
      self.img_vert_anchors = self.cropped_height // 32
      self.img_horz_anchors = self.cropped_width // 32
    else:
      self.img_vert_anchors = self.camera_height // 32
      self.img_horz_anchors = self.camera_width // 32

    self.lidar_vert_anchors = self.lidar_resolution_height // 32
    self.lidar_horz_anchors = self.lidar_resolution_width // 32

    # Resolution at which the perspective auxiliary tasks are predicted
    self.perspective_downsample_factor = 1

    self.bev_features_chanels = 64  # Number of channels for the BEV feature pyramid
    # Resolution at which the BEV auxiliary tasks are predicted
    self.bev_down_sample_factor = 4
    self.bev_upsample_factor = 2

    # GPT Encoder
    self.block_exp = 4
    self.n_layer = 2  # Number of transformer layers used in the vision backbone
    self.n_head = 4
    self.embd_pdrop = 0.1
    self.resid_pdrop = 0.1
    self.attn_pdrop = 0.1
    # Mean of the normal distribution initialization for linear layers in the GPT
    self.gpt_linear_layer_init_mean = 0.0
    # Std of the normal distribution initialization for linear layers in the GPT
    self.gpt_linear_layer_init_std = 0.02
    # Initial weight of the layer norms in the gpt.
    self.gpt_layer_norm_init_weight = 1.0

    # Number of route checkpoints to predict. Needs to be smaller than num_route_points!
    self.predict_checkpoint_len = 10

    # Whether to normalize the camera image by the imagenet distribution
    self.normalize_imagenet = True
    self.use_wp_gru = False  # Whether to use the WP output GRU.

    # Semantic Segmentation
    self.use_semantic = True  # Whether to use semantic segmentation as auxiliary loss
    self.num_semantic_classes = 7
    self.classes = {
        0: [0, 0, 0],  # unlabeled
        1: [30, 170, 250],  # vehicle
        2: [200, 200, 200],  # road
        3: [255, 255, 0],  # light
        4: [0, 255, 0],  # pedestrian
        5: [0, 255, 255],  # road line
        6: [255, 255, 255],  # sidewalk
    }
    # Color format BGR
    self.classes_list = [
        [0, 0, 0],  # unlabeled
        [250, 170, 30],  # vehicle
        [200, 200, 200],  # road
        [0, 255, 255],  # light
        [0, 255, 0],  # pedestrian
        [255, 255, 0],  # road line
        [255, 255, 255],  # sidewalk
    ]

    # https://github.com/carla-simulator/carla/blob/43b5e7064872bb6a9529664c2218e29df38dca04/LibCarla/source/carla/image/CityScapesPalette.h#L56
    self.converter = [
        0,  # unlabeled
        2,  # road
        6,  # sidewalk
        0,  # building
        0,  # wall
        0,  # fence
        0,  # pole
        3,  # traffic light
        0,  # traffic sign
        0,  # vegetation
        0,  # terrain
        0,  # sky
        4,  # pedestrian
        0,  # rider
        1,  # Car
        1,  # truck
        1,  # bus
        0,  # train
        1,  # motorcycle
        1,  # bicycle
        0,  # static
        0,  # dynamic
        0,  # other
        0,  # water
        5,  # road line
        0,  # ground
        0,  # bridge
        0,  # rail track
        0,  # guard rail
    ]

    self.bev_converter = [
        0,  # unlabeled
        1,  # road
        2,  # sidewalk
        3,  # lane_markers
        4,  # lane_markers broken, you may cross them
        5,  # stop_signs
        6,  # traffic light green
        7,  # traffic light yellow
        8,  # traffic light red
        9,  # vehicle
        10,  # walker
    ]

    # Color format BGR
    self.bev_classes_list = [
        [0, 0, 0],  # unlabeled
        [200, 200, 200],  # road
        [255, 255, 255],  # sidewalk
        [255, 255, 0],  # road line
        [50, 234, 157],  # road line broken
        [160, 160, 0],  # stop sign
        [0, 255, 0],  # light green
        [255, 255, 0],  # light yellow
        [255, 0, 0],  # light red
        [250, 170, 30],  # vehicle
        [0, 255, 0],  # pedestrian
    ]

    self.num_bev_semantic_classes = len(self.bev_converter)

    self.deconv_channel_num_0 = 128  # Number of channels at the first deconvolution layer
    self.deconv_channel_num_1 = 64  # Number of channels at the second deconvolution layer
    self.deconv_channel_num_2 = 32  # Number of channels at the third deconvolution layer

    # Fraction of the down-sampling factor that will be up-sampled in the first Up-sample
    self.deconv_scale_factor_0 = 4
    # Fraction of the down-sampling factor that will be up-sampled in the second Up-sample
    self.deconv_scale_factor_1 = 8

    self.use_discrete_command = True  # Whether to input the discrete target point as input to the network.
    self.add_features = True  # Whether to add (true) or concatenate (false) the features at the end of the backbone.

    self.image_u_net_output_features = 512  # Channel dimension of the up-sampled encoded image in bev_encoder
    self.bev_latent_dim = 32  # Channel dimensions of the image projected to BEV in the bev_encoder

    # Whether to use a transformer decoder instead of global average pool + MLP for planning
    self.transformer_decoder_join = True
    self.num_transformer_decoder_layers = 6  # Number of layers in the TransFormer decoder
    self.num_decoder_heads = 8

    # Ratio by which the height size of the voxel grid in BEV decoder are larger than width and depth
    self.bev_grid_height_downsample_factor = 1.0

    self.wp_dilation = 1  # Factor by which the wp are dilated compared to full CARLA 20 FPS

    self.extra_sensor_channels = 128  # Number of channels the extra sensors are embedded to

    self.use_tp = True  # Whether to use the target point as input to TransFuser
    self.two_tp_input = False  # Whether to use the next two target points as input instead of just one

    # Unit meters. Points from the LiDAR higher than this threshold are discarded. Default uses all the points.
    self.max_height_lidar = 100.0

    self.tp_attention = False  # Adds a TP at the TF decoder and computes it with attention visualization.
    self.multi_wp_output = False  # Predicts 2 WP outputs and uses the min loss of both.

    # whether to give the predicted path as input to the target speed network (and use more layers)
    self.input_path_to_target_speed_network = False

    # -----------------------------------------------------------------------------
    # Agent file
    # -----------------------------------------------------------------------------
    self.carla_frame_rate = 1.0 / 20.0  # CARLA frame rate in milliseconds
    # Iou threshold used for non-maximum suppression on the Bounding Box
    # predictions for the ensembles
    self.iou_treshold_nms = 0.2
    self.route_planner_min_distance = 7.5
    self.route_planner_max_distance = 50.0
    # Min distance to the waypoint in the dense rout that the expert is trying to follow
    self.dense_route_planner_min_distance = 2.4
    # Number of frames after which the creep controller starts triggering. 1100 is larger than wait time at red light.
    self.stuck_threshold = 1100
    self.creep_duration = 20  # Number of frames we will creep forward
    self.creep_throttle = 0.4
    # CARLA needs some time to initialize in which the cars actions are blocked.
    # Number tuned empirically
    self.inital_frames_delay = 2.0 / self.carla_frame_rate
    self.slower_factor = 0.8  # Factor by which the target speed will be reduced during inference if slower is active

    # Extent of the ego vehicles bounding box
    self.ego_extent_x = 2.4508416652679443
    self.ego_extent_y = 1.0641621351242065
    self.ego_extent_z = 0.7553732395172119

    # Size of the safety box
    self.safety_box_z_min = 0.5
    self.safety_box_z_max = 1.5

    self.safety_box_y_min = -self.ego_extent_y * 0.8
    self.safety_box_y_max = self.ego_extent_y * 0.8

    self.safety_box_x_min = self.ego_extent_x
    self.safety_box_x_max = self.ego_extent_x + 2.5

    # Probability 0 - 1. If the confidence in the brake action is higher than this
    # value brake is chosen as the action.
    self.brake_uncertainty_threshold = 0.9  # 1 means that it is not used at all

    # -----------------------------------------------------------------------------
    # PlanT
    # -----------------------------------------------------------------------------
    self.use_plant = False
    self.plant_precision_pos = 7  # 7: 0.5 meters
    self.plant_precision_angle = 4  # 4: 1,875 km/h
    self.plant_precision_speed = 5  # 5: 22.5 degrees
    self.plant_precision_brake = 2  # 2: true, false
    self.plant_object_types = 6  # vehicle, pedestrian, traffic light, stop sign, route, other
    self.plant_num_attributes = 7  # x,y, extent x, extent y,yaw,speed, brake, (class)
    # Options: prajjwal1/bert-tiny, prajjwal1/bert-mini, prajjwal1/bert-small, prajjwal1/bert-medium
    self.plant_hf_checkpoint = 'prajjwal1/bert-medium'
    self.plant_embd_pdrop = 0.1
    self.plant_pretraining = None
    self.plant_max_speed_pred = 60.0  # Maximum speed we classify when forcasting cars.
    self.forcast_time = 0.5  # Number of seconds we forcast into the future

  def initialize(self, root_dir='', setting='all', **kwargs):
    for k, v in kwargs.items():
      setattr(self, k, v)

    self.root_dir = root_dir

    if setting == 'all':
      pass
    elif setting == '13_withheld':
      self.val_towns.append(13)
    elif setting == '12_only':
      self.val_towns.append(1)
      self.val_towns.append(2)
      self.val_towns.append(3)
      self.val_towns.append(4)
      self.val_towns.append(5)
      self.val_towns.append(6)
      self.val_towns.append(7)
      self.val_towns.append(10)
      self.val_towns.append(11)
      self.val_towns.append(13)
      self.val_towns.append(15)

    elif setting == 'eval':
      return
    else:
      raise ValueError(f'Error: Selected setting: {setting} does not exist.')

    print('Setting: ', setting)
    self.data_roots = []
    for td_path in self.root_dir:
      self.data_roots = self.data_roots + [os.path.join(td_path, name) for name in os.listdir(td_path)]
