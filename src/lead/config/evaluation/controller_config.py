"""Controller configuration of the learned agent: route planner, target points and PID."""

from lead.config.node import ConfigNode


class ControllerConfig(ConfigNode):
    """Route planner distances, target point handling and PID controller gains."""

    # --- Route Planner ---
    # Minimum distance for route planner
    route_planner_min_distance: float = 5.0
    # Maximum distance for route planner
    route_planner_max_distance: float = 50

    # --- Adaptive target point controller ---
    # Skip next target point if too far away
    sensor_agent_skip_distant_target_point: bool = True
    # Distance threshold for skipping next target point (meters)
    sensor_agent_skip_distant_target_point_threshold: float = 50
    # Adaptively reduce pop-distance if target points are dense
    sensor_agent_pop_distance_adaptive: bool = True

    # --- PID Controller Parameters ---
    # Maximum change in speed input to longitudinal controller
    wp_delta_clip: float = 0.99
    # Aim distance for fast driving (meters)
    aim_distance_fast: float = 3.0
    # Aim distance for slow driving (meters)
    aim_distance_slow: float = 2.25
    # Speed threshold for switching between fast and slow aim distances (m/s)
    aim_distance_threshold: float = 5.5
    # Turn PID controller proportional gain
    turn_kp: float = 1.25
    # Turn PID controller integral gain
    turn_ki: float = 0.75
    # Turn PID controller derivative gain
    turn_kd: float = 0.3
    # Turn PID controller buffer size
    turn_n: int = 20
    # Speed PID controller proportional gain
    speed_kp: float = 1.75
    # Speed PID controller integral gain
    speed_ki: float = 1.0
    # Speed PID controller derivative gain
    speed_kd: float = 2.0
    # Speed PID controller buffer size
    speed_n: int = 20
    # Desired speed below which brake is triggered
    brake_speed: float = 0.4
    # Ratio of speed to desired speed at which brake is triggered
    brake_ratio: float = 1.1
    # Lateral PID controller proportional gain
    lateral_k_p: float = 3.118357247806046
    # Lateral PID controller derivative gain
    lateral_k_d: float = 1.3782508892109167
    # Lateral PID controller integral gain
    lateral_k_i: float = 0.6406067986034124
    # Speed scaling factor for lateral control
    lateral_speed_scale: float = 0.9755321901954155
    # Speed offset for lateral control
    lateral_speed_offset: float = 1.9152884533402488
    # Default lookahead distance for lateral control
    lateral_default_lookahead: float = 24
    # Speed threshold for lateral control behavior switching
    lateral_speed_threshold: float = 23.150102938235136
    # Lateral PID controller buffer size
    lateral_n: int = 6
    # If true use tuned aim distance for navigation
    tuned_aim_distance: bool = False
