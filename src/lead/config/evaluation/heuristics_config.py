"""Driving heuristics configuration: creeping when stuck and stop sign handling."""

from lead.config.node import ConfigNode


class HeuristicsConfig(ConfigNode):
    """Heuristics that override the model's controls in special situations."""

    # --- Creeping Heuristic ---
    # If true, enable the creeping heuristic that forces the agent to move when stuck
    sensor_agent_creeping: bool = False
    # Number of frames after which the creep controller starts triggering (larger than red light wait time)
    sensor_agent_stuck_threshold: int = 1100
    # Number of frames to creep forward when stuck
    sensor_agent_stuck_move_duration: int = 20
    # Throttle value for creeping when stuck
    sensor_agent_stuck_throttle: float = 0.4

    # --- Stop Sign Heuristic ---
    # If true enable stop sign controller
    slower_for_stop_sign: bool = False
    # Distance threshold for stop sign controller activation
    slower_for_stop_sign_dist_threshold: float = 1.0
    # Cool down period for stop sign controller (frames)
    slower_for_stop_sign_cool_down: int = 120
    # Number of frames to apply slower behavior
    slower_for_stop_sign_count: int = 40
    # Throttle threshold for slower stop sign behavior
    slower_for_stop_sign_throttle_threshold: float = 0.1
