"""Navigation and ego-status inputs of the TransFuser model."""

from lead.config.node import ConfigNode


class TransfuserNavigationConfig(ConfigNode):
    """Target points and ego-state network inputs."""

    # If true use the velocity as input to the network.
    use_velocity: bool = True
    # Maximum speed limit for the vehicle in m/s.
    max_speed_mps: float = 25.0
    # If true use the previous/visited target point as input to the network.
    use_previous_target_point: bool = True
    # If true use the next/subsequent target point as input to the network.
    use_next_target_point: bool = True
    # If true use the current target point as input to the network.
    use_target_point: bool = True
    # Normalization constants for target points [x_norm, y_norm].
    target_point_normalization_xy: list[list[float]] = [[200.0, 50.0]]
    # Distance threshold for popping target points from route.
    target_point_pop_distance: float = 3.25
