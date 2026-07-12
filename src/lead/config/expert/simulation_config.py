"""Simulator timing, route planning and ego geometry configuration."""

from lead.config.node import ConfigNode


class SimulationConfig(ConfigNode):
    """Simulator timing, route planner distances, ego extents and the safety box."""

    # Simulator frames per second
    carla_fps: int = 20
    # Initial frames delay for CARLA initialization
    inital_frames_delay: int = 1
    # Points sampled per meter when interpolating route
    points_per_meter: int = 10
    # Number of future route points we save per step
    num_route_points_saved: int = 50

    # Minimum distance to route planner waypoints
    route_planner_min_distance: float = 7.5
    # Maximum distance to route planner waypoints
    route_planner_max_distance: float = 50.0
    # Minimum distance to waypoint in dense route that expert follows
    dense_route_planner_min_distance: float = 2.4
    # Target point distances for route planning (in meters). From 3.0m to 10.0m with step of 0.25m
    tp_distances: list[float] = [i / 100 for i in range(300, 1001, 25)]

    # Extent of the ego vehicle's bounding box in x direction
    ego_extent_x: float = 2.4508416652679443
    # Extent of the ego vehicle's bounding box in y direction
    ego_extent_y: float = 1.0641621351242065
    # Extent of the ego vehicle's bounding box in z direction
    ego_extent_z: float = 0.7553732395172119

    # Minimum z coordinate of the safety box
    safety_box_z_min: float = 0.5
    # Maximum z coordinate of the safety box
    safety_box_z_max: float = 1.5

    @property
    def carla_frame_rate(self) -> float:
        """CARLA frame rate in seconds."""
        return 1.0 / self.carla_fps

    @property
    def safety_box_y_min(self) -> float:
        """Minimum y coordinate of the safety box relative to ego vehicle."""
        return -self.ego_extent_y * 0.8

    @property
    def safety_box_y_max(self) -> float:
        """Maximum y coordinate of the safety box relative to ego vehicle."""
        return self.ego_extent_y * 0.8

    @property
    def safety_box_x_min(self) -> float:
        """Minimum x coordinate of the safety box relative to ego vehicle."""
        return self.ego_extent_x

    @property
    def safety_box_x_max(self) -> float:
        """Maximum x coordinate of the safety box relative to ego vehicle."""
        return self.ego_extent_x + 2.5
