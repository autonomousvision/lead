import logging
import math
import typing

import carla
import jaxtyping as jt
import numpy as np
import numpy.typing as npt
from agents.navigation.global_route_planner import GlobalRoutePlanner

from lead.common.constants import (
    SEMANTIC_SEGMENTATION_CONVERTER,
)
from lead.common.control.pid_controller import PIDController
from lead.config import ExpertConfig, load_lead_config

LOG = logging.getLogger(__name__)


def step_cached_property(func: typing.Callable) -> property:
    """Decorator to cache the result of a method based on the current step.
    This is useful for properties that are expensive to compute and
    should only be recalculated when the step changes.

    Args:
        func: Function to be decorated.

    Returns:
        A property that caches its value based on the step attribute of the instance.
    """
    cache_attr = f"_{func.__name__}_cache"
    step_attr = f"_{func.__name__}_step"

    def getter(self):
        if getattr(self, step_attr, None) != self.step:
            setattr(self, cache_attr, func(self))
            setattr(self, step_attr, self.step)
        return getattr(self, cache_attr)

    return property(getter)


def cached_property_by(key_getter: typing.Callable):
    """Decorator to cache the result of a method based on a custom key.
    This is useful for properties that are expensive to compute and should only be
    recalculated when a specific attribute or value changes.

    Args:
        key_getter: Function that takes the instance and returns the cache key.

    Returns:
        A decorator that creates a property cached by the key_getter result.

    Example:
        @cached_property_by(lambda self: self.privileged_route_planner.route_index)
        def some_expensive_calculation(self):
            return expensive_computation()
    """

    def decorator(func: typing.Callable) -> property:
        cache_attr = f"_{func.__name__}_cache"
        key_attr = f"_{func.__name__}_cache_key"

        def getter(self):
            current_key = key_getter(self)
            if getattr(self, key_attr, None) != current_key:
                setattr(self, cache_attr, func(self))
                setattr(self, key_attr, current_key)
            return getattr(self, cache_attr)

        return property(getter)

    return decorator


def get_horizontal_distance(actor1: carla.Actor, actor2: carla.Actor) -> float:
    """Get horizontal distance between two actors.

    Args:
        actor1: First actor.
        actor2: Second actor.

    Returns:
        float: Horizontal distance between the two actors.
    """
    location1, location2 = actor1.get_location(), actor2.get_location()

    # Compute the distance vector (ignoring the z-coordinate)
    diff_vector = carla.Vector3D(
        location1.x - location2.x,
        location1.y - location2.y,
        0,
    )

    norm: float = diff_vector.length()

    return norm


def distance_location_to_route(route: npt.NDArray, location: npt.NDArray) -> float:
    """
    Project a location onto the closest point on a route.

    Args:
        route: (N, 3) np.array of route points
        location: (3,) np.array of the location to project

    Returns:
        Distance from the projected point to the location
    """
    # Compute the distance between the location and each point on the route
    distances = np.linalg.norm(route - location, axis=1)

    # Find the minimum distance (i.e., the closest point)
    min_distance = np.min(distances)

    return min_distance


def compute_global_route(
    world: carla.World,
    source_location: carla.Location,
    sink_location: carla.Location,
    resolution: float = 1.0,
) -> jt.Float[npt.NDArray, "n 3"]:
    """
    Args:
        world: carla.World instance
        source_location: carla.Location of the source point
        sink_location: carla.Location of the sink point
        resolution: resolution for the global route planner
    Returns:
        array of waypoints in the format [[x, y, z], ...]
    """
    grp = GlobalRoutePlanner(world.get_map(), resolution)
    route = grp.trace_route(source_location, sink_location)
    return np.array(
        [
            [wp.transform.location.x, wp.transform.location.y, wp.transform.location.z]
            for wp, _ in route
        ],
    )


def intersection_of_routes(
    points_a: npt.NDArray,
    points_b: npt.NDArray,
    epsilon: float = 0.5,
) -> tuple[carla.Location | None, int | None]:
    """
    Args:
        points_a: (N, 3) np.array of route points for the first route
        points_b: (M, 3) np.array of route points for the second route
        epsilon: threshold distance to consider two points as intersecting
    Returns:
        The intersection point and its index if found, otherwise None.
    """
    points_a = np.array(points_a, dtype=np.float32)
    points_b = np.array(points_b, dtype=np.float32)
    diff = points_a[:, None, :2] - points_b[None, :, :2]  # (N, M, 2)
    dists = np.sqrt((diff**2).sum(axis=-1))  # (N, M)

    mask = dists < epsilon
    indices = np.argwhere(mask)

    if indices.shape[0] == 0:
        return None, None

    i, j = indices[0]
    z = (points_a[i, 2] + points_b[j, 2]) / 2.0
    x = (points_a[i, 0] + points_b[j, 0]) / 2.0
    y = (points_a[i, 1] + points_b[j, 1]) / 2.0
    return carla.Location(x=x, y=y, z=z), int(i)


def _obb_axes(obb: carla.BoundingBox) -> jt.Float[npt.NDArray, "3 3"]:
    """Return the three local axes of an oriented bounding box as matrix rows.

    Args:
        obb: The oriented bounding box.

    Returns:
        Matrix whose rows are the forward, right and up unit vectors.
    """
    forward = obb.rotation.get_forward_vector()
    right = obb.rotation.get_right_vector()
    up = obb.rotation.get_up_vector()
    return np.array(
        [
            [forward.x, forward.y, forward.z],
            [right.x, right.y, right.z],
            [up.x, up.y, up.z],
        ],
    )


def check_obb_intersection(obb1: carla.BoundingBox, obb2: carla.BoundingBox) -> bool:
    """
    Check if two 3D oriented bounding boxes (OBBs) intersect.

    Uses the separating axis theorem over the 15 candidate axes (3 + 3 box axes
    and their 9 cross products), preceded by a bounding-sphere early-out.

    Args:
        obb1: The first oriented bounding box.
        obb2: The second oriented bounding box.

    Returns:
        True if the two OBBs intersect, False otherwise.
    """
    relative_position = np.array(
        [
            obb2.location.x - obb1.location.x,
            obb2.location.y - obb1.location.y,
            obb2.location.z - obb1.location.z,
        ],
    )
    extent1 = np.array([obb1.extent.x, obb1.extent.y, obb1.extent.z])
    extent2 = np.array([obb2.extent.x, obb2.extent.y, obb2.extent.z])

    # Boxes further apart than the sum of their bounding spheres cannot intersect
    max_reach = np.linalg.norm(extent1) + np.linalg.norm(extent2)
    if relative_position @ relative_position > max_reach * max_reach:
        return False

    axes1 = _obb_axes(obb1)
    axes2 = _obb_axes(obb2)
    cross_axes = np.cross(axes1[:, None, :], axes2[None, :, :]).reshape(9, 3)
    candidate_axes = np.concatenate([axes1, axes2, cross_axes], axis=0)

    center_projections = np.abs(candidate_axes @ relative_position)
    reach_projections = (
        np.abs(candidate_axes @ axes1.T) @ extent1
        + np.abs(
            candidate_axes @ axes2.T,
        )
        @ extent2
    )

    # A separating axis exists iff the centers project further apart than the
    # combined box reach; degenerate cross products (parallel axes) project to
    # zero on both sides and never separate, as in the reference SAT.
    return not np.any(center_projections > reach_projections)


def get_angle_to(
    current_position: list[tuple[float, float]],
    current_heading: float,
    target_position: tuple | list,
) -> float:
    """
    Calculate the angle (in degrees) from the current position and heading to a target position.

    Args:
        current_position: A list of (x, y) coordinates representing the current position.
        current_heading: The current heading angle in radians.
        target_position: A tuple or list of (x, y) coordinates representing the target position.
    Returns:
        float: The angle (in degrees) from the current position and heading to the target position.
    """
    cos_heading = math.cos(current_heading)
    sin_heading = math.sin(current_heading)

    # Calculate the vector from the current position to the target position
    position_delta = target_position - current_position

    # Calculate the dot product of the position delta vector and the current heading vector
    aim_x = cos_heading * position_delta[0] + sin_heading * position_delta[1]
    aim_y = -sin_heading * position_delta[0] + cos_heading * position_delta[1]

    # Calculate the angle (in radians) from the current heading to the target position
    angle_radians = -math.atan2(-aim_y, aim_x)

    # Convert the angle from radians to degrees
    angle_degrees = float(math.degrees(angle_radians))

    return angle_degrees


def get_steer(
    config: ExpertConfig,
    turn_controller: PIDController,
    route_points: npt.NDArray,
    current_position: tuple[float, float],
    current_heading: float,
    current_speed: float,
) -> float:
    """
    Calculate the steering angle based on the current position, heading, speed, and the route points.

    Args:
        config: Configuration object containing PID controller parameters.
        turn_controller: An instance of the PIDController for steering.
        route_points: An array of (x, y) coordinates representing the route points.
        current_position: The current position (x, y) of the vehicle.
        current_heading: The current heading angle (in radians) of the vehicle.
        current_speed: The current speed of the vehicle (in m/s).

    Returns:
        The calculated steering angle.
    """
    # Calculate the steering angle using the turn controller
    steering_angle = turn_controller.step(
        route_points,
        current_speed,
        current_position,
        current_heading,
    )
    steering_angle = round(steering_angle, 3)
    return steering_angle


def get_previous_road_lane_ids(
    config: ExpertConfig,
    starting_waypoint: carla.Waypoint,
) -> list[tuple[int, int]]:
    """
    Retrieves the previous road and lane IDs for a given starting waypoint.

    Args:
        config: Configuration object containing parameters.
        starting_waypoint: The starting waypoint.
    Returns:
        A list of tuples containing road IDs and lane IDs.
    """
    current_waypoint = starting_waypoint
    previous_lane_ids = [(current_waypoint.road_id, current_waypoint.lane_id)]

    # Traverse backwards up to 100 waypoints to find previous lane IDs
    for _ in range(config.driving.previous_road_lane_retrieve_distance):
        previous_waypoints = current_waypoint.previous(1)

        # Check if the road ends and no previous route waypoints exist
        if len(previous_waypoints) == 0:
            break
        current_waypoint = previous_waypoints[0]

        if (
            current_waypoint.road_id,
            current_waypoint.lane_id,
        ) not in previous_lane_ids:
            previous_lane_ids.append(
                (current_waypoint.road_id, current_waypoint.lane_id),
            )

    return previous_lane_ids


def wps_next_until_lane_end(wp: carla.Waypoint) -> list[carla.Waypoint]:
    """
    Get all waypoints until the lane ends (i.e., road_id or lane_id changes).
    Args:
        wp: The starting waypoint.
    Returns:
        list[carla.Waypoint]: A list of waypoints until the lane ends.
    """
    try:
        road_id_cur = wp.road_id
        lane_id_cur = wp.lane_id
        road_id_next = road_id_cur
        lane_id_next = lane_id_cur
        curr_wp = [wp]
        next_wps = []
        # https://github.com/carla-simulator/carla/issues/2511#issuecomment-597230746
        while road_id_cur == road_id_next and lane_id_cur == lane_id_next:
            next_wp = curr_wp[0].next(1)
            if len(next_wp) == 0:
                break
            curr_wp = next_wp
            next_wps.append(next_wp[0])
            road_id_next = next_wp[0].road_id
            lane_id_next = next_wp[0].lane_id
    except:
        next_wps = []

    return next_wps


def enhance_semantics_segmentation(
    instance_image: npt.NDArray,
    semantics_segmentation_image: npt.NDArray,
    id_map: dict[int, int],
    boxes_new_semantic_id: int,
) -> npt.NDArray:
    """
    Enhance the semantics segmentation image by replacing specific semantic IDs based on instance segmentation.
    Args:
        instance_image: HxWx2 numpy array where the first channel is the semantic ID and the second channel is the instance ID.
        semantics_segmentation_image: HxW numpy array with the original semantic segmentation IDs.
        id_map: Dictionary mapping truncated CARLA actor IDs (as stored in the instance channel) to full actor IDs.
        boxes_new_semantic_id: The new semantic ID to assign to the pixels of the given actors.
    Returns:
        npt.NDArray: Enhanced semantics segmentation image.
    """
    if len(id_map) == 0:
        return semantics_segmentation_image
    enhanced_image = semantics_segmentation_image.copy()
    for instance_id in id_map.keys():
        mask = instance_image[..., 1] == instance_id
        enhanced_image[mask] = boxes_new_semantic_id
    return enhanced_image


def rotate_point(point: carla.Vector3D, angle: float) -> carla.Vector3D:
    """Rotate a given point by a given angle.

    Args:
        point: The point to be rotated.
        angle: The angle in degrees.
    Returns:
        The rotated point.
    """
    x_ = (
        math.cos(math.radians(angle)) * point.x
        - math.sin(math.radians(angle)) * point.y
    )
    y_ = (
        math.sin(math.radians(angle)) * point.x
        + math.cos(math.radians(angle)) * point.y
    )
    return carla.Vector3D(x_, y_, point.z)


def get_traffic_light_waypoints(traffic_light: carla.Actor, carla_map: carla.Map):
    """Get waypoints of roads controlled by a given traffic light.

    This function finds waypoints on roads that lead to intersections controlled
    by the traffic light. It discretizes the traffic light's trigger volume,
    finds corresponding road waypoints, and advances them toward the intersection
    while staying within the traffic light's influence area.

    Args:
        traffic_light: The traffic light object to analyze.
        carla_map: The CARLA map containing road network information.

    Returns:
        tuple: A tuple containing:
            - area_loc (carla.Location): The world location of the trigger volume center
            - wps (list[carla.Waypoint]): List of waypoints on roads controlled by this traffic light
    """
    base_transform = traffic_light.get_transform()
    base_loc = traffic_light.get_location()
    base_rot = base_transform.rotation.yaw
    area_loc = base_transform.transform(traffic_light.trigger_volume.location)

    # Discretize the trigger box into points
    area_ext = traffic_light.trigger_volume.extent
    x_values = np.arange(
        -0.9 * area_ext.x,
        0.9 * area_ext.x,
        1.0,
    )  # 0.9 to avoid crossing to adjacent lanes

    area = []
    for x in x_values:
        point = rotate_point(carla.Vector3D(x, 0, area_ext.z), base_rot)
        point_location = area_loc + carla.Location(x=point.x, y=point.y)
        area.append(point_location)

    # Get the waypoints of these points, removing duplicates
    ini_wps = []
    for pt in area:
        wpx = carla_map.get_waypoint(pt)
        # As x_values are arranged in order, only the last one has to be checked
        if (
            not ini_wps
            or ini_wps[-1].road_id != wpx.road_id
            or ini_wps[-1].lane_id != wpx.lane_id
        ):
            ini_wps.append(wpx)

    # Advance them until the intersection
    wps = []
    eu_wps = []
    for wpx in ini_wps:
        distance_to_light = base_loc.distance(wpx.transform.location)
        eu_wps.append(wpx)
        next_distance_to_light = distance_to_light + 1.0
        while not wpx.is_intersection:
            next_wp = wpx.next(0.5)[0]
            next_distance_to_light = base_loc.distance(next_wp.transform.location)
            if (
                next_wp
                and not next_wp.is_intersection
                and next_distance_to_light <= distance_to_light
            ):
                eu_wps.append(next_wp)
                distance_to_light = next_distance_to_light
                wpx = next_wp
            else:
                break

        if not next_distance_to_light <= distance_to_light and len(eu_wps) >= 4:
            wps.append(eu_wps[-4])
        else:
            wps.append(wpx)

    return area_loc, wps


def get_same_direction_lanes(
    waypoint: carla.Waypoint,
    max_lane_search: int = 10,
) -> list[carla.Waypoint]:
    """
    Find all waypoints in the same direction (left and right lanes)

    Args:
        waypoint: The reference waypoint
        max_lane_search: Maximum number of lanes to search in each direction

    Returns:
        List of waypoints in the same direction
    """
    same_direction_waypoints = []  # Include the original waypoint

    # Search left lanes
    current_wp = waypoint
    for _ in range(max_lane_search):
        left_wp = current_wp.get_left_lane()
        if left_wp is None:
            break
        if left_wp.lane_type != carla.LaneType.Driving:
            break
        if left_wp.lane_id == waypoint.lane_id:
            continue  # Skip if it's the same lane
        if left_wp.lane_id * waypoint.lane_id < 0:
            continue  # Skip if it's the same lane

        same_direction_waypoints.append(left_wp)
        current_wp = left_wp

    # Search right lanes
    current_wp = waypoint
    for _ in range(max_lane_search):
        right_wp = current_wp.get_right_lane()
        if right_wp is None:
            break
        if right_wp.lane_type != carla.LaneType.Driving:
            break
        if right_wp.lane_id == waypoint.lane_id:
            continue  # Skip if it's the same lane
        if right_wp.lane_id * waypoint.lane_id < 0:
            continue  # Skip if it's the same lane

        same_direction_waypoints.append(right_wp)
        current_wp = right_wp

    return same_direction_waypoints


def get_stop_waypoints(
    ego_waypoint: carla.Waypoint,
    traffic_light: carla.TrafficLight,
) -> list[carla.Waypoint]:
    """
    Get all waypoints at the intersection controlled by this traffic light in the same direction.

    Args:
        ego_waypoint: The ego vehicle's current waypoint for lane matching
        traffic_light: The traffic light to get stop waypoints from

    Returns:
        List of waypoints at the intersection in the same direction
    """
    waypoints = traffic_light.get_stop_waypoints()

    # Push all waypoints forward to intersection
    intersection_waypoints = []
    for wp in waypoints:
        current = previous = wp
        while current.next(1.0) and not current.get_junction():
            previous = current
            current = current.next(1.0)[0]
        intersection_waypoints.append(previous)

    if not intersection_waypoints:
        return []

    # Use intersection waypoint with same lane id as ego, otherwise use first
    base_wp = intersection_waypoints[0]
    for wp in intersection_waypoints:
        if wp.lane_id == ego_waypoint.lane_id:
            base_wp = wp
            break
    same_direction = get_same_direction_lanes(base_wp)

    # Include the base waypoint itself
    if len(same_direction) + 1 < len(intersection_waypoints):
        return intersection_waypoints
    return [base_wp] + same_direction


def create_bounding_box_for_waypoint(
    original_bbox: carla.BoundingBox,
    waypoint: carla.Waypoint,
) -> carla.BoundingBox:
    """
    Create a new bounding box positioned at the given waypoint

    Args:
        original_bbox: The original traffic light bounding box
        waypoint: The waypoint where to position the new bounding box

    Returns:
        New bounding box with updated location
    """
    new_bbox = carla.BoundingBox()
    new_bbox.location = waypoint.transform.location
    new_bbox.rotation = waypoint.transform.rotation
    new_bbox.extent = original_bbox.extent
    return new_bbox


class NegativeIdCounter:
    """ID generator that maps keys to negative IDs, creating new ones only when needed"""

    def __init__(self, start_value: int = -100000) -> None:
        """
        Initialize the ID generator

        Args:
            start_value: Starting value for new IDs (default: -100000)
        """
        self._id_map: dict[int, int] = {}
        self._current = start_value

    def __call__(self, key: int) -> int:
        """
        Get ID for a key. If key exists in map, return existing ID.
        If not, create and return a new negative ID.

        Args:
            key: The key to get an ID for

        Returns:
            The ID associated with the key
        """
        if key in self._id_map:
            return self._id_map[key]

        # Create new ID
        new_id = self._current
        self._id_map[key] = new_id
        self._current -= 1
        return new_id


def convert_depth(data: jt.UInt8[npt.NDArray, "h w 3"]) -> jt.Float[npt.NDArray, "h w"]:
    """Compute normalized depth from a CARLA depth map.

    Converts CARLA's RGB-encoded depth values to actual depth in meters.

    Args:
        data: CARLA depth map as RGB array of shape (height, width, 3).

    Returns:
        Depth values in meters as array of shape (height, width).
    """
    # Optimized implementation:
    # 1. Avoids astype(float32) on the whole 3D array (saves 66% memory allocation)
    # 2. Computes per-channel contributions directly

    scale = 1000.0 / 16777215.0  # 1000 / (256**3 - 1)

    # R * 65536 * scale + G * 256 * scale + B * scale
    # Initialize with the most significant byte (R)
    depth = data[:, :, 0].astype(np.float32)
    depth *= 65536.0 * scale

    # Add Middle byte (G)
    depth += data[:, :, 1].astype(np.float32) * (256.0 * scale)

    # Add Least significant byte (B)
    depth += data[:, :, 2].astype(np.float32) * scale

    return depth


def convert_instance_segmentation(
    data: jt.UInt8[npt.NDArray, "H W 3"],
) -> jt.Int32[npt.NDArray, "H W 2"]:
    """
    Args:
        data: Instance segmentation map from CARLA of shape (H, W, 3) with values in range [0, 255].
              R channel = semantic ID
              G+B*256 = instance ID
    Returns:
        data: Instance segmentation of shape (H, W, 2) with channels: [semantic_id, instance_id]
    """
    semantic_id = data[:, :, 0]
    semantic_id[semantic_id >= len(SEMANTIC_SEGMENTATION_CONVERTER)] = (
        0  # Carla semantic segmentation has some bug which give invalid semantic class.
    )
    instance_id = data[:, :, 1].astype(np.int32) + (data[:, :, 2].astype(np.int32) << 8)
    return np.stack([semantic_id, instance_id], axis=-1)


if __name__ == "__main__":
    config = load_lead_config().expert
