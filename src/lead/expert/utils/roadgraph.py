"""Queries against the CARLA road network: routes, lanes, and the waypoints of traffic controls."""

import carla
import jaxtyping as jt
import numpy as np
import numpy.typing as npt
from agents.navigation.global_route_planner import GlobalRoutePlanner

from lead.common.geometry import rotate_point
from lead.config import ExpertConfig


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


def get_traffic_light_waypoints(traffic_light: carla.Actor, carla_map: carla.Map):
    """Get waypoints of roads controlled by a given traffic light.

    Discretizes the trigger volume into road waypoints and advances them toward
    the intersection while staying within the traffic light's influence area.

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


def speed_limit_of_sign(type_id: str) -> float | None:
    """Speed limit a sign blueprint encodes, in km/h.

    Args:
        type_id: CARLA blueprint id of a traffic sign, e.g. ``traffic.speed_limit.30``.

    Returns:
        The speed limit in km/h, or None if the sign is not a speed limit sign.
    """
    if not type_id.startswith("traffic.speed_limit."):
        return None
    try:
        return float(type_id.rsplit(".", 1)[-1])
    except ValueError:
        return None
