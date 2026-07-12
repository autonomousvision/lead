"""Route smoothing for the planning route labels."""

import logging

import jaxtyping as jt
import numpy as np
import numpy.typing as npt

from lead.config import LeadConfig

LOG = logging.getLogger(__name__)


def smooth_path(
    config: LeadConfig,
    route: jt.Float[npt.NDArray, "N 2"],
    target_first_distance: float,
) -> jt.Float[npt.NDArray, "N 2"]:
    """Smooth a route by removing duplicates and creating evenly-spaced interpolated waypoints.

    This function preprocesses a route by removing duplicate waypoints while preserving the
    original path order, then generates a smoothed representation with evenly-spaced points
    using iterative line interpolation.

    The smoothing process:
    1. Identifies and removes duplicate consecutive waypoints
    2. Preserves the original route order (important for path following)
    3. Generates interpolated points at regular intervals along the cleaned route

    Args:
        config: Training configuration containing route smoothing parameters such as
            the number of interpolated points to generate.
        route: Array of shape (N, 2) containing input waypoints as (x, y) coordinates.
            May contain duplicate points.
        target_first_distance: Distance in meters for placing the first interpolated point
            from the origin (0, 0).

    Returns:
        Array of shape (num_route_points_smoothing, 2) containing the smoothed route
        with evenly-spaced waypoints. All duplicates are removed and spacing is regularized.
    """
    _, indices = np.unique(route, return_index=True, axis=0)
    # We need to remove the sorting of unique, because this algorithm assumes the order of the path is kept
    route = np.array(route)
    indices = np.sort(indices)
    indices = np.array(indices).astype(int)
    route = route[indices]
    interpolated_route_points = iterative_line_interpolation(
        config,
        route,
        target_first_distance,
    )

    return interpolated_route_points


def circle_line_segment_intersection(
    circle_center: jt.Float[npt.NDArray, " 2"],
    circle_radius: float,
    pt1: jt.Float[npt.NDArray, " 2"],
    pt2: jt.Float[npt.NDArray, " 2"],
    full_line: bool = True,
    tangent_tol: float = 1e-9,
) -> list[tuple[float, float]]:
    """Find the intersection points between a circle and a line segment.

    Computes geometric intersections using the analytical solution for circle-line
    intersection. The function can return 0, 1, or 2 intersection points depending
    on whether the line misses, is tangent to, or crosses through the circle.

    Args:
        circle_center: The (x, y) coordinates of the circle center.
        circle_radius: The radius of the circle.
        pt1: The (x, y) coordinates of the first point of the line segment.
        pt2: The (x, y) coordinates of the second point of the line segment.
        full_line: If True, find intersections along the infinite line extending through
            pt1 and pt2. If False, only return intersections within the segment [pt1, pt2].
        tangent_tol: Numerical tolerance for determining if the line is tangent to the circle.
            When discriminant is below this threshold, treat as tangent case.

    Returns:
        A list of (x, y) tuples representing intersection points. The list contains:
        - 0 elements if no intersection exists
        - 1 element if the line is tangent to the circle
        - 2 elements if the line crosses through the circle
        Points are ordered along the direction from pt1 to pt2.

    Note:
        Implementation follows the analytical solution from:
        http://mathworld.wolfram.com/Circle-LineIntersection.html
        Credit: https://stackoverflow.com/a/59582674/9173068
    """
    if np.linalg.norm(pt1 - pt2) < 0.000000001:
        LOG.warning("Problem")

    (p1x, p1y), (p2x, p2y), (cx, cy) = pt1, pt2, circle_center
    (x1, y1), (x2, y2) = (p1x - cx, p1y - cy), (p2x - cx, p2y - cy)
    dx, dy = (x2 - x1), (y2 - y1)
    dr = (dx**2 + dy**2) ** 0.5
    big_d = x1 * y2 - x2 * y1
    discriminant = circle_radius**2 * dr**2 - big_d**2

    if discriminant < 0:  # No intersection between circle and line
        return []
    else:  # There may be 0, 1, or 2 intersections with the segment
        # This makes sure the order along the segment is correct
        intersections = [
            (
                cx
                + (big_d * dy + sign * (-1 if dy < 0 else 1) * dx * discriminant**0.5)
                / dr**2,
                cy + (-big_d * dx + sign * abs(dy) * discriminant**0.5) / dr**2,
            )
            for sign in ((1, -1) if dy < 0 else (-1, 1))
        ]
        if not full_line:  # If only considering the segment, filter out intersections that do not fall within the segment
            fraction_along_segment = [
                (xi - p1x) / dx if abs(dx) > abs(dy) else (yi - p1y) / dy
                for xi, yi in intersections
            ]
            intersections = [
                pt
                for pt, frac in zip(intersections, fraction_along_segment, strict=False)
                if 0 <= frac <= 1
            ]
        # If line is tangent to circle, return just one point (as both intersections have same location)
        if len(intersections) == 2 and abs(discriminant) <= tangent_tol:
            return [intersections[0]]
        else:
            return intersections


def iterative_line_interpolation(
    config: LeadConfig,
    route: jt.Float[npt.NDArray, "n 2"],
    target_first_distance: float,
) -> jt.Float[npt.NDArray, "n 2"]:
    """Generate evenly-spaced interpolated points along a route using circle-line intersection.

    This function creates a smoothed route representation by iteratively finding points at
    fixed distances along the input route. It uses a geometric approach where each new point
    is found at the intersection of a circle (centered at the last interpolated point) with
    the line segments of the original route.

    The algorithm:
    1. Places the first point at `target_first_distance` from the origin (0, 0)
    2. Places subsequent points exactly 1.0 meter apart along the route
    3. When multiple intersections exist, selects the one in the forward direction
    4. Extrapolates beyond the route end if needed to reach the target number of points

    Args:
        config: Training configuration containing route planning parameters, specifically:
            - num_route_points_smoothing: Number of interpolated points to generate
            - dense_route_planner_min_distance: Initial minimum distance (unused, overwritten)
        route: Array of shape (N, 2) containing the input waypoints as (x, y) coordinates.
        target_first_distance: Distance in meters for the first interpolated point from origin.

    Returns:
        Array of shape (num_route_points_smoothing, 2) containing evenly-spaced interpolated
        waypoints along the route.

    Raises:
        Exception: If no intersection is found during interpolation (should not occur under
            normal circumstances).
    """
    interpolated_route_points = []

    # this value is actually not used anymore, it is overwritten in the loop
    min_distance = config.expert.simulation.dense_route_planner_min_distance
    last_interpolated_point = np.array([0.0, 0.0])
    current_route_index = 0
    current_point = route[current_route_index]
    last_point = np.array([0.0, 0.0])
    first_iteration = True

    while (
        len(interpolated_route_points)
        < config.agent.transfuser.num_route_points_smoothing
    ):
        # First point should be target_first_distance away from the vehicle.
        if not first_iteration:
            current_route_index += 1
            last_point = current_point

        if current_route_index < route.shape[0]:
            current_point = route[current_route_index]
            intersection = circle_line_segment_intersection(
                circle_center=last_interpolated_point,
                circle_radius=(
                    min_distance if not first_iteration else target_first_distance
                ),
                pt1=last_interpolated_point,
                pt2=current_point,
                full_line=True,
            )

        else:  # We hit the end of the input route. We extrapolate the last 2 points
            current_point = route[-1]
            last_point = route[-2]
            intersection = circle_line_segment_intersection(
                circle_center=last_interpolated_point,
                circle_radius=min_distance,
                pt1=last_point,
                pt2=current_point,
                full_line=True,
            )

        # 3 cases: 0 intersection, 1 intersection, 2 intersection
        if len(intersection) > 1:  # 2 intersections
            # Take the one that is closer to current point
            point_1 = np.array(intersection[0])
            point_2 = np.array(intersection[1])
            direction = current_point - last_point
            dot_p1_to_last = np.dot(point_1, direction)
            dot_p2_to_last = np.dot(point_2, direction)

            if dot_p1_to_last > dot_p2_to_last:
                intersection_point = point_1
            else:
                intersection_point = point_2
            add_point = True
        elif len(intersection) == 1:  # 1 Intersections
            intersection_point = np.array(intersection[0])
            add_point = True
        else:  # 0 Intersection
            add_point = False
            raise Exception("No intersection found. This should never occur.")

        if add_point:
            last_interpolated_point = intersection_point
            interpolated_route_points.append(intersection_point)
            min_distance = 1.0  # After the first point we want each point to be 1 m away from the last.

        first_iteration = False

    interpolated_route_points = np.array(interpolated_route_points)
    return interpolated_route_points
