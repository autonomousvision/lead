"""Route smoothing for the planning route labels."""

import logging

import jaxtyping as jt
import numpy as np
import numpy.typing as npt

from lead.config import LeadConfig

LOG = logging.getLogger(__name__)


def smooth_route(
    lead_config: LeadConfig,
    route: jt.Float[npt.NDArray, "N 2"],
    target_first_distance: float,
) -> jt.Float[npt.NDArray, "N 2"]:
    """Smooth a route by removing duplicates and creating evenly-spaced interpolated waypoints.

    Removes duplicate waypoints while preserving path order, then generates
    evenly-spaced points via iterative line interpolation.

    Args:
        lead_config: Root config tree with route smoothing parameters such as
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
    return _interpolate_evenly_spaced_route(
        lead_config,
        route,
        target_first_distance,
    )


def _circle_line_segment_intersection(
    circle_center: jt.Float[npt.NDArray, " 2"],
    circle_radius: float,
    segment_start: jt.Float[npt.NDArray, " 2"],
    segment_end: jt.Float[npt.NDArray, " 2"],
    extend_to_infinite_line: bool = True,
    tangent_tolerance: float = 1e-9,
) -> list[tuple[float, float]]:
    """Find the intersection points between a circle and a line segment.

    Returns 0, 1, or 2 intersection points depending on whether the line misses,
    is tangent to, or crosses through the circle.
    """
    if np.linalg.norm(segment_start - segment_end) < 0.000000001:
        LOG.warning("Problem")

    (p1x, p1y), (p2x, p2y), (cx, cy) = segment_start, segment_end, circle_center
    (x1, y1), (x2, y2) = (p1x - cx, p1y - cy), (p2x - cx, p2y - cy)
    dx, dy = (x2 - x1), (y2 - y1)
    dr = (dx**2 + dy**2) ** 0.5
    big_d = x1 * y2 - x2 * y1
    discriminant = circle_radius**2 * dr**2 - big_d**2

    if discriminant < 0:  # No intersection between circle and line
        return []
    # There may be 0, 1, or 2 intersections with the segment
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
    if not extend_to_infinite_line:  # If only considering the segment, filter out intersections that do not fall within the segment
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
    if len(intersections) == 2 and abs(discriminant) <= tangent_tolerance:
        return [intersections[0]]
    return intersections


def _interpolate_evenly_spaced_route(
    lead_config: LeadConfig,
    route: jt.Float[npt.NDArray, "n 2"],
    target_first_distance: float,
) -> jt.Float[npt.NDArray, "n 2"]:
    """Generate evenly-spaced interpolated points along a route using circle-line intersection.

    Places the first point at ``target_first_distance`` from the origin and each
    subsequent point 1.0 m further along the route, extrapolating past the route
    end when needed to reach the target number of points.
    """
    interpolated_route_points = []

    point_spacing_meter = 1.0
    last_interpolated_point = np.array([0.0, 0.0])
    current_route_index = 0
    current_point = route[current_route_index]
    last_point = np.array([0.0, 0.0])
    first_iteration = True

    while (
        len(interpolated_route_points)
        < lead_config.policy.transfuser.num_route_points_smoothing
    ):
        # First point should be target_first_distance away from the vehicle.
        if not first_iteration:
            current_route_index += 1
            last_point = current_point

        if current_route_index < route.shape[0]:
            current_point = route[current_route_index]
            intersection = _circle_line_segment_intersection(
                circle_center=last_interpolated_point,
                circle_radius=(
                    point_spacing_meter
                    if not first_iteration
                    else target_first_distance
                ),
                segment_start=last_interpolated_point,
                segment_end=current_point,
                extend_to_infinite_line=True,
            )

        else:  # We hit the end of the input route. We extrapolate the last 2 points
            current_point = route[-1]
            last_point = route[-2]
            intersection = _circle_line_segment_intersection(
                circle_center=last_interpolated_point,
                circle_radius=point_spacing_meter,
                segment_start=last_point,
                segment_end=current_point,
                extend_to_infinite_line=True,
            )

        # 2 intersections: take the one closer to the current point.
        if len(intersection) > 1:
            point_1 = np.array(intersection[0])
            point_2 = np.array(intersection[1])
            direction = current_point - last_point
            projection_a_on_direction = np.dot(point_1, direction)
            projection_b_on_direction = np.dot(point_2, direction)

            if projection_a_on_direction > projection_b_on_direction:
                intersection_point = point_1
            else:
                intersection_point = point_2
        elif len(intersection) == 1:
            intersection_point = np.array(intersection[0])
        else:
            raise RuntimeError("No intersection found. This should never occur.")

        last_interpolated_point = intersection_point
        interpolated_route_points.append(intersection_point)
        # After the first point, every point is 1 m from the last.
        point_spacing_meter = 1.0

        first_iteration = False

    return np.array(interpolated_route_points)
