"""Lean numba ground removal, equivalent to :mod:`lead.common.sensors.ransac`."""

import jaxtyping as jt
import numpy as np
import numpy.typing as npt
from numba import njit, prange

from lead.common.sensors.ransac import generate_center_grid
from lead.config import ExpertConfig

# Points within 20 m (L1) of a center required to fit planes around it.
_MIN_POINTS_NEAR_CENTER = 3
# Ground percentile below which points are always marked as ground.
_GROUND_PERCENTILE = 30.0


@njit(cache=True)
def _median_of_lowest(values: jt.Float[npt.NDArray, " m"], count: int) -> float:
    """Median of the ``count`` smallest values, via partial selection."""
    if count % 2 == 0:
        partitioned = np.partition(values, count // 2)
        upper = partitioned[count // 2]
        lower = np.max(partitioned[: count // 2])
        return 0.5 * (lower + upper)
    partitioned = np.partition(values, count // 2)
    return partitioned[count // 2]


@njit(cache=True)
def _percentile(values: jt.Float[npt.NDArray, " m"], q: float) -> float:
    """Linear-interpolation percentile via partial selection."""
    position = (values.shape[0] - 1) * (q / 100.0)
    upper_index = int(np.ceil(position))
    fraction = position - np.floor(position)
    partitioned = np.partition(values, upper_index)
    upper = partitioned[upper_index]
    if fraction == 0.0:
        return upper
    lower = np.max(partitioned[:upper_index])
    return lower * (1.0 - fraction) + upper * fraction


@njit(cache=True)
def _fit_segment(
    x: jt.Float[npt.NDArray, " m"],
    y: jt.Float[npt.NDArray, " m"],
    z: jt.Float[npt.NDArray, " m"],
    distances: jt.Float[npt.NDArray, " m"],
    ground: jt.Bool[npt.NDArray, " m"],
    N_iter: int,
    N_LPR: int,
    Th_seeds: float,
    Th_dist: float,
    Th_center: float,
) -> None:
    """Iterative ground-plane fit of one radial segment, writing into ``ground``.

    Same estimator as ``fit_plane_on_radial_segment``: seeds from the median
    height of the ``N_LPR`` lowest near-center points, refined ``N_iter`` times,
    plus everything below the segment's 30th height percentile.
    """
    num_points = x.shape[0]
    if num_points < _MIN_POINTS_NEAR_CENTER:
        return

    num_candidates = 0
    for k in range(num_points):
        if distances[k] <= Th_center:
            num_candidates += 1
    if num_candidates < N_LPR:
        return
    candidate_z = np.empty(num_candidates, np.float64)
    index = 0
    for k in range(num_points):
        if distances[k] <= Th_center:
            candidate_z[index] = z[k]
            index += 1
    lpr_height = _median_of_lowest(candidate_z, N_LPR)

    # Initial seeds: near-center points close to the lowest-point median
    seeds = np.empty(num_points, np.bool_)
    num_seeds = 0
    for k in range(num_points):
        seeds[k] = distances[k] <= Th_center and abs(z[k] - lpr_height) < Th_seeds
        if seeds[k]:
            num_seeds += 1
    if num_seeds < _MIN_POINTS_NEAR_CENTER:
        return

    normal_matrix = np.empty((3, 3), np.float64)
    rhs = np.empty(3, np.float64)
    for _ in range(N_iter):
        sxx = sxy = sx = syy = sy = s1 = sxz = syz = sz = 0.0
        for k in range(num_points):
            if not seeds[k]:
                continue
            sxx += x[k] * x[k]
            sxy += x[k] * y[k]
            sx += x[k]
            syy += y[k] * y[k]
            sy += y[k]
            s1 += 1.0
            sxz += x[k] * z[k]
            syz += y[k] * z[k]
            sz += z[k]
        normal_matrix[0, 0] = sxx
        normal_matrix[0, 1] = sxy
        normal_matrix[0, 2] = sx
        normal_matrix[1, 0] = sxy
        normal_matrix[1, 1] = syy
        normal_matrix[1, 2] = sy
        normal_matrix[2, 0] = sx
        normal_matrix[2, 1] = sy
        normal_matrix[2, 2] = s1
        rhs[0] = sxz
        rhs[1] = syz
        rhs[2] = sz
        try:
            coefficients = np.linalg.solve(normal_matrix, rhs)
        except:
            return
        plane_a, plane_b, plane_d = (
            coefficients[0],
            coefficients[1],
            coefficients[2],
        )
        denominator = np.sqrt(plane_a**2 + plane_b**2 + 1)
        for k in range(num_points):
            height = plane_a * x[k] + plane_b * y[k] + plane_d
            seeds[k] = (z[k] - height) / denominator < Th_dist

    percentile_height = _percentile(z, _GROUND_PERCENTILE)
    for k in range(num_points):
        ground[k] = seeds[k] or z[k] < percentile_height


@njit(cache=True)
def _process_center(
    P: jt.Float[npt.NDArray, "n 3"],
    center_x: float,
    center_y: float,
    member_point: jt.Int[npt.NDArray, " e"],
    member_ground: jt.Bool[npt.NDArray, " e"],
    base: int,
    num_members: int,
    min_angles: jt.Float[npt.NDArray, " s"],
    max_angles: jt.Float[npt.NDArray, " s"],
    n_segments: int,
    N_iter: int,
    N_LPR: int,
    Th_seeds: float,
    Th_dist: float,
    Th_center: float,
    max_r: float,
    min_r: float,
) -> None:
    """Fit the radial segments of one center, writing member results in place."""
    num_points = P.shape[0]

    member_x = np.empty(num_members, np.float64)
    member_y = np.empty(num_members, np.float64)
    member_z = np.empty(num_members, np.float64)
    member_distance = np.empty(num_members, np.float64)
    segment_of_member = np.empty(num_members, np.int64)
    segment_counts = np.zeros(n_segments, np.int64)
    angle_step = max_angles[0] - min_angles[0]

    index = 0
    for k in range(num_points):
        distance = abs(P[k, 0] - center_x) + abs(P[k, 1] - center_y)
        if not (min_r < distance < max_r):
            continue
        member_point[base + index] = k
        member_x[index] = P[k, 0]
        member_y[index] = P[k, 1]
        member_z[index] = P[k, 2]
        member_distance[index] = distance

        # Segment with the reference boundary floats; on the (ulp-rare) overlap
        # of adjacent segments the higher one wins, like the reference loop
        angle = np.arctan2(P[k, 1] - center_y, P[k, 0] - center_x)
        guess = int((angle + np.pi) / angle_step)
        segment = -1
        high = min(guess + 1, n_segments - 1)
        low = max(guess - 1, 0)
        for candidate in range(high, low - 1, -1):
            if min_angles[candidate] <= angle < max_angles[candidate]:
                segment = candidate
                break
        segment_of_member[index] = segment
        if segment >= 0:
            segment_counts[segment] += 1
        index += 1

    # Group members by segment (stable counting sort keeps the point order)
    segment_starts = np.zeros(n_segments + 1, np.int64)
    for s in range(n_segments):
        segment_starts[s + 1] = segment_starts[s] + segment_counts[s]
    fill = segment_starts[:-1].copy()
    member_of_slot = np.empty(num_members, np.int64)
    for k in range(num_members):
        segment = segment_of_member[k]
        if segment < 0:
            continue
        member_of_slot[fill[segment]] = k
        fill[segment] += 1

    for s in range(n_segments):
        start, end = segment_starts[s], segment_starts[s + 1]
        size = end - start
        if size == 0:
            continue
        x = np.empty(size, np.float64)
        y = np.empty(size, np.float64)
        z = np.empty(size, np.float64)
        distances = np.empty(size, np.float64)
        ground = np.zeros(size, np.bool_)
        for slot in range(size):
            k = member_of_slot[start + slot]
            x[slot] = member_x[k]
            y[slot] = member_y[k]
            z[slot] = member_z[k]
            distances[slot] = member_distance[k]
        _fit_segment(
            x,
            y,
            z,
            distances,
            ground,
            N_iter,
            N_LPR,
            Th_seeds,
            Th_dist,
            Th_center,
        )
        for slot in range(size):
            member_ground[base + member_of_slot[start + slot]] = ground[slot]


@njit(parallel=True, cache=True)
def fit_ground_fast(
    P: jt.Float[npt.NDArray, "n 3"],
    centers: jt.Float[npt.NDArray, "c 2"],
    n_segments: int,
    N_iter: int,
    N_LPR: int,
    Th_seeds: float,
    Th_dist: float,
    Th_center: float,
    max_r: float,
    min_r: float,
) -> jt.Bool[npt.NDArray, " n"]:
    """Fit ground planes around all centers, returning the union of ground masks.

    Centers run in parallel into disjoint slices of flat member arrays; the
    final union is a serial scatter, so the result is deterministic.
    """
    num_points = P.shape[0]
    num_centers = centers.shape[0]

    ring_counts = np.zeros(num_centers, np.int64)
    for c in prange(num_centers):
        center_x, center_y = centers[c, 0], centers[c, 1]
        near = 0
        ring = 0
        for k in range(num_points):
            distance = abs(P[k, 0] - center_x) + abs(P[k, 1] - center_y)
            if distance < 20.0:
                near += 1
            if min_r < distance < max_r:
                ring += 1
        if near >= _MIN_POINTS_NEAR_CENTER:
            ring_counts[c] = ring

    offsets = np.zeros(num_centers + 1, np.int64)
    for c in range(num_centers):
        offsets[c + 1] = offsets[c] + ring_counts[c]
    num_members = offsets[num_centers]
    member_point = np.zeros(num_members, np.int64)
    member_ground = np.zeros(num_members, np.bool_)

    angle_step = 2 * np.pi / n_segments
    min_angles = np.empty(n_segments, np.float64)
    max_angles = np.empty(n_segments, np.float64)
    for s in range(n_segments):
        min_angles[s] = -np.pi + s * angle_step
        max_angles[s] = min_angles[s] + angle_step

    for c in prange(num_centers):
        if ring_counts[c] == 0:
            continue
        _process_center(
            P,
            centers[c, 0],
            centers[c, 1],
            member_point,
            member_ground,
            offsets[c],
            ring_counts[c],
            min_angles,
            max_angles,
            n_segments,
            N_iter,
            N_LPR,
            Th_seeds,
            Th_dist,
            Th_center,
            max_r,
            min_r,
        )

    union_ground_mask = np.zeros(num_points, dtype=np.bool_)
    for e in range(num_members):
        if member_ground[e]:
            union_ground_mask[member_point[e]] = True
    return union_ground_mask


def remove_ground_fast(
    P: jt.Float[npt.NDArray, "n 3"],
    config: ExpertConfig,
    n_segments: int = 8,
    N_iter: int = 2,
    N_LPR: int = 64,
    Th_seeds: float = 0.2,
    Th_dist: float = 0.15,
    Th_center: float = 50.0,
    min_r: float = 1.0,
    max_r: float = 32.0,
    center_resolution: float = 28.0,
) -> jt.Bool[npt.NDArray, " n"]:
    """Remove ground points from a point cloud with the lean numba pipeline.

    Same algorithm and parameters as :func:`lead.common.sensors.ransac.remove_ground`,
    restructured for machines with few CPU cores: order statistics use partial
    selection instead of full sorts, per-segment buffers are filled in single
    fused passes, and the center union is race-free.

    Args:
        P: shape=(N, 3): Point cloud to remove ground points from.
        config: Configuration object containing dataset parameters.
        n_segments: Number of radial segments to divide around each center.
        N_iter: Number of iterations for ground plane fitting.
        N_LPR: Number of lowest points to consider for initial seed points.
        Th_seeds: Threshold for selecting initial seed points.
        Th_dist: Threshold for inlier points to the fitted plane.
        Th_center: Threshold for selecting points near the center.
        min_r: Minimum radius for selecting points around a center.
        max_r: Maximum radius for selecting points around a center.
        center_resolution: Resolution for generating centers.
    Returns:
        Boolean mask shape (n,) indicating whether a point is on a ground or not.
            True indicates ground point.
            False indicates non-ground point.
    """
    if P.dtype != np.float64:
        P = P.astype(np.float64)
    centers = generate_center_grid(
        center_resolution,
        config.data_collection.min_x_meter,
        config.data_collection.max_x_meter,
        config.data_collection.min_y_meter,
        config.data_collection.max_y_meter,
    ).astype(np.float64)
    return fit_ground_fast(
        P,
        centers,
        n_segments,
        N_iter,
        N_LPR,
        Th_seeds,
        Th_dist,
        Th_center,
        max_r,
        min_r,
    )
