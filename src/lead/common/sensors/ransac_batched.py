"""Batched numpy/torch ground removal, equivalent to :mod:`lead.common.sensors.ransac`."""

from __future__ import annotations

import typing

import jaxtyping as jt
import numpy as np
import numpy.typing as npt

if typing.TYPE_CHECKING:
    import torch

from lead.common.sensors.ransac import generate_center_grid
from lead.config import ExpertConfig

# Points within 20 m (L1) of a center required to fit planes around it.
_MIN_POINTS_NEAR_CENTER = 3
# Ground percentile below which points are always marked as ground.
_GROUND_PERCENTILE = 30.0


def remove_ground_batched(
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
    backend: str = "numpy",
) -> jt.Bool[npt.NDArray, " n"]:
    """Remove ground points from a point cloud, batched over all centers and segments.

    Same algorithm and parameters as :func:`lead.common.sensors.ransac.remove_ground`;
    instead of looping over centers and radial segments, every (center, segment)
    bucket is processed in one array pipeline, which is faster when only a
    couple of CPU cores are available.

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
        backend: Array backend executing the pipeline, "numpy" or "torch".
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
    if backend == "torch":
        return _fit_ground_torch(
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
    return _fit_ground_numpy(
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


def _percentile_positions(
    counts: jt.Int[npt.NDArray, " B"],
    q: float,
) -> tuple[
    jt.Int[npt.NDArray, " B"],
    jt.Int[npt.NDArray, " B"],
    jt.Float[npt.NDArray, " B"],
]:
    """Order-statistic indices and interpolation weight of a linear percentile."""
    positions = (counts - 1) * (q / 100.0)
    lower = np.floor(positions)
    fraction = positions - lower
    lower_index = np.clip(lower.astype(np.int64), 0, None)
    upper_index = np.clip(np.ceil(positions).astype(np.int64), 0, None)
    return lower_index, upper_index, fraction


def _fit_ground_numpy(
    P: jt.Float[npt.NDArray, "n 3"],
    centers: jt.Float[npt.NDArray, "C 2"],
    n_segments: int,
    N_iter: int,
    N_LPR: int,
    Th_seeds: float,
    Th_dist: float,
    Th_center: float,
    max_r: float,
    min_r: float,
) -> jt.Bool[npt.NDArray, " n"]:
    """Fit ground planes for all (center, segment) buckets in one numpy pipeline."""
    num_points = P.shape[0]
    ground_mask = np.zeros(num_points, dtype=np.bool_)
    if num_points == 0:
        return ground_mask

    x, y, z = P[:, 0], P[:, 1], P[:, 2]

    # One (center, point) entry per point within a center's ring, center-major
    # like the reference loop over centers
    l1_distances = np.abs(x[None, :] - centers[:, 0, None]) + np.abs(
        y[None, :] - centers[:, 1, None],
    )
    center_active = (l1_distances < 20).sum(axis=1) >= _MIN_POINTS_NEAR_CENTER
    in_ring = (l1_distances > min_r) & (l1_distances < max_r) & center_active[:, None]
    center_index, point_index = np.nonzero(in_ring)
    if point_index.size == 0:
        return ground_mask
    entry_distance = l1_distances[in_ring]
    entry_x = x[point_index]
    entry_y = y[point_index]
    entry_z = z[point_index]

    # Radial segment of each entry, with the reference per-segment boundary floats
    angles = np.arctan2(
        entry_y - centers[center_index, 1],
        entry_x - centers[center_index, 0],
    )
    angle_step = 2 * np.pi / n_segments
    min_angles = -np.pi + np.arange(n_segments) * angle_step
    segment = np.searchsorted(min_angles, angles, side="right") - 1
    segment_valid = angles < (min_angles[segment] + angle_step)

    num_buckets = centers.shape[0] * n_segments
    bucket = center_index * n_segments + segment
    bucket[~segment_valid] = num_buckets  # spill bucket, never marked as ground

    # Group entries by bucket with ascending z inside each bucket: float sort by
    # z, then a stable integer (radix) sort by bucket
    z_order = np.argsort(entry_z)
    order = z_order[np.argsort(bucket[z_order], kind="stable")]
    bucket_sorted = bucket[order]
    z_sorted = entry_z[order]
    x_sorted = entry_x[order]
    y_sorted = entry_y[order]
    point_sorted = point_index[order]

    counts = np.bincount(bucket_sorted, minlength=num_buckets + 1)[:num_buckets]
    starts = np.concatenate(([0], np.cumsum(counts)))[:-1]

    # Median height of the N_LPR lowest seed candidates per bucket
    if Th_center >= max_r:
        candidate_counts = counts
        candidate_starts = starts
        candidate_z = z_sorted
        candidate_entry = np.ones(bucket_sorted.shape[0], dtype=np.bool_)
    else:
        candidate_entry = entry_distance[order] <= Th_center
        candidate_counts = np.bincount(
            bucket_sorted[candidate_entry],
            minlength=num_buckets + 1,
        )[:num_buckets]
        candidate_starts = np.concatenate(([0], np.cumsum(candidate_counts)))[:-1]
        candidate_z = z_sorted[candidate_entry]

    valid = (counts >= _MIN_POINTS_NEAR_CENTER) & (candidate_counts >= N_LPR)
    last_candidate = max(candidate_z.shape[0] - 1, 0)
    safe_starts = np.where(valid, candidate_starts, 0)
    if N_LPR % 2 == 0:
        lpr_height = 0.5 * (
            candidate_z[np.clip(safe_starts + N_LPR // 2 - 1, 0, last_candidate)]
            + candidate_z[np.clip(safe_starts + N_LPR // 2, 0, last_candidate)]
        )
    else:
        lpr_height = candidate_z[np.clip(safe_starts + N_LPR // 2, 0, last_candidate)]

    lpr_of_entry = lpr_height[np.clip(bucket_sorted, 0, num_buckets - 1)]
    seeds = (
        candidate_entry
        & (np.abs(z_sorted - lpr_of_entry) < Th_seeds)
        & (bucket_sorted < num_buckets)
    )

    # Iterative plane fits: per-bucket normal equations via weighted bincounts
    ones = np.ones_like(z_sorted)
    features = (x_sorted, y_sorted, ones)
    is_ground = np.zeros(bucket_sorted.shape[0], dtype=np.bool_)
    for _ in range(N_iter):
        weights = seeds.astype(np.float64)
        normal_matrix = np.empty((num_buckets, 3, 3))
        rhs = np.empty((num_buckets, 3))
        for i in range(3):
            for j in range(i, 3):
                normal_matrix[:, i, j] = np.bincount(
                    bucket_sorted,
                    weights=weights * features[i] * features[j],
                    minlength=num_buckets + 1,
                )[:num_buckets]
                normal_matrix[:, j, i] = normal_matrix[:, i, j]
            rhs[:, i] = np.bincount(
                bucket_sorted,
                weights=weights * features[i] * z_sorted,
                minlength=num_buckets + 1,
            )[:num_buckets]

        determinant = np.linalg.det(normal_matrix)
        solvable = np.isfinite(determinant) & (determinant != 0.0)
        valid = valid & solvable
        normal_matrix[~solvable] = np.eye(3)
        coefficients = np.linalg.solve(normal_matrix, rhs)

        plane_a = coefficients[np.clip(bucket_sorted, 0, num_buckets - 1), 0]
        plane_b = coefficients[np.clip(bucket_sorted, 0, num_buckets - 1), 1]
        plane_d = coefficients[np.clip(bucket_sorted, 0, num_buckets - 1), 2]
        plane_heights = plane_a * x_sorted + plane_b * y_sorted + plane_d
        distances = (z_sorted - plane_heights) / np.sqrt(
            plane_a**2 + plane_b**2 + 1,
        )
        is_ground = distances < Th_dist
        seeds = is_ground

    # Points below the bucket's 30th height percentile are ground as well
    lower_index, upper_index, fraction = _percentile_positions(
        counts,
        _GROUND_PERCENTILE,
    )
    last_entry = max(z_sorted.shape[0] - 1, 0)
    end_index = np.clip(counts - 1, 0, None)
    z_lower = z_sorted[
        np.clip(starts + np.minimum(lower_index, end_index), 0, last_entry)
    ]
    z_upper = z_sorted[
        np.clip(starts + np.minimum(upper_index, end_index), 0, last_entry)
    ]
    percentile_height = z_lower * (1 - fraction) + z_upper * fraction
    below_percentile = (
        z_sorted < percentile_height[np.clip(bucket_sorted, 0, num_buckets - 1)]
    )

    entry_is_ground = valid[np.clip(bucket_sorted, 0, num_buckets - 1)] & (
        (is_ground | below_percentile) & (bucket_sorted < num_buckets)
    )
    ground_mask[point_sorted[entry_is_ground]] = True
    return ground_mask


def _segment_min_angles(n_segments: int) -> jt.Float[npt.NDArray, " s"]:
    """Lower angle boundaries of the radial segments, reference floats."""
    angle_step = 2 * np.pi / n_segments
    return -np.pi + np.arange(n_segments) * angle_step


def _fit_ground_torch(
    P: jt.Float[npt.NDArray, "n 3"],
    centers: jt.Float[npt.NDArray, "C 2"],
    n_segments: int,
    N_iter: int,
    N_LPR: int,
    Th_seeds: float,
    Th_dist: float,
    Th_center: float,
    max_r: float,
    min_r: float,
) -> jt.Bool[npt.NDArray, " n"]:
    """Fit ground planes for all (center, segment) buckets in one torch pipeline."""
    import torch

    if P.shape[0] == 0:
        return np.zeros(0, dtype=np.bool_)
    mask = _fit_ground_torch_core(
        torch.from_numpy(P),
        torch.from_numpy(centers),
        torch.from_numpy(_segment_min_angles(n_segments)),
        n_segments,
        N_iter,
        N_LPR,
        Th_seeds,
        Th_dist,
        Th_center,
        max_r,
        min_r,
    )
    return mask.numpy()


class TorchGroundRemover:
    """Reusable torch ground removal keeping constants and buffers on one device.

    Centers and segment boundaries are uploaded once at construction; each call
    performs a single host-to-device copy of the points (through a pinned
    staging buffer on CUDA) and a single device-to-host copy of the mask.
    Intermediate device memory is reused by torch's caching allocator.
    """

    def __init__(
        self,
        config: ExpertConfig,
        device: str = "cuda",
        n_segments: int = 8,
        N_iter: int = 2,
        N_LPR: int = 64,
        Th_seeds: float = 0.2,
        Th_dist: float = 0.15,
        Th_center: float = 50.0,
        min_r: float = 1.0,
        max_r: float = 32.0,
        center_resolution: float = 28.0,
    ) -> None:
        """Upload the constant tensors of the pipeline to the target device.

        Args:
            config: Configuration object containing dataset parameters.
            device: Torch device running the pipeline.
            n_segments: Number of radial segments to divide around each center.
            N_iter: Number of iterations for ground plane fitting.
            N_LPR: Number of lowest points to consider for initial seed points.
            Th_seeds: Threshold for selecting initial seed points.
            Th_dist: Threshold for inlier points to the fitted plane.
            Th_center: Threshold for selecting points near the center.
            min_r: Minimum radius for selecting points around a center.
            max_r: Maximum radius for selecting points around a center.
            center_resolution: Resolution for generating centers.
        """
        import torch

        self._torch = torch
        self._device = torch.device(device)
        self._params = (
            n_segments,
            N_iter,
            N_LPR,
            Th_seeds,
            Th_dist,
            Th_center,
            max_r,
            min_r,
        )
        centers = generate_center_grid(
            center_resolution,
            config.data_collection.min_x_meter,
            config.data_collection.max_x_meter,
            config.data_collection.min_y_meter,
            config.data_collection.max_y_meter,
        ).astype(np.float64)
        self._centers = torch.from_numpy(centers).to(self._device)
        self._min_angles = torch.from_numpy(_segment_min_angles(n_segments)).to(
            self._device,
        )
        self._staging: torch.Tensor | None = None

    def __call__(
        self,
        P: jt.Float[npt.NDArray, "n 3"],
    ) -> jt.Bool[npt.NDArray, " n"]:
        """Remove ground points from a point cloud on the wrapped device.

        Args:
            P: shape=(N, 3): Point cloud to remove ground points from.

        Returns:
            Boolean mask shape (n,) indicating whether a point is on a ground or not.
        """
        torch = self._torch
        if P.dtype != np.float64:
            P = P.astype(np.float64)
        num_points = P.shape[0]
        if num_points == 0:
            return np.zeros(0, dtype=np.bool_)

        if self._staging is None or self._staging.shape[0] < num_points:
            self._staging = torch.empty(
                (num_points, 3),
                dtype=torch.float64,
                pin_memory=self._device.type == "cuda",
            )
        self._staging[:num_points].copy_(torch.from_numpy(P))
        points = self._staging[:num_points].to(self._device, non_blocking=True)

        mask = _fit_ground_torch_core(
            points,
            self._centers,
            self._min_angles,
            *self._params,
        )
        return mask.cpu().numpy()


def _fit_ground_torch_core(
    points: torch.Tensor,
    centers_t: torch.Tensor,
    min_angles: torch.Tensor,
    n_segments: int,
    N_iter: int,
    N_LPR: int,
    Th_seeds: float,
    Th_dist: float,
    Th_center: float,
    max_r: float,
    min_r: float,
) -> torch.Tensor:
    """Torch pipeline over prepared tensors; runs on the tensors' device."""
    import torch

    device = points.device
    num_points = points.shape[0]
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    l1_distances = torch.abs(x[None, :] - centers_t[:, 0, None]) + torch.abs(
        y[None, :] - centers_t[:, 1, None],
    )
    center_active = (l1_distances < 20).sum(dim=1) >= _MIN_POINTS_NEAR_CENTER
    in_ring = (l1_distances > min_r) & (l1_distances < max_r) & center_active[:, None]
    entry = in_ring.nonzero(as_tuple=True)
    center_index, point_index = entry[0], entry[1]
    if point_index.numel() == 0:
        return torch.zeros(num_points, dtype=torch.bool, device=device)
    entry_distance = l1_distances[in_ring]
    entry_x = x[point_index]
    entry_y = y[point_index]
    entry_z = z[point_index]

    angles = torch.atan2(
        entry_y - centers_t[center_index, 1],
        entry_x - centers_t[center_index, 0],
    )
    angle_step = 2 * np.pi / n_segments
    segment = torch.searchsorted(min_angles, angles, right=True) - 1
    segment_valid = angles < (min_angles[segment] + angle_step)

    num_buckets = centers_t.shape[0] * n_segments
    bucket = center_index * n_segments + segment
    bucket[~segment_valid] = num_buckets

    z_order = torch.argsort(entry_z, stable=True)
    order = z_order[torch.argsort(bucket[z_order], stable=True)]
    bucket_sorted = bucket[order]
    z_sorted = entry_z[order]
    x_sorted = entry_x[order]
    y_sorted = entry_y[order]
    point_sorted = point_index[order]

    counts = torch.bincount(bucket_sorted, minlength=num_buckets + 1)[:num_buckets]
    starts = torch.cumsum(counts, dim=0) - counts

    if Th_center >= max_r:
        candidate_counts = counts
        candidate_starts = starts
        candidate_z = z_sorted
        candidate_entry = torch.ones_like(bucket_sorted, dtype=torch.bool)
    else:
        candidate_entry = entry_distance[order] <= Th_center
        candidate_counts = torch.bincount(
            bucket_sorted[candidate_entry],
            minlength=num_buckets + 1,
        )[:num_buckets]
        candidate_starts = torch.cumsum(candidate_counts, dim=0) - candidate_counts
        candidate_z = z_sorted[candidate_entry]

    valid = (counts >= _MIN_POINTS_NEAR_CENTER) & (candidate_counts >= N_LPR)
    last_candidate = max(candidate_z.shape[0] - 1, 0)
    safe_starts = torch.where(
        valid,
        candidate_starts,
        torch.zeros_like(candidate_starts),
    )
    if N_LPR % 2 == 0:
        lpr_height = 0.5 * (
            candidate_z[(safe_starts + N_LPR // 2 - 1).clamp(0, last_candidate)]
            + candidate_z[(safe_starts + N_LPR // 2).clamp(0, last_candidate)]
        )
    else:
        lpr_height = candidate_z[(safe_starts + N_LPR // 2).clamp(0, last_candidate)]

    bucket_clamped = bucket_sorted.clamp(max=num_buckets - 1)
    seeds = (
        candidate_entry
        & (torch.abs(z_sorted - lpr_height[bucket_clamped]) < Th_seeds)
        & (bucket_sorted < num_buckets)
    )

    ones = torch.ones_like(z_sorted)
    features = (x_sorted, y_sorted, ones)
    is_ground = torch.zeros_like(seeds)
    for _ in range(N_iter):
        weights = seeds.to(torch.float64)
        normal_matrix = torch.empty(
            (num_buckets, 3, 3),
            dtype=torch.float64,
            device=device,
        )
        rhs = torch.empty((num_buckets, 3), dtype=torch.float64, device=device)
        for i in range(3):
            for j in range(i, 3):
                normal_matrix[:, i, j] = torch.bincount(
                    bucket_sorted,
                    weights=weights * features[i] * features[j],
                    minlength=num_buckets + 1,
                )[:num_buckets]
                normal_matrix[:, j, i] = normal_matrix[:, i, j]
            rhs[:, i] = torch.bincount(
                bucket_sorted,
                weights=weights * features[i] * z_sorted,
                minlength=num_buckets + 1,
            )[:num_buckets]

        determinant = torch.linalg.det(normal_matrix)
        solvable = torch.isfinite(determinant) & (determinant != 0.0)
        valid = valid & solvable
        normal_matrix[~solvable] = torch.eye(3, dtype=torch.float64, device=device)
        coefficients = torch.linalg.solve(normal_matrix, rhs)

        plane = coefficients[bucket_clamped]
        plane_heights = plane[:, 0] * x_sorted + plane[:, 1] * y_sorted + plane[:, 2]
        distances = (z_sorted - plane_heights) / torch.sqrt(
            plane[:, 0] ** 2 + plane[:, 1] ** 2 + 1,
        )
        is_ground = distances < Th_dist
        seeds = is_ground

    positions = (counts - 1).to(torch.float64) * (_GROUND_PERCENTILE / 100.0)
    lower = torch.floor(positions)
    fraction = positions - lower
    last_entry = max(z_sorted.shape[0] - 1, 0)
    end_index = (counts - 1).clamp(min=0)
    lower_index = torch.minimum(lower.to(torch.int64).clamp(min=0), end_index)
    upper_index = torch.minimum(torch.ceil(positions).to(torch.int64), end_index)
    z_lower = z_sorted[(starts + lower_index).clamp(0, last_entry)]
    z_upper = z_sorted[(starts + upper_index).clamp(0, last_entry)]
    percentile_height = z_lower * (1 - fraction) + z_upper * fraction
    below_percentile = z_sorted < percentile_height[bucket_clamped]

    entry_is_ground = valid[bucket_clamped] & (
        (is_ground | below_percentile) & (bucket_sorted < num_buckets)
    )
    ground_mask = torch.zeros(num_points, dtype=torch.bool, device=device)
    ground_mask[point_sorted[entry_is_ground]] = True
    return ground_mask
