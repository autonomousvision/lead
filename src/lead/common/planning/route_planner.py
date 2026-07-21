"""Tracking of the ego's progress along a route's target points."""

from __future__ import annotations

import typing
from typing import Any

import jaxtyping as jt
import numpy as np
import numpy.typing as npt

from lead.common.localization import gps as gps_utils

if typing.TYPE_CHECKING:
    import carla

__all__ = ["RoutePlanner"]


class RoutePlanner:
    """Tracks which target point of a route the ego is currently heading to.

    The target points are fixed once set; navigating the route advances
    ``target_point_index``, the index of the last target point the ego has
    come within ``min_distance`` of.
    """

    def __init__(self, min_distance: float, max_distance: float) -> None:
        """Initialize the route planner with distance constraints.

        Args:
            min_distance: Minimum distance threshold for route planning.
            max_distance: Maximum distance threshold for route planning.
        """
        self.target_points: jt.Float[npt.NDArray, "n 3"] = np.zeros((0, 3))
        self.target_point_distances: jt.Float[npt.NDArray, " n"] = np.zeros(0)
        self.target_point_index: int = 0

        self.min_distance = min_distance
        self.max_distance = max_distance

    @property
    def is_last(self) -> bool:
        """Whether the ego has reached the last target point of the route."""
        return self.target_point_index >= len(self.target_points) - 2

    def set_route(
        self,
        global_plan: list[tuple[Any, Any]],
        is_gps: bool = False,
        carla_map: carla.Map | None = None,
        lat_ref: float | None = None,
        lon_ref: float | None = None,
    ) -> None:
        """Set the global route plan for navigation.

        Consecutive duplicates are merged, so every target point is distinct.

        Args:
            global_plan: List of tuples containing positions and (ignored) commands.
            is_gps: Whether the positions are in GPS coordinates.
            carla_map: CARLA map object for waypoint extension.
            lat_ref: Latitude reference for GPS conversion.
            lon_ref: Longitude reference for GPS conversion.
        """
        route: list[jt.Float[npt.NDArray, " 3"]] = []

        for pos, _ in global_plan:
            if is_gps:
                assert lat_ref is not None and lon_ref is not None, (
                    "lat_ref and lon_ref are required when is_gps=True"
                )
                pos = np.array([pos["lat"], pos["lon"], pos["z"]])
                pos = gps_utils.convert_gps_to_carla(pos, lat_ref, lon_ref)
            else:
                # important to use the z variable, otherwise there are some rare bugs at carla.map.get_waypoint(carla.Location)
                pos = np.array([pos.location.x, pos.location.y, pos.location.z])

            route.append(pos)

        if carla_map is not None:
            import carla

            for _ in range(50):
                loc = carla.Location(
                    x=route[-1][0],
                    y=route[-1][1],
                    z=route[-1][2],
                )
                next_loc = carla_map.get_waypoint(loc).next(1)[0].transform.location
                route.append(np.array([next_loc.x, next_loc.y, next_loc.z]))

        self.target_points = np.array(
            [
                point
                for i, point in enumerate(route)
                if i == 0 or not np.allclose(point[:2], route[i - 1][:2])
            ],
        )
        self.target_point_index = 0

        # We do the calculations in the beginning once so that we don't have
        # to do them every time in run_step
        self.target_point_distances = np.zeros(len(self.target_points))
        self.target_point_distances[1:] = np.linalg.norm(
            np.diff(self.target_points[:, :2], axis=0),
            axis=1,
        )

    def run_step(self, gps: jt.Float[npt.NDArray, " 3"]) -> int:
        """Advance the target point index to the ego's current position.

        Args:
            gps: Current GPS position as a 3D array (x, y, z).

        Returns:
            The index of the previous target point.
        """
        if self.is_last:
            return self.target_point_index

        farthest_in_range = -np.inf
        cumulative_distance = 0.0
        for i in range(self.target_point_index + 1, len(self.target_points)):
            if cumulative_distance > self.max_distance:
                break

            cumulative_distance += self.target_point_distances[i]

            diff = self.target_points[i] - gps
            distance = (diff[0] ** 2 + diff[1] ** 2) ** 0.5

            if farthest_in_range < distance <= self.min_distance:
                farthest_in_range = distance
                # The last target point is never the current one: it stays
                # ahead of the ego as the route's final destination.
                self.target_point_index = min(i, len(self.target_points) - 2)
        return self.target_point_index
