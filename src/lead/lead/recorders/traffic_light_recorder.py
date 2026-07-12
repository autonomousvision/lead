"""Recorder for the traffic-light detections modality stream."""

import logging
import typing

import carla
import numpy as np
import shapely.geometry as geom
from py123d.api.map.arrow.arrow_map_api import ArrowMapAPI, get_lru_cached_map_api
from py123d.datatypes import (
    BaseModality,
    EgoStateSE3,
    Lane,
    MapLayer,
    Timestamp,
    TrafficLightDetection,
    TrafficLightDetections,
    TrafficLightStatus,
)
from shapely.ops import substring

from lead.lead.recorders.base_recorder import BaseRecorder
from lead.lead.utils import carla_to_123d

if typing.TYPE_CHECKING:
    from pathlib import Path

    from lead.lead.data_collection import ExpertData

LOG = logging.getLogger(__name__)

# Max ego-to-waypoint distance (m) for logging a TL detection.
TRAFFIC_LIGHT_LOG_RADIUS_M: float = 80.0

# Extra margin (m) beyond the log radius for the cheap per-light pre-filter. A
# light's affected waypoints sit up to ~40 m from the light body, so a light
# further than the radius plus this margin cannot have an in-radius waypoint.
TL_PREFILTER_MARGIN_M: float = 50.0

# Max distance (m) from a TL waypoint to a 123D lane centerline for a match.
LANE_CENTERLINE_MATCH_THRESHOLD_M: float = 0.1

# Length (m) of the lane centerline start segment used to match TL waypoints.
LANE_START_MATCH_LENGTH_M: float = 5.0

# Whether to only assign TL to lanes on intersections.
FORCE_TL_ON_INTERSECTIONS: bool = True


class TrafficLightRecorder(BaseRecorder):
    """Records per-lane traffic-light states via 123D map queries.

    For each CARLA traffic light, takes its affected-lane waypoints, keeps
    those within ``TRAFFIC_LIGHT_LOG_RADIUS_M`` of the ego, queries the 123D
    map, and emits one detection per lane whose centerline runs within
    ``LANE_CENTERLINE_MATCH_THRESHOLD_M`` of the waypoint.
    """

    def __init__(self, expert: "ExpertData", map_arrow_path: "Path") -> None:
        """Initialize recorder and open the queryable 123D map.

        Args:
            expert: The expert agent owning the CARLA state to record.
            map_arrow_path: Path to the town's converted 123D map file.
        """
        super().__init__(expert)
        self.map_api: ArrowMapAPI | None = None
        if map_arrow_path.exists():
            self.map_api = get_lru_cached_map_api(map_arrow_path)
        else:
            LOG.warning(
                f"Map Arrow file not found at {map_arrow_path.absolute()}. "
                "Traffic-light lane lookups will be skipped.",
            )

        # Lights and their affected waypoints never move; both are fetched from
        # CARLA once (waypoints lazily per light) instead of every save tick.
        self._lights: list[carla.TrafficLight] | None = None
        self._light_locations: list[carla.Location] = []
        self._light_positions: np.ndarray = np.zeros((0, 3))
        self._affected_lane_locations: dict[int, list[carla.Location]] = {}

    def _ensure_light_cache(self) -> None:
        """Fetch all traffic lights and their world positions once."""
        if self._lights is not None:
            return
        self._lights = [
            actor
            for actor in self.expert.carla_world.get_actors().filter("*traffic_light*")
            if isinstance(actor, carla.TrafficLight)
        ]
        self._light_locations = [actor.get_location() for actor in self._lights]
        self._light_positions = np.array(
            [[loc.x, loc.y, loc.z] for loc in self._light_locations],
        ).reshape(-1, 3)

    def record(
        self,
        input_data: dict,
        timestamp: Timestamp,
        ego_state: EgoStateSE3,
    ) -> list[BaseModality]:
        """Extract per-lane traffic-light detections for this tick.

        Args:
            input_data: Post-tick sensor data (unused; actors are read live).
            timestamp: Current simulation timestamp.
            ego_state: Ego state of the current tick (unused; the CARLA ego
                location is read directly).

        Returns:
            A single-element list with this tick's traffic-light detections
            (also written when no lights are nearby).
        """
        if self.map_api is None:
            return [TrafficLightDetections(detections=[], timestamp=timestamp)]

        expert = self.expert
        ego_loc = expert.ego_location

        # One query point per affected-lane waypoint (CARLA: +y right → 123D /
        # ISO 8855: +y left, hence the y negation). Filter is per-waypoint so
        # only lanes near the ego are logged, even at wide intersections.
        query_points: list[geom.Point] = []
        point_status: list[TrafficLightStatus] = []
        point_actor_ids: list[int] = []
        actor_ids_in_radius: set[int] = set()
        prefilter_radius = TRAFFIC_LIGHT_LOG_RADIUS_M + TL_PREFILTER_MARGIN_M
        self._ensure_light_cache()
        ego_position = np.array([ego_loc.x, ego_loc.y, ego_loc.z])
        light_distances = np.linalg.norm(
            self._light_positions - ego_position,
            axis=1,
        )
        for index in np.where(light_distances <= prefilter_radius + 1.0)[0]:
            actor = self._lights[index]
            # Skip the expensive affected-waypoint query for far-away lights.
            if ego_loc.distance(self._light_locations[index]) > prefilter_radius:
                continue
            lane_locations = self._affected_lane_locations.get(actor.id)
            if lane_locations is None:
                lane_locations = [
                    waypoint.transform.location
                    for waypoint in actor.get_affected_lane_waypoints()
                ]
                self._affected_lane_locations[actor.id] = lane_locations
            status = carla_to_123d.carla_traffic_light_status_to_py123d(actor.state)
            for lane_loc in lane_locations:
                if ego_loc.distance(lane_loc) > TRAFFIC_LIGHT_LOG_RADIUS_M:
                    continue
                actor_ids_in_radius.add(actor.id)
                query_points.append(geom.Point(lane_loc.x, -lane_loc.y))
                point_status.append(status)
                point_actor_ids.append(actor.id)

        if not actor_ids_in_radius:
            return [TrafficLightDetections(detections=[], timestamp=timestamp)]

        # Single batched map call. `dwithin` (not `intersects`) absorbs sub-cm
        # noise on polygon edges; same threshold reused as the centerline filter.
        result = self.map_api.query_object_ids(
            query_points,
            [MapLayer.LANE],
            predicate="dwithin",
            distance=LANE_CENTERLINE_MATCH_THRESHOLD_M,
        )
        lane_dict = typing.cast(dict[int, list[int]], result.get(MapLayer.LANE, {}))

        # Reject candidates whose centerline doesn't actually pass near the
        # waypoint (the polygon query can grab adjacent/overlapping lanes).
        # One status per lane: first match wins; conflicts are warned and dropped.
        lane_status: dict[int, TrafficLightStatus] = {}
        assigned_actor_ids: set[int] = set()
        rejected = 0
        for point_idx, lane_ids in lane_dict.items():
            status = point_status[point_idx]
            query_point = query_points[point_idx]
            actor_id = point_actor_ids[point_idx]
            for lane_id in lane_ids:
                lane = self.map_api.get_map_object_in_layer(lane_id, MapLayer.LANE)
                assert isinstance(lane, Lane)
                if lane.lane_group is not None:
                    lane_in_intersection = lane.lane_group.intersection_id is not None
                else:
                    lane_in_intersection = False
                lane_in_intersection = (
                    lane_in_intersection or not FORCE_TL_ON_INTERSECTIONS
                )

                centerline = lane.centerline.linestring
                start_segment = substring(
                    centerline,
                    0.0,
                    min(LANE_START_MATCH_LENGTH_M, centerline.length),
                )
                if (
                    start_segment.distance(query_point)
                    > LANE_CENTERLINE_MATCH_THRESHOLD_M
                    or not lane_in_intersection
                ):
                    rejected += 1
                    continue

                assigned_actor_ids.add(actor_id)
                existing = lane_status.get(lane_id)
                if existing is None:
                    lane_status[lane_id] = status
                elif existing != status:
                    LOG.warning(
                        f"Conflicting traffic-light status for lane {lane_id}: "
                        f"keeping {existing.name}, ignoring {status.name}.",
                    )

        unassigned_actor_ids = actor_ids_in_radius - assigned_actor_ids
        if unassigned_actor_ids:
            LOG.warning(
                f"Traffic-light assignment incomplete: "
                f"{len(assigned_actor_ids)}/{len(actor_ids_in_radius)} assigned. "
                f"assigned={sorted(assigned_actor_ids)}, "
                f"unassigned={sorted(unassigned_actor_ids)}, "
                f"rejected_candidates={rejected}.",
            )

        detections = [
            TrafficLightDetection(lane_id=lane_id, status=status)
            for lane_id, status in lane_status.items()
        ]
        return [TrafficLightDetections(detections=detections, timestamp=timestamp)]
