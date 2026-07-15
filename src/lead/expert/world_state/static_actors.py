"""Cached world positions of static signs and traffic lights for fast radius queries."""

import dataclasses
import logging

import carla
import numpy as np

import lead.common.geometry as geometry
from lead.common import constants

LOG = logging.getLogger(__name__)

# Margin absorbing float32 (CARLA) vs float64 (numpy) rounding in the prefilter;
# candidates inside it are re-checked with CARLA's own distance.
_PREFILTER_MARGIN_M = 1.0

# How far a level bounding box may be from an actor and still be taken to be that
# actor's geometry. Overhead lights hang some way from their pole, so this is
# generous; beyond it the box is kept without an actor rather than mis-attributed.
_SIGN_ASSOCIATION_RADIUS_M = 15.0


@dataclasses.dataclass(frozen=True)
class PhysicalSign:
    """A traffic light or sign at its true world pose.

    Attributes:
        bounding_box: The level bounding box holding the rendered geometry.
        actor: The ``traffic.*`` actor this geometry belongs to, if one could be
            associated; the box has no state or type without it.
        lane: ``(road_id, lane_id)`` of the lane the sign stands by, or None if
            it does not project onto a driving lane.
    """

    bounding_box: carla.BoundingBox
    actor: carla.Actor | None
    lane: tuple[int, int] | None


class StaticActorsMixin:
    """Radius queries against stop signs and traffic lights, whose poses never change.

    The world positions are fetched from CARLA once per episode; per-tick
    queries are a vectorized distance prefilter followed by the exact CARLA
    distance check on the few candidates, so results match the uncached loops.
    """

    def _ensure_stop_sign_cache(self) -> None:
        """Fetch all stop signs and their trigger-volume world positions once."""
        if getattr(self, "_stop_signs", None) is not None:
            return
        stop_signs: list[carla.Actor] = []
        trigger_locations: list[carla.Location] = []
        for actor in self.carla_world.get_actors().filter("*traffic.stop*"):
            try:
                trigger_position = actor.get_transform().transform(
                    actor.trigger_volume.location,
                )
            except:
                LOG.info(
                    "Warning! Error caught in get_nearby_objects. (probably AttributeError: actor.trigger_volume)",
                )
                LOG.info("Skipping this object.")
                continue
            stop_signs.append(actor)
            trigger_locations.append(
                carla.Location(
                    x=trigger_position.x,
                    y=trigger_position.y,
                    z=trigger_position.z,
                ),
            )
        self._stop_signs = stop_signs
        self._stop_sign_trigger_locations = trigger_locations
        self._stop_sign_trigger_positions = np.array(
            [[loc.x, loc.y, loc.z] for loc in trigger_locations],
        ).reshape(-1, 3)
        self._stop_signs_by_id = {actor.id: actor for actor in stop_signs}

    def nearby_stop_signs(self, search_radius: float) -> list[carla.Actor]:
        """Stop signs whose trigger boxes are within the radius around the ego.

        Args:
            search_radius: The radius (in meters) around the ego vehicle.

        Returns:
            The stop-sign actors within the search radius.
        """
        self._ensure_stop_sign_cache()
        ego_location = self.ego_location
        ego_position = np.array([ego_location.x, ego_location.y, ego_location.z])
        distances = np.linalg.norm(
            self._stop_sign_trigger_positions - ego_position,
            axis=1,
        )
        nearby = []
        for index in np.where(distances < search_radius + _PREFILTER_MARGIN_M)[0]:
            if (
                self._stop_sign_trigger_locations[index].distance(ego_location)
                < search_radius
            ):
                nearby.append(self._stop_signs[index])
        return nearby

    def stop_sign_actor(self, actor_id: int) -> carla.Actor | None:
        """Return the cached stop-sign actor with this ID, or None.

        Args:
            actor_id: CARLA actor ID of a stop sign.

        Returns:
            The stop-sign actor, or None if the ID is not a stop sign.
        """
        self._ensure_stop_sign_cache()
        return self._stop_signs_by_id.get(actor_id)

    def _ensure_physical_sign_cache(self) -> None:
        """Fetch the true geometry of every traffic light and sign once.

        ``actor.bounding_box`` of a ``traffic.*`` actor is its trigger volume on
        the governed road, so the rendered geometry is cached from the level
        bounding boxes, each associated with the nearest actor of a matching
        kind so that its ID and state can travel with the box.
        """
        if getattr(self, "_physical_signs", None) is not None:
            return

        actors = self.carla_world.get_actors()
        lights = list(actors.filter("*traffic.traffic_light*"))
        signs = [
            actor
            for actor in actors.filter("*traffic.*")
            if actor.type_id not in constants.NON_SIGN_TRAFFIC_TYPES
            or actor.type_id == "traffic.stop"
        ]

        physical_signs: list[PhysicalSign] = []
        for city_object_label, candidates in (
            (carla.CityObjectLabel.TrafficLight, lights),
            (carla.CityObjectLabel.TrafficSigns, signs),
        ):
            level_bounding_boxes = self.carla_world.get_level_bbs(city_object_label)
            candidate_positions = np.array(
                [
                    [actor.get_location().x, actor.get_location().y]
                    for actor in candidates
                ],
            ).reshape(-1, 2)
            for level_bounding_box in level_bounding_boxes:
                actor = None
                if len(candidates) > 0:
                    location = level_bounding_box.location
                    distances = np.linalg.norm(
                        candidate_positions - np.array([location.x, location.y]),
                        axis=1,
                    )
                    nearest = int(np.argmin(distances))
                    if distances[nearest] <= _SIGN_ASSOCIATION_RADIUS_M:
                        actor = candidates[nearest]
                physical_signs.append(
                    PhysicalSign(
                        bounding_box=level_bounding_box,
                        actor=actor,
                        lane=self.sign_lane(level_bounding_box.location),
                    ),
                )

        self._physical_signs = physical_signs
        self._physical_sign_positions = np.array(
            [
                [
                    sign.bounding_box.location.x,
                    sign.bounding_box.location.y,
                    sign.bounding_box.location.z,
                ]
                for sign in physical_signs
            ],
        ).reshape(-1, 3)

    def nearby_physical_signs(
        self,
        search_radius: float,
    ) -> list[tuple[int, PhysicalSign]]:
        """Traffic lights and signs whose true geometry is within the radius.

        Args:
            search_radius: The radius (in meters) around the ego vehicle.

        Returns:
            Pairs of (cache index, physical sign) within the search radius. The
            index is stable for a town and is what gives each box its ID.
        """
        self._ensure_physical_sign_cache()
        if len(self._physical_signs) == 0:
            return []
        ego_location = self.ego_location
        ego_position = np.array([ego_location.x, ego_location.y, ego_location.z])
        distances = np.linalg.norm(
            self._physical_sign_positions - ego_position,
            axis=1,
        )
        return [
            (int(index), self._physical_signs[index])
            for index in np.where(distances < search_radius)[0]
        ]

    def sign_lane(self, location: carla.Location) -> tuple[int, int] | None:
        """Road and lane a sign projects onto.

        Args:
            location: World location of the sign.

        Returns:
            ``(road_id, lane_id)`` of the nearest driving lane, or None if the
            sign does not project onto one.
        """
        waypoint = self.carla_world_map.get_waypoint(
            location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        return None if waypoint is None else (waypoint.road_id, waypoint.lane_id)

    def close_traffic_light_indices(self, radius: float) -> list[int]:
        """Indices into ``list_traffic_lights`` of lights within the radius.

        Args:
            radius: The radius (in meters) around the ego vehicle.

        Returns:
            Indices of nearby traffic lights, in ``list_traffic_lights`` order.
        """
        if getattr(self, "_traffic_light_centers", None) is None:
            self._traffic_light_centers = [
                carla.Location(center) for _, center, _ in self.list_traffic_lights
            ]
            self._traffic_light_center_positions = np.array(
                [[loc.x, loc.y, loc.z] for loc in self._traffic_light_centers],
            ).reshape(-1, 3)
        ego_location = self.ego_location
        ego_position = np.array([ego_location.x, ego_location.y, ego_location.z])
        distances = np.linalg.norm(
            self._traffic_light_center_positions - ego_position,
            axis=1,
        )
        return [
            int(index)
            for index in np.where(distances < radius + _PREFILTER_MARGIN_M)[0]
            if self._traffic_light_centers[index].distance(ego_location) <= radius
        ]

    def traffic_light_stop_boxes(
        self,
        traffic_light: carla.TrafficLight,
        traffic_light_waypoints: list[carla.Waypoint],
    ) -> list[tuple[carla.Waypoint, carla.BoundingBox]]:
        """Stop-line bounding boxes of a traffic light, one per affected waypoint.

        The box poses are static, so their location and rotation are computed
        once per light and only the CARLA objects are rebuilt per call.

        Args:
            traffic_light: The traffic light actor.
            traffic_light_waypoints: The light's affected-lane waypoints.

        Returns:
            Pairs of (waypoint, stop-line bounding box).
        """
        if getattr(self, "_traffic_light_stop_box_params", None) is None:
            self._traffic_light_stop_box_params = {}
        params = self._traffic_light_stop_box_params.get(traffic_light.id)
        if params is None:
            traffic_light_transform = traffic_light.get_transform()
            traffic_light_matrix = np.array(traffic_light_transform.get_matrix())
            rotation = traffic_light_transform.rotation
            params = []
            for wp in traffic_light_waypoints:
                # The z of the traffic light is relative to the street
                traffic_light_pos_on_street = geometry.get_relative_transform(
                    ego_matrix=np.array(wp.transform.get_matrix()),
                    vehicle_matrix=traffic_light_matrix,
                )
                wp_location = wp.transform.location
                params.append(
                    (
                        wp,
                        (
                            wp_location.x,
                            wp_location.y,
                            wp_location.z + traffic_light_pos_on_street[-1],
                        ),
                        (rotation.pitch, rotation.yaw, rotation.roll),
                    ),
                )
            self._traffic_light_stop_box_params[traffic_light.id] = params

        boxes = []
        for wp, (x, y, z), (pitch, yaw, roll) in params:
            bounding_box = carla.BoundingBox(
                carla.Location(x=x, y=y, z=z),
                carla.Vector3D(1.5, 1.5, 0.5),
            )
            bounding_box.rotation = carla.Rotation(pitch=pitch, yaw=yaw, roll=roll)
            boxes.append((wp, bounding_box))
        return boxes
