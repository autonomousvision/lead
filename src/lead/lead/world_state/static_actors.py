"""Cached world positions of static signs and traffic lights for fast radius queries."""

import logging

import carla
import numpy as np

import lead.common.common_utils as common_utils

LOG = logging.getLogger(__name__)

# Margin absorbing float32 (CARLA) vs float64 (numpy) rounding in the prefilter;
# candidates inside it are re-checked with CARLA's own distance.
_PREFILTER_MARGIN_M = 1.0


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
                traffic_light_pos_on_street = common_utils.get_relative_transform(
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
