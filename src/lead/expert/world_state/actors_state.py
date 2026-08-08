"""Cached per-step view of the surrounding actors."""

import carla

from lead.common.runtime_property_caching import step_cached_property


def _is_static_car(box: dict) -> bool:
    """Whether a static prop's mesh is a parked car.

    Args:
        box: A bounding box dict of a static prop.

    Returns:
        True if the mesh path names a car.
    """
    mesh_path = box.get("mesh_path")
    return mesh_path is not None and "Car" in mesh_path


class ActorsStateMixin:
    """Cached per-step surrounding vehicles, walkers, bikers and statics."""

    @step_cached_property
    def actors(self) -> carla.ActorList:
        return self.carla_world.get_actors()

    @step_cached_property
    def leading_vehicle_ids(self):
        return self.privileged_route_planner.compute_leading_vehicles(
            self.vehicles_inside_bev,
            self.ego_vehicle.id,
        )

    @step_cached_property
    def trailing_vehicle_ids(self):
        return self.privileged_route_planner.compute_trailing_vehicles(
            self.vehicles_inside_bev,
            self.ego_vehicle.id,
        )

    def _actors_of_class_inside_bev(self, bb_class: str) -> list[carla.Actor]:
        """Actors of a box class inside the BEV, from this step's extracted boxes.

        The boxes already carry the ego-relative position and distance of every
        actor within the save radius, so no world query is needed.

        Args:
            bb_class: The ``bb["class"]`` value to select.

        Returns:
            The matching actors inside the BEV, sorted by actor ID.
        """
        boxes = [
            bb
            for bb in self.stored_bounding_boxes_of_this_step
            if bb["class"] == bb_class and self.is_bounding_box_inside_bev(bb)
        ]
        return [
            self.id2actor_map[bb["id"]] for bb in sorted(boxes, key=lambda bb: bb["id"])
        ]

    @step_cached_property
    def vehicles_inside_bev(self) -> list[carla.Actor]:
        # The ego's own box is class "ego_car", so it is added separately
        vehicles = self._actors_of_class_inside_bev("car")
        vehicles.append(self.ego_vehicle)
        vehicles.sort(key=lambda vehicle: vehicle.id)
        if (
            self.config_expert.data_collection.datagen
            and self.config_expert.occlusion.vehicle_occlusion_check
        ):  # Can only perform occlusion check if we have sensor data
            vehicles = [
                vehicle
                for vehicle in vehicles
                if not (
                    0
                    <= self.id2bb_map[vehicle.id]["num_points"]
                    < self.config_expert.occlusion.vehicle_occlusion_check_min_num_points
                    and 0
                    <= self.id2bb_map[vehicle.id]["visible_pixels"]
                    < self.config_expert.occlusion.vehicle_min_num_visible_pixels
                )
            ]
        return vehicles

    @step_cached_property
    def walkers_inside_bev(self) -> list[carla.Actor]:
        walkers = self._actors_of_class_inside_bev("walker")
        if (
            self.config_expert.data_collection.datagen
        ):  # Can only perform occlusion check if we have sensor data
            walkers = [
                walker
                for walker in walkers
                if not (
                    0
                    <= self.id2bb_map[walker.id]["visible_pixels"]
                    < self.config_expert.occlusion.pedestrian_min_num_visible_pixels
                )
            ]
        return walkers

    @step_cached_property
    def bikers_inside_bev(self) -> list[carla.Actor]:
        bikers = [
            b
            for b in self._actors_of_class_inside_bev("car")
            if b.type_id
            in [
                "vehicle.diamondback.century",
                "vehicle.gazelle.omafiets",
                "vehicle.bh.crossbike",
            ]
        ]
        if (
            self.config_expert.data_collection.datagen
            and self.config_expert.occlusion.bikers_occlusion_check
        ):  # Can only perform occlusion check if we have sensor data
            bikers = [
                biker
                for biker in bikers
                if not (
                    0
                    <= self.id2bb_map[biker.id]["visible_pixels"]
                    < self.config_expert.occlusion.bikers_occlusion_check_min_visible_pixels
                )
            ]
        return bikers

    @step_cached_property
    def static_inside_bev(self) -> list[carla.Actor]:
        """
        Get static actors inside the BEV (Bird's Eye View) range.
        This includes traffic lights and other static objects that are not vehicles or walkers.

        Returns:
            list: A list of static actors inside the BEV.
        """
        static_actors = self.actors.filter("*static*")
        return [actor for actor in static_actors if self.is_actor_inside_bev(actor)]

    @step_cached_property
    def distance_to_pedestrian(self) -> float:
        """
        Calculate the distance to the closest pedestrian within the BEV (Bird's Eye View) range.

        Returns:
            float: The distance to the closest pedestrian, or infinity if no pedestrians are inside the BEV.
        """
        pedestrians = self.walkers_inside_bev
        if not pedestrians:
            return float("inf")

        # Get the location of the ego vehicle
        ego_location = self.ego_vehicle.get_location()

        # Find the closest pedestrian
        closest_pedestrian = min(
            pedestrians,
            key=lambda p: ego_location.distance(p.get_location()),
        )
        return ego_location.distance(closest_pedestrian.get_location())

    @step_cached_property
    def distance_to_biker(self) -> float:
        """
        Calculate the distance to the closest biker within the BEV (Bird's Eye View) range.

        Returns:
            float: The distance to the closest biker, or infinity if no bikers are inside the BEV.
        """
        bikers = self.bikers_inside_bev
        if not bikers:
            return float("inf")

        # Get the location of the ego vehicle
        ego_location = self.ego_vehicle.get_location()

        # Find the closest biker
        closest_biker = min(
            bikers,
            key=lambda b: ego_location.distance(b.get_location()),
        )
        return ego_location.distance(closest_biker.get_location())

    @step_cached_property
    def num_parking_vehicles_in_proximity(self):
        count = 0
        for bb in self.stored_bounding_boxes_of_this_step:
            if not (-8 <= bb["position"][0] <= 32 and abs(bb["position"][1]) <= 10):
                continue
            if bb["class"] == "static" and _is_static_car(bb):
                count += 1
            if bb["class"] == "static_prop_car":
                count += 1
        return count

    @step_cached_property
    def second_highest_speed(self):
        speeds = []
        for vehicle in self.vehicles_inside_bev:
            if vehicle.id == self.ego_vehicle.id:
                continue
            speeds.append(vehicle.get_velocity().length())
        if len(speeds) == 0:
            return 0.0
        if len(speeds) == 1:
            return speeds[0]
        speeds = sorted(speeds, reverse=True)
        return speeds[1]

    @step_cached_property
    def second_highest_speed_limit(self):
        speed_limits = []
        for vehicle in self.vehicles_inside_bev:
            if vehicle.id == self.ego_vehicle.id:
                continue
            speed_limits.append(vehicle.get_speed_limit() / 3.6)
        if len(speed_limits) == 0:
            return 0.0
        if len(speed_limits) == 1:
            return speed_limits[0]
        speed_limits = sorted(speed_limits, reverse=True)
        return speed_limits[1]
