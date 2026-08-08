"""Bounding-box extraction of all relevant actors around the ego vehicle."""

import logging
import typing

import carla
import numpy as np

from lead.common import constants, geometry
from lead.expert.driving import occlusion
from lead.expert.utils import roadgraph
from lead.expert.world_state import static_actors

LOG = logging.getLogger(__name__)

# Physical boxes are keyed by their index in the per-town geometry cache, which
# would collide with CARLA actor IDs, so they are shifted into their own ID
# namespace to stay distinguishable for per-box tracking and future-pose grafting.
PHYSICAL_BOX_ID_OFFSET = 200_000_000


def physical_box_id(index: int) -> int:
    """Return the box ID of a physical traffic light or sign.

    Args:
        index: The sign's index in the per-town physical geometry cache.

    Returns:
        The ID under which the sign's physical box is logged.
    """
    return PHYSICAL_BOX_ID_OFFSET + int(index)


def physical_box_class(type_id: str | None) -> str:
    """Return the box class of a physical traffic light or sign.

    Args:
        type_id: Blueprint ID of the sign's actor, or None if no actor could be
            associated with the geometry.

    Returns:
        The ``class`` field to store on the box.
    """
    if type_id == "traffic.traffic_light":
        return "traffic_light_physical"
    if type_id == "traffic.stop":
        return "stop_sign_physical"
    return "traffic_sign"


class BoundingBoxesMixin:
    """Extracts per-step bounding boxes of vehicles, walkers, statics and signs."""

    def physical_sign_box(
        self,
        index: int,
        physical_sign: static_actors.PhysicalSign,
        ego_matrix: np.ndarray,
        ego_yaw: float,
    ) -> dict[str, typing.Any]:
        """Build a bounding box at a traffic light's or sign's true world pose.

        Unlike the lane-projected traffic light and stop sign boxes, this keeps
        the geometry where it physically hangs.

        Args:
            index: The sign's index in the per-town cache, used as its box ID.
            physical_sign: The sign's level geometry and its associated actor.
            ego_matrix: The ego's 4x4 world transform.
            ego_yaw: The ego's yaw in radians.

        Returns:
            The sign's bounding box in the expert's box format.
        """
        bounding_box = physical_sign.bounding_box
        actor = physical_sign.actor
        # A level bounding box already holds world-space geometry. Its z is the
        # box centre, whereas the expert's boxes carry the floor centre.
        box_transform = carla.Transform(
            carla.Location(
                x=bounding_box.location.x,
                y=bounding_box.location.y,
                z=bounding_box.location.z - bounding_box.extent.z,
            ),
            bounding_box.rotation,
        )
        box_matrix = np.array(box_transform.get_matrix())
        relative_pos = geometry.get_relative_transform(ego_matrix, box_matrix)
        relative_yaw = geometry.normalize_angle_rad(
            np.deg2rad(bounding_box.rotation.yaw) - ego_yaw,
        )
        extent = bounding_box.extent
        type_id = None if actor is None else actor.type_id
        return {
            "class": physical_box_class(type_id),
            "extent": [extent.x, extent.y, extent.z],
            "position": [relative_pos[0], relative_pos[1], relative_pos[2]],
            "yaw": relative_yaw,
            "distance": float(np.linalg.norm(relative_pos)),
            "id": physical_box_id(index),
            "actor_id": None if actor is None else int(actor.id),
            "type_id": type_id,
            "matrix": box_transform.get_matrix(),
            "num_points": -1,
            "visible_pixels": -1,
        }

    def is_actor_inside_bev(self, actor: carla.Actor) -> bool:
        """
        Check if actor is visible in TransFuser++'s planning visible range.
        This is used to filter out actors that are not visible to TransFuser++'s
        planning module even though they might be visible in the camera.
        """
        actor_in_ego = geometry.get_relative_transform(
            self.ego_matrix,
            np.array(actor.get_transform().get_matrix()),
        )
        x_ego, y_ego, _ = actor_in_ego
        return bool(
            self.config_expert.data_collection.min_x_meter - 2
            < x_ego
            < self.config_expert.data_collection.max_x_meter + 2
            and self.config_expert.data_collection.min_y_meter - 2
            < y_ego
            < self.config_expert.data_collection.max_y_meter + 2
            and np.linalg.norm(actor_in_ego)
            < self.config_expert.driving.bb_save_radius,
        )

    def is_bounding_box_inside_bev(self, bounding_box: dict[str, typing.Any]) -> bool:
        """Same check as :meth:`is_actor_inside_bev` on an already extracted box.

        Args:
            bounding_box: One entry of this step's extracted bounding boxes.

        Returns:
            True if the box lies inside TransFuser++'s planning visible range.
        """
        x_ego, y_ego = bounding_box["position"][0], bounding_box["position"][1]
        return bool(
            self.config_expert.data_collection.min_x_meter - 2
            < x_ego
            < self.config_expert.data_collection.max_x_meter + 2
            and self.config_expert.data_collection.min_y_meter - 2
            < y_ego
            < self.config_expert.data_collection.max_y_meter + 2
            and bounding_box["distance"] < self.config_expert.driving.bb_save_radius,
        )

    def get_bounding_boxes(self, input_data: dict) -> list[dict]:
        boxes = []

        ego_transform = self.ego_vehicle.get_transform()
        ego_control = self.ego_vehicle.get_control()
        ego_velocity = self.ego_vehicle.get_velocity()
        ego_matrix = np.array(ego_transform.get_matrix())
        ego_rotation = ego_transform.rotation
        ego_extent = self.ego_vehicle.bounding_box.extent
        ego_speed = self._get_forward_speed(
            transform=ego_transform,
            velocity=ego_velocity,
        )
        ego_dx = np.array([ego_extent.x, ego_extent.y, ego_extent.z])
        ego_yaw = np.deg2rad(ego_rotation.yaw)
        ego_brake = ego_control.brake

        relative_yaw = 0.0
        relative_pos = geometry.get_relative_transform(ego_matrix, ego_matrix)

        ego_wp = self.carla_world_map.get_waypoint(
            self.ego_vehicle.get_location(),
            project_to_road=True,
            lane_type=carla.libcarla.LaneType.Any,
        )

        # Check for possible vehicle obstacles
        # Retrieve all relevant actors (one world query per step)
        self._actors = self.actors
        vehicle_list = self._actors.filter("*vehicle*")

        # Actor handles of every box appended below, keyed by actor ID
        actor_map: dict[int, carla.Actor] = {
            int(self.ego_vehicle.id): self.ego_vehicle,
        }

        # --- Start iterating through actors ---
        result = {
            "class": "ego_car",
            "extent": [ego_dx[0], ego_dx[1], ego_dx[2]],
            "position": [0, 0, 0],
            "yaw": 0.0,
            "num_points": -1,
            "distance": 0,
            "speed": ego_speed,
            "brake": ego_brake,
            "id": int(self.ego_vehicle.id),
            "matrix": ego_transform.get_matrix(),
            "visible_pixels": -1,
        }
        boxes.append(result)

        # Per-actor visible pixel counts from the instance segmentation cameras
        pixel_counts = input_data["instance_pixel_counts"]
        self._pixel_counts = pixel_counts

        for vehicle in vehicle_list:
            if (
                vehicle.get_location().distance(self.ego_location)
                < self.config_expert.driving.bb_save_radius
            ) and vehicle.id != self.ego_vehicle.id:
                vehicle_transform = vehicle.get_transform()
                vehicle_rotation = vehicle_transform.rotation
                vehicle_matrix = np.array(vehicle_transform.get_matrix())
                vehicle_control: carla.VehicleControl = vehicle.get_control()
                vehicle_velocity = vehicle.get_velocity()
                vehicle_extent = vehicle.bounding_box.extent
                vehicle_id = vehicle.id

                vehicle_extent_list = [
                    vehicle_extent.x,
                    vehicle_extent.y,
                    vehicle_extent.z,
                ]
                yaw = np.deg2rad(vehicle_rotation.yaw)

                relative_yaw = geometry.normalize_angle_rad(yaw - ego_yaw)
                relative_pos = geometry.get_relative_transform(
                    ego_matrix,
                    vehicle_matrix,
                )
                vehicle_speed = self._get_forward_speed(
                    transform=vehicle_transform,
                    velocity=vehicle_velocity,
                )
                vehicle_brake = vehicle_control.brake

                distance = np.linalg.norm(relative_pos)

                # LiDAR hit count per bounding box; used to filter invisible boxes.
                vehicle_extent_array = np.array(vehicle_extent_list)
                if input_data.get("lidar") is not None:
                    num_in_bbox_points = len(
                        occlusion.get_points_in_actor_frame_and_in_bbox(
                            ego_matrix,
                            vehicle_matrix,
                            vehicle_extent_array,
                            input_data["lidar"],
                            pad=True,
                        ),
                    )
                else:
                    num_in_bbox_points = -1

                if input_data.get("radar") is not None:
                    num_in_bb_radar_points = len(
                        occlusion.get_points_in_actor_frame_and_in_bbox(
                            ego_matrix,
                            vehicle_matrix,
                            vehicle_extent_array,
                            input_data["radar"],
                            pad=True,
                        ),
                    )
                else:
                    num_in_bb_radar_points = -1

                result = {
                    "class": "car",
                    "extent": vehicle_extent_list,
                    "position": [relative_pos[0], relative_pos[1], relative_pos[2]],
                    "yaw": relative_yaw,
                    "num_points": int(num_in_bbox_points),
                    "num_radar_points": int(num_in_bb_radar_points),
                    "distance": distance,
                    "speed": vehicle_speed,
                    "brake": vehicle_brake,
                    "id": int(vehicle_id),
                    "role_name": vehicle.attributes["role_name"],
                    "type_id": vehicle.type_id,
                    "matrix": vehicle_transform.get_matrix(),
                    "visible_pixels": occlusion.visible_pixels_of_actor(
                        pixel_counts,
                        vehicle_id,
                    ),
                }
                boxes.append(result)
                actor_map[int(vehicle_id)] = vehicle

        walkers = self._actors.filter("*walker*")
        for walker in walkers:
            if (
                walker.get_location().distance(self.ego_location)
                < self.config_expert.driving.bb_save_radius
            ):
                walker_transform = walker.get_transform()
                walker_velocity = walker.get_velocity()
                walker_rotation = walker_transform.rotation
                walker_matrix = np.array(walker_transform.get_matrix())
                walker_id = walker.id
                walker_extent = walker.bounding_box.extent
                walker_extent = [walker_extent.x, walker_extent.y, walker_extent.z]
                yaw = np.deg2rad(walker_rotation.yaw)

                relative_yaw = geometry.normalize_angle_rad(yaw - ego_yaw)
                relative_pos = geometry.get_relative_transform(
                    ego_matrix,
                    walker_matrix,
                )

                walker_speed = self._get_forward_speed(
                    transform=walker_transform,
                    velocity=walker_velocity,
                )

                # LiDAR hit count per bounding box; used to filter invisible boxes.
                walker_extent_array = np.array(walker_extent)
                if input_data.get("lidar") is not None:
                    num_in_bbox_points = len(
                        occlusion.get_points_in_actor_frame_and_in_bbox(
                            ego_matrix,
                            walker_matrix,
                            walker_extent_array,
                            input_data["lidar"],
                            pad=True,
                        ),
                    )
                else:
                    num_in_bbox_points = -1

                if input_data.get("radar") is not None:
                    num_in_bb_radar_points = len(
                        occlusion.get_points_in_actor_frame_and_in_bbox(
                            ego_matrix,
                            walker_matrix,
                            walker_extent_array,
                            input_data["radar"],
                            pad=True,
                        ),
                    )
                else:
                    num_in_bb_radar_points = -1

                distance = np.linalg.norm(relative_pos)

                result = {
                    "class": "walker",
                    "role_name": walker.attributes["role_name"],
                    "extent": walker_extent,
                    "position": [relative_pos[0], relative_pos[1], relative_pos[2]],
                    "yaw": relative_yaw,
                    "num_points": int(num_in_bbox_points),
                    "num_radar_points": int(num_in_bb_radar_points),
                    "distance": distance,
                    "speed": walker_speed,
                    "id": int(walker_id),
                    "matrix": walker_transform.get_matrix(),
                    "visible_pixels": occlusion.visible_pixels_of_actor(
                        pixel_counts,
                        walker_id,
                    ),
                }
                boxes.append(result)
                actor_map[int(walker_id)] = walker

        # Note this only saves static actors, which does not include static background objects
        if self.config_expert.data_collection.eval_expert:
            LOG.info("Skipping static actor bounding boxes in eval_expert mode.")
        else:
            static_list = self._actors.filter("*static*")
            LOG.debug(f"Found {len(static_list)} static actors in the scene.")
            for static in static_list:
                if (
                    static.get_location().distance(self.ego_location)
                    < self.config_expert.driving.bb_save_radius
                ):
                    static_transform = static.get_transform()
                    static_velocity = static.get_velocity()
                    static_rotation = static_transform.rotation
                    static_matrix = np.array(static_transform.get_matrix())
                    static_id = static.id
                    type_id = static.type_id
                    mesh_path = static.attributes.get("mesh_path", None)
                    static_extent = static.bounding_box.extent
                    static_extent = [static_extent.x, static_extent.y, static_extent.z]
                    if mesh_path is not None and mesh_path in constants.LOOKUP_TABLE:
                        static_extent = constants.LOOKUP_TABLE[mesh_path]
                    if type_id == "static.prop.trafficwarning":
                        static_extent[0], static_extent[1] = (
                            self.config_expert.data_collection.traffic_warning_bb_size[
                                0
                            ],
                            self.config_expert.data_collection.traffic_warning_bb_size[
                                1
                            ],
                        )
                    elif type_id == "static.prop.constructioncone":
                        static_extent[0], static_extent[1] = (
                            self.config_expert.data_collection.construction_cone_bb_size[
                                0
                            ],
                            self.config_expert.data_collection.construction_cone_bb_size[
                                1
                            ],
                        )

                    yaw = np.deg2rad(static_rotation.yaw)

                    relative_yaw = geometry.normalize_angle_rad(yaw - ego_yaw)
                    relative_pos = geometry.get_relative_transform(
                        ego_matrix,
                        static_matrix,
                    )

                    static_speed = self._get_forward_speed(
                        transform=static_transform,
                        velocity=static_velocity,
                    )

                    distance = np.linalg.norm(relative_pos)

                    result = {
                        "class": "static",
                        "extent": static_extent,
                        "position": [relative_pos[0], relative_pos[1], relative_pos[2]],
                        "yaw": relative_yaw,
                        "num_points": -1,
                        "distance": distance,
                        "speed": static_speed,
                        "id": int(static_id),
                        "matrix": static_transform.get_matrix(),
                        "type_id": type_id,
                        "mesh_path": mesh_path,
                    }
                    result["visible_pixels"] = occlusion.visible_pixels_of_actor(
                        pixel_counts,
                        static_id,
                    )

                    boxes.append(result)
                    actor_map[int(static_id)] = static

        # --- Traffic light ---
        # New logic needed to handle some weird traffic lights in Town12 and Town13
        for (
            traffic_light,
            original_traffic_light_bounding_box,
            traffic_light_state,
            traffic_light_id,
            traffic_light_affects_ego,
        ) in self.close_traffic_lights:
            original_waypoint = self.carla_world_map.get_waypoint(
                original_traffic_light_bounding_box.location,
            )
            waypoint_transform_matrix = np.array(
                original_waypoint.transform.get_matrix(),
            )
            traffic_light_transform_matrix = np.array(
                traffic_light.get_transform().get_matrix(),
            )

            traffic_light_in_waypoint = geometry.get_relative_transform(
                ego_matrix=waypoint_transform_matrix,
                vehicle_matrix=traffic_light_transform_matrix,
            )

            is_over_head_traffic_light = (
                self.town in ["Town11", "Town12", "Town13", "Town15"]
                and abs(traffic_light_in_waypoint[0]) < 4.0
            )
            is_europe_traffic_light = (
                self.town
                in [
                    "Town01",
                    "Town02",
                    "Town03",
                    "Town04",
                    "Town05",
                    "Town06",
                    "Town07",
                    "Town10HD",
                ]
                and abs(traffic_light_in_waypoint[0]) < 4.0
            )
            if traffic_light_affects_ego:
                red_light_stop_waypoints = roadgraph.get_stop_waypoints(
                    self.ego_wp,
                    traffic_light,
                )

                # Create bounding boxes for each additional lane
                for i, red_light_stop_waypoint in enumerate(red_light_stop_waypoints):
                    # Create bounding box for this waypoint
                    duplicated_traffic_light_bounding_box = (
                        roadgraph.create_bounding_box_for_waypoint(
                            original_traffic_light_bounding_box,
                            red_light_stop_waypoint,
                        )
                    )

                    traffic_light_extent = [
                        duplicated_traffic_light_bounding_box.extent.x,
                        duplicated_traffic_light_bounding_box.extent.y,
                        duplicated_traffic_light_bounding_box.extent.z,
                    ]

                    # Keep original rotation/yaw, only change position
                    traffic_light_transform = carla.Transform(
                        duplicated_traffic_light_bounding_box.location,
                        original_traffic_light_bounding_box.rotation,
                    )
                    traffic_light_rotation = traffic_light_transform.rotation
                    traffic_light_matrix = np.array(
                        traffic_light_transform.get_matrix(),
                    )
                    yaw = np.deg2rad(traffic_light_rotation.yaw)

                    relative_yaw = geometry.normalize_angle_rad(yaw - ego_yaw)
                    relative_pos = geometry.get_relative_transform(
                        ego_matrix,
                        traffic_light_matrix,
                    )

                    distance = np.linalg.norm(relative_pos)

                    # Keep the original distance_to_physical_traffic_light
                    distance_traffic_light_to_bounding_box = np.linalg.norm(
                        np.array(
                            [
                                traffic_light.get_transform().location.x
                                - duplicated_traffic_light_bounding_box.location.x,
                                traffic_light.get_transform().location.y
                                - duplicated_traffic_light_bounding_box.location.y,
                            ],
                        ),
                    )

                    # Duplicated bounding box result
                    if i == 0:
                        result = {
                            "class": "traffic_light",
                            "extent": traffic_light_extent,
                            "position": [
                                relative_pos[0],
                                relative_pos[1],
                                relative_pos[2],
                            ],
                            "yaw": relative_yaw,
                            "distance": distance,
                            "state": str(traffic_light_state),
                            "id": int(traffic_light_id),  # Keep original ID
                            "actor_id": int(traffic_light_id),
                            "affects_ego": traffic_light_affects_ego,
                            "matrix": traffic_light_transform.get_matrix(),
                            "distance_to_physical_traffic_light": distance_traffic_light_to_bounding_box,
                            "dummy_traffic_light_bounding_box": False,
                            "same_lane_as_ego": bool(
                                red_light_stop_waypoint.lane_id == ego_wp.lane_id,
                            ),
                            "is_over_head_traffic_light": is_over_head_traffic_light,
                            "is_europe_traffic_light": is_europe_traffic_light,
                        }
                    else:
                        result = {
                            "class": "traffic_light",
                            "extent": traffic_light_extent,
                            "position": [
                                relative_pos[0],
                                relative_pos[1],
                                relative_pos[2],
                            ],
                            "yaw": relative_yaw,
                            "distance": distance,
                            "state": str(traffic_light_state),
                            # One ID per lane of this light, so that the lanes
                            # neither collide with each other nor with the light
                            # itself, and keep their ID across frames.
                            "id": self.negative_id_counter(
                                (
                                    int(traffic_light.id),
                                    int(red_light_stop_waypoint.road_id),
                                    int(red_light_stop_waypoint.lane_id),
                                ),
                            ),
                            "actor_id": int(traffic_light_id),
                            "affects_ego": traffic_light_affects_ego,
                            "matrix": traffic_light_transform.get_matrix(),
                            "distance_to_physical_traffic_light": distance_traffic_light_to_bounding_box,
                            "dummy_traffic_light_bounding_box": True,
                            "same_lane_as_ego": bool(
                                red_light_stop_waypoint.lane_id == ego_wp.lane_id,
                            ),
                            "is_over_head_traffic_light": is_over_head_traffic_light,
                            "is_europe_traffic_light": is_europe_traffic_light,
                        }
                    boxes.append(result)
                    if i == 0:
                        actor_map[int(traffic_light_id)] = traffic_light

        for stop_sign in self.close_stop_signs:
            stop_sign_extent = [
                stop_sign[0].extent.x,
                stop_sign[0].extent.y,
                stop_sign[0].extent.z,
            ]

            stop_sign_transform = carla.Transform(
                stop_sign[0].location,
                stop_sign[0].rotation,
            )
            stop_sign_rotation = stop_sign_transform.rotation
            stop_sign_matrix = np.array(stop_sign_transform.get_matrix())
            yaw = np.deg2rad(stop_sign_rotation.yaw)

            relative_yaw = geometry.normalize_angle_rad(yaw - ego_yaw)
            relative_pos = geometry.get_relative_transform(
                ego_matrix,
                stop_sign_matrix,
            )

            distance = np.linalg.norm(relative_pos)

            result = {
                "class": "stop_sign",
                "extent": stop_sign_extent,
                "position": [relative_pos[0], relative_pos[1], relative_pos[2]],
                "yaw": relative_yaw,
                "distance": distance,
                "id": int(stop_sign[1]),
                "actor_id": int(stop_sign[1]),
                "affects_ego": stop_sign[2],
                "matrix": stop_sign_transform.get_matrix(),
            }
            boxes.append(result)
            sign_actor = self.stop_sign_actor(int(stop_sign[1]))
            if sign_actor is not None:
                actor_map[int(stop_sign[1])] = sign_actor

        # --- Physical poses of the signs and lights themselves ---
        # The traffic light and stop sign boxes above are projected onto the
        # lanes they govern, which is what TransFuser++ consumes but discards
        # where the sign physically hangs. These boxes keep that pose, and are
        # collected for every nearby sign regardless of its relation to the ego.
        for index, physical_sign in self.nearby_physical_signs(
            self.config_expert.driving.bb_save_radius,
        ):
            result = self.physical_sign_box(
                index,
                physical_sign,
                ego_matrix,
                ego_yaw,
            )
            actor = physical_sign.actor
            if result["class"] == "traffic_light_physical":
                result["state"] = str(actor.state)
            elif result["class"] == "traffic_sign":
                result["speed_limit"] = (
                    None
                    if actor is None
                    else roadgraph.speed_limit_of_sign(actor.type_id)
                )
            lane = physical_sign.lane
            result["road_id"] = None if lane is None else int(lane[0])
            result["lane_id"] = None if lane is None else int(lane[1])
            boxes.append(result)
            if actor is not None:
                actor_map[result["id"]] = actor

        # Others static meshes that dont belong to an actors but are still relevant for perception
        if self.config_expert.data_collection.eval_expert:
            LOG.info("Skipping static actor bounding boxes in eval_expert mode.")
        else:
            static_parking_id_start = 1e8
            existed_bboxes_ids = set([box["id"] for box in boxes])

            # Level bounding boxes are static map geometry, fetch them once
            if not hasattr(self, "_level_car_bounding_boxes"):
                self._level_car_bounding_boxes = self.carla_world.get_level_bbs(
                    carla.CityObjectLabel.Car,
                )
                self._level_car_bb_centers = np.array(
                    [
                        [bb.location.x, bb.location.y, bb.location.z]
                        for bb in self._level_car_bounding_boxes
                    ],
                ).reshape(-1, 3)

            # Global-space boxes of existing actors, used to make sure the level
            # bounding boxes don't duplicate anything
            actor_obbs = []
            for other in list(vehicle_list) + list(static_list):
                if other.id not in existed_bboxes_ids:
                    continue
                bb = other.bounding_box  # local-space BB (relative to actor origin)
                global_location = other.get_transform().transform(bb.location)
                other_bb = carla.BoundingBox(
                    carla.Location(
                        x=global_location.x,
                        y=global_location.y,
                        z=global_location.z,
                    ),
                    bb.extent,
                )
                other_bb.rotation = bb.rotation
                actor_obbs.append(other_bb)

            level_bb_distances = np.linalg.norm(
                self._level_car_bb_centers - ego_matrix[:3, 3],
                axis=1,
            )
            for i in np.where(
                level_bb_distances <= self.config_expert.driving.bb_save_radius,
            )[0]:
                vehicle_bounding_box = self._level_car_bounding_boxes[i]
                extent = vehicle_bounding_box.extent
                location = vehicle_bounding_box.location
                rotation = vehicle_bounding_box.rotation
                matrix = carla.Transform(location, rotation).get_matrix()
                relative_pos = geometry.get_relative_transform(
                    ego_matrix,
                    np.array(matrix),
                )
                distance = np.linalg.norm(relative_pos)
                relative_yaw = geometry.normalize_angle_rad(
                    np.deg2rad(rotation.yaw) - ego_yaw,
                )
                result = {
                    "class": "static_prop_car",
                    "extent": [extent.x, extent.y, extent.z],
                    "position": [relative_pos[0], relative_pos[1], relative_pos[2]],
                    "yaw": relative_yaw,
                    "num_points": -1,
                    "visible_pixels": -1,
                    "distance": distance,
                    "id": int(static_parking_id_start + i),
                    "matrix": matrix,
                }

                too_close = any(
                    geometry.check_obb_intersection(vehicle_bounding_box, other_bb)
                    for other_bb in actor_obbs
                )
                if not too_close:
                    boxes.append(result)

        # Filter bounding boxes with duplicate id
        bounding_box_ids = set()
        filtered_bounding_boxes = []
        for box in boxes:
            if box["id"] not in bounding_box_ids:
                bounding_box_ids.add(box["id"])
                filtered_bounding_boxes.append(box)
        # Sort by distances to ego
        filtered_bounding_boxes = sorted(
            filtered_bounding_boxes,
            key=lambda x: x["distance"],
        )

        self.id2actor_map = actor_map

        return filtered_bounding_boxes
