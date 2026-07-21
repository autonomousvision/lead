"""Label builders for the TransFuser sample dict.

All geometry consumed here is already expressed in the sample's view frame.
"""

from __future__ import annotations

import logging
import typing

import cv2
import jaxtyping as jt
import numpy as np
import numpy.typing as npt

from lead.common import constants, geometry
from lead.common.constants import (
    CONSTRUCTION_CONE_BB_SIZE,
    TRAFFIC_WARNING_BB_SIZE,
    LeadSemanticSegmentationClass,
    RadarLabels,
)
from lead.config import LeadConfig, TransfuserConfig
from lead.dataloader import Frame, view_geometry
from lead.policy.transfuser.dataloader import bev_raster
from lead.policy.transfuser.decoder import center_net_decoder as g_t
from lead.policy.transfuser.labels import (
    BEVOccupancyClass,
    BEVSemanticClass,
    BoundingBoxClass,
    BoundingBoxIndex,
)

LOG = logging.getLogger(__name__)

# Boxes at the true world pose of a sign or light. TransFuser++ instead consumes
# the boxes projected onto the lanes those actors govern, so these are skipped.
PHYSICAL_BOX_CLASSES = frozenset(
    {"traffic_light_physical", "stop_sign_physical", "traffic_sign"},
)

_CARLA_TO_LEAD_SEMANTIC_LUT = np.array(
    [
        constants.SEMANTIC_SEGMENTATION_CONVERTER[carla_class]
        for carla_class in constants.CarlaSemanticSegmentationClass
    ],
    dtype=np.uint8,
)


def build_labels(
    frame: Frame,
    view_boxes: list[dict] | None,
    lead_config: LeadConfig,
) -> dict[str, typing.Any]:
    """Build the training targets read from the privileged fields of the frame.

    The future-dependent labels (ego waypoints, per-box futures) are not built
    here: they need the log's future ticks, which ``view_boxes`` already
    carries grafted per box.

    Args:
        frame: The privileged frame to build the labels from.
        view_boxes: The tick's view-frame box dicts, unfiltered, with the
            per-box future poses grafted on.
        lead_config: Root config tree.

    Returns:
        The label entries of one unbatched sample.
    """
    config = lead_config.policy.transfuser
    data_config = lead_config.expert.data_collection
    meta = frame.meta
    assert meta is not None
    data: dict[str, typing.Any] = {}

    current_active_scenario_type = meta.get("current_active_scenario_type")
    previous_active_scenario_type = meta.get("previous_active_scenario_type")
    if current_active_scenario_type is not None:
        data["scenario_type"] = current_active_scenario_type
    elif previous_active_scenario_type is not None:
        data["scenario_type"] = previous_active_scenario_type
    else:
        data["scenario_type"] = "NA"

    # Bounding boxes
    boxes = None
    if config.detect_boxes or config.use_bev_semantic:
        assert view_boxes is not None
        boxes, _, _ = _get_bbox_labels(
            data,
            lead_config,
            view_boxes,
            meta,
        )

    # Radar detections
    if frame.radar_sweeps is not None:
        data["radar_detections"] = _parse_radar_detection_labels(lead_config, boxes)

    # Semantic segmentation, reduced from the frame's class and actor ids and
    # the tick's boxes (raw CARLA classes into the LEAD label space).
    if frame.semantics is not None and frame.instances is not None:
        assert view_boxes is not None
        instances = [
            np.stack(
                [
                    semantic_camera.image.astype(np.int32),
                    instance_camera.image.astype(np.int32),
                ],
                axis=-1,
            )
            for semantic_camera, instance_camera in zip(
                frame.semantics,
                frame.instances,
                strict=True,
            )
        ]
        semantics = _semantics_from_instances(instances, view_boxes)
        semantic = np.concatenate(semantics, axis=1)
        data["semantic"] = semantic[
            :: config.perspective_downsample_factor,
            :: config.perspective_downsample_factor,
        ]

    # Depth, decoded into metric depth by the camera's own metadata.
    if frame.depths is not None:
        depth = np.concatenate(
            [
                camera.metadata.decode_depth(camera.image).astype(np.float32)
                for camera in frame.depths
            ],
            axis=1,
        )
        if config.perspective_downsample_factor > 1:
            depth = cv2.resize(
                depth,
                dsize=(
                    depth.shape[1] // config.perspective_downsample_factor,
                    depth.shape[0] // config.perspective_downsample_factor,
                ),
                interpolation=cv2.INTER_LINEAR,
            )
        data["depth"] = depth

    # BEV semantic: static map raster + dynamic occupancy overlay
    if config.use_bev_semantic:
        assert config.detect_boxes
        assert view_boxes is not None
        assert frame.map_api is not None
        assert frame.ego_state is not None
        view_center_se2 = view_geometry.view_center_se2(
            frame.ego_state.center_se2,
            frame.perturbation.translation if frame.perturbation else 0.0,
            frame.perturbation.rotation if frame.perturbation else 0.0,
        )
        loaded_hdmap = bev_raster.rasterize_bev_semantic_map(
            frame.map_api,
            view_center_se2,
            stop_sign_hazard=bool(meta["stop_sign_hazard"]),
            lead_config=lead_config,
        )
        data["hdmap"] = loaded_hdmap

        bev_occupancy = _build_bev_occupancy(data, meta, view_boxes, lead_config)
        assert bev_occupancy.shape[0] == bev_occupancy.shape[1]
        bev_occupancy_center = bev_occupancy.shape[0] / 2
        x_cut = (
            bev_occupancy_center
            + np.array([data_config.min_x_meter, data_config.max_x_meter]) * 4
        ).astype(int)
        y_cut = (
            bev_occupancy_center
            + np.array([data_config.min_y_meter, data_config.max_y_meter]) * 4
        ).astype(int)
        loaded_bev_occupancy = bev_occupancy[y_cut[0] : y_cut[1], x_cut[0] : x_cut[1]]
        mask = loaded_bev_occupancy != BEVOccupancyClass.UNLABELED
        loaded_hdmap = loaded_hdmap.copy()
        loaded_hdmap[mask] = loaded_bev_occupancy[mask] + (
            len(BEVSemanticClass) - len(BEVOccupancyClass)
        )
        data["bev_semantic"] = loaded_hdmap

    # 2D bounding boxes for CenterNet
    if config.detect_boxes:
        assert boxes is not None
        data.update(_get_centernet_labels(boxes, lead_config, config.num_bb_classes))

    return data


def _semantics_from_instances(
    instances: list[jt.Int32[npt.NDArray, "h w 2"]],
    boxes: list[dict],
) -> list[jt.UInt8[npt.NDArray, "h w"]]:
    """Derive TransFuser's semantic labels from the stored class and actor ids.

    The logs store the raw CARLA classes; the classes CARLA's camera cannot
    emit (cones/traffic warnings, emergency vehicles, stop signs) are painted
    over the pixels whose instance id belongs to a matching recorded box.

    Args:
        instances: Per-camera ``[semantic_id, instance_id]`` maps.
        boxes: The tick's recorded box dicts, unfiltered.

    Returns:
        One label image per camera in the LEAD label space.
    """
    painted: list[tuple[int, LeadSemanticSegmentationClass]] = []
    for box in boxes:
        if box["class"] == "stop_sign_physical":
            # Physical sign boxes carry a synthetic id; the CARLA actor id
            # matching the instance pixels sits in ``actor_id``.
            actor_id = box.get("actor_id")
            lead_class = LeadSemanticSegmentationClass.STOP_SIGN
        elif box.get("type_id") in constants.CONSTRUCTION_MESHES:
            actor_id = box["id"]
            lead_class = LeadSemanticSegmentationClass.OBSTACLE
        elif box.get("type_id") in constants.EMERGENCY_MESHES:
            actor_id = box["id"]
            lead_class = LeadSemanticSegmentationClass.SPECIAL_VEHICLE
        else:
            continue
        if actor_id is None:
            continue
        painted.append((actor_id & constants.INSTANCE_ID_MASK, lead_class))

    semantics = []
    for instance in instances:
        semantic = _CARLA_TO_LEAD_SEMANTIC_LUT[instance[..., 0]]
        for instance_id, lead_class in painted:
            semantic[instance[..., 1] == instance_id] = lead_class
        semantics.append(semantic)
    return semantics


def _bbox_json2array(
    bbox_dict: dict,
    config: TransfuserConfig,
) -> tuple[jt.Float[npt.NDArray, " 9"], jt.Float[npt.NDArray, " timesteps 2"], int]:
    """Extract a bounding box label from a view-frame CARLA bounding box dictionary.

    Args:
        bbox_dict: Dictionary containing bounding box information from CARLA,
            already expressed in the sample's view frame.
        config: Transfuser architecture configuration.

    Returns:
        Array with bounding boxes. Each row is a bounding box.
        Array with waypoints for each bounding box in the view frame.
        Number of valid waypoints.
    """
    x, y = bbox_dict["position"][:2]
    num_radar_points = bbox_dict.get("num_radar_points", -1)

    # center_x, center_y, w, h, yaw
    bbox = np.array(
        [
            x,
            y,
            bbox_dict["extent"][0],
            bbox_dict["extent"][1],
            0,
            0,
            0,
            0,
            num_radar_points,
        ],
        dtype=np.float32,
    )
    bbox[BoundingBoxIndex.YAW] = geometry.normalize_angle(
        bbox_dict["yaw"],
    )

    if bbox_dict["class"] == "car":  # static class = parking vehicle = an implicit car
        bbox[BoundingBoxIndex.VELOCITY] = bbox_dict["speed"]
        # check for nans
        if np.isnan(bbox_dict["brake"]):
            bbox[BoundingBoxIndex.BRAKE] = 0
        else:
            bbox[BoundingBoxIndex.BRAKE] = bbox_dict["brake"]
        if (
            "role_name" in bbox_dict
            and "scenario" in bbox_dict["role_name"]
            and bbox_dict["type_id"] in constants.EMERGENCY_MESHES
        ):
            # this is an emergency vehicle that we need to yield to (or dodge in the RunningRedLight scenario)
            # so we give it a different label
            bbox[BoundingBoxIndex.CLASS] = BoundingBoxClass.SPECIAL
        else:
            bbox[BoundingBoxIndex.CLASS] = BoundingBoxClass.VEHICLE
    elif bbox_dict["class"] == "walker":
        bbox[BoundingBoxIndex.VELOCITY] = bbox_dict["speed"]
        bbox[BoundingBoxIndex.CLASS] = BoundingBoxClass.WALKER
    elif bbox_dict["class"] == "traffic_light":
        bbox[BoundingBoxIndex.CLASS] = BoundingBoxClass.TRAFFIC_LIGHT
    elif bbox_dict["class"] == "stop_sign":
        bbox[BoundingBoxIndex.CLASS] = BoundingBoxClass.STOP_SIGN
    elif bbox_dict["class"] == "static":
        bbox[BoundingBoxIndex.CLASS] = BoundingBoxClass.VEHICLE

    waypoints = np.array(
        [(bbox_dict["position"][0], bbox_dict["position"][1])]
        * config.num_way_points_prediction,
        dtype=np.float32,
    )
    num_waypoints = 0
    if bbox_dict.get("future_positions") is not None:
        future_waypoint_indices = [config.waypoints_spacing]
        for _ in range(config.num_way_points_prediction - 1):
            future_waypoint_indices.append(
                future_waypoint_indices[-1] + config.waypoints_spacing,
            )

        # In CARLA, disappearing vehicles are not removed but teleported far away,
        # so reject future waypoints that jump too far from the last known position.
        last_pos = np.array(
            [
                bbox_dict["position"][0],
                bbox_dict["position"][1],
            ],
        )
        last_valid_yaw = bbox_dict["yaw"]
        last_valid_speed = bbox_dict.get("speed", 0.0)
        for i, future_waypoint_index in enumerate(future_waypoint_indices):
            if future_waypoint_index < len(bbox_dict["future_positions"]):
                future_position = np.asarray(
                    bbox_dict["future_positions"][future_waypoint_index],
                )
                dist = np.linalg.norm(last_pos - future_position)
                if dist > config.max_distance_future_waypoint:
                    break
                waypoints[i] = future_position
                num_waypoints += 1
                last_pos = future_position
                last_valid_yaw = bbox_dict["future_yaws"][future_waypoint_index]
                last_valid_speed = (
                    bbox_dict["future_speeds"][future_waypoint_index]
                    if "future_speeds" in bbox_dict
                    else last_valid_speed
                )

        # Extrapolate last valid waypoint to mitigate disappearing boxes
        last_valid_waypoint = (
            waypoints[num_waypoints - 1]
            if num_waypoints > 0
            else np.array([bbox_dict["position"][0], bbox_dict["position"][1]])
        )
        if num_waypoints < config.num_way_points_prediction:
            dt = 1 / (config.waypoints_spacing - 1)  # seconds between waypoints
            for i in range(num_waypoints, config.num_way_points_prediction):
                # Extrapolate using constant velocity model
                dx = last_valid_speed * dt * np.cos(last_valid_yaw)
                dy = last_valid_speed * dt * np.sin(last_valid_yaw)

                # Update the waypoint position
                last_valid_waypoint = last_valid_waypoint + np.array([dx, dy])
                waypoints[i] = last_valid_waypoint.copy()

    return bbox, waypoints, num_waypoints


def _get_bbox_labels(
    data: dict,
    lead_config: LeadConfig,
    boxes: list[dict],
    current_measurement: dict,
) -> tuple[
    jt.Float[npt.NDArray, "max_num_bbs 9"],
    jt.Float[npt.NDArray, "max_num_bbs timesteps 2"],
    jt.Int[npt.NDArray, " max_num_bbs"],
]:
    """Parse and filter bounding boxes from CARLA data.

    Args:
        data: The data dictionary containing scenario information.
        lead_config: Root config tree.
        boxes: List of view-frame bounding box dictionaries from CARLA.
        current_measurement: Current measurement data, including scenario obstacle IDs.

    Returns:
        Array containing parsed and filtered bounding boxes in image coordinates for center net.
    """
    config = lead_config.policy.transfuser
    data_config = lead_config.expert.data_collection
    bboxes, waypoints, num_waypoints = [], [], []

    for _, current_box in enumerate(boxes):
        if current_box["class"] in PHYSICAL_BOX_CLASSES:
            continue

        bbox, waypoint, num_waypoint = _bbox_json2array(
            current_box,
            config,
        )
        if current_box["class"] in ["ego_car"]:
            continue

        # Occlusion check
        if "num_points" in current_box:
            num_points = current_box["num_points"]
            visible_pixels = -1
            if "visible_pixels" in current_box:
                visible_pixels = current_box["visible_pixels"]

            semantics = constants.semantic_class(current_box)
            if (
                semantics == LeadSemanticSegmentationClass.PEDESTRIAN
                and 0
                <= num_points
                < lead_config.expert.occlusion.pedestrian_min_num_lidar_points
                and 0
                <= visible_pixels
                < lead_config.expert.occlusion.pedestrian_min_num_visible_pixels
            ):
                continue
            if (
                semantics == LeadSemanticSegmentationClass.VEHICLE
                and 0
                <= num_points
                < lead_config.expert.occlusion.vehicle_min_num_lidar_points
                and 0
                <= visible_pixels
                < lead_config.expert.occlusion.vehicle_min_num_visible_pixels
            ):
                continue

        # Only use/detect boxes that are red and affect the ego vehicle
        if current_box["class"] == "traffic_light":
            if not current_box["affects_ego"] or current_box["state"] == "Green":
                continue

        if current_box["class"] == "stop_sign":
            # Don't detect cleared stop signs.
            if not current_box["affects_ego"]:
                continue

        # Filter bb that are outside of the LiDAR grid in the view frame.
        height = current_box["position"][2]
        if (
            bbox[BoundingBoxIndex.X] <= data_config.min_x_meter
            or bbox[BoundingBoxIndex.X] >= data_config.max_x_meter
            or bbox[BoundingBoxIndex.Y] <= data_config.min_y_meter
            or bbox[BoundingBoxIndex.Y] >= data_config.max_y_meter
            or height <= lead_config.training.data.min_z
            or height >= lead_config.training.data.max_z
        ):
            continue

        is_parking_vehicle = (
            current_box["class"] == "static"
            and current_box.get("mesh_path") is not None
            and "ParkedVehicles" in current_box["mesh_path"]
        )
        is_parking_vehicle = (
            is_parking_vehicle or current_box["class"] == "static_prop_car"
        )
        if (
            current_box["class"] == "static"
            and "type_id" in current_box
            and current_box["type_id"] not in config.data_bb_static_types_white_list
        ):
            if not is_parking_vehicle:
                continue

        if "type_id" in current_box:
            if current_box["type_id"] == "static.prop.trafficwarning":
                (
                    bbox[BoundingBoxIndex.W],
                    bbox[BoundingBoxIndex.H],
                ) = (
                    TRAFFIC_WARNING_BB_SIZE[0],
                    TRAFFIC_WARNING_BB_SIZE[1],
                )
                bbox[BoundingBoxIndex.CLASS] = BoundingBoxClass.OBSTACLE
            elif current_box["type_id"] == "static.prop.constructioncone":
                (
                    bbox[BoundingBoxIndex.W],
                    bbox[BoundingBoxIndex.H],
                ) = (
                    CONSTRUCTION_CONE_BB_SIZE[0],
                    CONSTRUCTION_CONE_BB_SIZE[1],
                )
                bbox[BoundingBoxIndex.CLASS] = BoundingBoxClass.OBSTACLE

        if "mesh_path" in current_box:
            if current_box["mesh_path"] in constants.LOOKUP_TABLE:
                bbox[BoundingBoxIndex.W] = constants.LOOKUP_TABLE[
                    current_box["mesh_path"]
                ][0]
                bbox[BoundingBoxIndex.H] = constants.LOOKUP_TABLE[
                    current_box["mesh_path"]
                ][1]
            if is_parking_vehicle:
                bbox[BoundingBoxIndex.CLASS] = BoundingBoxClass.VEHICLE

        if current_box["class"] == "static_prop_car":
            bbox[BoundingBoxIndex.CLASS] = BoundingBoxClass.VEHICLE

        # In some CARLA's scenarios, special vehicles have sometimes wrong labels.
        # We relabel those labels to be consistent with the scene
        if (
            current_box["class"] == "car"
            and "role_name" in current_box
            and "scenario" in current_box["role_name"]
            and current_box["speed"] < 0.1
        ):
            if (
                data["scenario_type"] == "VehicleOpensDoorTwoWays"
                and current_box["id"] in current_measurement["scenario_obstacles_ids"]
            ):
                # If the car open door, we extend the bounding box's width to consider the door
                if current_measurement["vehicle_opened_door"]:
                    bbox[BoundingBoxIndex.H] += config.car_open_door_extra_width / 2
                    if current_measurement["vehicle_door_side"] == "left":
                        bbox[BoundingBoxIndex.Y] += config.car_open_door_extra_width / 2
                    else:
                        bbox[BoundingBoxIndex.Y] -= config.car_open_door_extra_width / 2

                if (
                    bbox[BoundingBoxIndex.CLASS] != BoundingBoxClass.SPECIAL
                ) and current_measurement["vehicle_opened_door"]:
                    bbox[BoundingBoxIndex.CLASS] = BoundingBoxClass.OBSTACLE
            elif bbox[BoundingBoxIndex.CLASS] != BoundingBoxClass.SPECIAL:
                if (
                    data["scenario_type"]
                    in [
                        "Accident",
                        "AccidentTwoWays",
                        "BlockedIntersection",
                        "ParkedObstacle",
                        "ParkedObstacleTwoWays",
                    ]
                    and current_box["id"]
                    in current_measurement["scenario_obstacles_ids"]
                ):
                    bbox[BoundingBoxIndex.CLASS] = BoundingBoxClass.OBSTACLE

        if current_box.get("type_id") in constants.BIKER_MESHES:
            bbox[BoundingBoxIndex.CLASS] = BoundingBoxClass.BIKER

        bbox = bb_vehicle_to_image_system(
            bbox.reshape(1, -1),
            data_config.pixels_per_meter,
            data_config.min_x_meter,
            data_config.min_y_meter,
        ).squeeze()
        if not (0 <= bbox[BoundingBoxIndex.X] < data_config.lidar_width_pixel):
            LOG.warning(
                f"{bbox[BoundingBoxIndex.X]=} is larger than {data_config.lidar_width_pixel=}",
            )
            continue
        if not (0 <= bbox[BoundingBoxIndex.Y] < data_config.lidar_height_pixel):
            LOG.warning(
                f"{bbox[BoundingBoxIndex.Y]=} is larger than {data_config.lidar_height_pixel=}",
            )
            continue
        bboxes.append(bbox)
        waypoints.append(waypoint)
        num_waypoints.append(num_waypoint)

    bounding_boxes_array = np.array(bboxes)
    waypoints_array = np.array(waypoints)
    num_waypoints_array = np.array(num_waypoints)

    # Pad bounding boxes to a fixed number
    padded_bounding_boxes_array = np.zeros((config.max_num_bbs, 9), dtype=np.float32)
    padded_waypoints_array = np.zeros(
        (config.max_num_bbs, config.num_way_points_prediction, 2),
        dtype=np.float32,
    )
    padded_num_waypoints_array = np.zeros((config.max_num_bbs,), dtype=np.int32)

    if bounding_boxes_array.shape[0] > 0:
        if bounding_boxes_array.shape[0] <= config.max_num_bbs:
            padded_bounding_boxes_array[: bounding_boxes_array.shape[0], :] = (
                bounding_boxes_array
            )
            padded_waypoints_array[: bounding_boxes_array.shape[0], :, :] = (
                waypoints_array
            )
            padded_num_waypoints_array[: bounding_boxes_array.shape[0]] = (
                num_waypoints_array
            )
        else:
            padded_bounding_boxes_array[: config.max_num_bbs, :] = bounding_boxes_array[
                : config.max_num_bbs
            ]
            padded_waypoints_array[: config.max_num_bbs, :, :] = waypoints_array[
                : config.max_num_bbs,
                :,
                :,
            ]
            padded_num_waypoints_array[: config.max_num_bbs] = num_waypoints_array[
                : config.max_num_bbs
            ]

    return (
        padded_bounding_boxes_array,
        padded_waypoints_array,
        padded_num_waypoints_array,
    )


def _get_centernet_labels(
    gt_bboxes: jt.Float[npt.NDArray, "N 9"],
    lead_config: LeadConfig,
    num_bb_classes: int,
) -> dict[str, npt.NDArray | int]:
    """
    Compute regression and classification targets for CenterNet.

    Args:
        gt_bboxes: Ground truth bboxes for each image with shape (N, 11). Coordinates in image frame.
        lead_config: Root config tree.
        num_bb_classes: Number of bounding box classes.
    Returns:
        A dictionary containing various target tensors for training the CenterNet model.
    """
    config = lead_config.policy.transfuser
    data_config = lead_config.expert.data_collection
    feat_h = data_config.lidar_height_meter
    feat_w = data_config.lidar_width_meter

    center_heatmap_target = np.zeros([num_bb_classes, feat_h, feat_w], dtype=np.float32)
    wh_target = np.zeros([2, feat_h, feat_w], dtype=np.float32)
    offset_target = np.zeros([2, feat_h, feat_w], dtype=np.float32)
    yaw_class_target = np.zeros([1, feat_h, feat_w], dtype=np.int32)
    yaw_res_target = np.zeros([1, feat_h, feat_w], dtype=np.float32)
    velocity_target = np.zeros([1, feat_h, feat_w], dtype=np.float32)
    brake_target = np.zeros([1, feat_h, feat_w], dtype=np.int32)
    pixel_weight = np.zeros(
        [2, feat_h, feat_w],
        dtype=np.float32,
    )  # 2 is the max of the channels above here.

    if not gt_bboxes.shape[0] > 0:
        return {
            "center_net_bounding_boxes": gt_bboxes,
            "center_net_heatmap": center_heatmap_target,
            "center_net_wh": wh_target,
            "center_net_yaw_class": yaw_class_target.squeeze(0),
            "center_net_yaw_res": yaw_res_target,
            "center_net_offset": offset_target,
            "center_net_velocity": velocity_target,
            "center_net_brake": brake_target.squeeze(0),
            "center_net_pixel_weight": pixel_weight,
            "center_net_avg_factor": np.array([1]),
        }

    center_x = gt_bboxes[:, [BoundingBoxIndex.X]] / config.bev_down_sample_factor
    center_y = gt_bboxes[:, [BoundingBoxIndex.Y]] / config.bev_down_sample_factor
    gt_centers = np.concatenate((center_x, center_y), axis=1)

    for j, ct in enumerate(gt_centers):
        ctx_int, cty_int = ct.astype(int)
        ctx, cty = ct
        if ctx_int < 0 or ctx_int >= feat_w or cty_int < 0 or cty_int >= feat_h:
            LOG.warning(
                f"Be cautious! Bounding box center {ct} is out of bounds for image size ({feat_h}, {feat_w}).",
            )
            continue

        extent_x = gt_bboxes[j, BoundingBoxIndex.W] / config.bev_down_sample_factor
        extent_y = gt_bboxes[j, BoundingBoxIndex.H] / config.bev_down_sample_factor

        radius = g_t.gaussian_radius([extent_y, extent_x], min_overlap=0.1)
        radius = max(2, int(radius))
        ind = gt_bboxes[j, BoundingBoxIndex.CLASS].astype(int)

        g_t.gen_gaussian_target(center_heatmap_target[ind], [ctx_int, cty_int], radius)

        wh_target[0, cty_int, ctx_int] = extent_x
        wh_target[1, cty_int, ctx_int] = extent_y

        yaw_class, yaw_res = geometry.angle2class(
            gt_bboxes[j, BoundingBoxIndex.YAW],
            config.num_dir_bins,
        )

        yaw_class_target[0, cty_int, ctx_int] = yaw_class
        yaw_res_target[0, cty_int, ctx_int] = yaw_res

        velocity_target[0, cty_int, ctx_int] = gt_bboxes[
            j,
            BoundingBoxIndex.VELOCITY,
        ]
        # Brakes can potentially be continuous but we classify them now.
        # Using mathematical rounding the split is applied at 0.5
        brake_target[0, cty_int, ctx_int] = int(
            round(gt_bboxes[j, BoundingBoxIndex.BRAKE]),
        )

        offset_target[0, cty_int, ctx_int] = ctx - ctx_int
        offset_target[1, cty_int, ctx_int] = cty - cty_int
        # All pixels with a bounding box have a weight of 1 all others have a weight of 0.
        # Used to ignore the pixels without bbs in the loss.
        pixel_weight[:, cty_int, ctx_int] = 1.0

    avg_factor = max(1, np.equal(center_heatmap_target, 1).sum())
    return {
        "center_net_bounding_boxes": gt_bboxes,
        "center_net_heatmap": center_heatmap_target,
        "center_net_wh": wh_target,
        "center_net_yaw_class": yaw_class_target.squeeze(0),
        "center_net_yaw_res": yaw_res_target,
        "center_net_offset": offset_target,
        "center_net_velocity": velocity_target,
        "center_net_brake": brake_target.squeeze(0),
        "center_net_pixel_weight": pixel_weight,
        "center_net_avg_factor": avg_factor,
    }


def _build_bev_occupancy(
    data: dict,
    current_measurement: dict,
    json_boxes: list,
    lead_config: LeadConfig,
) -> jt.UInt8[npt.NDArray, "H W"]:
    """Build bird's eye view occupancy map from bounding box data.

    Projects view-frame bounding boxes (vehicles, pedestrians, traffic lights,
    construction zones) onto a semantic grid.

    Args:
        data: Dictionary containing scenario information.
        current_measurement: Current measurement data including scenario obstacles.
        json_boxes: List of view-frame bounding box dictionaries from CARLA.
        lead_config: Root config tree.

    Returns:
        Bird's eye view occupancy map as integer array of shape.
    """
    config = lead_config.policy.transfuser
    scale = 4
    grid_size = 256 * scale
    bev = np.zeros((grid_size, grid_size), dtype=np.uint8)

    obstacle_scenario_corners = []
    cone_corners = []
    warning_corners = []
    normal_red_light_corners = []
    unnormal_red_light_corners = []
    green_light_corners = []

    for current_box in json_boxes:
        cls = current_box["class"]
        if cls not in ["car", "walker", "static", "static_prop_car", "traffic_light"]:
            continue
        extra_scale = 1.0
        min_extent = 0.0
        if cls in ["walker"] or (
            "type_id" in current_box
            and current_box["type_id"] in constants.BIKER_MESHES
        ):
            extra_scale = config.scale_pedestrian_bev_semantic_size
            min_extent = config.pedestrian_bev_min_extent
        x, y = current_box["position"][:2]
        y = -y

        cx = (x + 128.0) * scale
        cy = (128.0 - y) * scale
        if not (0 <= cx < grid_size and 0 <= cy < grid_size):
            continue

        yaw = geometry.normalize_angle(current_box["yaw"])
        extent_x, extent_y = current_box["extent"][:2]
        extent_x = max(extent_x, min_extent)
        extent_y = max(extent_y, min_extent)
        if (
            current_box["class"] == "car"
            and "role_name" in current_box
            and "scenario" in current_box["role_name"]
            and current_box["speed"] < 0.1
            and current_box["id"] in current_measurement["scenario_obstacles_ids"]
        ):
            if data["scenario_type"] == "VehicleOpensDoorTwoWays":
                if current_measurement["vehicle_opened_door"]:
                    # This is a special case where we open the door, so we extend the bounding box's width
                    extent_y += config.car_open_door_extra_width / 2
                    if current_measurement["vehicle_door_side"] == "left":
                        cy += config.car_open_door_extra_width / 2 * scale
                    else:
                        cy -= config.car_open_door_extra_width / 2 * scale

        if "mesh_path" in current_box:
            if current_box["mesh_path"] in constants.LOOKUP_TABLE:
                extent_x = constants.LOOKUP_TABLE[current_box["mesh_path"]][0]
                extent_y = constants.LOOKUP_TABLE[current_box["mesh_path"]][1]
        elif "type_id" in current_box:
            if current_box["type_id"] == "static.prop.trafficwarning":
                extent_x, extent_y = (
                    TRAFFIC_WARNING_BB_SIZE[0],
                    TRAFFIC_WARNING_BB_SIZE[1],
                )
            elif current_box["type_id"] == "static.prop.constructioncone":
                extent_x, extent_y = (
                    CONSTRUCTION_CONE_BB_SIZE[0],
                    CONSTRUCTION_CONE_BB_SIZE[1],
                )
        rect = (
            (cx, cy),
            (extent_x * 2 * scale * extra_scale, extent_y * 2 * scale * extra_scale),
            np.rad2deg(yaw),
        )
        box_pts = cv2.boxPoints(rect).astype(np.int32)

        # Assign label
        is_parking_car = False
        label = BEVOccupancyClass.VEHICLE
        if cls in ["car"]:
            if current_box.get("type_id", "") in constants.EMERGENCY_MESHES:
                if (
                    data["scenario_type"]
                    in [
                        "Accident",
                        "AccidentTwoWays",
                        "BlockedIntersection",
                        "ParkedObstacle",
                        "ParkedObstacleTwoWays",
                    ]
                    and current_box["id"]
                    in current_measurement["scenario_obstacles_ids"]
                ):
                    label = BEVOccupancyClass.OBSTACLE
                    obstacle_scenario_corners.extend(box_pts.tolist())
                else:
                    label = BEVOccupancyClass.SPECIAL_VEHICLE
            elif current_box.get("type_id", "") in constants.BIKER_MESHES:
                label = BEVOccupancyClass.BIKER
            elif (
                "role_name" in current_box
                and "scenario" in current_box["role_name"]
                and current_box["speed"] < 0.1
            ):
                if (
                    data["scenario_type"]
                    in [
                        "Accident",
                        "AccidentTwoWays",
                        "BlockedIntersection",
                        "ParkedObstacle",
                        "ParkedObstacleTwoWays",
                    ]
                    and current_box["id"]
                    in current_measurement["scenario_obstacles_ids"]
                ):
                    obstacle_scenario_corners.extend(box_pts.tolist())
                    label = BEVOccupancyClass.OBSTACLE
                elif (
                    data["scenario_type"] in ["VehicleOpensDoorTwoWays"]
                    and current_box["id"]
                    in current_measurement["scenario_obstacles_ids"]
                    and current_measurement["vehicle_opened_door"]
                ):
                    label = BEVOccupancyClass.OBSTACLE
                    obstacle_scenario_corners.extend(box_pts.tolist())
        elif cls == "walker":
            label = BEVOccupancyClass.WALKER
        elif cls == "static_prop_car":
            label = BEVOccupancyClass.VEHICLE
        elif cls == "static":
            type_id = current_box.get("type_id", "")
            is_parking_car = (
                current_box.get("mesh_path") is not None
                and "ParkedVehicles" in current_box["mesh_path"]
            )
            if type_id in config.data_bb_static_types_white_list:
                label = BEVOccupancyClass.OBSTACLE
                if type_id == "static.prop.constructioncone":
                    cone_corners.extend(box_pts.tolist())
                elif type_id == "static.prop.trafficwarning":
                    warning_corners.extend(box_pts.tolist())
            elif type_id in constants.EMERGENCY_MESHES:
                label = BEVOccupancyClass.SPECIAL_VEHICLE
                obstacle_scenario_corners.extend(box_pts.tolist())
            elif is_parking_car:
                label = BEVOccupancyClass.VEHICLE
            else:
                continue
        elif cls == "traffic_light":
            if current_box["affects_ego"] and current_box["state"] in ["Red", "Yellow"]:
                if (
                    not current_measurement["over_head_traffic_light"]
                    and not current_measurement["europe_traffic_light"]
                ):
                    label = BEVOccupancyClass.TRAFFIC_RED_NORMAL
                    normal_red_light_corners.extend(box_pts.tolist())
                else:
                    label = BEVOccupancyClass.TRAFFIC_RED_NOT_NORMAL
                    unnormal_red_light_corners.extend(box_pts.tolist())
            elif current_box["affects_ego"]:
                label = BEVOccupancyClass.TRAFFIC_GREEN
                green_light_corners.extend(box_pts.tolist())
            else:
                continue

        # Occlusion check
        if (
            cls == "walker"
            and (
                0
                <= current_box["num_points"]
                < lead_config.expert.occlusion.pedestrian_min_num_lidar_points
            )
            and (
                0
                <= current_box["visible_pixels"]
                < lead_config.expert.occlusion.pedestrian_min_num_visible_pixels
            )
        ):
            continue
        if (
            cls == "car"
            and (
                0
                <= current_box["num_points"]
                < lead_config.expert.occlusion.vehicle_min_num_lidar_points
            )
            and (
                0
                <= current_box["visible_pixels"]
                < lead_config.expert.occlusion.vehicle_min_num_visible_pixels
            )
        ):
            continue
        if (
            cls in ["static", "static_prop_car"]
            and (
                0
                <= current_box["num_points"]
                < lead_config.expert.occlusion.vehicle_min_num_lidar_points
            )
            and (
                0
                <= current_box["visible_pixels"]
                < lead_config.expert.occlusion.vehicle_min_num_visible_pixels
            )
        ):
            continue
        cv2.fillPoly(bev, [box_pts], label)

    # Construction site detection
    if len(warning_corners) >= 3 and len(cone_corners) >= 24:
        all_pts = np.array(cone_corners + warning_corners)
        mean = np.mean(all_pts, axis=0)
        dists = np.linalg.norm(all_pts - mean, axis=1)

        if np.all(dists <= 24 * scale):
            hull = cv2.convexHull(all_pts)
            cv2.fillPoly(bev, [hull], BEVOccupancyClass.OBSTACLE)

    # Accident and obstacle scenario detection
    if len(obstacle_scenario_corners) >= 3:
        all_pts = np.array(obstacle_scenario_corners)
        mean = np.mean(all_pts, axis=0)
        dists = np.linalg.norm(all_pts - mean, axis=1)

        if np.all(dists <= 48 * scale):
            hull = cv2.convexHull(all_pts)
            cv2.fillPoly(bev, [hull], BEVOccupancyClass.OBSTACLE)

    # Red light detection
    if len(unnormal_red_light_corners) >= 3:
        hull = cv2.convexHull(np.array(unnormal_red_light_corners))
        cv2.fillPoly(bev, [hull], BEVOccupancyClass.TRAFFIC_RED_NOT_NORMAL)
    elif len(normal_red_light_corners) >= 3:
        hull = cv2.convexHull(np.array(normal_red_light_corners))
        cv2.fillPoly(bev, [hull], BEVOccupancyClass.TRAFFIC_RED_NORMAL)
    elif len(green_light_corners) >= 3:
        hull = cv2.convexHull(np.array(green_light_corners))
        cv2.fillPoly(bev, [hull], BEVOccupancyClass.TRAFFIC_GREEN)
    return bev


def bb_vehicle_to_image_system(
    box: jt.Float[npt.NDArray, "N D"],
    pixels_per_meter: float,
    min_x: float,
    min_y: float,
) -> jt.Float[npt.NDArray, "N D"]:
    """
    Changed a bounding box from the vehicle coordinate system to the image coordinate system.

    Args:
        box: bounding box in the vehicle coordinate system.
        pixels_per_meter: scaling factor from meters to pixels
        min_x: minimum x value of the image in the vehicle coordinate system
        min_y: minimum y value of the image in the vehicle coordinate system

    Returns:
        box: bounding box in the image coordinate system.
    """
    box = box.copy()
    box[:, :2] = box[:, :2] - np.array([min_x, min_y])
    box[:, :4] = box[:, :4] * pixels_per_meter
    return box


def bb_image_to_vehicle_system(
    box: jt.Float[npt.NDArray, "N D"],
    pixels_per_meter: float,
    min_x: float,
    min_y: float,
) -> jt.Float[npt.NDArray, "N D"]:
    """Inverse of bb_vehicle_to_image_system.

    Args:
        box: bounding box in the image coordinate system.
        pixels_per_meter: scaling factor from meters to pixels
        min_x: minimum x value of the image in the vehicle coordinate system
        min_y: minimum y value of the image in the vehicle coordinate system

    Returns:
        box: bounding box in the vehicle coordinate system.
    """
    box = box.copy()
    box[:, :4] = box[:, :4] / pixels_per_meter
    box[:, :2] = box[:, :2] + np.array([min_x, min_y])
    return box


def _parse_radar_detection_labels(
    lead_config: LeadConfig,
    boxes: jt.Float32[npt.NDArray, "num_boxes features"] | None,
) -> jt.Float32[npt.NDArray, "num_queries features"]:
    """Parse and filter radar detection labels from the tick's box labels.

    Converts detections to vehicle coordinates and fills a fixed-size array,
    prioritizing higher velocity, class importance (SPECIAL > VEHICLE > WALKER >
    OBSTACLE), and radar point count.

    Args:
        lead_config: Root config tree with the radar query and BEV geometry knobs.
        boxes: The tick's box label array in the image system, or None.

    Returns:
        Array of shape (num_radar_queries, num_features) containing radar detection labels.
        Each row represents one detection with features [x, y, velocity, valid_flag].
        Unused slots are zero-padded. Features are in vehicle coordinate system.
    """
    config = lead_config.policy.transfuser
    data_config = lead_config.expert.data_collection

    # Initialize default values (all zeros)
    radar_detections = np.zeros(
        (config.num_radar_queries, len(RadarLabels)),
        dtype=np.float32,
    )

    if (
        lead_config.expert.sensor_rig.use_radars
        and boxes is not None
        and boxes.shape[0] > 0
    ):
        priority_classes = [
            BoundingBoxClass.SPECIAL,
            BoundingBoxClass.VEHICLE,
            BoundingBoxClass.WALKER,
            BoundingBoxClass.OBSTACLE,
        ]

        loaded_boxes_vehicle_system = bb_image_to_vehicle_system(
            boxes.copy(),
            data_config.pixels_per_meter,
            data_config.min_x_meter,
            data_config.min_y_meter,
        )

        # Remove zero-padded data
        non_zero_mask = (loaded_boxes_vehicle_system[:, BoundingBoxIndex.X] != 0.0) | (
            loaded_boxes_vehicle_system[:, BoundingBoxIndex.Y] != 0.0
        )
        loaded_boxes_vehicle_system = loaded_boxes_vehicle_system[non_zero_mask]

        # Filter data with minimally one radar point
        radar_mask = (
            loaded_boxes_vehicle_system[:, BoundingBoxIndex.NUM_RADAR_POINTS] > 0
        )
        loaded_boxes_vehicle_system = loaded_boxes_vehicle_system[radar_mask]

        selected_boxes: jt.Float[npt.NDArray, "n 9"] = np.zeros(
            (0, loaded_boxes_vehicle_system.shape[1]),
            dtype=np.float32,
        )

        if loaded_boxes_vehicle_system.shape[0] > 0:
            # Compute class priority index for each box
            class_priorities = {int(cls): i for i, cls in enumerate(priority_classes)}
            class_priority = np.array(
                [
                    class_priorities.get(int(c), len(priority_classes))
                    for c in loaded_boxes_vehicle_system[
                        :,
                        BoundingBoxIndex.CLASS,
                    ]
                ],
            )

            # Stack into sortable array: (-velocity, class_priority, -num_radar_points)
            sortable = np.stack(
                [
                    -loaded_boxes_vehicle_system[
                        :,
                        BoundingBoxIndex.VELOCITY,
                    ],
                    class_priority,
                    -loaded_boxes_vehicle_system[
                        :,
                        BoundingBoxIndex.NUM_RADAR_POINTS,
                    ],
                ],
                axis=1,
            )

            # Sort lexicographically: we prioritize higher velocity, then class priority, then more radar points
            sorted_indices = np.lexsort(sortable.T[::-1])

            # Apply sorting to all three arrays
            sorted_boxes = loaded_boxes_vehicle_system[sorted_indices]

            # Take up to num_radar_queries
            selected_boxes = sorted_boxes[: config.num_radar_queries]

        if len(selected_boxes) > 0:
            # Extract [x, y, velocity]
            n_boxes = selected_boxes.shape[0]
            radar_detections[:n_boxes, RadarLabels.X] = selected_boxes[
                :,
                BoundingBoxIndex.X,
            ]
            radar_detections[:n_boxes, RadarLabels.Y] = selected_boxes[
                :,
                BoundingBoxIndex.Y,
            ]
            radar_detections[:n_boxes, RadarLabels.V] = selected_boxes[
                :,
                BoundingBoxIndex.VELOCITY,
            ]
            radar_detections[:n_boxes, RadarLabels.VALID] = 1.0  # Valid box indicator

    return radar_detections
