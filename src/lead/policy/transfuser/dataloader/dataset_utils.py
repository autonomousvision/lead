"""Label builders and input preprocessing for the TransFuser sample dict.

All geometry consumed here is already expressed in the sample's view frame
(the ``Py123DDataLoader`` re-expresses ego-frame data when the perturbated
view is chosen), so these builders contain no perturbation handling.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import jaxtyping as jt
import numpy as np
import numpy.typing as npt

from lead.common import common_utils, constants
from lead.common.constants import (
    CONSTRUCTION_CONE_BB_SIZE,
    TRAFFIC_WARNING_BB_SIZE,
    RadarLabels,
    TransfuserBEVOccupancyClass,
    TransfuserBoundingBoxClass,
    TransfuserBoundingBoxIndex,
    TransfuserSemanticSegmentationClass,
)
from lead.common.sensors import ransac
from lead.config import LeadConfig, TransfuserConfig
from lead.policy.transfuser.decoder import center_net_decoder as g_t

LOG = logging.getLogger(__name__)


@dataclass
class SensorData:
    """A sample's loaded sensor data — the shared shape between the dataset and the label builders."""

    image: jt.UInt8[npt.NDArray, "img_height img_width 3"] | None
    rasterized_lidar: jt.Float32[npt.NDArray, "bev_height bev_width"] | None
    semantic: jt.UInt8[npt.NDArray, "img_height img_width"] | None
    hdmap: jt.UInt8[npt.NDArray, "bev_semantic_height bev_semantic_width"] | None
    depth: jt.Float32[npt.NDArray, "depth_img_height depth_img_width"] | None
    boxes: jt.Float32[npt.NDArray, "num_boxes features"] | None
    boxes_waypoints: jt.Float32[npt.NDArray, "num_boxes timesteps 2"] | None
    boxes_num_waypoints: jt.Int32[npt.NDArray, " num_boxes"] | None
    bev_occupancy: (
        jt.UInt8[npt.NDArray, "bev_occupancy_height bev_occupancy_width"] | None
    )
    # One entry per radar sensor (1..num_radar_sensors).
    radars: tuple[jt.Float16[npt.NDArray, "num_points 4"], ...] | None
    radar_detections: jt.Float32[npt.NDArray, "num_radar_queries radar_features"] | None


def rasterize_lidar(
    lead_config: LeadConfig,
    lidar: jt.Float[npt.NDArray, "N 3"],
    remove_ground_plane: bool = False,
) -> jt.Float[npt.NDArray, "H W"]:
    """
    Convert LiDAR point cloud into pseudo-image.

    Args:
        lead_config: Root config tree.
        lidar: LiDAR point cloud.
        remove_ground_plane: whether to remove ground plane points.
    Returns:
        Sparse pseudo-image.
    """
    data_config = lead_config.expert.data_collection

    def splat_points(point_cloud):
        # 256 x 256 grid
        xbins = np.linspace(
            data_config.min_x_meter,
            data_config.max_x_meter,
            (data_config.max_x_meter - data_config.min_x_meter)
            * int(data_config.pixels_per_meter)
            + 1,
        )
        ybins = np.linspace(
            data_config.min_y_meter,
            data_config.max_y_meter,
            (data_config.max_y_meter - data_config.min_y_meter)
            * int(data_config.pixels_per_meter)
            + 1,
        )
        hist = np.histogramdd(point_cloud[:, :2], bins=(xbins, ybins))[0]
        hist[hist > lead_config.training.data.hist_max_per_pixel] = (
            lead_config.training.data.hist_max_per_pixel
        )
        overhead_splat = hist / lead_config.training.data.hist_max_per_pixel
        # The transpose here is an efficient axis swap.
        # Comes from the fact that carla is x front, y right, whereas the image is y front, x right
        # (x height channel, y width channel)
        return overhead_splat.T

    # Remove points above the vehicle
    features = splat_points(lidar)
    lidar = lidar[
        (lidar[..., 2] <= lead_config.expert.storage.max_height_lidar)
        & (lead_config.expert.storage.min_height_lidar <= lidar[..., 2])
    ]
    if remove_ground_plane:
        is_ground_mask = ransac.remove_ground(
            lidar,
            lead_config.expert,
            parallel=True,
        )  # Torch parallel and dataloader seem to have issues with parallel numba.
        above = lidar[~is_ground_mask]
        features = splat_points(above)
    else:
        features = np.stack([splat_points(point_cloud=lidar)], axis=-1)
    return (features).squeeze().astype(np.float32)


def image_augmenter(lead_config: LeadConfig, prob: float = 0.2):
    """Create an image augmenter for data perturbation.

    Args:
        lead_config: Root config tree.
        prob: Probability of applying each perturbation.

    Returns:
        Image augmenter.
    """
    import imgaug
    from imgaug import augmenters as ia

    imgaug.imgaug.seed(lead_config.training.experiment.seed)
    # imgaug's stubs declare per_channel as bool, but a float probability is a
    # documented and intended usage.
    perturbations = [
        ia.Sometimes(prob, ia.GaussianBlur((0, 1.0))),
        ia.Sometimes(
            prob,
            ia.AdditiveGaussianNoise(
                loc=0,
                scale=(0.0, 0.05 * 255),
                per_channel=0.5,  # pyright: ignore[reportArgumentType]
            ),
        ),
        ia.Sometimes(
            prob,
            ia.Dropout(
                (0.01, 0.1),
                per_channel=0.5,  # pyright: ignore[reportArgumentType]
            ),
        ),  # Strong
        ia.Sometimes(
            prob,
            ia.Multiply(
                (1 / 1.2, 1.2),
                per_channel=0.5,  # pyright: ignore[reportArgumentType]
            ),
        ),
        ia.Sometimes(
            prob,
            ia.LinearContrast(
                (1 / 1.2, 1.2),
                per_channel=0.5,  # pyright: ignore[reportArgumentType]
            ),
        ),
        ia.Sometimes(prob, ia.ElasticTransformation(alpha=(0.5, 1.5), sigma=0.25)),
    ]
    return ia.Sequential(perturbations, random_order=True)


def bbox_json2array(
    bbox_dict: dict,
    config: TransfuserConfig,
) -> tuple[jt.Float[npt.NDArray, " 9"], jt.Float[npt.NDArray, " timesteps 2"], int]:
    """Extract a bounding box label from a view-frame CARLA bounding box dictionary.

    Args:
        bbox_dict: Dictionary containing bounding box information from CARLA,
            already expressed in the sample's view frame.
        config: Transfuser architecture configuration.

    Returns:
        Array with bounding boxs. Each row is a bounding box.
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
    bbox[TransfuserBoundingBoxIndex.YAW] = common_utils.normalize_angle(
        bbox_dict["yaw"],
    )

    if bbox_dict["class"] == "car":  # static class = parking vehicle = an implicit car
        bbox[TransfuserBoundingBoxIndex.VELOCITY] = bbox_dict["speed"]
        # check for nans
        if np.isnan(bbox_dict["brake"]):
            bbox[TransfuserBoundingBoxIndex.BRAKE] = 0
        else:
            bbox[TransfuserBoundingBoxIndex.BRAKE] = bbox_dict["brake"]
        if (
            "role_name" in bbox_dict
            and "scenario" in bbox_dict["role_name"]
            and bbox_dict["type_id"] in constants.EMERGENCY_MESHES
        ):
            # this is an emergency vehicle that we need to yield to (or dodge in the RunningRedLight scenario)
            # so we give it a different label
            bbox[TransfuserBoundingBoxIndex.CLASS] = TransfuserBoundingBoxClass.SPECIAL
        else:
            bbox[TransfuserBoundingBoxIndex.CLASS] = TransfuserBoundingBoxClass.VEHICLE
    elif bbox_dict["class"] == "walker":
        bbox[TransfuserBoundingBoxIndex.VELOCITY] = bbox_dict["speed"]
        bbox[TransfuserBoundingBoxIndex.CLASS] = TransfuserBoundingBoxClass.WALKER
    elif bbox_dict["class"] == "traffic_light":
        bbox[TransfuserBoundingBoxIndex.CLASS] = (
            TransfuserBoundingBoxClass.TRAFFIC_LIGHT
        )
    elif bbox_dict["class"] == "stop_sign":
        bbox[TransfuserBoundingBoxIndex.CLASS] = TransfuserBoundingBoxClass.STOP_SIGN
    elif bbox_dict["class"] == "static":
        bbox[TransfuserBoundingBoxIndex.CLASS] = TransfuserBoundingBoxClass.VEHICLE

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

        # In CARLA, often vehicles disappear. But they are not removed but rather teleported far away.
        # To mitigate this, we check the distance between the last known position and the future waypoints.
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
                dist = np.linalg.norm(
                    last_pos - bbox_dict["future_positions"][future_waypoint_index],
                )
                if dist > config.max_distance_future_waypoint:
                    break
                waypoints[i] = bbox_dict["future_positions"][future_waypoint_index]
                num_waypoints += 1
                last_pos = bbox_dict["future_positions"][future_waypoint_index]
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


def get_bbox_labels(
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
    config = lead_config.agent.transfuser
    data_config = lead_config.expert.data_collection
    bboxes, waypoints, num_waypoints = [], [], []

    for _, current_box in enumerate(boxes):
        bbox, waypoint, num_waypoint = bbox_json2array(
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

            if (
                current_box["transfuser_semantics_id"]
                == TransfuserSemanticSegmentationClass.PEDESTRIAN
                and 0
                <= num_points
                < lead_config.expert.occlusion.pedestrian_min_num_lidar_points
                and 0
                <= visible_pixels
                < lead_config.expert.occlusion.pedestrian_min_num_visible_pixels
            ):
                continue
            if (
                current_box["transfuser_semantics_id"]
                == TransfuserSemanticSegmentationClass.VEHICLE
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
            bbox[TransfuserBoundingBoxIndex.X] <= data_config.min_x_meter
            or bbox[TransfuserBoundingBoxIndex.X] >= data_config.max_x_meter
            or bbox[TransfuserBoundingBoxIndex.Y] <= data_config.min_y_meter
            or bbox[TransfuserBoundingBoxIndex.Y] >= data_config.max_y_meter
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
                    bbox[TransfuserBoundingBoxIndex.W],
                    bbox[TransfuserBoundingBoxIndex.H],
                ) = (
                    TRAFFIC_WARNING_BB_SIZE[0],
                    TRAFFIC_WARNING_BB_SIZE[1],
                )
                bbox[TransfuserBoundingBoxIndex.CLASS] = (
                    TransfuserBoundingBoxClass.OBSTACLE
                )
            elif current_box["type_id"] == "static.prop.constructioncone":
                (
                    bbox[TransfuserBoundingBoxIndex.W],
                    bbox[TransfuserBoundingBoxIndex.H],
                ) = (
                    CONSTRUCTION_CONE_BB_SIZE[0],
                    CONSTRUCTION_CONE_BB_SIZE[1],
                )
                bbox[TransfuserBoundingBoxIndex.CLASS] = (
                    TransfuserBoundingBoxClass.OBSTACLE
                )

        if "mesh_path" in current_box:
            if current_box["mesh_path"] in constants.LOOKUP_TABLE:
                bbox[TransfuserBoundingBoxIndex.W] = constants.LOOKUP_TABLE[
                    current_box["mesh_path"]
                ][0]
                bbox[TransfuserBoundingBoxIndex.H] = constants.LOOKUP_TABLE[
                    current_box["mesh_path"]
                ][1]
            if is_parking_vehicle:
                bbox[TransfuserBoundingBoxIndex.CLASS] = (
                    TransfuserBoundingBoxClass.VEHICLE
                )

        if current_box["class"] == "static_prop_car":
            bbox[TransfuserBoundingBoxIndex.CLASS] = TransfuserBoundingBoxClass.VEHICLE

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
                    bbox[TransfuserBoundingBoxIndex.H] += (
                        config.car_open_door_extra_width / 2
                    )
                    if current_measurement["vehicle_door_side"] == "left":
                        bbox[TransfuserBoundingBoxIndex.Y] += (
                            config.car_open_door_extra_width / 2
                        )
                    else:
                        bbox[TransfuserBoundingBoxIndex.Y] -= (
                            config.car_open_door_extra_width / 2
                        )

                if (
                    bbox[TransfuserBoundingBoxIndex.CLASS]
                    != TransfuserBoundingBoxClass.SPECIAL
                ) and current_measurement["vehicle_opened_door"]:
                    bbox[TransfuserBoundingBoxIndex.CLASS] = (
                        TransfuserBoundingBoxClass.OBSTACLE
                    )
            elif (
                bbox[TransfuserBoundingBoxIndex.CLASS]
                != TransfuserBoundingBoxClass.SPECIAL
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
                    bbox[TransfuserBoundingBoxIndex.CLASS] = (
                        TransfuserBoundingBoxClass.OBSTACLE
                    )

        if current_box.get("type_id") in constants.BIKER_MESHES:
            bbox[TransfuserBoundingBoxIndex.CLASS] = TransfuserBoundingBoxClass.BIKER

        bbox = bb_vehicle_to_image_system(
            bbox.reshape(1, -1),
            data_config.pixels_per_meter,
            data_config.min_x_meter,
            data_config.min_y_meter,
        ).squeeze()
        if not (
            0 <= bbox[TransfuserBoundingBoxIndex.X] < data_config.lidar_width_pixel
        ):
            LOG.warning(
                f"{bbox[TransfuserBoundingBoxIndex.X]=} is larger than {data_config.lidar_width_pixel=}",
            )
            continue
        if not (
            0 <= bbox[TransfuserBoundingBoxIndex.Y] < data_config.lidar_height_pixel
        ):
            LOG.warning(
                f"{bbox[TransfuserBoundingBoxIndex.Y]=} is larger than {data_config.lidar_height_pixel=}",
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


def get_centernet_labels(
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
    config = lead_config.agent.transfuser
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

    center_x = (
        gt_bboxes[:, [TransfuserBoundingBoxIndex.X]] / config.bev_down_sample_factor
    )
    center_y = (
        gt_bboxes[:, [TransfuserBoundingBoxIndex.Y]] / config.bev_down_sample_factor
    )
    gt_centers = np.concatenate((center_x, center_y), axis=1)

    for j, ct in enumerate(gt_centers):
        ctx_int, cty_int = ct.astype(int)
        ctx, cty = ct
        if ctx_int < 0 or ctx_int >= feat_w or cty_int < 0 or cty_int >= feat_h:
            LOG.warning(
                f"Be cautious! Bounding box center {ct} is out of bounds for image size ({feat_h}, {feat_w}).",
            )
            continue

        extent_x = (
            gt_bboxes[j, TransfuserBoundingBoxIndex.W] / config.bev_down_sample_factor
        )
        extent_y = (
            gt_bboxes[j, TransfuserBoundingBoxIndex.H] / config.bev_down_sample_factor
        )

        radius = g_t.gaussian_radius([extent_y, extent_x], min_overlap=0.1)
        radius = max(2, int(radius))
        ind = gt_bboxes[j, TransfuserBoundingBoxIndex.CLASS].astype(int)

        g_t.gen_gaussian_target(center_heatmap_target[ind], [ctx_int, cty_int], radius)

        wh_target[0, cty_int, ctx_int] = extent_x
        wh_target[1, cty_int, ctx_int] = extent_y

        yaw_class, yaw_res = common_utils.angle2class(
            gt_bboxes[j, TransfuserBoundingBoxIndex.YAW],
            config.num_dir_bins,
        )

        yaw_class_target[0, cty_int, ctx_int] = yaw_class
        yaw_res_target[0, cty_int, ctx_int] = yaw_res

        velocity_target[0, cty_int, ctx_int] = gt_bboxes[
            j,
            TransfuserBoundingBoxIndex.VELOCITY,
        ]
        # Brakes can potentially be continous but we classify them now.
        # Using mathematical rounding the split is applied at 0.5
        brake_target[0, cty_int, ctx_int] = int(
            round(gt_bboxes[j, TransfuserBoundingBoxIndex.BRAKE]),
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


def build_bev_occupancy(
    data: dict,
    current_measurement: dict,
    json_boxes: list,
    lead_config: LeadConfig,
) -> jt.UInt8[npt.NDArray, "H W"]:
    """Build bird's eye view occupancy map from bounding box data.

    Creates a semantic occupancy map by projecting view-frame bounding boxes
    onto a grid. Handles various object types including vehicles, pedestrians,
    traffic lights, and construction zones.

    Args:
        data: Dictionary containing scenario information.
        current_measurement: Current measurement data including scenario obstacles.
        json_boxes: List of view-frame bounding box dictionaries from CARLA.
        lead_config: Root config tree.

    Returns:
        Bird's eye view occupancy map as integer array of shape.
    """
    config = lead_config.agent.transfuser
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

        yaw = common_utils.normalize_angle(current_box["yaw"])
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
        label = TransfuserBEVOccupancyClass.VEHICLE
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
                    label = TransfuserBEVOccupancyClass.OBSTACLE
                    obstacle_scenario_corners.extend(box_pts.tolist())
                else:
                    label = TransfuserBEVOccupancyClass.SPECIAL_VEHICLE
            elif current_box.get("type_id", "") in constants.BIKER_MESHES:
                label = TransfuserBEVOccupancyClass.BIKER
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
                    label = TransfuserBEVOccupancyClass.OBSTACLE
                elif (
                    data["scenario_type"] in ["VehicleOpensDoorTwoWays"]
                    and current_box["id"]
                    in current_measurement["scenario_obstacles_ids"]
                    and current_measurement["vehicle_opened_door"]
                ):
                    label = TransfuserBEVOccupancyClass.OBSTACLE
                    obstacle_scenario_corners.extend(box_pts.tolist())
        elif cls == "walker":
            label = TransfuserBEVOccupancyClass.WALKER
        elif cls == "static_prop_car":
            label = TransfuserBEVOccupancyClass.VEHICLE
        elif cls == "static":
            type_id = current_box.get("type_id", "")
            is_parking_car = (
                current_box.get("mesh_path") is not None
                and "ParkedVehicles" in current_box["mesh_path"]
            )
            if type_id in config.data_bb_static_types_white_list:
                label = TransfuserBEVOccupancyClass.OBSTACLE
                if type_id == "static.prop.constructioncone":
                    cone_corners.extend(box_pts.tolist())
                elif type_id == "static.prop.trafficwarning":
                    warning_corners.extend(box_pts.tolist())
            elif type_id in constants.EMERGENCY_MESHES:
                label = TransfuserBEVOccupancyClass.SPECIAL_VEHICLE
                obstacle_scenario_corners.extend(box_pts.tolist())
            elif is_parking_car:
                label = TransfuserBEVOccupancyClass.VEHICLE
            else:
                continue
        elif cls == "traffic_light":
            if current_box["affects_ego"] and current_box["state"] in ["Red", "Yellow"]:
                if (
                    not current_measurement["over_head_traffic_light"]
                    and not current_measurement["europe_traffic_light"]
                ):
                    label = TransfuserBEVOccupancyClass.TRAFFIC_RED_NORMAL
                    normal_red_light_corners.extend(box_pts.tolist())
                else:
                    label = TransfuserBEVOccupancyClass.TRAFFIC_RED_NOT_NORMAL
                    unnormal_red_light_corners.extend(box_pts.tolist())
            elif current_box["affects_ego"]:
                label = TransfuserBEVOccupancyClass.TRAFFIC_GREEN
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
            cv2.fillPoly(bev, [hull], TransfuserBEVOccupancyClass.OBSTACLE)

    # Accident and obstacle scenario detection
    if len(obstacle_scenario_corners) >= 3:
        all_pts = np.array(obstacle_scenario_corners)
        mean = np.mean(all_pts, axis=0)
        dists = np.linalg.norm(all_pts - mean, axis=1)

        if np.all(dists <= 48 * scale):
            hull = cv2.convexHull(all_pts)
            cv2.fillPoly(bev, [hull], TransfuserBEVOccupancyClass.OBSTACLE)

    # Red light detection
    if len(unnormal_red_light_corners) >= 3:
        hull = cv2.convexHull(np.array(unnormal_red_light_corners))
        cv2.fillPoly(bev, [hull], TransfuserBEVOccupancyClass.TRAFFIC_RED_NOT_NORMAL)
    elif len(normal_red_light_corners) >= 3:
        hull = cv2.convexHull(np.array(normal_red_light_corners))
        cv2.fillPoly(bev, [hull], TransfuserBEVOccupancyClass.TRAFFIC_RED_NORMAL)
    elif len(green_light_corners) >= 3:
        hull = cv2.convexHull(np.array(green_light_corners))
        cv2.fillPoly(bev, [hull], TransfuserBEVOccupancyClass.TRAFFIC_GREEN)
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


def preprocess_radar_input(
    lead_config: LeadConfig,
    radar_data_dict: dict,
) -> list[jt.Float[npt.NDArray, "N 5"]]:
    """Preprocess radar input data for model inference.

    Args:
        lead_config: Root config tree.
        radar_data_dict: Dictionary containing radar data from sensors (e.g., {"radar1": array, "radar2": array, ...}).

    Returns:
        List of preprocessed radar data with sensor ID as last column.
    """
    if not lead_config.expert.sensor_rig.use_radars:
        return []

    config = lead_config.agent.transfuser
    data_config = lead_config.expert.data_collection

    def filter_and_pad_radars(arr):
        # Filter points within spatial bounds
        x_mask = (arr[:, 0] >= data_config.min_x_meter) & (
            arr[:, 0] <= data_config.max_x_meter
        )
        y_mask = (arr[:, 1] >= data_config.min_y_meter) & (
            arr[:, 1] <= data_config.max_y_meter
        )
        valid_mask = x_mask & y_mask
        filtered_arr = arr[valid_mask]

        # Pad the filtered array
        n = filtered_arr.shape[0]
        if n >= config.num_radar_points_per_sensor:
            return filtered_arr[: config.num_radar_points_per_sensor]
        out = np.zeros(
            (config.num_radar_points_per_sensor, filtered_arr.shape[1]),
            dtype=np.float32,
        )
        out[:n] = filtered_arr.astype(np.float32)
        return out

    radar_list = []
    for i in range(1, lead_config.expert.sensor_rig.num_radar_sensors + 1):
        padded_radar = filter_and_pad_radars(radar_data_dict[f"radar{i}"])

        # Add sensor identity column (0-indexed)
        sensor_id = np.full((padded_radar.shape[0], 1), float(i - 1), dtype=np.float32)
        radar_with_id = np.concatenate([padded_radar, sensor_id], axis=1)
        radar_list.append(radar_with_id)
    return radar_list


def parse_radar_detection_labels(
    lead_config: LeadConfig,
    sensor_data: SensorData,
) -> jt.Float32[npt.NDArray, "num_queries features"]:
    """Parse and filter radar detection labels from sensor data for model training.

    This function extracts radar-based object detections from bounding box data, filtering
    and prioritizing detections based on radar point coverage, object velocity, and class
    importance. It converts detections to vehicle coordinates and outputs a fixed-size array
    suitable for model consumption.

    The selection process prioritizes detections by:
    1. Higher velocity (more relevant for collision avoidance)
    2. Class priority (SPECIAL > VEHICLE > WALKER > OBSTACLE)
    3. More radar measurement points (higher confidence)

    Args:
        lead_config: Root config tree with the radar query and BEV geometry knobs.
        sensor_data: Sensor data container with bounding boxes, waypoints, and metadata.
            Must have non-None boxes attribute if radar processing is enabled.

    Returns:
        Array of shape (num_radar_queries, num_features) containing radar detection labels.
        Each row represents one detection with features [x, y, velocity, valid_flag].
        Unused slots are zero-padded. Features are in vehicle coordinate system.
    """
    config = lead_config.agent.transfuser
    data_config = lead_config.expert.data_collection

    # Initialize default values (all zeros)
    radar_detections = np.zeros(
        (config.num_radar_queries, len(RadarLabels)),
        dtype=np.float32,
    )

    if (
        lead_config.expert.sensor_rig.use_radars
        and sensor_data.boxes is not None
        and sensor_data.boxes.shape[0] > 0
    ):
        assert sensor_data.boxes_waypoints is not None
        assert sensor_data.boxes_num_waypoints is not None
        priority_classes = [
            TransfuserBoundingBoxClass.SPECIAL,
            TransfuserBoundingBoxClass.VEHICLE,
            TransfuserBoundingBoxClass.WALKER,
            TransfuserBoundingBoxClass.OBSTACLE,
        ]

        # Copy data
        loaded_boxes_image_system = sensor_data.boxes.copy()
        loaded_waypoints = sensor_data.boxes_waypoints.copy()
        loaded_num_waypoints = sensor_data.boxes_num_waypoints.copy()
        loaded_boxes_vehicle_system = bb_image_to_vehicle_system(
            loaded_boxes_image_system,
            data_config.pixels_per_meter,
            data_config.min_x_meter,
            data_config.min_y_meter,
        )

        # Remove zero-padded data
        non_zero_mask = (
            loaded_boxes_vehicle_system[:, TransfuserBoundingBoxIndex.X] != 0.0
        ) | (loaded_boxes_vehicle_system[:, TransfuserBoundingBoxIndex.Y] != 0.0)
        loaded_boxes_vehicle_system = loaded_boxes_vehicle_system[non_zero_mask]
        loaded_waypoints = loaded_waypoints[non_zero_mask]
        loaded_num_waypoints = loaded_num_waypoints[non_zero_mask]

        # Filter data with minimally one radar point
        radar_mask = (
            loaded_boxes_vehicle_system[:, TransfuserBoundingBoxIndex.NUM_RADAR_POINTS]
            > 0
        )
        loaded_boxes_vehicle_system = loaded_boxes_vehicle_system[radar_mask]
        loaded_waypoints = loaded_waypoints[radar_mask]
        loaded_num_waypoints = loaded_num_waypoints[radar_mask]

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
                        TransfuserBoundingBoxIndex.CLASS,
                    ]
                ],
            )

            # Stack into sortable array: (-velocity, class_priority, -num_radar_points)
            sortable = np.stack(
                [
                    -loaded_boxes_vehicle_system[
                        :,
                        TransfuserBoundingBoxIndex.VELOCITY,
                    ],
                    class_priority,
                    -loaded_boxes_vehicle_system[
                        :,
                        TransfuserBoundingBoxIndex.NUM_RADAR_POINTS,
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
                TransfuserBoundingBoxIndex.X,
            ]
            radar_detections[:n_boxes, RadarLabels.Y] = selected_boxes[
                :,
                TransfuserBoundingBoxIndex.Y,
            ]
            radar_detections[:n_boxes, RadarLabels.V] = selected_boxes[
                :,
                TransfuserBoundingBoxIndex.VELOCITY,
            ]
            radar_detections[:n_boxes, RadarLabels.VALID] = 1.0  # Valid box indicator

    return radar_detections
