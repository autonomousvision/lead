"""Label builders for the TransFuser sample dict.

All geometry consumed here is already expressed in the sample's view frame.
"""

from __future__ import annotations

import logging

import cv2
import jaxtyping as jt
import numpy as np
import numpy.typing as npt
from py123d.datatypes import DepthCameraMetadata

from lead.api.driving_meta import DrivingMeta
from lead.common import constants, geometry
from lead.common.constants import (
    CONSTRUCTION_CONE_BB_SIZE,
    TRAFFIC_WARNING_BB_SIZE,
    RadarLabels,
)
from lead.config import LeadConfig
from lead.config.policy.transfuser.label_classes import (
    BEVOccupancyClass,
    BEVSemanticClass,
    BoundingBoxClass,
    BoundingBoxIndex,
    PerspectiveSemanticClass,
)
from lead.log_reader import SceneData, view_geometry
from lead.policy.transfuser.dataloader import bev_raster
from lead.policy.transfuser.dataloader.sample import TransfuserOutputs
from lead.policy.transfuser.decoder import center_net_decoder
from lead.policy.transfuser.utils import semantics

LOG = logging.getLogger(__name__)

# Boxes at the true world pose of a sign or light. TransFuser++ instead consumes
# the boxes projected onto the lanes those actors govern, so these are skipped.
_PHYSICAL_BOX_CLASSES = frozenset(
    {"traffic_light_physical", "stop_sign_physical", "traffic_sign"},
)

# BEV occupancy raster geometry: a 256 m grid centred on the ego, drawn at the
# BEV raster resolution so the crop of it lands on the semantic grid one to one.
_OCCUPANCY_GRID_SIZE_METER = 256
_OCCUPANCY_GRID_HALF_SIZE_METER = 128.0
# Convex hulls need at least a triangle; a construction site needs enough cones.
_MIN_HULL_CORNERS = 3
_MIN_CONSTRUCTION_SITE_CONE_CORNERS = 24
# A hull is drawn only when its corners cluster within these radii.
_CONSTRUCTION_SITE_MAX_SPREAD_METER = 24
_OBSTACLE_SCENARIO_MAX_SPREAD_METER = 48

_CARLA_TO_PERSPECTIVE_SEMANTIC_LUT = np.array(
    [
        semantics.SEMANTIC_SEGMENTATION_CONVERTER[carla_class]
        for carla_class in constants.CarlaSemanticSegmentationClass
    ],
    dtype=np.uint8,
)


def build_labels(
    scene_data: SceneData,
    view_boxes: list[dict] | None,
    lead_config: LeadConfig,
) -> TransfuserOutputs:
    """Build the training targets read from the privileged fields of the scene_data.

    Args:
        scene_data: The privileged scene data to build the labels from.
        view_boxes: The tick's view-frame box dicts, unfiltered.
        lead_config: Root config tree.

    Returns:
        The label entries of one unbatched sample.
    """
    config = lead_config.policy.transfuser
    driving_meta = scene_data.driving_meta
    assert driving_meta is not None
    data: TransfuserOutputs = {}

    current_active_scenario_type = driving_meta.get("current_active_scenario_type")
    previous_active_scenario_type = driving_meta.get("previous_active_scenario_type")
    if current_active_scenario_type is not None:
        scenario_type = current_active_scenario_type
    elif previous_active_scenario_type is not None:
        scenario_type = previous_active_scenario_type
    else:
        scenario_type = "NA"

    # Bounding boxes
    box_rows_pixel_frame = None
    if config.detect_boxes or config.use_bev_semantic:
        assert view_boxes is not None
        box_rows_pixel_frame = _build_bounding_box_rows(
            scenario_type,
            lead_config,
            view_boxes,
            driving_meta,
        )

    # Radar detections
    if scene_data.radar_sweeps is not None:
        data["radar_detections"] = _build_radar_detection_targets(
            lead_config,
            box_rows_pixel_frame,
        )

    # Semantic segmentation, reduced from the scene's class and actor ids and
    # the tick's boxes (raw CARLA classes into TransFuser's label space).
    if (
        scene_data.semantic_cameras is not None
        and scene_data.instance_cameras is not None
    ):
        assert view_boxes is not None
        semantic_instance_maps = [
            np.stack(
                [
                    semantic_camera.image.astype(np.int32),
                    instance_camera.image.astype(np.int32),
                ],
                axis=-1,
            )
            for semantic_camera, instance_camera in zip(
                scene_data.semantic_cameras,
                scene_data.instance_cameras,
                strict=True,
            )
        ]
        per_camera_semantic_labels = _semantics_from_instances(
            semantic_instance_maps,
            view_boxes,
        )
        data["semantic"] = np.concatenate(per_camera_semantic_labels, axis=1)

    # BEV semantic: static map raster + dynamic occupancy overlay
    if config.use_bev_semantic:
        assert view_boxes is not None
        assert scene_data.map_api is not None
        assert scene_data.ego_state is not None
        view_frame_center_se2 = view_geometry.view_frame_center_se2(
            scene_data.ego_state.center_se2,
            scene_data.rig_perturbation.lateral_translation_m
            if scene_data.rig_perturbation
            else 0.0,
            scene_data.rig_perturbation.yaw_rotation_deg
            if scene_data.rig_perturbation
            else 0.0,
        )
        bev_semantic_grid = bev_raster.rasterize_bev_semantic_map(
            scene_data.map_api,
            view_frame_center_se2,
            stop_sign_hazard=bool(driving_meta["stop_sign_hazard"]),
            lead_config=lead_config,
        )
        bev_occupancy = _rasterize_bev_occupancy(
            scenario_type,
            driving_meta,
            view_boxes,
            lead_config,
        )
        assert bev_occupancy.shape[0] == bev_occupancy.shape[1]
        bev_occupancy_center = bev_occupancy.shape[0] / 2
        x_cut = (
            bev_occupancy_center
            + np.array([config.bev_min_x_meter, config.bev_max_x_meter])
            * config.bev_pixels_per_meter
        ).astype(int)
        y_cut = (
            bev_occupancy_center
            + np.array([config.bev_min_y_meter, config.bev_max_y_meter])
            * config.bev_pixels_per_meter
        ).astype(int)
        cropped_bev_occupancy = bev_occupancy[y_cut[0] : y_cut[1], x_cut[0] : x_cut[1]]
        mask = cropped_bev_occupancy != BEVOccupancyClass.UNLABELED
        bev_semantic_grid = bev_semantic_grid.copy()
        bev_semantic_grid[mask] = cropped_bev_occupancy[mask] + (
            len(BEVSemanticClass) - len(BEVOccupancyClass)
        )
        data["bev_semantic"] = bev_semantic_grid

    # 2D bounding boxes for CenterNet
    if config.detect_boxes:
        assert box_rows_pixel_frame is not None
        data.update(
            _build_center_net_targets(
                box_rows_pixel_frame,
                lead_config,
                config.num_box_classes,
            ),
        )

    # scenario_type is an internal input of the helpers above; the sample's
    # copy comes from the driving_meta builder.
    del scenario_type
    return data


def build_depth_target(scene_data: SceneData) -> TransfuserOutputs:
    """The depth target as the stored quantized raster, dequantized on the GPU.

    Shipping the raw codes instead of decoded float32 metres quarters the
    field's bytes; :func:`~lead.policy.transfuser.utils.ops.dequantize_depth`
    is the decode counterpart. Requires the dataset's linear, sentinel-free
    depth encoding.

    Args:
        scene_data: The privileged scene data to build the target from.

    Returns:
        The depth target, or an empty dict without depth cameras.
    """
    if scene_data.depth_cameras is None:
        return {}
    depth_grids = []
    for camera in scene_data.depth_cameras:
        assert isinstance(camera.metadata, DepthCameraMetadata)
        assert camera.metadata.depth_transform == "linear"
        assert not camera.metadata.has_invalid
        depth_grids.append(camera.image)
    return {"depth": np.concatenate(depth_grids, axis=1)}


def box_rows_to_bev_pixel_frame(
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


def box_rows_to_ego_frame(
    box: jt.Float[npt.NDArray, "N D"],
    pixels_per_meter: float,
    min_x: float,
    min_y: float,
) -> jt.Float[npt.NDArray, "N D"]:
    """Inverse of :func:`box_rows_to_bev_pixel_frame`.

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


def _semantics_from_instances(
    semantic_instance_maps: list[jt.Int32[npt.NDArray, "h w 2"]],
    view_frame_boxes: list[dict],
) -> list[jt.UInt8[npt.NDArray, "h w"]]:
    """Derive TransFuser's semantic labels from the stored class and actor ids.

    The logs store the raw CARLA classes; the classes CARLA's camera cannot
    emit (cones/traffic warnings, emergency vehicles, stop signs) are painted
    over the pixels whose instance id belongs to a matching recorded box.
    """
    painted: list[tuple[int, PerspectiveSemanticClass]] = []
    for box in view_frame_boxes:
        if box["class"] == "stop_sign_physical":
            # Physical sign boxes carry a synthetic id; the CARLA actor id
            # matching the instance pixels sits in ``actor_id``.
            actor_id = box.get("actor_id")
            painted_class = PerspectiveSemanticClass.STOP_SIGN
        elif box.get("type_id") in constants.CONSTRUCTION_MESHES:
            actor_id = box["id"]
            painted_class = PerspectiveSemanticClass.OBSTACLE
        elif box.get("type_id") in constants.EMERGENCY_MESHES:
            actor_id = box["id"]
            painted_class = PerspectiveSemanticClass.SPECIAL_VEHICLE
        else:
            continue
        if actor_id is None:
            continue
        painted.append((actor_id & constants.INSTANCE_ID_MASK, painted_class))

    per_camera_semantic_labels = []
    for instance_map in semantic_instance_maps:
        semantic_labels = _CARLA_TO_PERSPECTIVE_SEMANTIC_LUT[instance_map[..., 0]]
        for instance_id, painted_class in painted:
            semantic_labels[instance_map[..., 1] == instance_id] = painted_class
        per_camera_semantic_labels.append(semantic_labels)
    return per_camera_semantic_labels


def _box_dict_to_row(box_dict: dict) -> jt.Float32[npt.NDArray, " 9"]:
    """Extract a bounding box label from a view-frame CARLA bounding box dictionary."""
    x, y = box_dict["position"][:2]
    num_radar_points = box_dict.get("num_radar_points", -1)

    # center_x, center_y, w, h, yaw
    box_row_view_frame = np.array(
        [
            x,
            y,
            box_dict["extent"][0],
            box_dict["extent"][1],
            0,
            0,
            0,
            0,
            num_radar_points,
        ],
        dtype=np.float32,
    )
    box_row_view_frame[BoundingBoxIndex.YAW] = geometry.normalize_angle_rad(
        box_dict["yaw"],
    )

    if box_dict["class"] == "car":  # static class = parking vehicle = an implicit car
        box_row_view_frame[BoundingBoxIndex.VELOCITY] = box_dict["speed"]
        # check for nans
        if np.isnan(box_dict["brake"]):
            box_row_view_frame[BoundingBoxIndex.BRAKE] = 0
        else:
            box_row_view_frame[BoundingBoxIndex.BRAKE] = box_dict["brake"]
        if (
            "role_name" in box_dict
            and "scenario" in box_dict["role_name"]
            and box_dict["type_id"] in constants.EMERGENCY_MESHES
        ):
            # this is an emergency vehicle that we need to yield to (or dodge in the RunningRedLight scenario)
            # so we give it a different label
            box_row_view_frame[BoundingBoxIndex.CLASS] = BoundingBoxClass.SPECIAL
        else:
            box_row_view_frame[BoundingBoxIndex.CLASS] = BoundingBoxClass.VEHICLE
    elif box_dict["class"] == "walker":
        box_row_view_frame[BoundingBoxIndex.VELOCITY] = box_dict["speed"]
        box_row_view_frame[BoundingBoxIndex.CLASS] = BoundingBoxClass.WALKER
    elif box_dict["class"] == "traffic_light":
        box_row_view_frame[BoundingBoxIndex.CLASS] = BoundingBoxClass.TRAFFIC_LIGHT
    elif box_dict["class"] == "stop_sign":
        box_row_view_frame[BoundingBoxIndex.CLASS] = BoundingBoxClass.STOP_SIGN
    elif box_dict["class"] == "static":
        box_row_view_frame[BoundingBoxIndex.CLASS] = BoundingBoxClass.VEHICLE

    return box_row_view_frame


def _build_bounding_box_rows(
    scenario_type: str,
    lead_config: LeadConfig,
    view_frame_boxes: list[dict],
    driving_meta: DrivingMeta,
) -> jt.Float32[npt.NDArray, "n_boxes 9"]:
    """Parse and filter bounding boxes from CARLA data."""
    config = lead_config.policy.transfuser
    box_rows = []

    for current_box in view_frame_boxes:
        if current_box["class"] in _PHYSICAL_BOX_CLASSES:
            continue

        box_row_view_frame = _box_dict_to_row(current_box)
        if current_box["class"] in ["ego_car"]:
            continue

        # Occlusion check
        if "num_points" in current_box:
            num_points = current_box["num_points"]
            visible_pixels = -1
            if "visible_pixels" in current_box:
                visible_pixels = current_box["visible_pixels"]

            box_semantic_class = semantics.semantic_class(current_box)
            if (
                box_semantic_class == PerspectiveSemanticClass.PEDESTRIAN
                and 0
                <= num_points
                < lead_config.expert.occlusion.pedestrian_min_num_lidar_points
                and 0
                <= visible_pixels
                < lead_config.expert.occlusion.pedestrian_min_num_visible_pixels
            ):
                continue
            if (
                box_semantic_class == PerspectiveSemanticClass.VEHICLE
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
        center_z_view_frame_meter = current_box["position"][2]
        if (
            box_row_view_frame[BoundingBoxIndex.X] <= config.bev_min_x_meter
            or box_row_view_frame[BoundingBoxIndex.X] >= config.bev_max_x_meter
            or box_row_view_frame[BoundingBoxIndex.Y] <= config.bev_min_y_meter
            or box_row_view_frame[BoundingBoxIndex.Y] >= config.bev_max_y_meter
            or center_z_view_frame_meter
            <= lead_config.policy.transfuser.min_box_center_z_meter
            or center_z_view_frame_meter
            >= lead_config.policy.transfuser.max_box_center_z_meter
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
            and current_box["type_id"] not in config.detected_static_prop_type_ids
        ) and not is_parking_vehicle:
            continue

        if "type_id" in current_box:
            if current_box["type_id"] == "static.prop.trafficwarning":
                (
                    box_row_view_frame[BoundingBoxIndex.W],
                    box_row_view_frame[BoundingBoxIndex.H],
                ) = (
                    TRAFFIC_WARNING_BB_SIZE[0],
                    TRAFFIC_WARNING_BB_SIZE[1],
                )
                box_row_view_frame[BoundingBoxIndex.CLASS] = BoundingBoxClass.OBSTACLE
            elif current_box["type_id"] == "static.prop.constructioncone":
                (
                    box_row_view_frame[BoundingBoxIndex.W],
                    box_row_view_frame[BoundingBoxIndex.H],
                ) = (
                    CONSTRUCTION_CONE_BB_SIZE[0],
                    CONSTRUCTION_CONE_BB_SIZE[1],
                )
                box_row_view_frame[BoundingBoxIndex.CLASS] = BoundingBoxClass.OBSTACLE

        if "mesh_path" in current_box:
            if current_box["mesh_path"] in constants.LOOKUP_TABLE:
                box_row_view_frame[BoundingBoxIndex.W] = constants.LOOKUP_TABLE[
                    current_box["mesh_path"]
                ][0]
                box_row_view_frame[BoundingBoxIndex.H] = constants.LOOKUP_TABLE[
                    current_box["mesh_path"]
                ][1]
            if is_parking_vehicle:
                box_row_view_frame[BoundingBoxIndex.CLASS] = BoundingBoxClass.VEHICLE

        if current_box["class"] == "static_prop_car":
            box_row_view_frame[BoundingBoxIndex.CLASS] = BoundingBoxClass.VEHICLE

        # In some CARLA's scenarios, special vehicles have sometimes wrong labels.
        # We relabel those labels to be consistent with the scene
        if (
            current_box["class"] == "car"
            and "role_name" in current_box
            and "scenario" in current_box["role_name"]
            and current_box["speed"] < 0.1
        ):
            if (
                scenario_type == "VehicleOpensDoorTwoWays"
                and current_box["id"] in driving_meta["scenario_obstacles_ids"]
            ):
                # If the car open door, we extend the bounding box's width to consider the door
                if driving_meta["vehicle_opened_door"]:
                    box_row_view_frame[BoundingBoxIndex.H] += (
                        config.open_door_extra_width_meter / 2
                    )
                    if driving_meta["vehicle_door_side"] == "left":
                        box_row_view_frame[BoundingBoxIndex.Y] += (
                            config.open_door_extra_width_meter / 2
                        )
                    else:
                        box_row_view_frame[BoundingBoxIndex.Y] -= (
                            config.open_door_extra_width_meter / 2
                        )

                if (
                    box_row_view_frame[BoundingBoxIndex.CLASS]
                    != BoundingBoxClass.SPECIAL
                ) and driving_meta["vehicle_opened_door"]:
                    box_row_view_frame[BoundingBoxIndex.CLASS] = (
                        BoundingBoxClass.OBSTACLE
                    )
            elif box_row_view_frame[BoundingBoxIndex.CLASS] != BoundingBoxClass.SPECIAL:
                if (
                    scenario_type
                    in [
                        "Accident",
                        "AccidentTwoWays",
                        "BlockedIntersection",
                        "ParkedObstacle",
                        "ParkedObstacleTwoWays",
                    ]
                    and current_box["id"] in driving_meta["scenario_obstacles_ids"]
                ):
                    box_row_view_frame[BoundingBoxIndex.CLASS] = (
                        BoundingBoxClass.OBSTACLE
                    )

        if current_box.get("type_id") in constants.BIKER_MESHES:
            box_row_view_frame[BoundingBoxIndex.CLASS] = BoundingBoxClass.BIKER

        # Some CARLA level meshes carry a zero or sign-flipped extent; drop them
        # once the mesh and type-id patches have had their chance to supply one.
        if (
            box_row_view_frame[BoundingBoxIndex.W] <= 0
            or box_row_view_frame[BoundingBoxIndex.H] <= 0
        ):
            continue

        box_row_pixel_frame = box_rows_to_bev_pixel_frame(
            box_row_view_frame.reshape(1, -1),
            config.bev_pixels_per_meter,
            config.bev_min_x_meter,
            config.bev_min_y_meter,
        ).squeeze()
        if not (
            0 <= box_row_pixel_frame[BoundingBoxIndex.X] < config.lidar_width_pixel
        ):
            LOG.warning(
                f"{box_row_pixel_frame[BoundingBoxIndex.X]=} is larger than "
                f"{config.lidar_width_pixel=}",
            )
            continue
        if not (
            0 <= box_row_pixel_frame[BoundingBoxIndex.Y] < config.lidar_height_pixel
        ):
            LOG.warning(
                f"{box_row_pixel_frame[BoundingBoxIndex.Y]=} is larger than "
                f"{config.lidar_height_pixel=}",
            )
            continue
        box_rows.append(box_row_pixel_frame)

    bounding_boxes_array = np.array(box_rows)

    # Pad bounding boxes to a fixed number
    padded_bounding_boxes_array = np.zeros((config.max_num_boxes, 9), dtype=np.float32)
    if bounding_boxes_array.shape[0] > 0:
        kept = min(bounding_boxes_array.shape[0], config.max_num_boxes)
        padded_bounding_boxes_array[:kept, :] = bounding_boxes_array[:kept]

    return padded_bounding_boxes_array


def _build_center_net_targets(
    box_rows_pixel_frame: jt.Float[npt.NDArray, "N 9"],
    lead_config: LeadConfig,
    num_box_classes: int,
) -> TransfuserOutputs:
    """Compute the regression and classification targets for CenterNet."""
    config = lead_config.policy.transfuser
    # The CenterNet target grid is the BEV pixel grid after downsampling.
    bev_cell_rows = config.lidar_height_pixel // config.bev_downsample_factor
    bev_cell_cols = config.lidar_width_pixel // config.bev_downsample_factor

    center_heatmap_target = np.zeros(
        [num_box_classes, bev_cell_rows, bev_cell_cols],
        dtype=np.float32,
    )
    wh_target = np.zeros([2, bev_cell_rows, bev_cell_cols], dtype=np.float32)
    offset_target = np.zeros([2, bev_cell_rows, bev_cell_cols], dtype=np.float32)
    yaw_class_target = np.zeros([1, bev_cell_rows, bev_cell_cols], dtype=np.int32)
    yaw_residual_target = np.zeros([1, bev_cell_rows, bev_cell_cols], dtype=np.float32)
    velocity_target = np.zeros([1, bev_cell_rows, bev_cell_cols], dtype=np.float32)
    brake_target = np.zeros([1, bev_cell_rows, bev_cell_cols], dtype=np.int32)
    pixel_weight = np.zeros(
        [2, bev_cell_rows, bev_cell_cols],
        dtype=np.float32,
    )  # 2 is the max of the channels above here.

    if not box_rows_pixel_frame.shape[0] > 0:
        return {
            "center_net_bounding_boxes": box_rows_pixel_frame,
            "center_net_heatmap": center_heatmap_target,
            "center_net_wh": wh_target,
            "center_net_yaw_class": yaw_class_target.squeeze(0),
            "center_net_yaw_res": yaw_residual_target,
            "center_net_offset": offset_target,
            "center_net_velocity": velocity_target,
            "center_net_brake": brake_target.squeeze(0),
            "center_net_pixel_weight": pixel_weight,
            "center_net_avg_factor": np.asarray(1, dtype=np.int64),
        }

    center_col_cells = (
        box_rows_pixel_frame[:, [BoundingBoxIndex.X]] / config.bev_downsample_factor
    )
    center_row_cells = (
        box_rows_pixel_frame[:, [BoundingBoxIndex.Y]] / config.bev_downsample_factor
    )
    box_centers_cells = np.concatenate((center_col_cells, center_row_cells), axis=1)

    for j, center_cells in enumerate(box_centers_cells):
        # Python ints, not numpy scalars: these index into and are passed on as
        # plain grid coordinates.
        col_cell, row_cell = (int(cell) for cell in center_cells)
        col_subcell, row_subcell = center_cells
        if (
            col_cell < 0
            or col_cell >= bev_cell_cols
            or row_cell < 0
            or row_cell >= bev_cell_rows
        ):
            LOG.warning(
                f"Be cautious! Bounding box center {center_cells} is out of bounds "
                f"for image size ({bev_cell_rows}, {bev_cell_cols}).",
            )
            continue

        half_extent_x_cells = (
            box_rows_pixel_frame[j, BoundingBoxIndex.W] / config.bev_downsample_factor
        )
        half_extent_y_cells = (
            box_rows_pixel_frame[j, BoundingBoxIndex.H] / config.bev_downsample_factor
        )

        radius = center_net_decoder.gaussian_radius(
            [half_extent_y_cells, half_extent_x_cells],
            min_overlap=0.1,
        )
        radius = max(2, int(radius))
        ind = box_rows_pixel_frame[j, BoundingBoxIndex.CLASS].astype(int)

        center_net_decoder.gen_gaussian_target(
            center_heatmap_target[ind],
            [col_cell, row_cell],
            radius,
        )

        wh_target[0, row_cell, col_cell] = half_extent_x_cells
        wh_target[1, row_cell, col_cell] = half_extent_y_cells

        yaw_class, yaw_res = geometry.angle_to_yaw_class(
            float(box_rows_pixel_frame[j, BoundingBoxIndex.YAW]),
            config.num_yaw_bins,
        )

        yaw_class_target[0, row_cell, col_cell] = yaw_class
        yaw_residual_target[0, row_cell, col_cell] = yaw_res

        velocity_target[0, row_cell, col_cell] = box_rows_pixel_frame[
            j,
            BoundingBoxIndex.VELOCITY,
        ]
        # Brakes can potentially be continuous but we classify them now.
        # Using mathematical rounding the split is applied at 0.5
        brake_target[0, row_cell, col_cell] = int(
            round(box_rows_pixel_frame[j, BoundingBoxIndex.BRAKE]),
        )

        offset_target[0, row_cell, col_cell] = col_subcell - col_cell
        offset_target[1, row_cell, col_cell] = row_subcell - row_cell
        # All pixels with a bounding box have a weight of 1 all others have a weight of 0.
        # Used to ignore the pixels without bbs in the loss.
        pixel_weight[:, row_cell, col_cell] = 1.0

    # A 0-d array, not a scalar: the same type on both return paths, so it
    # collates to (batch,) and the configured array codec applies to it.
    num_positive_pixels = np.asarray(
        max(1, np.equal(center_heatmap_target, 1).sum()),
        dtype=np.int64,
    )
    return {
        "center_net_bounding_boxes": box_rows_pixel_frame,
        "center_net_heatmap": center_heatmap_target,
        "center_net_wh": wh_target,
        "center_net_yaw_class": yaw_class_target.squeeze(0),
        "center_net_yaw_res": yaw_residual_target,
        "center_net_offset": offset_target,
        "center_net_velocity": velocity_target,
        "center_net_brake": brake_target.squeeze(0),
        "center_net_pixel_weight": pixel_weight,
        "center_net_avg_factor": num_positive_pixels,
    }


def _rasterize_bev_occupancy(
    scenario_type: str,
    driving_meta: DrivingMeta,
    view_frame_boxes: list,
    lead_config: LeadConfig,
) -> jt.UInt8[npt.NDArray, "H W"]:
    """Build bird's eye view occupancy map from bounding box data.

    Projects view-frame bounding boxes (vehicles, pedestrians, traffic lights,
    construction zones) onto a semantic grid.
    """
    config = lead_config.policy.transfuser
    scale = config.bev_pixels_per_meter
    grid_size = int(_OCCUPANCY_GRID_SIZE_METER * scale)
    bev = np.zeros((grid_size, grid_size), dtype=np.uint8)

    obstacle_scenario_corners = []
    cone_corners = []
    warning_corners = []
    normal_red_light_corners = []
    unnormal_red_light_corners = []
    green_light_corners = []

    for current_box in view_frame_boxes:
        cls = current_box["class"]
        if cls not in ["car", "walker", "static", "static_prop_car", "traffic_light"]:
            continue
        extra_scale = 1.0
        min_extent = 0.0
        if cls in ["walker"] or (
            "type_id" in current_box
            and current_box["type_id"] in constants.BIKER_MESHES
        ):
            extra_scale = config.pedestrian_bev_extent_scale
            min_extent = config.pedestrian_bev_min_half_extent_meter
        x, y = current_box["position"][:2]
        y = -y

        cx = (x + _OCCUPANCY_GRID_HALF_SIZE_METER) * scale
        cy = (_OCCUPANCY_GRID_HALF_SIZE_METER - y) * scale
        if not (0 <= cx < grid_size and 0 <= cy < grid_size):
            continue

        yaw = geometry.normalize_angle_rad(current_box["yaw"])
        half_extent_x_meter, half_extent_y_meter = current_box["extent"][:2]
        half_extent_x_meter = max(half_extent_x_meter, min_extent)
        half_extent_y_meter = max(half_extent_y_meter, min_extent)
        if (
            current_box["class"] == "car"
            and "role_name" in current_box
            and "scenario" in current_box["role_name"]
            and current_box["speed"] < 0.1
            and current_box["id"] in driving_meta["scenario_obstacles_ids"]
        ):
            if scenario_type == "VehicleOpensDoorTwoWays":
                if driving_meta["vehicle_opened_door"]:
                    # This is a special case where we open the door, so we extend the bounding box's width
                    half_extent_y_meter += config.open_door_extra_width_meter / 2
                    if driving_meta["vehicle_door_side"] == "left":
                        cy += config.open_door_extra_width_meter / 2 * scale
                    else:
                        cy -= config.open_door_extra_width_meter / 2 * scale

        if "mesh_path" in current_box:
            if current_box["mesh_path"] in constants.LOOKUP_TABLE:
                half_extent_x_meter = constants.LOOKUP_TABLE[current_box["mesh_path"]][
                    0
                ]
                half_extent_y_meter = constants.LOOKUP_TABLE[current_box["mesh_path"]][
                    1
                ]
        elif "type_id" in current_box:
            if current_box["type_id"] == "static.prop.trafficwarning":
                half_extent_x_meter, half_extent_y_meter = (
                    TRAFFIC_WARNING_BB_SIZE[0],
                    TRAFFIC_WARNING_BB_SIZE[1],
                )
            elif current_box["type_id"] == "static.prop.constructioncone":
                half_extent_x_meter, half_extent_y_meter = (
                    CONSTRUCTION_CONE_BB_SIZE[0],
                    CONSTRUCTION_CONE_BB_SIZE[1],
                )

        # Some CARLA level meshes carry a zero or sign-flipped extent; drop them
        # once the mesh and type-id patches have had their chance to supply one.
        if half_extent_x_meter <= 0 or half_extent_y_meter <= 0:
            continue

        rect = (
            (cx, cy),
            (
                half_extent_x_meter * 2 * scale * extra_scale,
                half_extent_y_meter * 2 * scale * extra_scale,
            ),
            np.rad2deg(yaw),
        )
        box_pts = cv2.boxPoints(rect).astype(np.int32)

        # Assign label
        is_parking_car = False
        label = BEVOccupancyClass.VEHICLE
        if cls in ["car"]:
            if current_box.get("type_id", "") in constants.EMERGENCY_MESHES:
                if (
                    scenario_type
                    in [
                        "Accident",
                        "AccidentTwoWays",
                        "BlockedIntersection",
                        "ParkedObstacle",
                        "ParkedObstacleTwoWays",
                    ]
                    and current_box["id"] in driving_meta["scenario_obstacles_ids"]
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
                    scenario_type
                    in [
                        "Accident",
                        "AccidentTwoWays",
                        "BlockedIntersection",
                        "ParkedObstacle",
                        "ParkedObstacleTwoWays",
                    ]
                    and current_box["id"] in driving_meta["scenario_obstacles_ids"]
                ):
                    obstacle_scenario_corners.extend(box_pts.tolist())
                    label = BEVOccupancyClass.OBSTACLE
                elif (
                    scenario_type in ["VehicleOpensDoorTwoWays"]
                    and current_box["id"] in driving_meta["scenario_obstacles_ids"]
                    and driving_meta["vehicle_opened_door"]
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
            if type_id in config.detected_static_prop_type_ids:
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
                    not driving_meta["over_head_traffic_light"]
                    and not driving_meta["europe_traffic_light"]
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
        cv2.fillPoly(bev, [box_pts], (int(label),))

    # Construction site detection
    if (
        len(warning_corners) >= _MIN_HULL_CORNERS
        and len(cone_corners) >= _MIN_CONSTRUCTION_SITE_CONE_CORNERS
    ):
        all_pts = np.array(cone_corners + warning_corners)
        mean = np.mean(all_pts, axis=0)
        dists = np.linalg.norm(all_pts - mean, axis=1)

        if np.all(dists <= _CONSTRUCTION_SITE_MAX_SPREAD_METER * scale):
            hull = cv2.convexHull(all_pts)
            cv2.fillPoly(bev, [hull], (int(BEVOccupancyClass.OBSTACLE),))

    # Accident and obstacle scenario detection
    if len(obstacle_scenario_corners) >= _MIN_HULL_CORNERS:
        all_pts = np.array(obstacle_scenario_corners)
        mean = np.mean(all_pts, axis=0)
        dists = np.linalg.norm(all_pts - mean, axis=1)

        if np.all(dists <= _OBSTACLE_SCENARIO_MAX_SPREAD_METER * scale):
            hull = cv2.convexHull(all_pts)
            cv2.fillPoly(bev, [hull], (int(BEVOccupancyClass.OBSTACLE),))

    # Red light detection
    if len(unnormal_red_light_corners) >= _MIN_HULL_CORNERS:
        hull = cv2.convexHull(np.array(unnormal_red_light_corners))
        cv2.fillPoly(bev, [hull], (int(BEVOccupancyClass.TRAFFIC_RED_NOT_NORMAL),))
    elif len(normal_red_light_corners) >= _MIN_HULL_CORNERS:
        hull = cv2.convexHull(np.array(normal_red_light_corners))
        cv2.fillPoly(bev, [hull], (int(BEVOccupancyClass.TRAFFIC_RED_NORMAL),))
    elif len(green_light_corners) >= _MIN_HULL_CORNERS:
        hull = cv2.convexHull(np.array(green_light_corners))
        cv2.fillPoly(bev, [hull], (int(BEVOccupancyClass.TRAFFIC_GREEN),))
    return bev


def _build_radar_detection_targets(
    lead_config: LeadConfig,
    box_rows_pixel_frame: jt.Float32[npt.NDArray, "num_boxes box_features"] | None,
) -> jt.Float32[npt.NDArray, "num_queries radar_labels"]:
    """Build the radar detection labels from the tick's box rows.

    Converts detections to vehicle coordinates and fills a fixed-size array,
    prioritizing higher velocity, class importance (SPECIAL > VEHICLE > WALKER >
    OBSTACLE), and radar point count.
    """
    config = lead_config.policy.transfuser

    # Initialize default values (all zeros)
    radar_detections = np.zeros(
        (config.num_radar_queries, len(RadarLabels)),
        dtype=np.float32,
    )

    if (
        lead_config.expert.sensor_rig.use_radars
        and box_rows_pixel_frame is not None
        and box_rows_pixel_frame.shape[0] > 0
    ):
        priority_classes = [
            BoundingBoxClass.SPECIAL,
            BoundingBoxClass.VEHICLE,
            BoundingBoxClass.WALKER,
            BoundingBoxClass.OBSTACLE,
        ]

        box_rows_ego_frame = box_rows_to_ego_frame(
            box_rows_pixel_frame.copy(),
            config.bev_pixels_per_meter,
            config.bev_min_x_meter,
            config.bev_min_y_meter,
        )

        # Remove zero-padded data
        non_zero_mask = (box_rows_ego_frame[:, BoundingBoxIndex.X] != 0.0) | (
            box_rows_ego_frame[:, BoundingBoxIndex.Y] != 0.0
        )
        box_rows_ego_frame = box_rows_ego_frame[non_zero_mask]

        # Filter data with minimally one radar point
        radar_mask = box_rows_ego_frame[:, BoundingBoxIndex.NUM_RADAR_POINTS] > 0
        box_rows_ego_frame = box_rows_ego_frame[radar_mask]

        selected_boxes: jt.Float[npt.NDArray, "n 9"] = np.zeros(
            (0, box_rows_ego_frame.shape[1]),
            dtype=np.float32,
        )

        if box_rows_ego_frame.shape[0] > 0:
            # Compute class priority index for each box
            class_priorities = {int(cls): i for i, cls in enumerate(priority_classes)}
            class_priority = np.array(
                [
                    class_priorities.get(int(c), len(priority_classes))
                    for c in box_rows_ego_frame[
                        :,
                        BoundingBoxIndex.CLASS,
                    ]
                ],
            )

            # Stack into sortable array: (-velocity, class_priority, -num_radar_points)
            sortable = np.stack(
                [
                    -box_rows_ego_frame[
                        :,
                        BoundingBoxIndex.VELOCITY,
                    ],
                    class_priority,
                    -box_rows_ego_frame[
                        :,
                        BoundingBoxIndex.NUM_RADAR_POINTS,
                    ],
                ],
                axis=1,
            )

            # Sort lexicographically: we prioritize higher velocity, then class priority, then more radar points
            sorted_indices = np.lexsort(sortable.T[::-1])

            # Apply sorting to all three arrays
            sorted_boxes = box_rows_ego_frame[sorted_indices]

            # Take up to num_radar_queries
            selected_boxes = sorted_boxes[: config.num_radar_queries]

        if len(selected_boxes) > 0:
            # Extract the x, y and velocity columns.
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
