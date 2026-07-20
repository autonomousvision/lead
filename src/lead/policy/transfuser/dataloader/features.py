"""TransFuser featurization: one frame of raw driving data into the model inputs.

Label branches are skipped when the frame is not privileged.
"""

import typing

import cv2
import numpy as np

from lead.config import LeadConfig
from lead.dataloader import Frame, view_geometry
from lead.dataloader.box_decoding import carla_forward_speed
from lead.dataloader.sensor_decoding import (
    lidar_sweep_to_carla_ego_frame,
    radar_to_carla_ego_frame,
    split_radar_by_sensor,
)
from lead.policy.transfuser.dataloader import bev_raster
from lead.policy.transfuser.dataloader.labels import (
    SensorData,
    build_bev_occupancy,
    get_bbox_labels,
    get_centernet_labels,
    parse_radar_detection_labels,
    preprocess_radar_input,
    rasterize_lidar,
    semantics_from_instances,
)
from lead.policy.transfuser.dataloader.point_cloud import accumulate_lidar_points
from lead.policy.transfuser.labels import BEVOccupancyClass, BEVSemanticClass


def build_features(
    frame: Frame,
    lead_config: LeadConfig,
    image_augmenter: typing.Any | None = None,
    boxes: list[dict] | None = None,
) -> dict[str, typing.Any]:
    """Turn one frame into the TransFuser model inputs, plus labels when privileged.

    Args:
        frame: One tick of raw driving data in the ego view frame.
        lead_config: Root config tree.
        image_augmenter: Color augmenter applied to the stitched camera image;
            None to leave the image untouched.
        boxes: The tick's view-frame box dicts derived from the native
            detections; None at inference time.

    Returns:
        The model inputs of one unbatched sample. Privileged frames also carry
        the training labels (semantics, depth, BEV semantic, CenterNet).
    """
    data: dict[str, typing.Any] = {}

    _add_model_inputs(data, frame, lead_config, image_augmenter)
    if frame.is_privileged:
        _add_labels(data, frame, lead_config, boxes)
    return data


def _add_model_inputs(
    data: dict[str, typing.Any],
    frame: Frame,
    lead_config: LeadConfig,
    image_augmenter: typing.Any | None,
) -> None:
    """Add the inputs the model consumes at both training and inference time.

    Args:
        data: Sample dict to add the inputs to, modified in place.
        frame: The frame to featurize.
        lead_config: Root config tree.
        image_augmenter: Color augmenter, or None.
    """
    config = lead_config.policy.transfuser

    # RGB: concatenate the per-camera images side by side, left to right.
    image = np.concatenate([camera.image for camera in frame.cameras], axis=1)
    if image_augmenter is not None:
        image = image_augmenter(image=image)
        assert isinstance(image, np.ndarray)
    data["rgb"] = np.transpose(image, (2, 0, 1))

    # LiDAR BEV: accumulate the sweep history into the anchor frame. The 123D
    # sweeps are decoded into the CARLA ego frame.
    if frame.lidar_sweeps is not None:
        assert frame.past_positions is not None
        assert frame.past_yaws is not None
        lidar_points = accumulate_lidar_points(
            {
                age: lidar_sweep_to_carla_ego_frame(lidar)
                for age, lidar in frame.lidar_sweeps.items()
            },
            (
                None
                if frame.radar_sweeps is None
                else {
                    age: radar_to_carla_ego_frame(radar)[:, :3].astype(np.float64)
                    for age, radar in frame.radar_sweeps.items()
                }
            ),
            frame.past_positions,
            frame.past_yaws,
            lead_config,
        )
        if frame.perturbation is not None:
            lidar_points = view_geometry.to_view_frame_lidar(
                lidar_points,
                frame.perturbation.translation,
                frame.perturbation.rotation,
            )
        points = lidar_points[
            lidar_points[:, 3] < lead_config.training.data.training_used_lidar_steps
        ]
        data["rasterized_lidar"] = np.array(
            rasterize_lidar(lead_config=lead_config, lidar=points[:, :3]),
        ).squeeze()[None]

    # Radars: the anchor tick's merged returns split per sensor, filtered and
    # padded to a fixed size.
    if frame.radar_sweeps is not None:
        radar_list = preprocess_radar_input(
            lead_config,
            {
                f"radar{i + 1}": radar
                for i, radar in enumerate(split_radar_by_sensor(frame.radar_sweeps[0]))
            },
        )
        for i, arr in enumerate(radar_list):
            data[f"radar{i + 1}"] = arr
        data["radar"] = np.concatenate(radar_list, axis=0)

    # Navigation and ego state.
    for key in ("target_point_previous", "target_point", "target_point_next"):
        point = getattr(frame, key)
        if point is not None:
            data[key] = point
    if config.use_velocity:
        assert frame.ego_state is not None
        data["speed"] = carla_forward_speed(frame.ego_state)
    assert frame.log_metadata is not None
    data["town"] = frame.log_metadata.location


def _add_labels(
    data: dict[str, typing.Any],
    frame: Frame,
    lead_config: LeadConfig,
    view_boxes: list[dict] | None,
) -> None:
    """Add the training targets read from the privileged fields of the frame.

    Args:
        data: Sample dict to add the labels to, modified in place.
        frame: The privileged frame to build the labels from.
        lead_config: Root config tree.
        view_boxes: The tick's view-frame box dicts, unfiltered.
    """
    config = lead_config.policy.transfuser
    data_config = lead_config.expert.data_collection
    meta = frame.meta
    assert meta is not None

    current_active_scenario_type = meta.get("current_active_scenario_type")
    previous_active_scenario_type = meta.get("previous_active_scenario_type")
    if current_active_scenario_type is not None:
        data["scenario_type"] = current_active_scenario_type
    elif previous_active_scenario_type is not None:
        data["scenario_type"] = previous_active_scenario_type
    else:
        data["scenario_type"] = "NA"

    # Bounding boxes
    boxes = boxes_waypoints = boxes_num_waypoints = None
    if config.detect_boxes or config.use_bev_semantic:
        assert view_boxes is not None
        boxes, boxes_waypoints, boxes_num_waypoints = get_bbox_labels(
            data,
            lead_config,
            view_boxes,
            meta,
        )

    # Radar detections
    if frame.radar_sweeps is not None:
        data["radar_detections"] = parse_radar_detection_labels(
            lead_config,
            SensorData(
                image=None,
                rasterized_lidar=None,
                semantic=None,
                hdmap=None,
                depth=None,
                boxes=boxes,
                boxes_waypoints=boxes_waypoints,
                boxes_num_waypoints=boxes_num_waypoints,
                bev_occupancy=None,
                radars=tuple(split_radar_by_sensor(frame.radar_sweeps[0])),
                radar_detections=None,
            ),
        )

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
        semantics = semantics_from_instances(instances, view_boxes)
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

        bev_occupancy = build_bev_occupancy(data, meta, view_boxes, lead_config)
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
        data.update(get_centernet_labels(boxes, lead_config, config.num_bb_classes))
