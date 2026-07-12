"""TransFuser training dataset (layer 2 of the data pipeline).

Turns the model-agnostic :class:`~lead.training.dataloader.py123d_data_loader.Frame`
into the legacy TransFuser sample dict (cameras, LiDAR BEV, radar, BEV
semantics, CenterNet labels).
"""

import time
import typing

import cv2
import numpy as np
from torch.utils.data import Dataset

if typing.TYPE_CHECKING:
    from imgaug.augmenters import Sequential

from lead.common.constants import (
    TransfuserBEVOccupancyClass,
    TransfuserBEVSemanticClass,
)
from lead.config import LeadConfig
from lead.policy.transfuser.dataloader import bev_raster
from lead.policy.transfuser.dataloader.dataset_utils import (
    SensorData,
    build_bev_occupancy,
    get_bbox_labels,
    get_centernet_labels,
    image_augmenter,
    parse_radar_detection_labels,
    preprocess_radar_input,
    rasterize_lidar,
)
from lead.training.dataloader.py123d_data_loader import Py123DDataLoader


class TransfuserDataset(Dataset):
    """Training dataset producing the legacy TransFuser sample dict."""

    def __init__(self, lead_config: LeadConfig) -> None:
        """Construct the dataset over the 123D scenes.

        Args:
            lead_config: Root config tree.
        """
        self.lead_config = lead_config
        self.config = lead_config.agent.transfuser
        self.image_augmenter_func: Sequential = image_augmenter(
            lead_config,
            lead_config.training.data.use_color_aug_prob,
        )
        self.data_loader: Py123DDataLoader = Py123DDataLoader(lead_config)

    def __len__(self) -> int:
        return len(self.data_loader)

    def __getitem__(self, index: int) -> dict[str, typing.Any]:
        # Disable threading because the data loader will already split in threads.
        cv2.setNumThreads(0)
        start_loading_time = time.time()

        config = self.config
        lead_config = self.lead_config
        data_config = lead_config.expert.data_collection
        frame, data = self.data_loader.load(index)

        # RGB: concatenate the per-camera images side by side, left to right.
        image = np.concatenate(frame.cameras, axis=1)
        if lead_config.training.data.use_color_aug:
            image = self.image_augmenter_func(image=image)
            assert isinstance(image, np.ndarray)
        data["rgb"] = np.transpose(image, (2, 0, 1))

        # LiDAR BEV
        if frame.lidar_points is not None:
            points = frame.lidar_points[
                frame.lidar_points[:, 3]
                < lead_config.training.data.training_used_lidar_steps
            ]
            rasterized_lidar = rasterize_lidar(
                lead_config=lead_config,
                lidar=points[:, :3],
            )
            data["rasterized_lidar"] = np.array(rasterized_lidar).squeeze()[None]

        # Bounding boxes
        boxes = boxes_waypoints = boxes_num_waypoints = None
        if config.detect_boxes or config.use_bev_semantic:
            boxes, boxes_waypoints, boxes_num_waypoints = get_bbox_labels(
                data,
                lead_config,
                frame.boxes,
                frame.meta,
            )

        # Radars
        if frame.radars is not None:
            radar_list = preprocess_radar_input(
                lead_config,
                {f"radar{i + 1}": arr for i, arr in enumerate(frame.radars)},
            )
            for i, arr in enumerate(radar_list):
                data[f"radar{i + 1}"] = arr
            data["radar"] = np.concatenate(radar_list, axis=0)
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
                    radars=tuple(frame.radars),
                    radar_detections=None,
                ),
            )

        # Semantic segmentation (stored already in the reduced label space)
        if frame.semantics is not None:
            semantic = np.concatenate(frame.semantics, axis=1)
            data["semantic"] = semantic[
                :: config.perspective_downsample_factor,
                :: config.perspective_downsample_factor,
            ]

        # Depth (metric, decoded by the data loader)
        if frame.depths is not None:
            depth = np.concatenate(frame.depths, axis=1)
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
            assert frame.map_api is not None
            assert frame.view_center_se2 is not None
            loaded_hdmap = bev_raster.rasterize_bev_semantic_map(
                frame.map_api,
                frame.view_center_se2,
                stop_sign_hazard=bool(frame.meta["stop_sign_hazard"]),
                lead_config=lead_config,
            )
            data["hdmap"] = loaded_hdmap

            bev_occupancy = build_bev_occupancy(
                data,
                frame.meta,
                frame.boxes,
                lead_config,
            )
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
            loaded_bev_occupancy = bev_occupancy[
                y_cut[0] : y_cut[1],
                x_cut[0] : x_cut[1],
            ]
            mask = loaded_bev_occupancy != TransfuserBEVOccupancyClass.UNLABELED
            loaded_hdmap = loaded_hdmap.copy()
            loaded_hdmap[mask] = loaded_bev_occupancy[mask] + (
                len(TransfuserBEVSemanticClass) - len(TransfuserBEVOccupancyClass)
            )
            data["bev_semantic"] = loaded_hdmap

        # 2D bounding boxes for CenterNet
        if config.detect_boxes:
            assert boxes is not None
            data.update(get_centernet_labels(boxes, lead_config, config.num_bb_classes))

        data["loading_time"] = time.time() - start_loading_time

        return data
