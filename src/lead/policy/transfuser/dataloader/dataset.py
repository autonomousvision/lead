"""TransFuser training dataset (layer 2 of the data pipeline).

Reads the generic :class:`~lead.dataloader.frame.Frame` of each sample, adds
what only TransFuser needs (meta fields, planning labels, per-box future poses)
and runs the frame through the policy's shared featurization.
"""

import logging
import time
import typing

import cv2
import jaxtyping as jt
import numpy as np
import numpy.typing as npt
from py123d.api.scene.scene_filter import SceneFilter

if typing.TYPE_CHECKING:
    from imgaug.augmenters import Sequential

import lead.common.geometry as geometry
from lead.api.abstract_policy import SizedDataset
from lead.api.py123d_log_api import (
    BOX_ATTRIBUTES_KEY,
    LOCALIZED_EGO_STATE_KEY,
    localized_position_yaw,
)
from lead.common import constants
from lead.config import LeadConfig
from lead.dataloader import Frame, box_decoding, view_geometry
from lead.dataloader.py123d_data_loader import Py123DDataLoader
from lead.policy.transfuser.dataloader import route_smoothing
from lead.policy.transfuser.dataloader.features import build_features
from lead.policy.transfuser.dataloader.labels import build_labels

if typing.TYPE_CHECKING:
    from py123d.api.scene.scene_api import SceneAPI
    from py123d.datatypes import BoxDetectionsSE3, EgoStateSE3

LOG = logging.getLogger(__name__)


# Meta keys copied into the sample dict (non-numeric).
_STRING_META_KEYS = [
    "current_active_scenario_type",
    "previous_active_scenario_type",
    "changed_route",
    "stop_sign_hazard",
    "walker_hazard",
    "light_hazard",
    "vehicle_hazard",
    "lane_type_str",
    "does_emergency_brake_for_pedestrians",
    "construction_obstacle_two_ways_stuck",
    "accident_two_ways_stuck",
    "parked_obstacle_two_ways_stuck",
    "vehicle_opens_door_two_ways_stuck",
    "vehicle_opened_door",
    "vehicle_door_side",
    "ego_lane_id",
    "rear_danger_8",
    "rear_danger_16",
    "brake_cutin",
    "weather_setting",
    "jpeg_storage_quality",
    "emergency_brake_for_special_vehicle",
    "visual_visibility",
    "num_parking_vehicles_in_proximity",
    "slower_bad_visibility",
    "slower_clutterness",
    "over_head_traffic_light",
    "europe_traffic_light",
    "stop_sign_close",
    "num_dangerous_adversarial",
    "num_safe_adversarial",
    "num_ignored_adversarial",
    "rear_adversarial_id",
]

# Meta keys copied into the sample dict (numeric, None → inf).
_NUMERIC_META_KEYS = [
    "steer",
    "throttle",
    "brake",
    "dist_to_construction_site",
    "dist_to_accident_site",
    "dist_to_parked_obstacle",
    "dist_to_vehicle_opens_door",
    "dist_to_cutin_vehicle",
    "dist_to_pedestrian",
    "dist_to_biker",
    "distance_to_next_junction",
    "signed_dist_to_lane_change",
    "speed_limit",
    "distance_to_intersection_index_ego",
    "ego_lane_width",
    "route_left_length",
    "distance_ego_to_route",
    "target_speed_limit",
    "target_speed",
    "traffic_light_height",
]


def _localized_yaw(meta: dict) -> float:
    """The tick's localized ego yaw, read from its SE(3) pose matrix.

    Args:
        meta: Driving meta of the tick.

    Returns:
        The yaw angle in radians, wrapped to [-pi, pi].
    """
    _, yaw = localized_position_yaw(
        np.asarray(meta[LOCALIZED_EGO_STATE_KEY], dtype=np.float64),
    )
    return yaw


def _image_augmenter(lead_config: LeadConfig, prob: float = 0.2):
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


def _not_town13(scene: "SceneAPI") -> bool:
    """Whether a scene is outside the held-out Town13 routes.

    Args:
        scene: The scene to test.

    Returns:
        True when the scene's log is not a Town13 route.
    """
    return not scene.log_name.startswith("Town13")


def build_scene_filter(config: LeadConfig) -> SceneFilter:
    """The scenes TransFuser trains on.

    Args:
        config: Root config tree.

    Returns:
        The scene filter of the TransFuser data pipeline.
    """
    transfuser = config.policy.transfuser

    # Future margin: planning labels need the full waypoint horizon; the
    # skip_first/skip_last margins are the history/future minimum.
    if config.training.is_pretraining:
        future_num_iterations = config.training.data.skip_last
    else:
        future_num_iterations = (
            transfuser.num_way_points_prediction * transfuser.waypoints_spacing
        )

    return SceneFilter(
        split_names=[config.training.data.py123d_split_name],
        history_num_iterations=config.training.data.skip_first,
        future_num_iterations=future_num_iterations,
        required_scene_modalities=[
            "ego_state_se3",
            "box_detections_se3@initial",
            "custom.driving_meta@initial",
            "camera:all@initial",
        ],
        custom_filter_fns=(
            [_not_town13] if config.training.data.hold_out_town13_routes else None
        ),
    )


def build_data_loader(config: LeadConfig) -> Py123DDataLoader:
    """The generic loader configured for the TransFuser training pipeline.

    Args:
        config: Root config tree.

    Returns:
        The loader over the training scenes.
    """
    transfuser = config.policy.transfuser
    return Py123DDataLoader(
        config.training.data.py123d_data_root,
        build_scene_filter(config),
        perturbation_prob=config.training.data.use_sensor_perburtation_prob,
        tp_pop_distance=transfuser.tp_pop_distance,
        sweep_window_ticks=2 * config.expert.data_collection.lidar_stack_size,
        tick_duration_us=round(1e6 / config.expert.simulation.carla_fps),
        lidar_sweeps=not transfuser.LTF,
        radar_sweeps=config.expert.sensor_rig.use_radars,
        semantics=transfuser.use_semantic,
        depths=transfuser.use_depth,
        map=transfuser.use_bev_semantic,
    )


class TransfuserDataset(SizedDataset):
    """Training dataset producing the TransFuser model inputs and labels."""

    def __init__(self, lead_config: LeadConfig) -> None:
        """Construct the dataset over the 123D frames.

        Args:
            lead_config: Root config tree.
        """
        self.lead_config = lead_config
        self.image_augmenter_func: Sequential = _image_augmenter(
            lead_config,
            lead_config.training.data.use_color_aug_prob,
        )
        self.future_waypoint_indices: list[int] = [
            lead_config.policy.transfuser.waypoints_spacing * (k + 1)
            for k in range(lead_config.policy.transfuser.num_way_points_prediction)
        ]
        self.data_loader: Py123DDataLoader = build_data_loader(lead_config)

    def __len__(self) -> int:
        return len(self.data_loader)

    def __getitem__(self, index: int) -> dict[str, typing.Any]:
        # Disable threading because the data loader will already split in threads.
        cv2.setNumThreads(0)
        start_loading_time = time.time()

        frame, data, boxes = self._load(index)
        data.update(
            build_features(
                frame,
                self.lead_config,
                image_augmenter=(
                    self.image_augmenter_func
                    if self.lead_config.training.data.use_color_aug
                    else None
                ),
            ),
        )
        data.update(build_labels(frame, boxes, self.lead_config))

        data["loading_time"] = time.time() - start_loading_time
        return data

    def _load(
        self,
        index: int,
    ) -> tuple[Frame, dict[str, typing.Any], list[dict] | None]:
        """Read one sample's frame and its TransFuser-specific sample entries.

        Args:
            index: Sample index into the frame sequence.

        Returns:
            The frame, the dict of collate-safe sample entries (meta lifts,
            planning labels) in the view frame, and the view-frame box dicts
            carrying the grafted futures.
        """
        start_loading_time: float = time.time()
        config: LeadConfig = self.lead_config
        frame: Frame = self.data_loader[index]
        meta: dict | None = frame.meta
        assert meta is not None
        assert frame.log_metadata is not None
        assert frame.scene_metadata is not None
        perturbation = frame.perturbation

        sample: dict[str, typing.Any] = {
            "perturbate_sensor": perturbation is not None,
            "perturbation_translation": (
                perturbation.translation if perturbation is not None else 0.0
            ),
            "perturbation_rotation": (
                perturbation.rotation if perturbation is not None else 0.0
            ),
            "index": index,
            "global_index": index,
            "route_number": frame.log_metadata.log_name,
            "frame_number": frame.scene_metadata.initial_idx,
            "scenario_type_dir": str(frame.log_metadata.split).split("/")[-1],
            "town": frame.log_metadata.location,
        }

        # --- Meta lifts ---
        for attr in _STRING_META_KEYS:
            if attr == "vehicle_door_side" and meta.get(attr) is None:
                sample[attr] = "NA"
            elif attr == "vehicle_door_side":
                value = meta[attr]
                sample[attr] = value[0] if isinstance(value, list) else value
            else:
                sample[attr] = meta[attr]

        for attr in _NUMERIC_META_KEYS:
            if attr in meta and meta[attr] is None:
                sample[attr] = np.inf
            elif attr in meta:
                sample[attr] = float(meta[attr])
        assert frame.ego_state is not None
        sample["speed"] = box_decoding.carla_forward_speed(frame.ego_state)

        # Scenario type
        for attr in [
            "current_active_scenario_type",
            "previous_active_scenario_type",
        ]:
            sample[attr] = "NA" if meta.get(attr) is None else meta[attr]
        if sample["current_active_scenario_type"] != "NA":
            sample["scenario_type"] = sample["current_active_scenario_type"]
        elif sample["previous_active_scenario_type"] != "NA":
            sample["scenario_type"] = sample["previous_active_scenario_type"]
        else:
            sample["scenario_type"] = "NA"
        sample["scenario_type_id"] = constants.SCENARIO_TYPES.index(
            sample["scenario_type"],
        )

        # --- Bounding boxes: dicts derived from the native detections ---
        boxes: list[dict] | None = None
        if frame.box_detections is not None:
            boxes = box_decoding.box_detections_to_carla_frame(
                frame.box_detections,
                frame.ego_state,
                meta[BOX_ATTRIBUTES_KEY],
            )
            if perturbation is not None:
                boxes = view_geometry.to_view_frame_boxes(
                    boxes,
                    perturbation.translation,
                    perturbation.rotation,
                )

        # --- Planning labels, derived from the future iterations ---
        needs_futures: bool = (
            config.policy.transfuser.use_planning_decoder
            or config.training.experiment.visualize_dataset
        )
        if needs_futures:
            self._add_future_waypoints(
                sample,
                frame,
                meta,
                self.data_loader.future_metas(index, self.future_waypoint_indices),
                self.data_loader.future_ego_states(
                    index,
                    self.future_waypoint_indices,
                ),
            )
            self._add_route(sample, frame, meta)

        if boxes is not None and (
            config.policy.transfuser.detect_boxes
            or config.policy.transfuser.use_bev_semantic
        ):
            self._graft_box_futures(
                frame,
                meta,
                boxes,
                self.data_loader.future_box_detections(
                    index,
                    self.future_waypoint_indices,
                ),
            )

        sample["loading_meta_time"] = time.time() - start_loading_time
        return frame, sample, boxes

    def _add_future_waypoints(
        self,
        sample: dict[str, typing.Any],
        frame: Frame,
        meta: dict,
        future_metas: dict[int, dict | None],
        future_ego_states: dict[int, "EgoStateSE3 | None"],
    ) -> None:
        """Add the ego's future poses as the planning labels of the sample.

        Args:
            sample: Sample dict to add the labels to, modified in place.
            frame: The frame of the sample.
            meta: Driving meta of the anchor tick.
            future_metas: The metas of the future waypoint iterations.
            future_ego_states: The ego states of the same iterations.
        """
        assert frame.ego_state is not None
        ego_position, _ = box_decoding.carla_ego_pose(frame.ego_state)
        ego_yaw = _localized_yaw(meta)

        future_waypoints: list[jt.Float[npt.NDArray, " 2"]] = []
        future_yaws: list[float] = []
        for k in self.future_waypoint_indices:
            future_meta: dict | None = future_metas.get(k)
            future_state = future_ego_states.get(k)
            if future_meta is None or future_state is None:
                continue
            future_position, _ = box_decoding.carla_ego_pose(future_state)
            future_waypoints.append(
                geometry.inverse_conversion_2d(future_position, ego_position, ego_yaw),
            )
            future_yaws.append(
                geometry.normalize_angle(_localized_yaw(future_meta) - ego_yaw),
            )
        if not future_waypoints:
            return

        waypoints: jt.Float[npt.NDArray, "n 2"] = np.array(future_waypoints).reshape(
            -1,
            2,
        )
        yaws: jt.Float[npt.NDArray, " n"] = np.array(future_yaws).reshape(-1)
        if frame.perturbation is not None:
            waypoints = view_geometry.to_view_frame_points(
                waypoints,
                frame.perturbation.translation,
                frame.perturbation.rotation,
            )
            yaws = view_geometry.to_view_frame_yaws(
                yaws,
                frame.perturbation.rotation,
            )
        sample["future_waypoints"] = waypoints
        sample["future_yaws"] = yaws

    def _add_route(
        self,
        sample: dict[str, typing.Any],
        frame: Frame,
        meta: dict,
    ) -> None:
        """Add the route ahead of the ego, smoothed and in the view frame.

        Args:
            sample: Sample dict to add the route to, modified in place.
            frame: The frame of the sample.
            meta: Driving meta of the anchor tick.
        """
        config: LeadConfig = self.lead_config
        transfuser = config.policy.transfuser

        # The route is stored in the global frame; convert to the ego frame.
        assert frame.ego_state is not None
        ego_position, _ = box_decoding.carla_ego_pose(frame.ego_state)
        ego_yaw = _localized_yaw(meta)
        route: jt.Float[npt.NDArray, "n 2"] = np.array(
            [
                geometry.inverse_conversion_2d(
                    np.array(point),
                    ego_position,
                    ego_yaw,
                )
                for point in meta["route"][: transfuser.num_route_points_smoothing]
            ],
        )
        if transfuser.smooth_route:
            route = route_smoothing.smooth_path(
                config,
                route,
                target_first_distance=2.5,
            )
        route = route[: transfuser.num_route_points_prediction]
        if frame.perturbation is not None:
            route = view_geometry.to_view_frame_points(
                route,
                frame.perturbation.translation,
                frame.perturbation.rotation,
            )
        sample["brake"] = meta["brake"]
        sample["throttle"] = meta["throttle"]
        sample["route"] = route

    def _graft_box_futures(
        self,
        frame: Frame,
        meta: dict,
        json_boxes: list[dict],
        future_box_detections: dict[int, "BoxDetectionsSE3 | None"],
    ) -> None:
        """Attach per-box future positions/yaws from the future iterations.

        Future world poses of each box (matched by id) are transformed into the
        view frame the frame's boxes already live in.

        Args:
            frame: The frame whose boxes are grafted onto.
            meta: Current driving meta with the ego heading.
            json_boxes: Current tick's box dicts (modified in place).
            future_box_detections: The native box detections of the future
                waypoint iterations.
        """
        assert frame.ego_state is not None
        ego_position, _ = box_decoding.carla_ego_pose(frame.ego_state)
        ego_yaw = _localized_yaw(meta)

        def to_view_frame(
            position: jt.Float[npt.NDArray, " 2"],
            yaw: float,
        ) -> tuple[jt.Float[npt.NDArray, " 2"], float]:
            if frame.perturbation is None:
                return position, yaw
            view_position = view_geometry.to_view_frame_points(
                position[None],
                frame.perturbation.translation,
                frame.perturbation.rotation,
            )[0]
            view_yaw = float(
                view_geometry.to_view_frame_yaws(
                    np.array([yaw]),
                    frame.perturbation.rotation,
                )[0],
            )
            return view_position, view_yaw

        future_boxes_by_id: dict[int, dict[str, list]] = {}
        for k in self.future_waypoint_indices:
            detections = future_box_detections.get(k)
            if detections is None:
                continue
            global_poses = box_decoding.box_detections_to_carla_global(detections)
            for box_id, (global_position, global_yaw) in global_poses.items():
                position: jt.Float[npt.NDArray, " 2"] = geometry.inverse_conversion_2d(
                    global_position,
                    ego_position,
                    ego_yaw,
                )
                yaw: float = geometry.normalize_angle(global_yaw - ego_yaw)
                position, yaw = to_view_frame(position, yaw)
                entry: dict[str, list] = future_boxes_by_id.setdefault(
                    box_id,
                    {"future_positions": [], "future_yaws": []},
                )
                entry["future_positions"].append(position.tolist())
                entry["future_yaws"].append(float(yaw))

        # Lay the sampled values out so that index spacing*k holds the k-th
        # future sample; the filler entries in between are never read.
        spacing: int = self.lead_config.policy.transfuser.waypoints_spacing
        for bb in json_boxes:
            box_entry: dict[str, list] | None = future_boxes_by_id.get(bb["id"])
            if box_entry is None or not box_entry["future_positions"]:
                continue
            expanded_positions: list[list[float]] = [box_entry["future_positions"][0]]
            expanded_yaws: list[float] = [box_entry["future_yaws"][0]]
            for future_position, future_yaw in zip(
                box_entry["future_positions"],
                box_entry["future_yaws"],
                strict=True,
            ):
                expanded_positions.extend([future_position] * spacing)
                expanded_yaws.extend([future_yaw] * spacing)
            bb["future_positions"] = expanded_positions
            bb["future_yaws"] = expanded_yaws
