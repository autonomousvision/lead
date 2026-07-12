"""Model-agnostic sample loading over the 123D logs (layer 1 of the data pipeline).

Resolves scene enumeration, the normal-vs-perturbated view choice,
frame-of-reference transforms, temporal alignment, and the native modality
reads into a :class:`Frame` plus collate-safe batch entries; model-specific
datasets turn the frame into input tensors.
"""

import logging
import time
import typing
from dataclasses import dataclass

import jaxtyping as jt
import numpy as np
import numpy.typing as npt
from py123d.api.map.map_api import MapAPI
from py123d.api.scene.scene_api import SceneAPI
from py123d.datatypes import DepthCameraMetadata, Lidar, ModalityType
from py123d.datatypes.sensors.base_camera import CameraID
from py123d.datatypes.sensors.radar import Radar, RadarID
from py123d.geometry import PoseSE2

import lead.common.common_utils as common_utils
from lead.common import constants
from lead.common.constants import SourceDataset
from lead.config import LeadConfig
from lead.lead.recorders.driving_meta_recorder import DRIVING_META_MODALITY_ID
from lead.lead.utils.carla_to_123d import CAMERA_ID_MAPPING
from lead.training.dataloader import (
    feature_assembly,
    route_smoothing,
    scene_index,
    view_geometry,
)

LOG = logging.getLogger(__name__)

T = typing.TypeVar("T")


def require_not_none(value: T | None) -> T:
    """Unwrap an Optional py123d read that the data contract guarantees.

    The py123d getters return None for missing data; the LEAD logs are
    complete for every modality the loader reads, so a None here is a data
    bug worth failing on immediately. Prefer a plain ``assert`` where a local
    variable exists; this is for comprehensions.

    Args:
        value: The Optional value to unwrap.

    Returns:
        The value, guaranteed not None.
    """
    assert value is not None
    return value


@dataclass
class Frame:
    """One sample's raw 123D data, expressed in the chosen view's frame.

    The data loader resolves the normal-vs-perturbated view choice and
    re-expresses every ego-frame quantity in the view frame (identity for the
    normal view), so consumers assemble model inputs without any view logic.
    Not collatable — model-specific datasets turn this into batch tensors.
    """

    # Whether the perturbated sensor views were chosen for this sample.
    perturbated: bool
    # Rig perturbation of the chosen view; zero for the normal view.
    perturbation_translation: float
    perturbation_rotation: float
    # Raw driving-meta dict of the anchor tick (ego-frame contents untouched).
    meta: dict
    # Driving-meta dicts of the future waypoint iterations (None when missing).
    future_metas: dict[int, dict | None]
    # Legacy bounding-box dicts in the view frame, future poses grafted.
    boxes: list[dict]
    # Per-camera images in LEAD camera order (1..num_cameras).
    cameras: list[jt.UInt8[npt.NDArray, "h w 3"]]
    # Per-camera semantic segmentation, same order; None when not loaded.
    semantics: list[jt.UInt8[npt.NDArray, "h w"]] | None
    # Per-camera metric depth, same order; None when not loaded.
    depths: list[jt.Float32[npt.NDArray, "h w"]] | None
    # Point cloud [x, y, z, frame_age] in the CARLA view frame; None when not loaded.
    lidar_points: jt.Float[npt.NDArray, "n 4"] | None
    # Per-radar point clouds [x, y, z, radial_velocity] in the CARLA view
    # frame, in LEAD radar order (1..num_radar_sensors); None when not loaded.
    radars: list[jt.Float16[npt.NDArray, "n 4"]] | None
    # Map of the town plus the view frame's global pose, for BEV rasterization;
    # None when the BEV semantic target is not loaded.
    map_api: MapAPI | None
    view_center_se2: PoseSE2 | None


# Legacy meta keys lifted verbatim into the sample dict (non-numeric).
_STRING_META_KEYS = [
    "town",
    "current_active_scenario_type",
    "previous_active_scenario_type",
    "changed_route",
    "stop_sign_hazard",
    "walker_hazard",
    "light_hazard",
    "vehicle_hazard",
    "lane_type_str",
    "is_intersection",
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

# Legacy meta keys lifted verbatim into the sample dict (numeric, None → inf).
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
    "speed",
    "accel_x",
    "accel_y",
    "accel_z",
    "angular_velocity_x",
    "angular_velocity_y",
    "angular_velocity_z",
    "speed_limit",
    "privileged_acceleration",
    "route_curvature",
    "distance_to_intersection_index_ego",
    "ego_lane_width",
    "route_left_length",
    "distance_ego_to_route",
    "target_speed_limit",
    "target_speed",
    "theta",
    "privileged_yaw",
    "traffic_light_height",
    "second_highest_speed",
    "second_highest_speed_limit",
]

# LEAD radar sensor index → 123D radar ID name (see RadarRecorder).
_RADAR_ID_BY_INDEX = {
    1: RadarID.RADAR_FRONT_LEFT,
    2: RadarID.RADAR_FRONT_RIGHT,
    3: RadarID.RADAR_BACK_RIGHT,
    4: RadarID.RADAR_BACK_LEFT,
}


class Py123DDataLoader:
    """Loads one 123D scene into a :class:`Frame` plus collatable sample entries."""

    def __init__(self, config: LeadConfig) -> None:
        """Construct the loader by enumerating scenes of both views.

        Args:
            config: Training configuration.
        """
        self.config: LeadConfig = config

        # Future margin: planning labels need the full waypoint horizon; the
        # legacy skip_first/skip_last margins are the history/future minimum.
        future_num_iterations: int
        if config.training.is_pretraining:
            future_num_iterations = config.training.data.skip_last
        else:
            future_num_iterations = (
                config.agent.transfuser.num_way_points_prediction
                * config.agent.transfuser.waypoints_spacing
            )
        self.future_waypoint_indices: list[int] = [
            config.agent.transfuser.waypoints_spacing * (k + 1)
            for k in range(
                config.agent.transfuser.num_way_points_prediction,
            )
        ]
        self.scenes: list[SceneAPI] = scene_index.get_scenes(
            config,
            history_num_iterations=config.training.data.skip_first,
            future_num_iterations=future_num_iterations,
        )
        self.future_num_iterations: int = future_num_iterations

        self.perturbated_scenes: dict[tuple[str, int], SceneAPI] = {}
        if config.training.data.use_sensor_perburtation_prob > 0.0:
            self.perturbated_scenes = scene_index.get_perturbated_scene_lookup(config)
        # Rig perturbation per log, derived lazily from the camera extrinsics.
        self._rig_perturbations: dict[str, tuple[float, float]] = {}

    def __len__(self) -> int:
        return len(self.scenes)

    def _driving_meta(self, scene: SceneAPI, iteration: int) -> dict | None:
        """Read the driving-meta dict at a scene iteration.

        Args:
            scene: The scene to read from.
            iteration: Scene iteration (0 = anchor frame).

        Returns:
            The meta dict, or None when unavailable.
        """
        modality = scene.get_custom_modality_at_iteration(
            iteration,
            DRIVING_META_MODALITY_ID,
        )
        return None if modality is None else modality.data

    def _used_camera_ids(self) -> list[CameraID]:
        """LEAD camera IDs in stored left-to-right order.

        Returns:
            Camera IDs for indices 1..num_cameras.
        """
        return [
            CAMERA_ID_MAPPING[cam_key]
            for cam_key in range(
                1,
                self.config.expert.sensor_rig.num_cameras + 1,
            )
        ]

    def _rig_perturbation(
        self,
        scene: SceneAPI,
        perturbated_scene: SceneAPI,
    ) -> tuple[float, float]:
        """Rig perturbation of a log, derived from the camera extrinsics.

        Args:
            scene: Normal-view scene of the log.
            perturbated_scene: Paired perturbated-view scene.

        Returns:
            The perturbation translation in meters and rotation in degrees.
        """
        log_name: str = scene.log_name
        if log_name not in self._rig_perturbations:
            camera_id: CameraID = self._used_camera_ids()[0]
            ego_metadata = scene.get_ego_state_se3_metadata()
            assert ego_metadata is not None
            self._rig_perturbations[log_name] = view_geometry.derive_rig_perturbation(
                scene.get_camera_metadatas()[camera_id].camera_to_imu_se3,
                perturbated_scene.get_camera_metadatas()[camera_id].camera_to_imu_se3,
                ego_metadata.rear_axle_to_center_longitudinal,
            )
        return self._rig_perturbations[log_name]

    def _read_radar(self, scene: SceneAPI, radar_id: RadarID) -> Radar:
        """Read one radar stream at the anchor iteration.

        The LEAD logs store one stream per radar without the per-point
        ``RadarFeature.IDS`` column, which py123d's per-sensor read path
        requires for masking merged clouds; request the merged read instead,
        which returns the stream's points as-is.

        Args:
            scene: The scene of the chosen view.
            radar_id: The radar stream to read.

        Returns:
            The radar modality at iteration 0.
        """
        radar = scene.get_modality_at_iteration(
            iteration=0,
            modality_type=ModalityType.RADAR,
            modality_id=radar_id,
            radar_id=RadarID.RADAR_MERGED,
        )
        assert isinstance(radar, Radar)
        return radar

    def load(self, index: int) -> tuple[Frame, dict[str, typing.Any]]:
        """Load one sample: choose the view, read and re-express its data.

        Args:
            index: Sample index into the scene list.

        Returns:
            The frame of raw view-frame data, and the dict of finished
            collate-safe sample entries (legacy meta lifts, planning labels,
            target points — all in the view frame).
        """
        start_loading_time: float = time.time()
        config: LeadConfig = self.config
        scene: SceneAPI = self.scenes[index]
        meta: dict | None = self._driving_meta(scene, 0)
        assert meta is not None
        log_name: str = scene.log_name

        # --- View choice ---
        perturbated_scene: SceneAPI | None = self.perturbated_scenes.get(
            (log_name, scene.get_scene_metadata().initial_idx),
        )
        perturbated: bool = False
        translation: float = 0.0
        rotation: float = 0.0
        sensor_scene: SceneAPI = scene
        if (
            perturbated_scene is not None
            and np.random.rand() < config.training.data.use_sensor_perburtation_prob
        ):
            perturbated = True
            translation, rotation = self._rig_perturbation(scene, perturbated_scene)
            sensor_scene = perturbated_scene

        sample: dict[str, typing.Any] = {
            "source_dataset": SourceDataset.CARLA,
            "perturbate_sensor": perturbated,
            "perturbation_translation": translation,
            "perturbation_rotation": rotation,
            "index": index,
            "global_index": index,
            "bucket_identity": log_name,
            "route_number": log_name,
            "frame_number": str(scene.scene_uuid),
        }

        # --- Meta lifts (verbatim from the legacy loader) ---
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

        # --- Temporal fields derived from future iterations ---
        future_metas: dict[int, dict | None] | None = None
        if (
            config.agent.transfuser.use_planning_decoder
            or config.training.experiment.visualize_dataset
        ):
            future_metas = {
                k: self._driving_meta(scene, k)
                for k in self.future_waypoint_indices
                if k <= self.future_num_iterations
            }
            ego_position: jt.Float[npt.NDArray, " 2"] = np.array(
                meta["pos_global"][:2],
            )
            ego_yaw: float = meta["theta"]
            future_waypoints: list[jt.Float[npt.NDArray, " 2"]] = []
            future_yaws: list[float] = []
            for k in self.future_waypoint_indices:
                future_meta: dict | None = future_metas.get(k)
                if future_meta is None:
                    continue
                future_waypoints.append(
                    common_utils.inverse_conversion_2d(
                        np.array(future_meta["pos_global"][:2]),
                        ego_position,
                        ego_yaw,
                    ),
                )
                future_yaws.append(
                    common_utils.normalize_angle(future_meta["theta"] - ego_yaw),
                )
            if future_waypoints:
                waypoints: jt.Float[npt.NDArray, "n 2"] = np.array(
                    future_waypoints,
                ).reshape(-1, 2)
                yaws: jt.Float[npt.NDArray, " n"] = np.array(future_yaws).reshape(-1)
                if perturbated:
                    waypoints = view_geometry.to_view_frame_points(
                        waypoints,
                        translation,
                        rotation,
                    )
                    yaws = view_geometry.to_view_frame_yaws(yaws, rotation)
                sample["future_waypoints"] = waypoints
                sample["future_yaws"] = yaws

        # --- Route and target speed features ---
        if (
            config.agent.transfuser.use_planning_decoder
            or config.training.experiment.visualize_dataset
        ):
            # The route is stored in the global frame; convert to the ego frame.
            route: jt.Float[npt.NDArray, "n 2"] = np.array(
                [
                    common_utils.inverse_conversion_2d(
                        np.array(point),
                        np.array(meta["pos_global"][:2]),
                        meta["theta"],
                    )
                    for point in meta["route"][
                        : config.agent.transfuser.num_route_points_smoothing
                    ]
                ],
            )
            if config.agent.transfuser.smooth_route:
                route = route_smoothing.smooth_path(
                    config,
                    route,
                    target_first_distance=2.5,
                )
            route = route[: config.agent.transfuser.num_route_points_prediction]
            if perturbated:
                route = view_geometry.to_view_frame_points(
                    route,
                    translation,
                    rotation,
                )
            sample["brake"] = meta["brake"]
            sample["throttle"] = meta["throttle"]
            sample["route"] = route

        # Velocity
        if config.agent.transfuser.use_velocity:
            sample["speed"] = meta["speed"]

        # --- Target point and navigation command ---
        ego_yaw = meta["theta"]
        ego_position = np.array(meta["pos_global"][:2])
        if config.agent.transfuser.use_noisy_tp:
            if config.agent.transfuser.use_kalman_filter_for_gps:
                ego_position = np.array(meta["filtered_pos_global"][:2])
            else:
                ego_position = np.array(meta["noisy_pos_global"][:2])

        def transform_to_view(point: list[float]) -> jt.Float[npt.NDArray, " 2"]:
            ego_point: jt.Float[npt.NDArray, " 2"] = common_utils.inverse_conversion_2d(
                np.array(point),
                ego_position,
                ego_yaw,
            )
            if not perturbated:
                return ego_point
            return view_geometry.to_view_frame_points(
                ego_point[None],
                translation,
                rotation,
            )[0]

        noisy_version: str = "_gps" if config.agent.transfuser.use_noisy_tp else ""
        next_tp_list: list[list[float]] = meta[
            f"next{noisy_version}_target_points_{config.agent.transfuser.tp_pop_distance}"
        ]

        # Merge duplicate target points
        filtered_tp_list: list[list[float]] = []
        for pt in next_tp_list:
            if (
                len(next_tp_list) == 2
                or not filtered_tp_list
                or not np.allclose(pt[:2], filtered_tp_list[-1][:2])
            ):
                filtered_tp_list.append(pt)
        next_tp_list = filtered_tp_list

        if len(next_tp_list) > 2:
            sample["target_point_next"] = transform_to_view(next_tp_list[2][:2])
            sample["target_point"] = transform_to_view(next_tp_list[1][:2])
            sample["target_point_previous"] = transform_to_view(next_tp_list[0][:2])
        else:
            assert len(next_tp_list) == 2
            sample["target_point"] = transform_to_view(next_tp_list[1][:2])
            sample["target_point_next"] = transform_to_view(next_tp_list[1][:2])
            sample["target_point_previous"] = transform_to_view(next_tp_list[0][:2])

        # --- Bounding boxes: legacy dicts with futures grafted, in view frame ---
        boxes: list[dict] = []
        if (
            config.agent.transfuser.detect_boxes
            or config.agent.transfuser.use_bev_semantic
        ):
            boxes = [dict(box) for box in meta["bounding_boxes"]]
            self._graft_box_futures(scene, meta, boxes, future_metas)
            if perturbated:
                boxes = view_geometry.to_view_frame_boxes(
                    boxes,
                    translation,
                    rotation,
                )

        sample["loading_meta_time"] = time.time() - start_loading_time

        # ----------------------------------------------------------------------------------------
        # Heavy sensor data
        # ----------------------------------------------------------------------------------------
        start_loading_sensor_time: float = time.time()
        camera_ids: list[CameraID] = self._used_camera_ids()

        cameras: list[jt.UInt8[npt.NDArray, "h w 3"]] = [
            require_not_none(sensor_scene.get_camera_at_iteration(0, camera_id)).image
            for camera_id in camera_ids
        ]

        # LiDAR is only recorded in the normal view; a rigid transform moves
        # the point cloud into the perturbated rig's frame exactly.
        lidar_points: jt.Float[npt.NDArray, "n 4"] | None = None
        if not config.agent.transfuser.LTF:
            lidar: Lidar | None = scene.get_lidar_at_iteration(0, "lidar_top")
            assert lidar is not None
            lidar_points = feature_assembly.lidar_to_carla_ego_frame(lidar, config)
            if perturbated:
                lidar_points = view_geometry.to_view_frame_lidar(
                    lidar_points,
                    translation,
                    rotation,
                )

        # The perturbated radar returns are stored in the view frame already
        # (converted with the nominal calibration at collection time).
        radars: list[jt.Float16[npt.NDArray, "n 4"]] | None = None
        if config.expert.sensor_rig.use_radars:
            radars = [
                feature_assembly.radar_to_carla_ego_frame(
                    self._read_radar(
                        sensor_scene,
                        _RADAR_ID_BY_INDEX[sensor_index],
                    ),
                )
                for sensor_index in range(
                    1,
                    config.expert.sensor_rig.num_radar_sensors + 1,
                )
            ]

        # Semantic segmentation (stored already in the reduced label space)
        semantics: list[jt.UInt8[npt.NDArray, "h w"]] | None = None
        if config.agent.transfuser.use_semantic:
            semantics = [
                require_not_none(
                    sensor_scene.get_camera_semantic_at_iteration(0, camera_id),
                ).image
                for camera_id in camera_ids
            ]

        # Depth (stored full resolution, metric after decoding)
        depths: list[jt.Float32[npt.NDArray, "h w"]] | None = None
        if config.agent.transfuser.use_depth:
            depth_metadatas: dict[CameraID, DepthCameraMetadata] = (
                sensor_scene.get_camera_depth_metadatas()
            )
            depths = [
                depth_metadatas[camera_id]
                .decode_depth(
                    require_not_none(
                        sensor_scene.get_camera_depth_at_iteration(0, camera_id),
                    ).image,
                )
                .astype(np.float32)
                for camera_id in camera_ids
            ]

        # Map plus the view frame's pose, for BEV rasterization downstream.
        map_api: MapAPI | None = None
        view_center_se2: PoseSE2 | None = None
        if config.agent.transfuser.use_bev_semantic:
            map_api = scene.get_map_api()
            ego_state = scene.get_ego_state_se3_at_iteration(0)
            assert ego_state is not None
            view_center_se2 = view_geometry.view_center_se2(
                ego_state.center_se2,
                translation,
                rotation,
            )

        sample["loading_sensor_time"] = time.time() - start_loading_sensor_time

        frame: Frame = Frame(
            perturbated=perturbated,
            perturbation_translation=translation,
            perturbation_rotation=rotation,
            meta=meta,
            future_metas=future_metas if future_metas is not None else {},
            boxes=boxes,
            cameras=cameras,
            semantics=semantics,
            depths=depths,
            lidar_points=lidar_points,
            radars=radars,
            map_api=map_api,
            view_center_se2=view_center_se2,
        )
        return frame, sample

    def _graft_box_futures(
        self,
        scene: SceneAPI,
        meta: dict,
        json_boxes: list,
        future_metas: dict | None,
    ) -> None:
        """Attach per-box future positions/yaws from future iterations.

        Replaces the deleted offline enhancement pass: future world poses of
        each box (matched by id) are transformed into the current ego frame.

        Args:
            scene: The scene to read future ticks from.
            meta: Current driving meta with the ego pose.
            json_boxes: Current tick's box dicts (modified in place).
            future_metas: Pre-read future metas keyed by iteration, or None.
        """
        if future_metas is None:
            future_metas = {
                k: self._driving_meta(scene, k)
                for k in self.future_waypoint_indices
                if k <= self.future_num_iterations
            }
        ego_position: jt.Float[npt.NDArray, " 2"] = np.array(meta["pos_global"][:2])
        ego_yaw: float = meta["theta"]

        future_boxes_by_id: dict[int, dict[str, list]] = {}
        for k in self.future_waypoint_indices:
            future_meta: dict | None = future_metas.get(k)
            if future_meta is None:
                continue
            for bb in future_meta["bounding_boxes"]:
                matrix: jt.Float[npt.NDArray, "4 4"] = np.array(bb["matrix"])
                position: jt.Float[npt.NDArray, " 2"] = (
                    common_utils.inverse_conversion_2d(
                        matrix[:2, 3],
                        ego_position,
                        ego_yaw,
                    )
                )
                yaw: float = common_utils.normalize_angle(
                    np.arctan2(matrix[1, 0], matrix[0, 0]) - ego_yaw,
                )
                entry: dict[str, list] = future_boxes_by_id.setdefault(
                    bb["id"],
                    {"future_positions": [], "future_yaws": []},
                )
                entry["future_positions"].append(position.tolist())
                entry["future_yaws"].append(float(yaw))

        # The legacy future arrays were indexed per save tick (index i = i
        # ticks ahead) and are only ever read at the waypoint indices
        # (spacing, 2*spacing, ...). Lay our sampled values out so that
        # index spacing*k hits the k-th future sample; the filler entries in
        # between are never read.
        spacing: int = self.config.agent.transfuser.waypoints_spacing
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
