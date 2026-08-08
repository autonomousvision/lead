"""Model-agnostic scene loading over the 123D logs: enumerates the scenes of a
log collection and reads each into a :class:`~lead.api.scene_data.SceneData`."""

from __future__ import annotations

import typing
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import jaxtyping as jt
import numpy as np
import numpy.typing as npt
from py123d.api.scene.scene_api import SceneAPI
from py123d.api.scene.scene_filter import SceneFilter
from py123d.datatypes import CustomModality, Lidar, ModalityType
from py123d.datatypes.sensors.base_camera import Camera, CameraID
from py123d.datatypes.sensors.lidar import LidarID
from py123d.datatypes.sensors.radar import Radar, RadarID

from lead.api.driving_meta import DrivingMeta
from lead.api.py123d_log_api import (
    CAMERA_ID_BY_LEAD_INDEX,
    DRIVING_META_MODALITY_ID,
    LOCALIZED_EGO_STATE_KEY,
    NORMAL_SENSOR_VIEW,
    PERTURBATED_SENSOR_VIEW,
    TARGET_POINTS_METADATA_KEY,
    ordered_target_points,
    se3_matrix_to_localized_pose,
)
from lead.api.scene_data import RigPerturbation, SceneData
from lead.api.scene_loading_spec import SceneLoadingSpec
from lead.common import geometry
from lead.log_reader import view_geometry
from lead.log_reader.scene_index import (
    PerturbatedViewPairing,
    get_perturbated_view_pairing,
    get_scenes,
    scene_log_is_covered,
)


class SceneLoader:
    """Reads the 123D scenes of a log collection into scene data.

    An indexable sequence: ``len(loader)`` is the number of scenes the logs
    yield and ``loader[i]`` is the i-th
    :class:`~lead.api.scene_data.SceneData`, with the sensor-view choice
    resolved and only the requested modalities read; the py123d scene handles
    never leave the class.
    """

    def __init__(
        self,
        data_root: str | Path | None = None,
        scene_filter: SceneFilter | None = None,
        *,
        perturbation_probability: float = 0.0,
        target_point_pop_distance: float = 3.25,
        past_tick_ages: tuple[int, ...] = tuple(range(10)),
        tick_duration_us: int = 50_000,
        read_lidar_sweeps: bool = True,
        read_radar_sweeps: bool = True,
        read_semantic_cameras: bool = True,
        read_depth_cameras: bool = True,
        read_map_api: bool = True,
        camera_ids: list[CameraID] | None = None,
    ) -> None:
        """Enumerate the scenes both views provide.

        Args:
            data_root: Dataset root holding ``logs/`` and ``maps/``; None uses
                py123d's ``PY123D_DATA_ROOT``.
            scene_filter: py123d's own scene selection (splits, logs, the
                history/future window); None selects every LEAD scene.
            perturbation_probability: Probability of reading the perturbated
                sensor views, where recorded.
            target_point_pop_distance: Pop distance of the route planner whose
                target points the scene data carries.
            past_tick_ages: Tick ages the loader's own default spec reads the
                sweeps and the ego-pose history at, 0 being the anchor; a
                caller-supplied spec names its own.
            tick_duration_us: Duration of one simulator tick in microseconds.
            read_lidar_sweeps: Whether to read the lidar sweep window.
            read_radar_sweeps: Whether to read the radar sweep window.
            read_semantic_cameras: Whether to read the semantic and instance cameras.
            read_depth_cameras: Whether to read the depth cameras.
            read_map_api: Whether to read the map API.
            camera_ids: Cameras to read, in stitch order; None reads every
                camera the log provides, in LEAD index order.
        """
        # One scene per save tick and no margins, unless the caller asks for
        # more; the modalities every read touches are required either way.
        scene_filter = scene_filter or SceneFilter(
            history_num_iterations=0,
            future_num_iterations=0,
        )
        self._scene_filter: SceneFilter = replace(
            scene_filter,
            required_scene_modalities=list(
                dict.fromkeys(
                    (scene_filter.required_scene_modalities or [])
                    + _REQUIRED_SCENE_MODALITIES,
                ),
            ),
        )
        self._perturbation_probability = perturbation_probability
        self.target_point_pop_distance = target_point_pop_distance
        self._tick_duration_us = tick_duration_us
        self._default_loading_spec = SceneLoadingSpec(
            rgb_tick_ages=(0,),
            lidar_tick_ages=past_tick_ages if read_lidar_sweeps else (),
            radar_tick_ages=past_tick_ages if read_radar_sweeps else (),
            ego_pose_tick_ages=past_tick_ages,
            read_depth_cameras=read_depth_cameras,
            read_semantic_cameras=read_semantic_cameras,
            read_map_api=read_map_api,
        )
        self._requested_camera_ids = camera_ids

        self.scenes: Sequence[SceneAPI] = get_scenes(
            data_root,
            self._scene_filter,
        )
        self._perturbated_view_pairing: PerturbatedViewPairing | None = None
        if perturbation_probability > 0.0:
            self._perturbated_view_pairing = get_perturbated_view_pairing(
                data_root,
                self._scene_filter.log_names,
            )
        # Resolved once per sample: the lookup runs on every view draw.
        self._has_perturbated_view: npt.NDArray[np.bool_] = np.zeros(
            len(self.scenes),
            dtype=bool,
        )
        if self._perturbated_view_pairing is not None:
            self._has_perturbated_view = scene_log_is_covered(
                self.scenes,
                self._perturbated_view_pairing.covered_logs(),
            )

        # The rig is constant across a log collection; read it lazily from the
        # logs themselves.
        self._camera_ids: list[CameraID] | None = None
        # Rig perturbation per log, derived lazily from the camera extrinsics.
        self._rig_perturbation_by_log: dict[str, RigPerturbation] = {}
        # Target points per log, read lazily from the driving-meta metadata.
        self._target_points_by_log: dict[str, jt.Float[npt.NDArray, "_ 3"]] = {}

    def __len__(self) -> int:
        return len(self.scenes)

    def __getitem__(self, scene_index: int) -> SceneData:
        """Read the scene data of one sample, its sensor view chosen at random.

        Args:
            scene_index: Sample index into the scene list.

        Returns:
            The scene data, with every ego-frame quantity in the chosen view's
            frame and only the requested modalities populated.
        """
        sensor_view = self.draw_sensor_view(scene_index)
        return self.read(scene_index, sensor_view == PERTURBATED_SENSOR_VIEW)

    def read(
        self,
        scene_index: int,
        use_perturbated_view: bool,
        loading_spec: SceneLoadingSpec | None = None,
    ) -> SceneData:
        """Read the scene data of one sample in an explicit sensor view.

        Args:
            scene_index: Sample index into the scene list.
            use_perturbated_view: Whether to read the perturbated rig view; the
                sample must provide it (see :meth:`has_perturbated_sensor_view`).
            loading_spec: The modalities to populate; None reads the loader's
                default spec (the constructor's modality switches).

        Returns:
            The scene data, with every ego-frame quantity in the chosen view's
            frame and only the requested modalities populated.
        """
        if loading_spec is None:
            loading_spec = self._default_loading_spec
        resolved_view: _ResolvedSensorView = self._resolve_sensor_view(
            scene_index,
            use_perturbated_view,
        )
        scene: SceneAPI = resolved_view.normal_scene
        driving_meta: DrivingMeta | None = self._read_driving_meta(scene, 0)
        assert driving_meta is not None

        # LiDAR is only recorded in the normal view; consumers accumulate the
        # sweeps and move them into the view frame themselves.
        lidar_sweeps: dict[int, Lidar] | None = None
        if loading_spec.lidar_tick_ages:
            lidar_sweeps = self._read_lidar_sweeps(scene, loading_spec.lidar_tick_ages)

        radar_sweeps: dict[int, Radar] | None = None
        if loading_spec.radar_tick_ages:
            radar_sweeps = self._read_radar_sweeps(
                resolved_view.sensor_scene,
                loading_spec.radar_tick_ages,
            )

        # Age 0 is the anchor rig the scene API serves directly; only the older
        # ages cost a timestamp lookup, so the anchor is never decoded twice.
        cameras: list[Camera] = []
        if 0 in loading_spec.rgb_tick_ages:
            cameras = self._read_camera_stream(
                resolved_view.sensor_scene,
                resolved_view.sensor_scene.get_camera_at_iteration,
            )
        past_rgb_tick_ages = tuple(age for age in loading_spec.rgb_tick_ages if age)
        past_cameras: dict[int, list[Camera]] | None = None
        if past_rgb_tick_ages:
            past_cameras = self._read_past_cameras(
                resolved_view.sensor_scene,
                past_rgb_tick_ages,
            )

        semantic_cameras: list[Camera] | None = None
        instance_cameras: list[Camera] | None = None
        if loading_spec.read_semantic_cameras:
            semantic_cameras = self._read_camera_stream(
                resolved_view.sensor_scene,
                resolved_view.sensor_scene.get_camera_semantic_at_iteration,
            )
            instance_cameras = self._read_camera_stream(
                resolved_view.sensor_scene,
                resolved_view.sensor_scene.get_camera_instance_at_iteration,
            )

        depth_cameras: list[Camera] | None = None
        if loading_spec.read_depth_cameras:
            depth_cameras = self._read_camera_stream(
                resolved_view.sensor_scene,
                resolved_view.sensor_scene.get_camera_depth_at_iteration,
            )

        map_api = scene.get_map_api() if loading_spec.read_map_api else None

        target_points = self._target_points_in_view(scene, driving_meta, resolved_view)
        # Dense over every age any modality of this read touches: the sweep
        # alignment indexes it by the sweeps' own tick ages, so a gap there
        # silently drops a sweep.
        past_ego_positions, past_ego_yaws = self._read_past_ego_motion(
            scene,
            driving_meta,
            num_ticks=1
            + max(
                (
                    *loading_spec.ego_pose_tick_ages,
                    *loading_spec.lidar_tick_ages,
                    *loading_spec.radar_tick_ages,
                    *loading_spec.rgb_tick_ages,
                ),
                default=0,
            ),
        )

        future_iterations = list(loading_spec.future_iterations)
        return SceneData(
            cameras=cameras,
            past_cameras=past_cameras,
            lidar_sweeps=lidar_sweeps,
            radar_sweeps=radar_sweeps,
            semantic_cameras=semantic_cameras,
            instance_cameras=instance_cameras,
            depth_cameras=depth_cameras,
            ego_state=scene.get_ego_state_se3_at_iteration(0),
            box_detections=scene.get_box_detections_se3_at_iteration(0),
            traffic_lights=scene.get_traffic_light_detections_at_iteration(0),
            map_api=map_api,
            log_metadata=scene.log_metadata,
            scene_metadata=scene.scene_metadata,
            previous_target_point=target_points[0],
            target_point=target_points[1],
            next_target_point=target_points[2],
            past_ego_positions=past_ego_positions,
            past_ego_yaws=past_ego_yaws,
            rig_perturbation=resolved_view.rig_perturbation,
            driving_meta=driving_meta,
            future_driving_metas=(
                self.read_future_driving_metas(scene_index, future_iterations)
                if future_iterations
                else None
            ),
            future_ego_states=(
                self.read_future_ego_states(scene_index, future_iterations)
                if future_iterations
                else None
            ),
        )

    @property
    def future_num_iterations(self) -> int:
        """Simulator ticks of future every scene provides."""
        return self._scene_filter.future_num_iterations or 0

    @property
    def any_perturbated_sensor_views(self) -> bool:
        """Whether any enumerated sample provides a perturbated sensor view."""
        return bool(self._perturbated_view_pairing)

    def _read_driving_meta(
        self,
        scene: SceneAPI,
        iteration: int,
    ) -> DrivingMeta | None:
        """Read the driving-meta dict at a scene iteration."""
        modality = scene.get_custom_modality_at_iteration(
            iteration,
            DRIVING_META_MODALITY_ID,
        )
        # The stream is untyped on the py123d side; this read is where a LEAD
        # log's driving-meta contract is asserted.
        if modality is None:
            return None
        return typing.cast("DrivingMeta", modality.data)

    def read_future_driving_metas(
        self,
        scene_index: int,
        iterations: list[int],
    ) -> dict[int, DrivingMeta | None]:
        """Read the driving metas at future iterations of one sample's scene.

        Args:
            scene_index: Sample index into the scene list.
            iterations: Scene iterations ahead of the anchor to read.

        Returns:
            The driving metas keyed by iteration; None for the ones the scene
            lacks.
        """
        scene: SceneAPI = self.scenes[scene_index]
        return {
            iteration: self._read_driving_meta(scene, iteration)
            for iteration in iterations
            if iteration <= self.future_num_iterations
        }

    def read_future_ego_states(
        self,
        scene_index: int,
        iterations: list[int],
    ) -> dict[int, typing.Any]:
        """Read the native ego states at future iterations of one sample's scene.

        Args:
            scene_index: Sample index into the scene list.
            iterations: Scene iterations ahead of the anchor to read.

        Returns:
            The ego states keyed by iteration; None for the ones the scene lacks.
        """
        scene: SceneAPI = self.scenes[scene_index]
        return {
            iteration: scene.get_ego_state_se3_at_iteration(iteration)
            for iteration in iterations
            if iteration <= self.future_num_iterations
        }

    def has_perturbated_sensor_view(self, scene_index: int) -> bool:
        """Whether one sample's scene has a recorded perturbated rig view.

        Args:
            scene_index: Sample index into the scene list.

        Returns:
            True when a perturbated view exists for the sample.
        """
        return bool(self._has_perturbated_view[scene_index])

    def draw_sensor_view(self, scene_index: int) -> str:
        """Roll the view choice: perturbated where available, at the configured probability.

        Args:
            scene_index: Sample index into the scene list.

        Returns:
            The sensor-view name to read for this draw.
        """
        use_perturbated_view = (
            self.has_perturbated_sensor_view(scene_index)
            and np.random.rand() < self._perturbation_probability
        )
        return PERTURBATED_SENSOR_VIEW if use_perturbated_view else NORMAL_SENSOR_VIEW

    def _resolve_sensor_view(
        self,
        scene_index: int,
        use_perturbated_view: bool,
    ) -> _ResolvedSensorView:
        """Resolve one sample's scene into an explicit sensor view."""
        scene: SceneAPI = self.scenes[scene_index]
        if not use_perturbated_view:
            return _ResolvedSensorView(
                normal_scene=scene,
                sensor_scene=scene,
                rig_perturbation=None,
            )
        assert self._perturbated_view_pairing is not None, (
            "requested a perturbated view the dataset does not provide"
        )
        perturbated_scene: SceneAPI = self._perturbated_view_pairing.scene_at(
            scene.log_name,
            scene.get_timestamp_at_iteration(0).time_us,
        )
        return _ResolvedSensorView(
            normal_scene=scene,
            sensor_scene=perturbated_scene,
            rig_perturbation=self._rig_perturbation(scene, perturbated_scene),
        )

    def _target_points_in_view(
        self,
        scene: SceneAPI,
        driving_meta: DrivingMeta,
        resolved_view: _ResolvedSensorView,
    ) -> tuple[
        jt.Float[npt.NDArray, " 2"],
        jt.Float[npt.NDArray, " 2"],
        jt.Float[npt.NDArray, " 2"],
    ]:
        """Place the route's target points in the sample's view frame.

        The ego's own localization places them, as it does at inference.
        """
        ego_position, ego_yaw = se3_matrix_to_localized_pose(
            np.asarray(driving_meta[LOCALIZED_EGO_STATE_KEY], dtype=np.float64),
        )

        def to_view(
            point: jt.Float[npt.NDArray, " 3"],
        ) -> jt.Float[npt.NDArray, " 2"]:
            ego_point: jt.Float[npt.NDArray, " 2"] = geometry.to_local_frame_2d(
                point[:2],
                ego_position,
                ego_yaw,
            )
            if resolved_view.rig_perturbation is None:
                return ego_point
            return view_geometry.to_view_frame_points(
                ego_point[None],
                resolved_view.rig_perturbation.lateral_translation_m,
                resolved_view.rig_perturbation.yaw_rotation_deg,
            )[0]

        previous_target_point, current_target_point, next_target_point = (
            ordered_target_points(
                self._read_log_target_points(scene),
                driving_meta["target_point_indices"][
                    str(self.target_point_pop_distance)
                ],
            )
        )
        return (
            to_view(previous_target_point),
            to_view(current_target_point),
            to_view(next_target_point),
        )

    def _read_log_target_points(
        self,
        scene: SceneAPI,
    ) -> jt.Float[npt.NDArray, "n 3"]:
        """Read the log's target points, the ones its ticks index into."""
        log_name: str = scene.log_name
        if log_name not in self._target_points_by_log:
            metadata = scene.get_all_custom_modality_metadatas()[
                DRIVING_META_MODALITY_ID
            ]
            self._target_points_by_log[log_name] = np.array(
                metadata.metadata[TARGET_POINTS_METADATA_KEY],
            )
        return self._target_points_by_log[log_name]

    def _camera_ids_to_read(self, scene: SceneAPI) -> list[CameraID]:
        """Camera IDs to read for a scene, in left-to-right stitch order.

        The requested selection, or every recorded camera in LEAD index order
        when none was requested.
        """
        if self._camera_ids is None:
            available_camera_ids = set(scene.get_camera_metadatas())
            if self._requested_camera_ids is not None:
                missing_camera_ids = [
                    camera_id
                    for camera_id in self._requested_camera_ids
                    if camera_id not in available_camera_ids
                ]
                if missing_camera_ids:
                    raise ValueError(
                        f"requested cameras {missing_camera_ids} are not recorded "
                        f"in log {scene.log_name}; available: "
                        f"{sorted(available_camera_ids, key=int)}",
                    )
                self._camera_ids = list(self._requested_camera_ids)
            else:
                self._camera_ids = [
                    camera_id
                    for _, camera_id in sorted(CAMERA_ID_BY_LEAD_INDEX.items())
                    if camera_id in available_camera_ids
                ]
        return self._camera_ids

    def _rig_perturbation(
        self,
        scene: SceneAPI,
        perturbated_scene: SceneAPI,
    ) -> RigPerturbation:
        """Rig perturbation of a log, derived from the camera extrinsics."""
        log_name: str = scene.log_name
        if log_name not in self._rig_perturbation_by_log:
            camera_id: CameraID = self._camera_ids_to_read(scene)[0]
            ego_metadata = scene.get_ego_state_se3_metadata()
            assert ego_metadata is not None
            translation, rotation = view_geometry.derive_rig_perturbation(
                scene.get_camera_metadatas()[camera_id].camera_to_imu_se3,
                perturbated_scene.get_camera_metadatas()[camera_id].camera_to_imu_se3,
                ego_metadata.rear_axle_to_center_longitudinal,
            )
            self._rig_perturbation_by_log[log_name] = RigPerturbation(
                lateral_translation_m=translation,
                yaw_rotation_deg=rotation,
            )
        return self._rig_perturbation_by_log[log_name]

    def _read_lidar_sweeps(
        self,
        scene: SceneAPI,
        tick_ages: tuple[int, ...],
    ) -> dict[int, Lidar]:
        """Read the lidar sweeps at the requested tick ages."""
        tick_duration_us = self._tick_duration_us
        anchor_timestamp_us = scene.get_timestamp_at_iteration(0).time_us
        wanted = set(tick_ages)

        sweeps: dict[int, Lidar] = {}
        for lidar in scene.get_modality_between_timestamps(
            anchor_timestamp_us - max(tick_ages) * tick_duration_us,
            anchor_timestamp_us,
            ModalityType.LIDAR,
            LidarID.LIDAR_TOP,
            inclusive="both",
            lidar_id=LidarID.LIDAR_TOP,
        ):
            assert isinstance(lidar, Lidar)
            age = round(
                (anchor_timestamp_us - lidar.timestamp.time_us) / tick_duration_us,
            )
            if age in wanted:
                sweeps[age] = lidar
        return sweeps

    def _read_radar_sweeps(
        self,
        sensor_scene: SceneAPI,
        tick_ages: tuple[int, ...],
    ) -> dict[int, Radar]:
        """Read the merged radar returns at the requested tick ages."""
        tick_duration_us = self._tick_duration_us
        anchor_timestamp_us = sensor_scene.get_timestamp_at_iteration(0).time_us
        wanted = set(tick_ages)

        sweeps: dict[int, Radar] = {}
        for radar in sensor_scene.get_modality_between_timestamps(
            anchor_timestamp_us - max(tick_ages) * tick_duration_us,
            anchor_timestamp_us,
            ModalityType.RADAR,
            RadarID.RADAR_MERGED,
            inclusive="both",
            radar_id=RadarID.RADAR_MERGED,
        ):
            assert isinstance(radar, Radar)
            age = round(
                (anchor_timestamp_us - radar.timestamp.time_us) / tick_duration_us,
            )
            if age in wanted:
                sweeps[age] = radar
        return sweeps

    def _read_past_cameras(
        self,
        sensor_scene: SceneAPI,
        tick_ages: tuple[int, ...],
    ) -> dict[int, list[Camera]]:
        """Read the camera rig at the requested tick ages.

        The camera streams are anchor-locked in the log, so an age the dataset
        stored no frame at is absent from the result rather than an error.
        """
        tick_duration_us = self._tick_duration_us
        anchor_timestamp_us = sensor_scene.get_timestamp_at_iteration(0).time_us
        camera_ids = self._camera_ids_to_read(sensor_scene)

        rigs: dict[int, list[Camera]] = {}
        for age in tick_ages:
            timestamp_us = anchor_timestamp_us - age * tick_duration_us
            rig = [
                sensor_scene.get_modality_at_timestamp(
                    timestamp_us,
                    ModalityType.CAMERA,
                    camera_id,
                )
                for camera_id in camera_ids
            ]
            # A partial rig would silently change the stitch width downstream.
            if any(camera is None for camera in rig):
                continue
            rigs[age] = [typing.cast("Camera", camera) for camera in rig]
        return rigs

    def _read_past_ego_motion(
        self,
        scene: SceneAPI,
        driving_meta: DrivingMeta,
        num_ticks: int,
    ) -> tuple[jt.Float[npt.NDArray, "t 2"], jt.Float[npt.NDArray, " t"]]:
        """Localized ego pose history over the past window, in the anchor's frame.

        Derived from the per-tick localized ego states of the driving-meta
        stream, matching what the ego knows about its own motion online.

        Args:
            scene: The scene to read from.
            driving_meta: The anchor tick's driving meta.
            num_ticks: Ticks to cover, the anchor included; the result is
                shorter where the log runs out.

        Returns:
            The past ego positions and yaws, index i = i ticks ago.
        """
        tick_duration_us = self._tick_duration_us
        anchor_timestamp_us = scene.get_timestamp_at_iteration(0).time_us
        anchor_position, anchor_yaw = se3_matrix_to_localized_pose(
            np.asarray(driving_meta[LOCALIZED_EGO_STATE_KEY], dtype=np.float64),
        )

        poses_by_tick_age: dict[int, tuple[jt.Float[npt.NDArray, " 2"], float]] = {}
        for modality in scene.get_modality_between_timestamps(
            anchor_timestamp_us - (num_ticks - 1) * tick_duration_us,
            anchor_timestamp_us,
            ModalityType.CUSTOM,
            DRIVING_META_MODALITY_ID,
            inclusive="both",
        ):
            assert isinstance(modality, CustomModality)
            age = round(
                (anchor_timestamp_us - modality.timestamp.time_us) / tick_duration_us,
            )
            poses_by_tick_age[age] = se3_matrix_to_localized_pose(
                np.asarray(modality.data[LOCALIZED_EGO_STATE_KEY], dtype=np.float64),
            )

        cos_yaw, sin_yaw = np.cos(anchor_yaw), np.sin(anchor_yaw)
        to_anchor_rotation = np.array([[cos_yaw, sin_yaw], [-sin_yaw, cos_yaw]])
        positions: list[jt.Float[npt.NDArray, " 2"]] = []
        yaws: list[float] = []
        for age in range(num_ticks):
            pose = poses_by_tick_age.get(age)
            if pose is None:
                break
            position, yaw = pose
            positions.append(to_anchor_rotation @ (position - anchor_position))
            yaws.append(geometry.normalize_angle_rad(yaw - anchor_yaw))
        return np.array(positions).reshape(-1, 2), np.array(yaws)

    def _read_camera_stream(
        self,
        sensor_scene: SceneAPI,
        read_camera: Callable[[int, CameraID], Camera | None],
    ) -> list[Camera]:
        """Read one camera stream at the anchor tick, in LEAD camera order."""
        return [
            _require_not_none(read_camera(0, camera_id))
            for camera_id in self._camera_ids_to_read(sensor_scene)
        ]


T = typing.TypeVar("T")


def _require_not_none(optional_read: T | None) -> T:
    """Unwrap an Optional py123d read the data contract guarantees; for comprehensions."""
    assert optional_read is not None
    return optional_read


# Modalities every scene read touches; anchoring on ``camera:all@initial``
# keeps the scenes on the save ticks, where the camera streams exist.
_REQUIRED_SCENE_MODALITIES = [
    "ego_state_se3",
    "custom.driving_meta@initial",
    "camera:all@initial",
]


@dataclass
class _ResolvedSensorView:
    """One sample's 123D scene with the sensor-view choice resolved.

    ``normal_scene`` holds the labels and the view-independent modalities
    (lidar, boxes, driving meta); view-dependent sensors (cameras, instance
    segmentation, depth, radar) are read from ``sensor_scene``.
    """

    normal_scene: SceneAPI
    sensor_scene: SceneAPI
    # Rig perturbation of the chosen view; None for the normal view.
    rig_perturbation: RigPerturbation | None
