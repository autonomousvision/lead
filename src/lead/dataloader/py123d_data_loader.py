"""Model-agnostic frame loading over the 123D logs: enumerates the scenes of a
log collection and reads each into a :class:`~lead.dataloader.frame.Frame`."""

import typing
from dataclasses import dataclass
from pathlib import Path

import jaxtyping as jt
import numpy as np
import numpy.typing as npt
from py123d.api.scene.scene_api import SceneAPI
from py123d.api.scene.scene_filter import SceneFilter
from py123d.datatypes import Lidar, ModalityType
from py123d.datatypes.sensors.base_camera import Camera, CameraID
from py123d.datatypes.sensors.lidar import LidarID
from py123d.datatypes.sensors.radar import Radar, RadarID

from lead.common import geometry
from lead.dataloader import scene_index, view_geometry
from lead.dataloader.frame import Frame, RigPerturbation
from lead.dataloader.log_format import (
    CAMERA_ID_MAPPING,
    DRIVING_META_MODALITY_ID,
    TARGET_POINTS_KEY,
    ordered_target_points,
)

T = typing.TypeVar("T")


def require_not_none(value: T | None) -> T:
    """Unwrap an Optional py123d read that the data contract guarantees.

    The LEAD logs are complete for every modality the loader reads, so a None
    from a py123d getter is a data bug worth failing on immediately; prefer a
    plain ``assert`` where a local variable exists — this is for comprehensions.

    Args:
        value: The Optional value to unwrap.

    Returns:
        The value, guaranteed not None.
    """
    assert value is not None
    return value


def default_scene_filter() -> SceneFilter:
    """The scenes of a LEAD log collection: one per save tick, no margins.

    Returns:
        A filter over the modalities every LEAD log records.
    """
    return SceneFilter(
        history_num_iterations=0,
        future_num_iterations=0,
        required_scene_modalities=[
            "ego_state_se3",
            "custom.driving_meta@initial",
            "camera:all@initial",
        ],
    )


@dataclass
class _SceneView:
    """One sample's 123D scene with the view choice resolved.

    ``scene`` is always the normal view holding the labels and the
    view-independent modalities (lidar, boxes, driving meta); view-dependent
    sensors (cameras, instance segmentation, depth, radar) are read from
    ``sensor_scene``.
    """

    scene: SceneAPI
    sensor_scene: SceneAPI
    # Rig perturbation of the chosen view; None for the normal view.
    perturbation: RigPerturbation | None


class Py123DDataLoader:
    """Reads the 123D scenes of a log collection into frames.

    An indexable sequence: ``len(loader)`` is the number of frames the logs
    yield and ``loader[i]`` is the i-th :class:`~lead.dataloader.frame.Frame`,
    with the view choice resolved and only the requested modalities read; the
    py123d scene handles never leave the class.
    """

    def __init__(
        self,
        data_root: str | Path | None = None,
        scene_filter: SceneFilter | None = None,
        *,
        perturbation_prob: float = 0.0,
        tp_pop_distance: float = 3.25,
        sweep_window_ticks: int = 10,
        tick_duration_us: int = 50_000,
        lidar_sweeps: bool = True,
        radar_sweeps: bool = True,
        semantics: bool = True,
        depths: bool = True,
        map: bool = True,  # noqa: A002
    ) -> None:
        """Enumerate the scenes both views provide.

        Args:
            data_root: Dataset root holding ``logs/`` and ``maps/``; None uses
                py123d's ``PY123D_DATA_ROOT``.
            scene_filter: py123d's own scene selection (splits, logs, the
                history/future window); None selects every LEAD scene.
            perturbation_prob: Probability of reading the perturbated sensor
                views, where recorded.
            tp_pop_distance: Pop distance of the route planner whose target
                points the frames carry.
            sweep_window_ticks: Simulator ticks of lidar/radar sweeps to read
                per frame, ending at the anchor tick.
            tick_duration_us: Duration of one simulator tick in microseconds.
            lidar_sweeps: Whether to read the lidar sweep window.
            radar_sweeps: Whether to read the radar sweep window.
            semantics: Whether to read the semantic and instance cameras.
            depths: Whether to read the depth cameras.
            map: Whether to read the map API.
        """
        self.scene_filter: SceneFilter = (
            scene_filter if scene_filter is not None else default_scene_filter()
        )
        self.perturbation_prob = perturbation_prob
        self.tp_pop_distance = tp_pop_distance
        self.sweep_window_ticks = sweep_window_ticks
        self.tick_duration_us = tick_duration_us
        self._read_lidar_sweeps = lidar_sweeps
        self._read_radar_sweeps = radar_sweeps
        self._read_semantics = semantics
        self._read_depths = depths
        self._read_map = map

        self.scenes: list[SceneAPI] = scene_index.get_scenes(
            data_root,
            self.scene_filter,
        )
        self.perturbated_scenes: dict[tuple[str, int], SceneAPI] = {}
        if perturbation_prob > 0.0:
            self.perturbated_scenes = scene_index.get_perturbated_scene_lookup(
                data_root,
            )

        # The rig is constant across a log collection; read it lazily from the
        # logs themselves.
        self._camera_ids: list[CameraID] | None = None
        # Rig perturbation per log, derived lazily from the camera extrinsics.
        self._rig_perturbations: dict[str, RigPerturbation] = {}
        # Target points per log, read lazily from the driving-meta metadata.
        self._target_points: dict[str, jt.Float[npt.NDArray, "n 3"]] = {}

    def __len__(self) -> int:
        return len(self.scenes)

    def __getitem__(self, index: int) -> Frame:
        """Read the frame of one sample.

        Args:
            index: Sample index into the scene list.

        Returns:
            The frame, with every ego-frame quantity in the chosen view's frame
            and only the requested modalities populated.
        """
        view: _SceneView = self._choose_view(index)
        scene: SceneAPI = view.scene
        meta: dict | None = self.driving_meta(scene, 0)
        assert meta is not None

        # LiDAR is only recorded in the normal view; consumers accumulate the
        # sweeps and move them into the view frame themselves.
        lidar_sweeps: dict[int, Lidar] | None = None
        if self._read_lidar_sweeps:
            lidar_sweeps = self._load_lidar_sweeps(scene)

        radar_sweeps: dict[int, Radar] | None = None
        if self._read_radar_sweeps:
            radar_sweeps = self._load_radar_sweeps(view.sensor_scene)

        semantics: list[Camera] | None = None
        instances: list[Camera] | None = None
        if self._read_semantics:
            semantics = self._load_semantics(view.sensor_scene)
            instances = self._load_instances(view.sensor_scene)

        depths: list[Camera] | None = None
        if self._read_depths:
            depths = self._load_depths(view.sensor_scene)

        map_api = scene.get_map_api() if self._read_map else None

        target_points = self._target_points_in_view(scene, meta, view)

        return Frame(
            cameras=self._load_cameras(view.sensor_scene),
            lidar_sweeps=lidar_sweeps,
            radar_sweeps=radar_sweeps,
            semantics=semantics,
            instances=instances,
            depths=depths,
            ego_state=scene.get_ego_state_se3_at_iteration(0),
            box_detections=scene.get_box_detections_se3_at_iteration(0),
            traffic_lights=scene.get_traffic_light_detections_at_iteration(0),
            map_api=map_api,
            log_metadata=scene.log_metadata,
            scene_metadata=scene.scene_metadata,
            target_point_previous=target_points[0],
            target_point=target_points[1],
            target_point_next=target_points[2],
            past_positions=np.asarray(meta["past_positions"], dtype=np.float64),
            past_yaws=np.asarray(meta["past_yaws"], dtype=np.float64),
            perturbation=view.perturbation,
            meta=meta,
        )

    @property
    def future_num_iterations(self) -> int:
        """Save ticks of future every scene provides."""
        return self.scene_filter.future_num_iterations or 0

    def driving_meta(self, scene: SceneAPI, iteration: int) -> dict | None:
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

    def future_metas(self, index: int, iterations: list[int]) -> dict[int, dict | None]:
        """Read the driving metas at future iterations of one sample's scene.

        Args:
            index: Sample index into the scene list.
            iterations: Scene iterations ahead of the anchor to read.

        Returns:
            The metas keyed by iteration; None for the ones the scene lacks.
        """
        scene: SceneAPI = self.scenes[index]
        return {
            iteration: self.driving_meta(scene, iteration)
            for iteration in iterations
            if iteration <= self.future_num_iterations
        }

    def future_ego_states(
        self,
        index: int,
        iterations: list[int],
    ) -> dict[int, typing.Any]:
        """Read the native ego states at future iterations of one sample's scene.

        Args:
            index: Sample index into the scene list.
            iterations: Scene iterations ahead of the anchor to read.

        Returns:
            The ego states keyed by iteration; None for the ones the scene lacks.
        """
        scene: SceneAPI = self.scenes[index]
        return {
            iteration: scene.get_ego_state_se3_at_iteration(iteration)
            for iteration in iterations
            if iteration <= self.future_num_iterations
        }

    def future_box_detections(
        self,
        index: int,
        iterations: list[int],
    ) -> dict[int, typing.Any]:
        """Read the native box detections at future iterations of one sample's scene.

        Args:
            index: Sample index into the scene list.
            iterations: Scene iterations ahead of the anchor to read.

        Returns:
            The box detections keyed by iteration; None for the ones the scene
            lacks.
        """
        scene: SceneAPI = self.scenes[index]
        return {
            iteration: scene.get_box_detections_se3_at_iteration(iteration)
            for iteration in iterations
            if iteration <= self.future_num_iterations
        }

    def _choose_view(self, index: int) -> _SceneView:
        """Choose the normal or perturbated view of one sample's scene.

        Args:
            index: Sample index into the scene list.

        Returns:
            The scene with the view choice resolved and the rig perturbation
            of the chosen view.
        """
        scene: SceneAPI = self.scenes[index]
        perturbated_scene: SceneAPI | None = self.perturbated_scenes.get(
            (scene.log_name, scene.get_scene_metadata().initial_idx),
        )
        if perturbated_scene is not None and np.random.rand() < self.perturbation_prob:
            return _SceneView(
                scene=scene,
                sensor_scene=perturbated_scene,
                perturbation=self._rig_perturbation(scene, perturbated_scene),
            )
        return _SceneView(scene=scene, sensor_scene=scene, perturbation=None)

    def _target_points_in_view(
        self,
        scene: SceneAPI,
        meta: dict,
        view: _SceneView,
    ) -> tuple[
        jt.Float[npt.NDArray, " 2"],
        jt.Float[npt.NDArray, " 2"],
        jt.Float[npt.NDArray, " 2"],
    ]:
        """Place the route's target points in the frame's view.

        The ego's own localization places them, as it does at inference.

        Args:
            scene: The scene of the sample.
            meta: Driving meta of the anchor tick.
            view: The resolved view of the sample.

        Returns:
            The previous, current and next target point.
        """
        ego_position: jt.Float[npt.NDArray, " 2"] = np.array(
            meta["localized_pos_global"][:2],
        )
        ego_yaw: float = meta["theta"]

        def to_view(
            point: jt.Float[npt.NDArray, " 3"],
        ) -> jt.Float[npt.NDArray, " 2"]:
            ego_point: jt.Float[npt.NDArray, " 2"] = geometry.inverse_conversion_2d(
                point[:2],
                ego_position,
                ego_yaw,
            )
            if view.perturbation is None:
                return ego_point
            return view_geometry.to_view_frame_points(
                ego_point[None],
                view.perturbation.translation,
                view.perturbation.rotation,
            )[0]

        previous_tp, current_tp, next_tp = ordered_target_points(
            self._log_target_points(scene),
            meta["target_point_indices"][str(self.tp_pop_distance)],
        )
        return to_view(previous_tp), to_view(current_tp), to_view(next_tp)

    def _log_target_points(self, scene: SceneAPI) -> jt.Float[npt.NDArray, "n 3"]:
        """Read the log's target points, the ones its ticks index into.

        Args:
            scene: A scene of the log.

        Returns:
            The target points, in the global frame.
        """
        log_name: str = scene.log_name
        if log_name not in self._target_points:
            metadata = scene.get_all_custom_modality_metadatas()[
                DRIVING_META_MODALITY_ID
            ]
            self._target_points[log_name] = np.array(
                metadata.metadata[TARGET_POINTS_KEY],
            )
        return self._target_points[log_name]

    def _used_camera_ids(self, scene: SceneAPI) -> list[CameraID]:
        """LEAD camera IDs of the rig, in stored left-to-right order.

        Args:
            scene: A scene of the log collection.

        Returns:
            The recorded camera IDs, ordered by LEAD camera index.
        """
        if self._camera_ids is None:
            available = set(scene.get_camera_metadatas())
            self._camera_ids = [
                camera_id
                for _, camera_id in sorted(CAMERA_ID_MAPPING.items())
                if camera_id in available
            ]
        return self._camera_ids

    def _rig_perturbation(
        self,
        scene: SceneAPI,
        perturbated_scene: SceneAPI,
    ) -> RigPerturbation:
        """Rig perturbation of a log, derived from the camera extrinsics.

        Args:
            scene: Normal-view scene of the log.
            perturbated_scene: Paired perturbated-view scene.

        Returns:
            The rig perturbation of the log's perturbated view.
        """
        log_name: str = scene.log_name
        if log_name not in self._rig_perturbations:
            camera_id: CameraID = self._used_camera_ids(scene)[0]
            ego_metadata = scene.get_ego_state_se3_metadata()
            assert ego_metadata is not None
            translation, rotation = view_geometry.derive_rig_perturbation(
                scene.get_camera_metadatas()[camera_id].camera_to_imu_se3,
                perturbated_scene.get_camera_metadatas()[camera_id].camera_to_imu_se3,
                ego_metadata.rear_axle_to_center_longitudinal,
            )
            self._rig_perturbations[log_name] = RigPerturbation(
                translation=translation,
                rotation=rotation,
            )
        return self._rig_perturbations[log_name]

    def _load_lidar_sweeps(self, scene: SceneAPI) -> dict[int, Lidar]:
        """Read the lidar sweeps of the frame's temporal window.

        Args:
            scene: The scene providing the sweep stream.

        Returns:
            The sweeps keyed by frame age (0 = anchor tick), each in the IMU
            frame of its own tick. Ages without a stored sweep are absent.
        """
        tick_us = self.tick_duration_us
        anchor_us = scene.get_timestamp_at_iteration(0).time_us

        sweeps: dict[int, Lidar] = {}
        for lidar in scene.get_modality_between_timestamps(
            anchor_us - (self.sweep_window_ticks - 1) * tick_us,
            anchor_us,
            ModalityType.LIDAR,
            LidarID.LIDAR_TOP,
            inclusive="both",
            lidar_id=LidarID.LIDAR_TOP,
        ):
            assert isinstance(lidar, Lidar)
            sweeps[round((anchor_us - lidar.timestamp.time_us) / tick_us)] = lidar
        return sweeps

    def _load_radar_sweeps(self, sensor_scene: SceneAPI) -> dict[int, Radar]:
        """Read the merged radar returns of the frame's temporal window.

        Args:
            sensor_scene: The scene of the chosen view.

        Returns:
            The merged returns of all sensors keyed by frame age (0 = anchor
            tick), each in the IMU frame of its own tick.
        """
        tick_us = self.tick_duration_us
        anchor_us = sensor_scene.get_timestamp_at_iteration(0).time_us

        sweeps: dict[int, Radar] = {}
        for radar in sensor_scene.get_modality_between_timestamps(
            anchor_us - (self.sweep_window_ticks - 1) * tick_us,
            anchor_us,
            ModalityType.RADAR,
            RadarID.RADAR_MERGED,
            inclusive="both",
            radar_id=RadarID.RADAR_MERGED,
        ):
            assert isinstance(radar, Radar)
            sweeps[round((anchor_us - radar.timestamp.time_us) / tick_us)] = radar
        return sweeps

    def _load_cameras(self, sensor_scene: SceneAPI) -> list[Camera]:
        """Read the anchor tick's cameras in LEAD camera order.

        Args:
            sensor_scene: The scene of the chosen view.

        Returns:
            One camera per LEAD index (1..num_cameras), each carrying its
            image, its calibration and its pose.
        """
        return [
            require_not_none(sensor_scene.get_camera_at_iteration(0, camera_id))
            for camera_id in self._used_camera_ids(sensor_scene)
        ]

    def _load_semantics(self, sensor_scene: SceneAPI) -> list[Camera]:
        """Read the anchor tick's semantic cameras, same camera order.

        Args:
            sensor_scene: The scene of the chosen view.

        Returns:
            One semantic camera per LEAD index, its image the raw CARLA class
            of every pixel.
        """
        return [
            require_not_none(
                sensor_scene.get_camera_semantic_at_iteration(0, camera_id),
            )
            for camera_id in self._used_camera_ids(sensor_scene)
        ]

    def _load_instances(self, sensor_scene: SceneAPI) -> list[Camera]:
        """Read the anchor tick's instance cameras, same camera order.

        Args:
            sensor_scene: The scene of the chosen view.

        Returns:
            One instance camera per LEAD index, its image the CARLA actor id of
            every pixel.
        """
        return [
            require_not_none(
                sensor_scene.get_camera_instance_at_iteration(0, camera_id),
            )
            for camera_id in self._used_camera_ids(sensor_scene)
        ]

    def _load_depths(self, sensor_scene: SceneAPI) -> list[Camera]:
        """Read the anchor tick's depth cameras, same camera order.

        Args:
            sensor_scene: The scene of the chosen view.

        Returns:
            One depth camera per LEAD index, its image the stored encoding;
            ``camera.metadata.decode_depth(camera.image)`` makes it metric.
        """
        return [
            require_not_none(
                sensor_scene.get_camera_depth_at_iteration(0, camera_id),
            )
            for camera_id in self._used_camera_ids(sensor_scene)
        ]
