"""Py123D Arrow log writing: map conversion, modality recorders and teardown."""

import logging
from pathlib import Path

from leaderboard.utils.statistics_manager_local import RouteRecord
from py123d.api.map.arrow.arrow_map_writer import ArrowMapWriter
from py123d.api.scene.arrow.arrow_log_writer import ArrowLogWriter, SyncConfig
from py123d.api.scene.arrow.utils.log_writer_config import LogWriterConfig
from py123d.common.io.lidar.point_cloud_codec_config import PointCloudCodecConfig
from py123d.common.runtime.dataset_paths import get_dataset_paths
from py123d.datatypes import (
    BaseModality,
    EgoStateSE3,
    LogMetadata,
    MapMetadata,
    Timestamp,
)
from py123d.parser.base_dataset_parser import ModalitiesSync
from py123d.parser.opendrive import opendrive_map_parser
from py123d.parser.opendrive.opendrive_map_parser import iter_xodr_map_objects

from lead.expert.logs_writing.async_writer import AsyncWriter
from lead.expert.logs_writing.recorders import (
    BoxDetectionsRecorder,
    CameraRecorder,
    DepthRecorder,
    DrivingMetaRecorder,
    EgoStateRecorder,
    InstanceRecorder,
    LidarRecorder,
    RadarRecorder,
    TrafficLightRecorder,
)
from lead.expert.logs_writing.recorders.driving_meta_recorder import (
    DRIVING_META_MODALITY_ID,
)

LOG = logging.getLogger(__name__)

# OpenDRIVE maps of all CARLA towns, bundled with py123d
CARLA_MAPS_DIR = Path(opendrive_map_parser.__file__).parent / "carla_maps"


class Py123dLoggingMixin:
    """Writes the expert's per-tick data as Py123D Arrow logs."""

    @property
    def _writes_py123d(self) -> bool:
        """Whether this run produces a 123D log (datagen, not expert evaluation)."""
        return (
            self.config_expert.data_collection.datagen
            and not self.config_expert.data_collection.eval_expert
        )

    @property
    def _writes_py123d_perturbated(self) -> bool:
        """Whether this run additionally logs the perturbated sensor views."""
        return (
            self._writes_py123d and self.config_expert.perturbation.perturbate_sensors
        )

    def _setup_py123d_writers(self) -> None:
        """Create the Arrow log/map writers and their output directories."""
        dataset_paths = get_dataset_paths()
        self.py123d_logs_root = Path(dataset_paths.py123d_logs_root)
        self.py123d_maps_root = Path(dataset_paths.py123d_maps_root)
        self.py123d_sensors_root = Path(dataset_paths.py123d_sensors_root)
        self.py123d_logs_root.mkdir(parents=True, exist_ok=True)
        self.py123d_maps_root.mkdir(parents=True, exist_ok=True)
        # Not created eagerly: unused by the current inline (jpeg_binary/binary)
        # camera/lidar/radar store options, which embed sensor data in the log.

        storage = self.config_expert.storage
        log_writer_config = LogWriterConfig(
            force_log_conversion=True,
            camera_store_option="jpeg_binary",
            lidar_store_option="binary",
            lidar_codec="laz",
            lidar_codec_config=PointCloudCodecConfig(
                laz_point_format=storage.point_format,
                laz_scales=(
                    storage.point_precision_x,
                    storage.point_precision_y,
                    storage.point_precision_z,
                ),
            ),
            radar_store_option="binary",
            radar_codec="draco",
        )
        # Sync rows are built at close, one per driving-meta (save tick)
        # timestamp; scene iterations stay in save-tick units.
        self.py123d_log_writer = ArrowLogWriter(
            log_writer_config=log_writer_config,
            logs_root=self.py123d_logs_root,
            sensors_root=self.py123d_sensors_root,
            sync_config=SyncConfig(
                reference_column=f"custom.{DRIVING_META_MODALITY_ID}.timestamp_us",
                direction="backward",
            ),
        )
        if self._writes_py123d_perturbated:
            self.py123d_perturbated_log_writer = ArrowLogWriter(
                log_writer_config=log_writer_config,
                logs_root=self.py123d_logs_root,
                sensors_root=self.py123d_sensors_root,
            )
        self.py123d_map_writer = ArrowMapWriter(
            force_map_conversion=False,
            maps_root=self.py123d_maps_root,
            logs_root=self.py123d_logs_root,
        )

        # Image encoding and disk writes happen on a background thread so the
        # simulation never waits on them
        self._sensor_writer = AsyncWriter(self._write_tick)

    def _init_py123d(self) -> None:
        """Convert the town map once, open the 123D log, and build the recorders."""
        town_name = self.carla_world.get_map().name.split("/")[-1]
        location = town_name.lower()
        dataset = self.config_expert.data_collection.py123d_dataset
        log_name = self.log_name

        # Convert the OpenDRIVE map bundled with py123d once per town
        map_metadata = MapMetadata(
            dataset=dataset,
            location=location,
            map_has_z=True,
            map_is_per_log=False,
        )
        if self.py123d_map_writer.reset(map_metadata):
            xodr_file = CARLA_MAPS_DIR / f"{town_name}.xodr.gz"
            for map_object in iter_xodr_map_objects(xodr_file):
                self.py123d_map_writer.write_map_object(map_object)
            LOG.info(f"Converted 123D map for {location}")
        self.py123d_map_writer.close()

        # Logs are grouped by scenario type inside each split directory
        self.py123d_log_writer.reset(
            LogMetadata(
                dataset=dataset,
                split=f"{self.config_expert.data_collection.py123d_split}/{self.scenario_name}",
                log_name=log_name,
                location=location,
                map_metadata=map_metadata,
            ),
        )
        if self._writes_py123d_perturbated:
            self.py123d_perturbated_log_writer.reset(
                LogMetadata(
                    dataset=dataset,
                    split=f"{self.config_expert.data_collection.py123d_perturbated_split}/{self.scenario_name}",
                    log_name=log_name,
                    location=location,
                    map_metadata=map_metadata,
                ),
            )
        LOG.info(
            "123D log writer initialized: "
            f"{self.py123d_logs_root / self.config_expert.data_collection.py123d_split / self.scenario_name / log_name}",
        )

        # Modality recorders: ego state is extracted first and passed to the rest.
        # State recorders read live CARLA/queue state and must run on the driving
        # thread at the correct tick; array recorders are pure functions over the
        # tick's arrays and run on the background write thread.
        data_config = self.config_expert.data_collection
        self.ego_state_recorder = EgoStateRecorder(self)
        self.tick_state_recorders = [
            LidarRecorder(self, store_freq=data_config.lidar_store_freq),
            BoxDetectionsRecorder(
                self,
                store_freq=data_config.box_detections_store_freq,
            ),
        ]
        self.save_tick_state_recorders = [
            TrafficLightRecorder(
                self,
                map_arrow_path=self.py123d_maps_root
                / dataset
                / f"{dataset}_{location}.arrow",
                store_freq=data_config.traffic_light_store_freq,
            ),
        ]
        self.tick_array_recorders = []
        if self.config_expert.sensor_rig.use_radars:
            self.tick_array_recorders.append(
                RadarRecorder(self, store_freq=data_config.radar_store_freq),
            )
        self.save_tick_array_recorders = [
            CameraRecorder(self, store_freq=data_config.camera_store_freq),
            InstanceRecorder(self, store_freq=data_config.instance_store_freq),
        ]
        if data_config.save_depth:
            self.save_tick_array_recorders.append(
                DepthRecorder(self, store_freq=data_config.depth_store_freq),
            )
        for recorder in self.save_tick_state_recorders + self.save_tick_array_recorders:
            assert recorder.store_freq % data_config.data_save_freq == 0, (
                f"{type(recorder).__name__}: store_freq {recorder.store_freq} is not "
                f"a multiple of data_save_freq {data_config.data_save_freq}"
            )
        self.driving_meta_recorder = DrivingMetaRecorder(self)

        # Save ticks are detected by camera-data presence (the render phase is
        # not aligned with ``self.step``); tick recorders re-phase on them so
        # every save tick carries the every-tick modalities at age 0.
        self._num_save_ticks = 0
        self._last_save_tick_step = 0

        # Perturbated views go to their own split; view-independent modalities
        # (lidar, boxes, traffic lights, driving meta) stay in the normal split.
        self.perturbated_modality_recorders = []
        if self._writes_py123d_perturbated:
            self.perturbated_modality_recorders = [
                CameraRecorder(self, perturbated=True),
                InstanceRecorder(self, perturbated=True),
            ]
            if self.config_expert.data_collection.save_depth:
                self.perturbated_modality_recorders.append(
                    DepthRecorder(self, perturbated=True),
                )
            if self.config_expert.sensor_rig.use_radars:
                self.perturbated_modality_recorders.append(
                    RadarRecorder(self, perturbated=True),
                )
        LOG.info(
            f"123D perturbated log writer initialized: {self.py123d_logs_root / log_name}",
        )

    def _tick_store_due(self, recorder) -> bool:
        """Whether a tick recorder's storage frequency is met this step.

        Storage is phased on the last save tick, so every save tick stores
        the every-tick modalities regardless of the render phase.
        """
        return (self.step - self._last_save_tick_step) % recorder.store_freq == 0

    def _save_tick_store_due(self, recorder) -> bool:
        """Whether a save-tick recorder's storage frequency is met on this save tick."""
        data_save_freq = self.config_expert.data_collection.data_save_freq
        return self._num_save_ticks % (recorder.store_freq // data_save_freq) == 0

    def save_sensors(self, tick_data: dict) -> None:
        """Capture the current tick's state and queue it for background writing.

        Called every tick; each modality is recorded whenever its storage
        frequency is met, save-tick-only modalities (cameras, driving meta,
        traffic lights) additionally only when ``self.is_save_tick``.

        Args:
            tick_data: Post-tick sensor data from CARLA.
        """
        if not self._writes_py123d:
            # Expert evaluation: all outputs live in the 123D logs.
            return

        if self.is_save_tick:
            self._last_save_tick_step = self.step

        timestamp = Timestamp.from_s(self.step * self.config_expert.driving.fps_inv)
        ego_state = self.ego_state_recorder.extract(timestamp)
        modalities = [ego_state]
        for recorder in self.tick_state_recorders:
            if self._tick_store_due(recorder):
                modalities.extend(recorder.record(tick_data, timestamp, ego_state))
        if self.is_save_tick:
            for recorder in self.save_tick_state_recorders:
                if self._save_tick_store_due(recorder):
                    modalities.extend(recorder.record(tick_data, timestamp, ego_state))
            modalities.append(
                self.driving_meta_recorder.record_meta(
                    self.driving_meta,
                    tick_data,
                    timestamp,
                ),
            )

        # The write thread must not read the counters, so the due array
        # recorders are resolved here on the driving thread.
        due_array_recorders = [
            recorder
            for recorder in self.tick_array_recorders
            if self._tick_store_due(recorder)
        ]
        if self.is_save_tick:
            due_array_recorders += [
                recorder
                for recorder in self.save_tick_array_recorders
                if self._save_tick_store_due(recorder)
            ]
            self._num_save_ticks += 1
        self._sensor_writer.submit(
            tick_data,
            timestamp,
            ego_state,
            modalities,
            due_array_recorders,
            self.is_save_tick,
        )

    def _write_tick(
        self,
        tick_data: dict,
        timestamp: Timestamp,
        ego_state: EgoStateSE3,
        modalities: list[BaseModality],
        due_array_recorders: list,
        is_save_tick: bool,
    ) -> None:
        """Encode and write one tick; runs on the background write thread.

        Args:
            tick_data: Post-tick sensor data of the tick being written.
            timestamp: Simulation timestamp of the tick.
            ego_state: Ego state extracted on the driving thread.
            modalities: Modalities already recorded on the driving thread.
            due_array_recorders: Array recorders whose storage frequency is
                met this tick.
            is_save_tick: Whether this tick carries the save-tick modalities.
        """
        for recorder in due_array_recorders:
            modalities.extend(recorder.record(tick_data, timestamp, ego_state))
        for modality in modalities:
            self.py123d_log_writer.write_async(modality)

        if is_save_tick and self._writes_py123d_perturbated:
            perturbated_modalities = [ego_state]
            for recorder in self.perturbated_modality_recorders:
                perturbated_modalities.extend(
                    recorder.record(tick_data, timestamp, ego_state),
                )
            self.py123d_perturbated_log_writer.write_sync(
                ModalitiesSync(
                    timestamp=timestamp,
                    modalities=perturbated_modalities,
                ),
            )

    def destroy(self, results: RouteRecord | None = None) -> None:
        """
        Save the collected data and statistics to files, and clean up the data structures.
        This method should be called at the end of the data collection process.

        Args:
            results: Route record of the run; already stored in the leaderboard
                checkpoint under ``<data_root>/results``, so not written here.
        """
        if hasattr(self, "_sensor_writer"):
            self._sensor_writer.close()

        if self._writes_py123d and hasattr(self, "py123d_log_writer"):
            LOG.info(f"Closing 123D log writer: {self.py123d_logs_root}")
            self.py123d_log_writer.close()
        if self._writes_py123d_perturbated and hasattr(
            self,
            "py123d_perturbated_log_writer",
        ):
            self.py123d_perturbated_log_writer.close()
