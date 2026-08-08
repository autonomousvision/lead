"""Abstract CARLA leaderboard agent wrapping a driving policy for evaluation."""

import abc
import json
import logging
import os
import shutil
import typing
from collections import deque

import carla
import cv2
import jaxtyping as jt
import numpy as np
import numpy.typing as npt
import torch
import yaml
from agents.navigation.local_planner import RoadOption
from leaderboard.autoagents import autonomous_agent
from py123d.datatypes import DynamicStateSE3, EgoStateSE3, Lidar, Timestamp
from py123d.datatypes.metadata.log_metadata import LogMetadata
from py123d.datatypes.sensors.base_camera import Camera
from py123d.datatypes.sensors.radar import Radar, RadarFeature
from py123d.geometry import PoseSE3, Vector3D
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from lead.api.abstract_policy import AbstractPolicy
from lead.api.point_cloud_transforms import (
    lidar_sweep_from_carla_ego_frame,
    radar_returns_from_carla_ego_frame,
)
from lead.api.py123d_log_api import (
    CAMERA_ID_BY_LEAD_INDEX,
    CARLA_LINCOLN_MKZ_2020_METADATA,
    RADAR_ID_BY_LEAD_INDEX,
    RADIAL_VELOCITY_FEATURE,
    ordered_target_points,
)
from lead.api.scene_data import SceneData
from lead.common import carla_to_123d, geometry
from lead.common.base_agent import BaseAgent, CarlaSensorData
from lead.common.planning import RoutePlanner
from lead.common.sensors import ransac
from lead.common.sensors.av_sensor_setup import SensorSpec, av_sensor_setup
from lead.config import load_lead_config
from lead.evaluation.inference.policy_runner import PolicyRunner
from lead.evaluation.recorder.infraction_recorder import InfractionRecorder
from lead.evaluation.recorder.video_recorder import VideoRecorder

LOG = logging.getLogger(__name__)


class AbstractDrivingAgent(BaseAgent, autonomous_agent.AutonomousAgent, abc.ABC):
    """CARLA leaderboard protocol adapter around a driving policy.

    Assembles one :class:`~lead.api.scene_data.SceneData` per tick from the simulator's
    sensors and owns everything policy-agnostic: config, sensor rig, infraction and
    video recording, the :meth:`run_step` skeleton. Subclasses supply the policy and
    turn its predictions into a control (:meth:`setup_policy`, :meth:`compute_control`).
    """

    # The leaderboard entry point deliberately shadows `BaseAgent.setup`'s
    # internal (lead_config, sensor_agent) signature.
    def setup(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        path_to_conf_file: str,
        route_index: str | None = None,
        traffic_manager: carla.TrafficManager | None = None,
    ) -> None:
        self.config_path = path_to_conf_file
        # Bench2Drive appends "+..." to the checkpoint path.
        checkpoint_dir = path_to_conf_file.split("+")[0]

        # Rebuild the config tree saved during training; env/CLI overrides apply on top.
        with open(
            os.path.join(checkpoint_dir, "config.yaml"),
            encoding="utf-8",
        ) as f:
            stored_config = yaml.safe_load(f)
        # The stored tree describes how the model was trained; how it is
        # evaluated is decided by the running code and its overrides.
        stored_config.pop("evaluation", None)
        lead_config = load_lead_config(
            loaded_config=stored_config,
            raise_on_unknown_key=False,
        )

        self.step = -1
        self.initialized = False
        self.device = torch.device("cuda:0")

        # The policy named by ``policy.target``, loaded from the checkpoint.
        self.policy_runner = PolicyRunner(
            lead_config=lead_config,
            model_path=checkpoint_dir,
            device=self.device,
        )
        self.policy: AbstractPolicy = self.policy_runner.policy
        policy_config = self.policy.get_policy_config()

        super().setup(
            lead_config,
            sensor_agent=True,
            past_window_num_iterations=policy_config.past_window_num_iterations,
        )

        # Sweep history matching the temporal window the policy trains on; a
        # length of N ticks back means N + 1 buffered sweeps.
        self.lidar_sweep_queue: deque = deque(
            maxlen=policy_config.past_lidar_num_iterations + 1,
        )
        self.radar_sweep_queue: deque = deque(
            maxlen=policy_config.past_radar_num_iterations + 1,
        )
        ransac.remove_ground(
            np.random.rand(1000, 3),
            lead_config.expert,
        )  # Pre-compile numba code

        # The rig's calibration, identical to what the logs are written with.
        self._camera_metadatas = carla_to_123d.build_pinhole_camera_metadatas(
            lead_config.expert,
            CARLA_LINCOLN_MKZ_2020_METADATA,
            perturbation_translation=0.0,
            perturbation_rotation=0.0,
        )
        self._lidar_metadata = carla_to_123d.build_lidar_metadata(lead_config.expert)
        self._radar_metadatas, self._merged_radar_metadata = (
            carla_to_123d.build_radar_metadatas(lead_config.expert)
        )

        self.setup_policy(checkpoint_dir)

        self.metric_info = {}
        self.meters_travelled = 0.0

        # Infraction tracking
        self.infraction_recorder = InfractionRecorder(
            config_evaluation=lead_config.evaluation,
            agent_name=type(self).__name__,
        )

        self.track = autonomous_agent.Track.SENSORS

        if not shutil.which("ffmpeg"):
            raise RuntimeError(
                "ffmpeg is not installed or not found in PATH. Please install ffmpeg to use video compression.",
            )

    def setup_policy(self, checkpoint_dir: str) -> None:
        """Build what driving the wrapped policy needs beyond the policy itself, e.g. trackers.

        Called once during :meth:`setup`, after ``self.policy`` is loaded; no-op by default.

        Args:
            checkpoint_dir: Directory holding the trained model checkpoint(s).
        """

    @abc.abstractmethod
    def compute_control(
        self,
        prediction: typing.Any,
        features: dict[str, typing.Any],
    ) -> carla.VehicleControl:
        """Turn the policy's prediction for this tick into a vehicle control.

        Args:
            prediction: Raw prediction of the wrapped policy.
            features: The batched model inputs the prediction was computed on.

        Returns:
            The vehicle control to apply this step.
        """
        raise NotImplementedError

    def save_step_visualizations(self, sensor_data: CarlaSensorData) -> None:
        """Save per-step visualizations of the last control computation; no-op by default.

        Args:
            sensor_data: Sensor data processed by :meth:`tick`.
        """

    def save_input_log(self) -> None:
        """Dump this step's model inputs when input logging is enabled."""
        evaluation = self.lead_config.evaluation
        if evaluation.save_path is None or not evaluation.produce_input_log:
            return
        torch.save(
            {
                key: value.cpu() if isinstance(value, torch.Tensor) else value
                for key, value in self.features.items()
            },
            os.path.join(
                evaluation.input_log_path,
                f"{str(self.step).zfill(5)}.pth",
            ),
        )

    def tick(self, sensor_data: CarlaSensorData) -> CarlaSensorData:
        """Pre-process the tick's sensors and record the sweep history.

        Args:
            sensor_data: Raw sensor data provided by the leaderboard.

        Returns:
            The pre-processed sensor data.
        """
        sensor_data = super().tick(sensor_data)

        # The scene data carries the per-tick sweeps as the logs store them; the
        # policy's featurization owns the merging and ground removal.
        if self.lidar_sweep_queue.maxlen:
            self.lidar_sweep_queue.append(self._lidar_sweep(sensor_data))
        if (
            self.radar_sweep_queue.maxlen
            and self.lead_config.expert.sensor_rig.use_radars
        ):
            self.radar_sweep_queue.append(self._merged_radar_sweep(sensor_data))
        return sensor_data

    def _lidar_sweep(self, sensor_data: CarlaSensorData) -> Lidar:
        """The tick's lidar sweep in the 123D IMU frame, as the modality the logs store."""
        points = self._quantize(sensor_data["lidar_sweep"])
        return Lidar(
            timestamp=self._timestamp(),
            timestamp_end=self._timestamp(),
            metadata=self._lidar_metadata,
            point_cloud_3d=lidar_sweep_from_carla_ego_frame(
                points,
                self.lead_config,
            ).astype(np.float32),
        )

    def _quantize(
        self,
        points: jt.Float[npt.NDArray, "n 3"],
    ) -> jt.Float[npt.NDArray, "n 3"]:
        """Round a sweep to the precision the logs store it at.

        Training sweeps pass through the writer's laspy quantization; applying it here
        keeps the simulator input free of a train-test mismatch.
        """
        storage = self.lead_config.expert.storage
        precision = np.array(
            [
                storage.point_precision_x,
                storage.point_precision_y,
                storage.point_precision_z,
            ],
        )
        quantized = points.copy()
        quantized[:, :3] = np.round(quantized[:, :3] / precision) * precision
        return quantized

    def build_scene_data(self, sensor_data: CarlaSensorData) -> SceneData:
        """Assemble this tick's simulator data into the policy's input contract.

        Produces the same :class:`~lead.api.scene_data.SceneData` training reads from a
        123D log, minus the privileged fields the simulator cannot provide.

        Args:
            sensor_data: Sensor data pre-processed by :meth:`tick`.

        Returns:
            The scene data of this tick.
        """
        target_points = self._target_points()
        return SceneData(
            cameras=self._cameras(sensor_data),
            lidar_sweeps=dict(enumerate(reversed(self.lidar_sweep_queue))),
            radar_sweeps=(
                dict(enumerate(reversed(self.radar_sweep_queue)))
                if self.lead_config.expert.sensor_rig.use_radars
                else None
            ),
            ego_state=self._ego_state(sensor_data),
            log_metadata=LogMetadata(
                dataset=self.lead_config.expert.data_collection.py123d_dataset,
                split="",
                log_name="",
                location=self._world.get_map().name.split("/")[-1],
            ),
            past_ego_positions=np.array(self.ego_past_positions[::-1]),
            past_ego_yaws=np.array(self.ego_past_yaws[::-1]),
            previous_target_point=target_points["previous_target_point"],
            target_point=target_points["target_point"],
            next_target_point=target_points["next_target_point"],
        )

    def _ego_state(self, sensor_data: CarlaSensorData) -> EgoStateSE3:
        """The ego state the sensors provide: identity pose, speedometer velocity.

        The simulator's true pose is privileged, so the online ego state
        carries only what the featurization reads of it — the forward speed.
        """
        return EgoStateSE3.from_center(
            center_se3=PoseSE3.identity(),
            metadata=CARLA_LINCOLN_MKZ_2020_METADATA,
            timestamp=self._timestamp(),
            dynamic_state_se3=DynamicStateSE3(
                velocity=Vector3D(x=float(sensor_data["speed"]), y=0.0, z=0.0),
                acceleration=Vector3D(x=0.0, y=0.0, z=0.0),
                angular_velocity=Vector3D(x=0.0, y=0.0, z=0.0),
            ),
        )

    def _cameras(self, sensor_data: CarlaSensorData) -> list[Camera]:
        """The tick's cameras of the active policy, in stitch order, as 123D modalities.

        Each image passes the storage JPEG round-trip so the model sees the same
        compression artifacts as in training.
        """
        quality = self.lead_config.evaluation.inference.jpeg_quality
        timestamp = Timestamp.from_us(
            round(
                self.step * 1e6 / self.lead_config.expert.simulation.carla_fps,
            ),
        )
        lead_indices = {
            camera_id: index for index, camera_id in CAMERA_ID_BY_LEAD_INDEX.items()
        }

        cameras: list[Camera] = []
        for camera_id in self.policy.input_cameras:
            image = sensor_data[f"rgb_{lead_indices[camera_id]}"]
            _, encoded = cv2.imencode(
                ".jpg",
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                [int(cv2.IMWRITE_JPEG_QUALITY), quality],
            )
            metadata = self._camera_metadatas[camera_id]
            cameras.append(
                Camera(
                    metadata=metadata,
                    image=typing.cast(
                        "npt.NDArray[np.uint8]",
                        cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED),
                    ),
                    camera_to_global_se3=metadata.camera_to_imu_se3,
                    timestamp=timestamp,
                ),
            )
        return cameras

    def _timestamp(self) -> Timestamp:
        """The simulation timestamp of the current step."""
        return Timestamp.from_us(
            round(
                self.step * 1e6 / self.lead_config.expert.simulation.carla_fps,
            ),
        )

    def _radars(self, sensor_data: CarlaSensorData) -> list[Radar] | None:
        """The tick's radars in LEAD radar order, as 123D modalities."""
        if not self.lead_config.expert.sensor_rig.use_radars:
            return None

        timestamp = self._timestamp()
        radars: list[Radar] = []
        for radar_idx in range(
            1,
            self.lead_config.expert.sensor_rig.num_radar_sensors + 1,
        ):
            returns = sensor_data[f"radar{radar_idx}"]
            radars.append(
                Radar(
                    timestamp=timestamp,
                    metadata=self._radar_metadatas[RADAR_ID_BY_LEAD_INDEX[radar_idx]],
                    point_cloud_3d=radar_returns_from_carla_ego_frame(
                        returns[:, :3],
                    ).astype(np.float32),
                    point_cloud_features={
                        RADIAL_VELOCITY_FEATURE: returns[:, 3].astype(np.float32),
                    },
                ),
            )
        return radars

    def _merged_radar_sweep(self, sensor_data: CarlaSensorData) -> Radar:
        """The tick's radar returns of all sensors merged into one 123D radar, like the
        merged radar stream of the logs.
        """
        radars = self._radars(sensor_data) or []
        points = [radar.point_cloud_3d for radar in radars]
        velocities = []
        ids = []
        for radar_idx, radar in enumerate(radars, start=1):
            assert radar.point_cloud_features is not None
            velocities.append(radar.point_cloud_features[RADIAL_VELOCITY_FEATURE])
            ids.append(
                np.full(
                    radar.point_cloud_3d.shape[0],
                    int(RADAR_ID_BY_LEAD_INDEX[radar_idx].value),
                    dtype=np.uint8,
                ),
            )
        return Radar(
            timestamp=self._timestamp(),
            metadata=self._merged_radar_metadata,
            point_cloud_3d=(
                np.concatenate(points, axis=0).astype(np.float32)
                if points
                else np.zeros((0, 3), dtype=np.float32)
            ),
            point_cloud_features={
                RadarFeature.IDS.serialize(): (
                    np.concatenate(ids) if ids else np.zeros((0,), dtype=np.uint8)
                ),
                RADIAL_VELOCITY_FEATURE: (
                    np.concatenate(velocities)
                    if velocities
                    else np.zeros((0,), dtype=np.float32)
                ),
            },
        )

    def _target_points(self) -> dict[str, jt.Float[npt.NDArray, " 2"]]:
        """Plan the previous, current and next target point in the ego frame.

        The pop distance adapts when the route bunches target points together,
        and a far-away next point is dropped.
        """
        controller = self.lead_config.evaluation.controller
        points = self._plan_target_points(controller.route_planner_min_distance)

        if controller.sensor_agent_pop_distance_adaptive and self._points_are_dense(
            points,
        ):
            points = self._plan_target_points(4.0)

        # Ignore the next target point if it's too far away.
        if (
            controller.sensor_agent_skip_distant_target_point
            and np.linalg.norm(points["next_target_point"])
            > controller.sensor_agent_skip_distant_target_point_threshold
        ):
            points["next_target_point"] = points["target_point"]
        return points

    def _plan_target_points(
        self,
        pop_distance: float,
    ) -> dict[str, jt.Float[npt.NDArray, " 2"]]:
        """Read the target points of one route planner into the ego frame."""
        planner: RoutePlanner = self.gps_waypoint_planners_dict[pop_distance]
        previous_target_point, current_target_point, next_target_point = (
            ordered_target_points(
                planner.target_points,
                planner.target_point_index,
            )
        )

        compass = self.compass
        assert compass is not None, "tick() must run before build_scene_data()"

        def transform(
            point: jt.Float[npt.NDArray, " 3"],
        ) -> jt.Float[npt.NDArray, " 2"]:
            return geometry.to_local_frame_2d(
                point[:2],
                self.localized_position,
                compass,
            )

        return {
            "previous_target_point": transform(previous_target_point),
            "target_point": transform(current_target_point),
            "next_target_point": transform(next_target_point),
        }

    @staticmethod
    def _points_are_dense(points: dict[str, jt.Float[npt.NDArray, " 2"]]) -> bool:
        """Whether the route's target points bunch up around the ego."""
        previous_target_point = points["previous_target_point"]
        current_target_point = points["target_point"]
        next_target_point = points["next_target_point"]
        close_to_ego = (
            min(
                np.linalg.norm(previous_target_point),
                np.linalg.norm(current_target_point),
            )
            < 10.0
        )
        return bool(
            close_to_ego
            and (
                np.linalg.norm(current_target_point - next_target_point) < 10.0
                or np.linalg.norm(previous_target_point - current_target_point) < 10.0
            ),
        )

    def set_global_plan(
        self,
        global_plan_gps: list[tuple[dict[str, float], RoadOption]],
        privileged_org_dense_route_world_coord: list[
            tuple[carla.Transform, RoadOption]
        ],
    ) -> None:
        """Store the leaderboard's global plan before the scenario starts.

        The dense world-coordinate route is privileged information for offline logging
        and metrics and must not be used by the driving policy.

        Args:
            global_plan_gps: Global route waypoints in GPS space as provided by
                the leaderboard.
            privileged_org_dense_route_world_coord: Dense global route in world coordinates.
        """
        self.privileged_org_dense_route_world_coord = (
            privileged_org_dense_route_world_coord
        )
        LOG.info(
            "Set global plan with %d waypoints.",
            len(self.privileged_org_dense_route_world_coord),
        )
        super().set_global_plan(global_plan_gps, privileged_org_dense_route_world_coord)

    def set_scenario(self, scenario: typing.Any) -> None:
        """Set the scenario reference to track infractions; called by the leaderboard
        after loading the scenario."""
        self.infraction_recorder.set_scenario(scenario)
        LOG.info(
            "[%s] Scenario reference set for infraction tracking",
            type(self).__name__,
        )

    def _init(self) -> None:
        # Get the hero vehicle and the CARLA world
        self._vehicle: carla.Actor = CarlaDataProvider.get_hero_actor()
        self._world: carla.World = self._vehicle.get_world()

        # Set up video recorder
        self.video_recorder = VideoRecorder(
            config_evaluation=self.lead_config.evaluation,
            vehicle=self._vehicle,
            world=self._world,
            step_counter=self.step,
            lead_config=self.lead_config,
        )

        self.initialized = True

    def sensors(self) -> list[SensorSpec]:
        return av_sensor_setup(
            config=self.lead_config.expert,
            lidar=True,
            radar=True,
            sensor_agent=True,
            perturbate=False,
            perturbation_rotation=0.0,
            perturbation_translation=0.0,
        )

    def check_infractions(self) -> None:
        """Check and record infractions for the current rollout step."""
        self.infraction_recorder.check_infractions(
            step=self.step,
            meters_travelled=self.meters_travelled,
        )

    @torch.inference_mode()
    def run_step(
        self,
        sensor_data: CarlaSensorData,
        _,
        __=None,
    ) -> carla.VehicleControl:
        """Drive one simulation step: tick, run the policy, record the outcome.

        Args:
            sensor_data: Raw sensor data provided by the leaderboard.

        Returns:
            The vehicle control to apply this step.
        """
        self.step += 1

        if not self.initialized:
            self._init()
            self.control = carla.VehicleControl(steer=0.0, throttle=0.0, brake=1.0)
            sensor_data = self.tick(sensor_data)
            return self.control

        # Update video recorder step and demo cameras
        if hasattr(self, "video_recorder"):
            self.video_recorder.update_step(self.step)
            self.video_recorder.move_demo_cameras_with_ego()

        # Need to run this every step for GPS filtering
        sensor_data = self.tick(sensor_data)

        # One featurization path with training: the simulator's tick becomes
        # scene data, the policy turns it into its model inputs.
        scene_data = self.build_scene_data(sensor_data)
        self.features = self.policy.features_to_batch(
            self.policy.build_features(scene_data),
            self.device,
        )
        self.save_input_log()
        prediction = self.policy_runner.forward(self.features)
        self.control = self.compute_control(prediction, self.features)

        self.meters_travelled += (
            sensor_data["speed"].item()
            * self.lead_config.expert.simulation.carla_frame_rate
        )
        sensor_data["meters_travelled"] = self.meters_travelled

        # CARLA will not let the car drive in the initial frames. This help the filter not get confused.
        if self.step < self.lead_config.expert.simulation.inital_frames_delay:
            self.control = carla.VehicleControl(0.0, 0.0, 1.0)

        # Check for infractions at this step
        self.check_infractions()

        self.save_step_visualizations(sensor_data)

        # Save metric info if in Bench2Drive mode
        if self.lead_config.evaluation.is_bench2drive and hasattr(
            self,
            "get_metric_info",
        ):
            metric = self.get_metric_info()
            self.metric_info[self.step] = metric
            with open(
                f"{self.lead_config.evaluation.save_path}/metric_info.json",
                "w",
            ) as outfile:
                json.dump(self.metric_info, outfile, indent=4)
        return self.control

    def destroy(self, results: object = None) -> None:
        LOG.info(results)

        # Clean up video recorder
        if hasattr(self, "video_recorder"):
            self.video_recorder.cleanup()
