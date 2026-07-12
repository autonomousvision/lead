import logging
import os
from collections import deque
from copy import deepcopy

import carla
import cv2
import jaxtyping as jt
import matplotlib
import numpy as np
import numpy.typing as npt
import torch

from lead.common import common_utils
from lead.common.constants import TransfuserBoundingBoxClass
from lead.common.logging_config import setup_logging
from lead.common.planning import RoutePlanner
from lead.config import LeadConfig
from lead.evaluation.abstract_driving_agent import AbstractDrivingAgent
from lead.evaluation.inference.ensemble import (
    AgentPrediction,
    Ensemble,
    EnsemblePrediction,
)
from lead.evaluation.inference.trackers import PathSpeedTracker, WaypointTracker
from lead.policy.transfuser.dataloader import dataset_utils as carla_dataset_utils
from lead.policy.transfuser.dataloader.dataset_utils import rasterize_lidar
from lead.policy.transfuser.visualization.agent_prediction_visualizer import (
    AgentPredictionVisualizer,
)

matplotlib.use("Agg")  # non-GUI backend for headless servers

setup_logging()
LOG = logging.getLogger(__name__)

# Configure pytorch for maximum performance
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.allow_tf32 = True


def get_entry_point():  # dead: disable
    return "TransfuserAgent"


class TransfuserAgent(AbstractDrivingAgent):
    """Driving agent wrapping the TransFuser policy ensemble.

    Pre-processes the sensor data into TransFuser model inputs (target
    points, rasterized LiDAR, radar), tracks the ensemble's predicted plans
    into vehicle controls and applies stop-sign and anti-stuck
    post-processing heuristics on them.
    """

    def setup_policy(self, checkpoint_dir: str) -> None:
        """Load the model ensemble and build the trackers and control post-processors.

        Args:
            checkpoint_dir: Directory holding the trained model checkpoint(s).
        """
        self.ensemble = Ensemble(
            lead_config=self.lead_config,
            model_path=checkpoint_dir,
            device=self.device,
            prefix="model",
        )
        self.waypoint_tracker = WaypointTracker(self.lead_config)
        self.path_speed_tracker = PathSpeedTracker(self.lead_config)

        # Post-processing heuristics
        self.bb_buffer = deque(maxlen=1)
        self.stop_sign_post_processor = StopSignPostProcessor(
            lead_config=self.lead_config,
            bb_buffer=self.bb_buffer,
        )
        self.force_move_post_processor = ForceMovePostProcessor(
            lead_config=self.lead_config,
            lidar_queue=self.lidar_pc_queue,
        )

    def set_target_points(self, input_data: dict, pop_distance: float):
        """Defines local planning signals based on the input data.

        Args:
            input_data: The input data containing sensor information and state. Will be fed into model.
            pop_distance: Distance threshold to pop waypoints from the route planner.
        """
        planner: RoutePlanner = self.gps_waypoint_planners_dict[pop_distance]

        def transform(point: list[float]) -> jt.Float[npt.NDArray, " 2"]:
            # Use filtered or noisy position based on training config
            ego_position = (
                self.filtered_state[:2]
                if self.lead_config.evaluation.inference.use_kalman_filter
                else input_data["noisy_state"][:2]
            )
            return common_utils.inverse_conversion_2d(
                np.array(point),
                np.array(ego_position),
                self.compass,
            )

        next_target_points = [tp.tolist() for tp in planner.route]

        # Merge duplicate consecutive target points
        filtered_tp_list = []
        for pt in next_target_points:
            if (
                len(next_target_points) == 2
                or not filtered_tp_list
                or not np.allclose(pt[:2], filtered_tp_list[-1][:2])
            ):
                filtered_tp_list.append(pt)
        next_target_points = filtered_tp_list

        if len(next_target_points) > 2:
            input_data["target_point_next"] = transform(next_target_points[2][:2])
            input_data["target_point"] = transform(next_target_points[1][:2])
            input_data["target_point_previous"] = transform(next_target_points[0][:2])
        else:
            assert len(next_target_points) == 2
            input_data["target_point_next"] = transform(next_target_points[1][:2])
            input_data["target_point"] = transform(next_target_points[1][:2])
            input_data["target_point_previous"] = transform(next_target_points[0][:2])

    @torch.inference_mode()
    def tick(self, input_data: dict) -> dict:
        """Pre-processes sensor data"""
        input_data = super().tick(
            input_data,
            use_kalman_filter=self.lead_config.agent.transfuser.use_kalman_filter_for_gps,
        )

        # Simulate JPEG compression to avoid train-test mismatch
        rgb = input_data["rgb"]
        input_data["original_rgb"] = rgb.copy()
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        _, rgb = cv2.imencode(
            ".jpg",
            rgb,
            [
                int(cv2.IMWRITE_JPEG_QUALITY),
                self.lead_config.evaluation.inference.jpeg_quality,
            ],
        )
        rgb = cv2.imdecode(rgb, cv2.IMREAD_UNCHANGED)
        rgb = np.transpose(rgb, (2, 0, 1))
        input_data["rgb"] = rgb

        # Plan next target point and command.
        self.set_target_points(
            input_data,
            pop_distance=self.lead_config.evaluation.controller.route_planner_min_distance,
        )
        if self.lead_config.evaluation.controller.sensor_agent_pop_distance_adaptive:
            dense_points = (
                np.linalg.norm(
                    input_data["target_point"] - input_data["target_point_next"],
                )
                < 10.0
                and min(
                    np.linalg.norm(input_data["target_point_previous"]),
                    np.linalg.norm(input_data["target_point"]),
                )
                < 10.0
            )
            dense_points = dense_points or (
                np.linalg.norm(
                    input_data["target_point_previous"] - input_data["target_point"],
                )
                < 10.0
                and min(
                    np.linalg.norm(input_data["target_point_previous"]),
                    np.linalg.norm(input_data["target_point"]),
                )
                < 10.0
            )
            if dense_points:
                self.set_target_points(input_data, pop_distance=4.0)

        # Ignore the next target point if it's too far away
        if (
            self.lead_config.evaluation.controller.sensor_agent_skip_distant_target_point
            and np.linalg.norm(input_data["target_point_next"])
            > self.lead_config.evaluation.controller.sensor_agent_skip_distant_target_point_threshold
        ):
            # Skip the next target point if it's too far away
            input_data["target_point_next"] = input_data["target_point"]

        # Lidar input
        lidar = self.accumulate_lidar()
        # Use only part of the lidar history we trained on
        lidar = lidar[
            lidar[:, -1] < self.lead_config.training.data.training_used_lidar_steps
        ]

        # At inference time, simulate laspy quantization to avoid train-test mismatch
        lidar[:, 0] = (
            np.round(
                lidar[:, 0] / self.lead_config.expert.storage.point_precision_x,
            )
            * self.lead_config.expert.storage.point_precision_x
        )
        lidar[:, 1] = (
            np.round(
                lidar[:, 1] / self.lead_config.expert.storage.point_precision_y,
            )
            * self.lead_config.expert.storage.point_precision_y
        )
        lidar[:, 2] = (
            np.round(
                lidar[:, 2] / self.lead_config.expert.storage.point_precision_z,
            )
            * self.lead_config.expert.storage.point_precision_z
        )

        # Convert to pseudo image, shaped (1, 1, H, W) as the data loader provides it
        input_data["rasterized_lidar"] = rasterize_lidar(
            lead_config=self.lead_config,
            lidar=lidar[:, :3],
        )[None, None]

        # Radar input preprocessing
        if self.lead_config.expert.sensor_rig.use_radars:
            # Preprocess radar input using the same function as during training
            input_data["radar"] = np.concatenate(
                carla_dataset_utils.preprocess_radar_input(
                    self.lead_config,
                    input_data,
                ),
                axis=0,
            )

        return input_data

    def compute_control(self, input_data: dict) -> carla.VehicleControl:
        """Run the TransFuser ensemble and post-process its controls.

        Args:
            input_data: Sensor data processed by :meth:`tick`.

        Returns:
            The vehicle control to apply this step.
        """
        # Transform the data into torch tensor comforting with data loader's format.
        self.input_data_tensors = {
            "rgb": torch.Tensor(input_data["rgb"]).to(self.device, dtype=torch.float32)[
                None
            ],
            "rasterized_lidar": torch.Tensor(input_data["rasterized_lidar"]).to(
                self.device,
                dtype=torch.float32,
            ),
            "target_point_previous": torch.Tensor(input_data["target_point_previous"])
            .to(self.device, dtype=torch.float32)
            .view(1, 2),
            "target_point": torch.Tensor(input_data["target_point"])
            .to(self.device, dtype=torch.float32)
            .view(1, 2),
            "target_point_next": (
                torch.Tensor(input_data["target_point_next"]).to(
                    self.device,
                    dtype=torch.float32,
                )
            ).view(1, 2),
            "speed": torch.Tensor([input_data["speed"]])
            .to(self.device, dtype=torch.float32)
            .view(1),
            "town": np.array([self._world.get_map().name.split("/")[-1]]),
        }

        # Add radar data if available
        if self.lead_config.expert.sensor_rig.use_radars and "radar" in input_data:
            self.input_data_tensors["radar"] = torch.Tensor(input_data["radar"]).to(
                self.device,
                dtype=torch.float32,
            )[None]

        # Save input log if need
        if (
            self.lead_config.evaluation.save_path is not None
            and self.lead_config.evaluation.produce_input_log
        ):
            torch.save(
                {
                    k: v.to(torch.device("cpu")) if isinstance(v, torch.Tensor) else v
                    for k, v in self.input_data_tensors.items()
                },
                os.path.join(
                    self.lead_config.evaluation.input_log_path,
                    str(self.step).zfill(5),
                )
                + ".pth",
            )

        # Forward pass
        ensemble_prediction = self.ensemble.forward(data=self.input_data_tensors)
        agent_prediction = self._build_agent_prediction(ensemble_prediction)

        # Update bounding boxes
        if (
            agent_prediction.pred_bounding_box_vehicle_system is not None
            and len(agent_prediction.pred_bounding_box_vehicle_system) > 0
        ):
            self.bb_buffer.append(
                agent_prediction.pred_bounding_box_vehicle_system,
            )

        # Post-processing heuristic
        self.stop_sign_post_processor.update_stop_box(
            self.ego_past_positions[-2][0],
            self.ego_past_positions[-2][1],
            self.ego_past_yaws[-2],
            0.0,
            0.0,
            0.0,
        )
        agent_prediction.throttle, agent_prediction.brake = (
            self.force_move_post_processor.adjust(
                input_data["speed"].item(),
                agent_prediction.throttle,
                agent_prediction.brake,
            )
        )
        agent_prediction.throttle, agent_prediction.brake = (
            self.stop_sign_post_processor.adjust(
                input_data["speed"].item(),
                agent_prediction.throttle,
                agent_prediction.brake,
            )
        )
        self.agent_prediction = agent_prediction

        return carla.VehicleControl(
            steer=float(agent_prediction.steer),
            throttle=float(agent_prediction.throttle),
            brake=float(agent_prediction.brake),
        )

    def _build_agent_prediction(
        self,
        ensemble_prediction: EnsemblePrediction,
    ) -> AgentPrediction:
        """Track the predicted plans and select controls per the modality knobs.

        Args:
            ensemble_prediction: Aggregated predictions of the model ensemble.

        Returns:
            The ensemble prediction extended with the vehicle controls.
        """
        ego_speed = self.input_data_tensors["speed"].unsqueeze(1)

        steer = throttle = brake = waypoints_steer = waypoints_throttle = (
            waypoints_brake
        ) = route_steer = target_speed_throttle = target_speed_brake = None

        if ensemble_prediction.pred_route is not None:
            route_steer, target_speed_throttle, target_speed_brake = (
                self.path_speed_tracker.step(
                    ensemble_prediction.pred_route,
                    ensemble_prediction.pred_target_speed_scalar,
                    ego_speed,
                )
            )
        if ensemble_prediction.pred_future_waypoints is not None:
            waypoints_steer, waypoints_throttle, waypoints_brake = (
                self.waypoint_tracker.step(
                    ensemble_prediction.pred_future_waypoints,
                    ego_speed,
                )
            )

        # Select which high-level modality we want to use for each control
        if self.lead_config.evaluation.inference.steer_modality == "route":
            steer = route_steer
        elif self.lead_config.evaluation.inference.steer_modality == "waypoint":
            steer = waypoints_steer
        else:
            raise ValueError(
                f"Invalid steer_modality: {self.lead_config.evaluation.inference.steer_modality}",
            )

        if self.lead_config.evaluation.inference.throttle_modality == "target_speed":
            throttle = target_speed_throttle
        elif self.lead_config.evaluation.inference.throttle_modality == "waypoint":
            throttle = waypoints_throttle
        else:
            raise ValueError(
                f"Invalid throttle_modality: {self.lead_config.evaluation.inference.throttle_modality}",
            )

        if self.lead_config.evaluation.inference.brake_modality == "target_speed":
            brake = target_speed_brake
        elif self.lead_config.evaluation.inference.brake_modality == "waypoint":
            brake = waypoints_brake
        else:
            raise ValueError(
                f"Invalid brake_modality: {self.lead_config.evaluation.inference.brake_modality}",
            )

        # Turn off throttle if we brake
        if brake > 0.0:
            throttle = 0.0
            if (
                ego_speed < 0.01
            ):  # When we don't move we don't want the angle error to accumulate in the integral
                steer = 0.0

        return AgentPrediction(
            **vars(ensemble_prediction),
            steer=steer,
            throttle=throttle,
            brake=brake,
            waypoints_steer=waypoints_steer,
            waypoints_throttle=waypoints_throttle,
            waypoints_brake=waypoints_brake,
            route_steer=route_steer,
            target_speed_throttle=target_speed_throttle,
            target_speed_brake=target_speed_brake,
        )

    def save_step_visualizations(self, input_data: dict) -> None:
        """Save the input, demo and debug images and videos of this step.

        Args:
            input_data: Sensor data processed by :meth:`tick`.
        """
        # Visualization of prediction for debugging and video recording
        self.input_data_tensors.update(
            {
                "steer": torch.Tensor([self.control.steer]),
                "throttle": torch.Tensor([self.control.throttle]),
                "brake": torch.Tensor([self.control.brake]).bool(),
                "distance_to_stop_sign": torch.Tensor(
                    [
                        self.stop_sign_post_processor.stop_sign_buffer[0].norm
                        if len(self.stop_sign_post_processor.stop_sign_buffer) > 0
                        else np.inf,
                    ],
                ),
                "stuck_detector": torch.Tensor(
                    [int(self.force_move_post_processor.stuck_detector)],
                ).int(),
                "force_move": torch.Tensor(
                    [int(self.force_move_post_processor.force_move)],
                ).int(),
                "meters_travelled": torch.Tensor([self.meters_travelled]),
            },
        )

        # Save input images as PNG and video
        if (
            self.lead_config.evaluation.save_path is not None
            and self.step % self.lead_config.evaluation.produce_frame_frequency == 0
        ):
            # Get the RGB image for visualization (before JPEG compression)
            input_image = input_data["original_rgb"].copy()
            # Save input image and video using VideoRecorder
            if hasattr(self, "video_recorder"):
                self.video_recorder.save_input_image(input_image)
                self.video_recorder.save_input_video_frame(input_image)

        # Save demo images
        if (
            self.lead_config.evaluation.save_path is not None
            and self.step % self.lead_config.evaluation.produce_frame_frequency == 0
        ):
            # Get predicted route and waypoints (if available)
            pred_waypoints = (
                self.agent_prediction.pred_future_waypoints[0]
                if self.agent_prediction.pred_future_waypoints is not None
                else None
            )

            # Prepare target points dictionary for BEV visualization
            target_points = {
                "previous": input_data.get("target_point_previous"),
                "current": input_data.get("target_point"),
                "next": input_data.get("target_point_next"),
            }

            # Save demo cameras with visualization using VideoRecorder
            if hasattr(self, "video_recorder"):
                self.video_recorder.save_demo_cameras(pred_waypoints, target_points)
                # Save grid (demo + input stacked vertically) with planning visualization
                self.video_recorder.save_grid_image_and_video(
                    pred_waypoints=pred_waypoints,
                    target_points=target_points,
                )

        # Save abstract debug images
        if (
            self.lead_config.evaluation.save_path is not None
            and (
                self.lead_config.evaluation.produce_debug_video
                or self.lead_config.evaluation.produce_debug_image
            )
            and self.step % self.lead_config.evaluation.produce_frame_frequency == 0
        ):
            # Produce image
            image = AgentPredictionVisualizer(
                lead_config=self.lead_config,
                data=self.input_data_tensors,
                prediction=self.agent_prediction,
            ).visualize()
            image = np.array(image).astype(np.uint8)

            # Save debug image and video using VideoRecorder
            if hasattr(self, "video_recorder"):
                self.video_recorder.save_debug_video_frame(image)
                self.video_recorder.save_debug_image(image)


class StopSignPostProcessor:
    """Heuristics to obey stop sign law."""

    def __init__(
        self,
        lead_config: LeadConfig,
        bb_buffer: deque,
    ) -> None:
        self.lead_config = lead_config
        self.bb_buffer = bb_buffer
        self.stop_sign_buffer: deque = deque(maxlen=1)
        self.clear_stop_sign_cool_down = 0  # Counter if we recently cleared a stop sign
        self.slower_stop_sign_count = 0
        self.slower_for_stop_sign_cool_down = 0

    def adjust(self, ego_speed: float, current_throttle: float, current_brake: float):
        """Checks whether the car is intersecting with one of the detected stop signs"""
        if (
            not self.lead_config.evaluation.heuristics.slower_for_stop_sign
            or len(self.bb_buffer) == 0
        ):
            # LOG.info("No bounding box")
            return current_throttle, current_brake

        if self.clear_stop_sign_cool_down > 0:
            self.clear_stop_sign_cool_down -= 1
        if self.slower_for_stop_sign_cool_down > 0:
            self.slower_for_stop_sign_cool_down -= 1
        stop_sign_stop_predicted = False

        for bb in self.bb_buffer[-1]:
            if bb.clazz == TransfuserBoundingBoxClass.STOP_SIGN:  # Stop sign detected
                # LOG.info("Stop sign detected.")
                self.stop_sign_buffer.append(bb)

        if len(self.stop_sign_buffer) > 0:
            # Check if we need to stop
            stop_box = self.stop_sign_buffer[0]
            stop_origin = carla.Location(x=stop_box.x, y=stop_box.y, z=0.0)
            stop_extent = carla.Vector3D(stop_box.w, stop_box.h, 1.0)
            stop_carla_box = carla.BoundingBox(stop_origin, stop_extent)
            stop_carla_box.rotation = carla.Rotation(0.0, np.rad2deg(stop_box.yaw), 0.0)

            stop_sign_distance = np.linalg.norm([stop_box.x, stop_box.y])
            boxes_intersect = (
                stop_sign_distance
                < self.lead_config.evaluation.heuristics.slower_for_stop_sign_dist_threshold
            )
            if boxes_intersect and self.clear_stop_sign_cool_down <= 0:
                if ego_speed > 0.01:
                    # LOG.info("Stop sign intersection detected.")
                    stop_sign_stop_predicted = True
                else:
                    # LOG.info("Stop sign intersection detected but car is already stopped.")
                    # We have cleared the stop sign
                    stop_sign_stop_predicted = False
                    self.stop_sign_buffer.pop()
                    # Stop signs don't come in herds, so we know we don't need to clear one for a while.
                    self.clear_stop_sign_cool_down = self.lead_config.evaluation.heuristics.slower_for_stop_sign_cool_down
                    self.slower_stop_sign_count = 0
            elif (
                self.slower_for_stop_sign_cool_down <= 0
                and stop_sign_distance
                < self.lead_config.evaluation.heuristics.slower_for_stop_sign_dist_threshold
            ):
                # LOG.info("Stop sign in range for slower.")
                self.slower_stop_sign_count = (
                    self.lead_config.evaluation.heuristics.slower_for_stop_sign_count
                )
                self.slower_for_stop_sign_cool_down = self.lead_config.evaluation.heuristics.slower_for_stop_sign_cool_down

        if len(self.stop_sign_buffer) > 0:
            # Remove boxes that are too far away
            if self.stop_sign_buffer[0].norm > abs(
                self.lead_config.expert.data_collection.max_x_meter,
            ):
                # LOG.info("Stop sign removed")
                self.stop_sign_buffer.pop()

        if stop_sign_stop_predicted:
            # LOG.info("Stopping for stop sign.")
            current_throttle = 0.0
            current_brake = True

        if (
            self.lead_config.evaluation.heuristics.slower_for_stop_sign
            and self.slower_stop_sign_count > 0
        ):
            # LOG.info("Slowing down for stop sign.")
            current_throttle = np.clip(
                current_throttle,
                0.0,
                self.lead_config.evaluation.heuristics.slower_for_stop_sign_throttle_threshold,
            )
            self.slower_stop_sign_count -= 1

        return current_throttle, current_brake

    def update_stop_box(
        self,
        x: float,
        y: float,
        orientation: float,
        x_target: float,
        y_target: float,
        orientation_target: float,
    ):
        if not self.lead_config.evaluation.heuristics.slower_for_stop_sign:
            return
        if len(self.stop_sign_buffer) != 0:
            self.stop_sign_buffer.append(
                self.stop_sign_buffer[0].update(
                    x,
                    y,
                    orientation,
                    x_target,
                    y_target,
                    orientation_target,
                ),
            )


class ForceMovePostProcessor:
    """Forces the agent to move after a certain time of being stuck."""

    def __init__(
        self,
        lead_config: LeadConfig,
        lidar_queue: deque,
    ) -> None:
        self.lead_config = lead_config
        self.stuck_detector = 0
        self.force_move = 0
        self.lidar_buffer = lidar_queue

    def adjust(
        self,
        ego_speed: float,
        current_throttle: float,
        current_brake: float,
    ) -> tuple[float, float]:
        if not self.lead_config.evaluation.heuristics.sensor_agent_creeping:
            return current_throttle, current_brake
        if (
            ego_speed < 0.1
        ):  # 0.1 is just an arbitrary low number to threshold when the car is stopped
            self.stuck_detector += 1
        else:
            self.stuck_detector = 0

        # If last red light was encountered a long time ago, we can assume it was cleared
        stuck_threshold = (
            self.lead_config.evaluation.heuristics.sensor_agent_stuck_threshold
        )

        if self.stuck_detector > stuck_threshold:
            self.force_move = (
                self.lead_config.evaluation.heuristics.sensor_agent_stuck_move_duration
            )

        if self.force_move > 0:
            emergency_stop = False
            # safety check
            safety_box = deepcopy(self.lidar_buffer[-1])

            # z-axis
            safety_box = safety_box[
                safety_box[..., 2] > self.lead_config.expert.simulation.safety_box_z_min
            ]
            safety_box = safety_box[
                safety_box[..., 2] < self.lead_config.expert.simulation.safety_box_z_max
            ]

            # y-axis
            safety_box = safety_box[
                safety_box[..., 1] > self.lead_config.expert.simulation.safety_box_y_min
            ]
            safety_box = safety_box[
                safety_box[..., 1] < self.lead_config.expert.simulation.safety_box_y_max
            ]

            # x-axis
            safety_box = safety_box[
                safety_box[..., 0] > self.lead_config.expert.simulation.safety_box_x_min
            ]
            safety_box = safety_box[
                safety_box[..., 0] < self.lead_config.expert.simulation.safety_box_x_max
            ]
            if len(safety_box) > 0:  # Checks if the List is empty
                emergency_stop = True
                LOG.info("Creeping overriden by safety box.")
            if not emergency_stop:
                LOG.info("Detected agent being stuck.")
                current_throttle = max(
                    self.lead_config.evaluation.heuristics.sensor_agent_stuck_throttle,
                    current_throttle,
                )
                current_brake = 0.0
                self.force_move -= 1
            else:
                LOG.info("Forced moving stopped by safety box.")
                current_throttle = 0.0
                current_brake = 1.0
                self.force_move = self.lead_config.evaluation.heuristics.sensor_agent_stuck_move_duration
        return current_throttle, current_brake


if __name__ == "__main__":
    transfuser_agent = TransfuserAgent()
