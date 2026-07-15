import logging
import typing

import carla
import matplotlib
import numpy as np
import torch

from lead.common.logging import setup_logging
from lead.evaluation.agents.abstract_driving_agent import AbstractDrivingAgent
from lead.evaluation.inference.trackers import PathSpeedTracker, WaypointTracker
from lead.policy.transfuser.transfuser import AgentPrediction, Prediction
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
    """Driving agent wrapping the TransFuser policy.

    The abstract agent assembles the frame and the policy featurizes it; this
    agent only tracks the model's predicted plans into vehicle controls.
    """

    def setup_policy(self, checkpoint_dir: str) -> None:
        """Build the trackers turning the predicted plans into controls.

        Args:
            checkpoint_dir: Directory holding the trained model checkpoint.
        """
        self.waypoint_tracker = WaypointTracker(self.lead_config)
        self.path_speed_tracker = PathSpeedTracker(self.lead_config)

    def compute_control(
        self,
        prediction: Prediction,
        features: dict[str, typing.Any],
    ) -> carla.VehicleControl:
        """Track the TransFuser plans into a vehicle control.

        Args:
            prediction: Raw prediction of the TransFuser policy.
            features: The batched model inputs the prediction was computed on.

        Returns:
            The vehicle control to apply this step.
        """
        self.agent_prediction = self._build_agent_prediction(prediction, features)
        return carla.VehicleControl(
            steer=float(self.agent_prediction.steer),
            throttle=float(self.agent_prediction.throttle),
            brake=float(self.agent_prediction.brake),
        )

    def _post_process_target_speed(
        self,
        prediction: Prediction,
    ) -> torch.Tensor | None:
        """Apply the test-time brake threshold and speed lowering to the target speed.

        Args:
            prediction: Raw predictions of the model.

        Returns:
            The target speed to track, or None if the model does not predict it.
        """
        if (
            prediction.pred_target_speed_scalar is None
            or prediction.pred_target_speed_distribution is None
        ):
            return None

        target_speed = prediction.pred_target_speed_scalar.reshape(1, 1)
        target_speed_distribution = torch.softmax(
            prediction.pred_target_speed_distribution.float(),
            dim=-1,
        )
        if (
            target_speed_distribution[0, 0]
            > self.lead_config.evaluation.inference.brake_threshold
        ):  # Brake if we are confident enough.
            target_speed = torch.zeros_like(target_speed)
        if self.lead_config.evaluation.inference.lower_target_speed:
            target_speed = (
                target_speed
                * self.lead_config.evaluation.inference.lower_target_speed_factor
            )
        return target_speed

    def _build_agent_prediction(
        self,
        prediction: Prediction,
        features: dict[str, typing.Any],
    ) -> AgentPrediction:
        """Track the predicted plans and select controls per the modality knobs.

        Args:
            prediction: Raw predictions of the model.
            features: The batched model inputs the prediction was computed on.

        Returns:
            The model prediction extended with the vehicle controls.
        """
        ego_speed = features["speed"].unsqueeze(1)
        target_speed = self._post_process_target_speed(prediction)

        steer = throttle = brake = waypoints_steer = waypoints_throttle = (
            waypoints_brake
        ) = route_steer = target_speed_throttle = target_speed_brake = None

        if prediction.pred_route is not None and target_speed is not None:
            route_steer, target_speed_throttle, target_speed_brake = (
                self.path_speed_tracker.step(
                    prediction.pred_route,
                    target_speed,
                    ego_speed,
                )
            )
        if prediction.pred_future_waypoints is not None:
            waypoints_steer, waypoints_throttle, waypoints_brake = (
                self.waypoint_tracker.step(
                    prediction.pred_future_waypoints,
                    ego_speed,
                )
            )

        # Select which high-level modality we want to use for each control
        steer_modality = self.lead_config.evaluation.inference.steer_modality
        if steer_modality == "route":
            steer = route_steer
        elif steer_modality == "waypoint":
            steer = waypoints_steer
        else:
            raise ValueError(f"Invalid steer_modality: {steer_modality}")
        if steer is None:
            raise ValueError(
                f"steer_modality is {steer_modality!r} but the model did not predict it",
            )

        throttle_modality = self.lead_config.evaluation.inference.throttle_modality
        if throttle_modality == "target_speed":
            throttle = target_speed_throttle
        elif throttle_modality == "waypoint":
            throttle = waypoints_throttle
        else:
            raise ValueError(f"Invalid throttle_modality: {throttle_modality}")
        if throttle is None:
            raise ValueError(
                f"throttle_modality is {throttle_modality!r} but the model did not predict it",
            )

        brake_modality = self.lead_config.evaluation.inference.brake_modality
        if brake_modality == "target_speed":
            brake = target_speed_brake
        elif brake_modality == "waypoint":
            brake = waypoints_brake
        else:
            raise ValueError(f"Invalid brake_modality: {brake_modality}")
        if brake is None:
            raise ValueError(
                f"brake_modality is {brake_modality!r} but the model did not predict it",
            )

        # Turn off throttle if we brake
        if brake > 0.0:
            throttle = 0.0
            if (
                ego_speed < 0.01
            ):  # When we don't move we don't want the angle error to accumulate in the integral
                steer = 0.0

        return AgentPrediction(
            prediction=prediction,
            target_speed=target_speed,
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
        evaluation = self.lead_config.evaluation
        if evaluation.save_path is None:
            return
        if self.step % evaluation.produce_frame_frequency != 0:
            return

        # Visualization of prediction for debugging and video recording
        self.features.update(
            {
                "steer": torch.Tensor([self.control.steer]),
                "throttle": torch.Tensor([self.control.throttle]),
                "brake": torch.Tensor([self.control.brake]).bool(),
                "meters_travelled": torch.Tensor([self.meters_travelled]),
            },
        )

        if not hasattr(self, "video_recorder"):
            return

        # The uncompressed cameras of this tick, as the simulator produced them.
        input_image = np.concatenate(
            [
                input_data[f"rgb_{camera_idx}"]
                for camera_idx in range(
                    1,
                    self.lead_config.expert.sensor_rig.num_cameras + 1,
                )
            ],
            axis=1,
        )
        self.video_recorder.save_input_image(input_image)
        self.video_recorder.save_input_video_frame(input_image)

        # Get predicted route and waypoints (if available)
        pred_waypoints = (
            self.agent_prediction.prediction.pred_future_waypoints[0]
            if self.agent_prediction.prediction.pred_future_waypoints is not None
            else None
        )
        target_points = {
            "previous": self.features["target_point_previous"][0].cpu().numpy(),
            "current": self.features["target_point"][0].cpu().numpy(),
            "next": self.features["target_point_next"][0].cpu().numpy(),
        }
        self.video_recorder.save_demo_cameras(pred_waypoints, target_points)
        self.video_recorder.save_grid_image_and_video(
            pred_waypoints=pred_waypoints,
            target_points=target_points,
        )

        # Save abstract debug images
        if evaluation.produce_debug_video or evaluation.produce_debug_image:
            image = AgentPredictionVisualizer(
                lead_config=self.lead_config,
                data=self.features,
                prediction=self.agent_prediction,
            ).visualize()
            image = np.array(image).astype(np.uint8)
            self.video_recorder.save_debug_video_frame(image)
            self.video_recorder.save_debug_image(image)


if __name__ == "__main__":
    transfuser_agent = TransfuserAgent()
