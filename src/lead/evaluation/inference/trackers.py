"""Trackers turning one predicted plan representation into vehicle controls."""

import jaxtyping as jt
import numpy as np
import torch

from lead.common.pid import (
    LateralPIDController,
    LongitudinalController,
    PIDController,
)
from lead.config import LeadConfig


class WaypointTracker:
    """Follows predicted spatio-temporal waypoints with PID controllers.

    The waypoint spacing encodes the desired speed; steering aims at the
    first waypoint beyond a speed-dependent aim distance.
    """

    def __init__(self, lead_config: LeadConfig) -> None:
        self.lead_config = lead_config
        self.lateral_controller = PIDController(
            k_p=lead_config.evaluation.controller.turn_kp,
            k_i=lead_config.evaluation.controller.turn_ki,
            k_d=lead_config.evaluation.controller.turn_kd,
            n=lead_config.evaluation.controller.turn_n,
        )
        self.longitudinal_controller = PIDController(
            k_p=lead_config.evaluation.controller.speed_kp,
            k_i=lead_config.evaluation.controller.speed_ki,
            k_d=lead_config.evaluation.controller.speed_kd,
            n=lead_config.evaluation.controller.speed_n,
        )

    def step(
        self,
        waypoints: jt.Float[torch.Tensor, "1 num_waypoints 2"],
        velocity: jt.Float[torch.Tensor, "1 1"],
    ) -> tuple[float, float, float]:
        """Compute vehicle controls from the predicted waypoints.

        Args:
            waypoints: Predicted future waypoints in ego-vehicle coordinates.
            velocity: Current speed of the vehicle in m/s.

        Returns:
            steer: Steering command in [-1, 1]
            throttle: Throttle command in [0, 1]
            brake: Brake command in [0, 1]
        """
        waypoints_np = waypoints[0].data.cpu().float().numpy()
        speed = float(velocity[0].data.cpu().float().numpy())

        one_second = int(
            self.lead_config.expert.simulation.carla_fps
            // self.lead_config.expert.data_collection.data_save_freq,
        )
        half_second = one_second // 2

        desired_speed = (
            np.linalg.norm(
                waypoints_np[half_second - 1] - waypoints_np[one_second - 1],
            )
            * 2.0
        )
        delta_speed = float(
            np.clip(
                desired_speed - speed,
                0.0,
                self.lead_config.evaluation.controller.wp_delta_clip,
            ),
        )

        brake = (
            desired_speed < self.lead_config.evaluation.controller.brake_speed
        ) or (
            (speed / desired_speed) > self.lead_config.evaluation.controller.brake_ratio
        )
        throttle = self.longitudinal_controller.step(delta_speed)
        throttle = throttle if not brake else 0.0

        if (
            self.lead_config.evaluation.controller.tuned_aim_distance
        ):  # In LB2, we go faster, so we need to choose waypoints farther ahead
            # range [2.4, 10.5] same as in the disentangled rep.
            aim_distance = np.clip(0.975532 * speed + 1.915288, 24, 105) / 10
        else:
            # To replicate the slow TransFuser behaviour we have a different distance
            # inside and outside of intersections (detected by desired_speed)
            if (
                desired_speed
                < self.lead_config.evaluation.controller.aim_distance_threshold
            ):
                aim_distance = self.lead_config.evaluation.controller.aim_distance_slow
            else:
                aim_distance = self.lead_config.evaluation.controller.aim_distance_fast

        # We follow the waypoint that is at least a certain distance away
        aim_index = waypoints_np.shape[0] - 1
        for index, predicted_waypoint in enumerate(waypoints_np):
            if np.linalg.norm(predicted_waypoint) >= aim_distance:
                aim_index = index
                break

        aim = waypoints_np[aim_index]
        angle = np.degrees(np.arctan2(aim[1], aim[0])) / 90.0
        if speed < 0.01:
            # When we don't move we don't want the angle error to accumulate in the integral
            angle = 0.0
        if brake:
            angle = 0.0

        steer = self.lateral_controller.step(angle)
        steer = np.clip(steer, -1.0, 1.0)  # Valid steering values are in [-1,1]
        return float(steer), float(throttle), float(brake)


class PathSpeedTracker:
    """Follows the predicted spatial path at the predicted target speed.

    Steering tracks the path checkpoints with a lateral PID controller;
    throttle and brake come from the predicted target speed.
    """

    def __init__(self, lead_config: LeadConfig) -> None:
        self.lead_config = lead_config
        self.lateral_controller = LateralPIDController(lead_config.expert)
        self.longitudinal_controller = LongitudinalController(lead_config.expert)

    def step(
        self,
        pred_checkpoints: jt.Float[torch.Tensor, "1 num_checkpoints 2"],
        pred_target_speed: jt.Float[torch.Tensor, "1 1"],
        speed: jt.Float[torch.Tensor, "1 1"],
        ego_vehicle_location: float = 0.0,
        ego_vehicle_rotation: float = 0.0,
    ) -> tuple[float, float, float]:
        """Compute vehicle controls from the predicted path and target speed.

        Args:
            pred_checkpoints: Predicted path checkpoints in ego-vehicle coordinates.
            pred_target_speed: Predicted target speed in m/s.
            speed: Current speed of the vehicle in m/s.
            ego_vehicle_location: Current lateral location of the ego vehicle.
            ego_vehicle_rotation: Current rotation of the ego vehicle.

        Returns:
            steer: Steering command in [-1, 1]
            throttle: Throttle command in [0, 1]
            brake: Brake command in [0, 1]
        """
        pred_checkpoints_np = pred_checkpoints[0].data.cpu().float().numpy()
        speed_val = float(speed[0].data.cpu().float().numpy())
        pred_target_speed_val = float(pred_target_speed[0].data.cpu().float().numpy())

        brake = bool(
            pred_target_speed_val < 0.01
            or (speed_val / pred_target_speed_val)
            > self.lead_config.evaluation.controller.brake_ratio,
        )
        steer = round(
            self.lateral_controller.step(
                pred_checkpoints_np,
                speed_val,
                np.full(2, ego_vehicle_location),
                ego_vehicle_rotation,
                inference_mode=True,
            ),
            3,
        )
        throttle, brake = self.longitudinal_controller.get_throttle_and_brake(
            brake,
            pred_target_speed_val,
            speed_val,
        )

        return steer, throttle, float(brake)
