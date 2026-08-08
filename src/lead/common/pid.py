"""Vehicle controllers: PID steering and linear-regression longitudinal control."""

from collections import deque

import jaxtyping as jt
import numpy as np
import numpy.typing as npt

from lead.config import ExpertConfig


class PIDController:
    """Classical PID controller for general control applications.

    Implements proportional-integral-derivative control with a sliding
    window for error history management.
    """

    def __init__(
        self,
        k_p: float = 1.0,
        k_i: float = 0.0,
        k_d: float = 0.0,
        error_window_size: int = 20,
    ) -> None:
        """Initialize the PID controller with gain parameters.

        Args:
            k_p: Proportional gain coefficient.
            k_i: Integral gain coefficient.
            k_d: Derivative gain coefficient.
            error_window_size: Size of the sliding window for error history.
        """
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d

        self._window: deque[float] = deque(
            [0.0 for _ in range(error_window_size)],
            maxlen=error_window_size,
        )

    def step(self, error: float) -> float:
        """Compute the PID control output for the given error.

        Args:
            error: Current error value.

        Returns:
            PID control output value.
        """
        self._window.append(error)
        if len(self._window) >= 2:
            integral = sum(self._window) / len(self._window)
            derivative = self._window[-1] - self._window[-2]
        else:
            integral = 0.0
            derivative = 0.0

        return self.k_p * error + self.k_i * integral + self.k_d * derivative


class LateralPIDController:
    """PID controller steering the vehicle along a route.

    The lookahead distance adapts to speed; the expert runs it on its dense
    route (10cm point spacing) and, with ``inference_mode=True``, the learned
    agent runs it on predicted checkpoints (1m spacing).
    """

    def __init__(self, config: ExpertConfig) -> None:
        """Initialize the lateral PID controller.

        Args:
            config: expert configuration containing PID parameters.
        """
        self.config = config

        self.lateral_pid_kp = self.config.pid.lateral_pid_kp
        self.lateral_pid_kd = self.config.pid.lateral_pid_kd
        self.lateral_pid_ki = self.config.pid.lateral_pid_ki

        self.lateral_pid_speed_scale = self.config.pid.lateral_pid_speed_scale
        self.lateral_pid_speed_offset = self.config.pid.lateral_pid_speed_offset

        self.lateral_pid_window_size = self.config.pid.lateral_pid_window_size
        self.lateral_pid_minimum_lookahead_distance = (
            self.config.pid.lateral_pid_minimum_lookahead_distance
        )
        self.lateral_pid_maximum_lookahead_distance = (
            self.config.pid.lateral_pid_maximum_lookahead_distance
        )

        # The following lists are used as deques
        self.error_history = []  # Sliding window to store past errors

    def step(
        self,
        route_points: jt.Float[npt.NDArray, "n 2"],
        current_speed: float,
        vehicle_position: jt.Float[npt.NDArray, " 2"],
        vehicle_heading: float,
        inference_mode: bool = False,
    ) -> float:
        """Compute steering angle based on route following.

        Args:
            route_points: Array of (x, y) coordinates representing the route.
            current_speed: Current speed of the vehicle in m/s.
            vehicle_position: Array of (x, y) coordinates representing vehicle position.
            vehicle_heading: Current heading angle of the vehicle in radians.
            inference_mode: Controls whether TF or expert executes this method.

        Returns:
            Computed steering angle in the range [-1.0, 1.0].
        """
        current_speed_kph = current_speed * 3.6  # Convert speed from m/s to km/h

        # Compute the lookahead distance based on the current speed
        lookahead_distance = (
            self.lateral_pid_speed_scale * current_speed_kph
            + self.lateral_pid_speed_offset
        )
        lookahead_distance = np.clip(
            lookahead_distance,
            self.lateral_pid_minimum_lookahead_distance,
            self.lateral_pid_maximum_lookahead_distance,
        )
        if inference_mode:
            # Transfuser predicts checkpoints 1m apart, whereas in the expert the route points have distance 10cm.
            lookahead_distance = (
                lookahead_distance / self.config.simulation.points_per_meter - 2
            )

        lookahead_distance = int(min(lookahead_distance, route_points.shape[0] - 1))

        # Calculate the desired heading vector from the lookahead point
        desired_heading_vec = route_points[lookahead_distance] - vehicle_position
        desired_heading_angle = np.arctan2(
            desired_heading_vec[1],
            desired_heading_vec[0],
        )

        # Calculate the heading error
        heading_error = (desired_heading_angle - vehicle_heading) % (2 * np.pi)
        heading_error = (
            heading_error if heading_error < np.pi else heading_error - 2 * np.pi
        )

        # Scale the heading error (leftover from a previous implementation)
        heading_error = heading_error * 180.0 / np.pi / 90.0

        # Update the error history. Only use the last lateral_pid_window_size errors like in a deque.
        self.error_history.append(heading_error)
        self.error_history = self.error_history[-self.lateral_pid_window_size :]

        # Calculate the derivative and integral terms
        derivative = (
            0.0
            if len(self.error_history) == 1
            else self.error_history[-1] - self.error_history[-2]
        )
        integral = np.mean(self.error_history)

        # Compute the steering angle using the PID control law
        return np.clip(
            self.lateral_pid_kp * heading_error
            + self.lateral_pid_kd * derivative
            + self.lateral_pid_ki * integral,
            -1.0,
            1.0,
        ).item()


class LongitudinalController:
    """Linear regression-based longitudinal controller.

    Implements speed control using a linear regression model to determine
    optimal throttle and brake values based on speed error and current speed.
    """

    def __init__(self, config: ExpertConfig) -> None:
        """Initialize the longitudinal controller.

        Args:
            config: expert configuration containing regression parameters.
        """
        self.config = config
        self.minimum_target_speed = (
            self.config.pid.longitudinal_linear_regression_minimum_target_speed
        )
        self.params = self.config.pid.longitudinal_linear_regression_params
        self.maximum_acceleration = (
            self.config.pid.longitudinal_linear_regression_maximum_acceleration
        )
        self.maximum_deceleration = (
            self.config.pid.longitudinal_linear_regression_maximum_deceleration
        )

    def get_throttle_and_brake(
        self,
        hazard_brake: bool,
        target_speed: float,
        current_speed: float,
    ) -> tuple[float, bool]:
        """Get throttle and brake values using linear regression model.

        Args:
            hazard_brake: Flag indicating whether to apply hazard braking.
            target_speed: The desired target speed in m/s.
            current_speed: The current speed of the vehicle in m/s.

        Returns:
            A tuple containing the throttle and brake values.
        """
        if target_speed < 1e-5 or hazard_brake:
            return 0.0, True
        if target_speed < self.minimum_target_speed:  # Avoid very small target speeds
            target_speed = self.minimum_target_speed

        current_speed = current_speed * 3.6
        target_speed = target_speed * 3.6
        params = self.params
        speed_error = target_speed - current_speed

        # Maximum acceleration 1.9 m/tick
        if speed_error > self.maximum_acceleration:
            return 1.0, False

        if current_speed / target_speed > params[-1] or hazard_brake:
            throttle, control_brake = 0.0, True
            return throttle, control_brake

        speed_error_cl = np.clip(speed_error, 0.0, np.inf) / 100.0
        current_speed /= 100.0
        features = np.array(
            [
                current_speed,
                current_speed**2,
                100 * speed_error_cl,
                speed_error_cl**2,
                current_speed * speed_error_cl,
                current_speed**2 * speed_error_cl,
            ],
        )

        throttle, control_brake = np.clip(features @ params[:-1], 0.0, 1.0), False

        return float(throttle), control_brake

    def get_throttle_extrapolation(
        self,
        target_speed: float,
        current_speed: float,
    ) -> float:
        """Get throttle value for forecasting purposes.

        Computes throttle assuming no hazard brake condition, used for
        trajectory forecasting and planning.

        Args:
            target_speed: The desired target speed in m/s.
            current_speed: The current speed of the vehicle in m/s.

        Returns:
            The throttle value in range [0, 1].
        """
        current_speed = current_speed * 3.6  # Conversion to km/h
        target_speed = target_speed * 3.6  # Conversion to km/h
        params = self.params
        speed_error = target_speed - current_speed

        # Maximum acceleration 1.9 m/tick
        if speed_error > self.maximum_acceleration:
            return 1.0
        # Maximum deceleration -4.82 m/tick
        if speed_error < self.maximum_deceleration:
            return 0.0

        throttle = 0.0
        # 0.1 to ensure small distances are overcome fast
        if target_speed < 0.1 or current_speed / target_speed > params[-1]:
            return throttle

        speed_error_cl = (
            np.clip(speed_error, 0.0, np.inf) / 100.0
        )  # The scaling is a leftover from the optimization
        current_speed /= 100.0  # The scaling is a leftover from the optimization
        features = np.array(
            [
                current_speed,
                current_speed**2,
                100 * speed_error_cl,
                speed_error_cl**2,
                current_speed * speed_error_cl,
                current_speed**2 * speed_error_cl,
            ],
        ).flatten()

        return float(np.clip(features @ params[:-1], 0.0, 1.0))
