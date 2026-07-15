import math
from collections import deque

import carla
import jaxtyping as jt
import numpy as np
import numpy.typing as npt
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter as UKF

import lead.common.geometry as geometry
from lead.config import ExpertConfig


class KalmanFilter:
    """Unscented Kalman Filter for less noisy GPS localization."""

    def __init__(self, config: ExpertConfig) -> None:
        """Constructor.

        Args:
            config: Object containing the configuration parameters.
        """
        self.config = config
        self.points = MerweScaledSigmaPoints(
            n=4,
            alpha=0.00001,
            beta=2,
            kappa=0,
            subtract=self._residual_state_x,
        )

        self.ukf = UKF(
            dim_x=4,
            dim_z=4,
            fx=self._bicycle_model_forward,
            hx=self._measurement_function_hx,
            dt=self.config.simulation.carla_frame_rate,
            points=self.points,
            x_mean_fn=self._state_mean,
            z_mean_fn=self._measurement_mean,
            residual_x=self._residual_state_x,
            residual_z=self._residual_measurement_h,
        )

        # State noise, same as measurement because we
        # initialize with the first measurement later
        self.ukf.P = np.diag([0.5, 0.5, 0.000001, 0.000001])
        # Measurement noise
        self.ukf.R = np.diag([0.5, 0.5, 0.000000000000001, 0.000000000000001])
        self.ukf.Q = np.diag([0.0001, 0.0001, 0.001, 0.001])  # Model noise
        # Used to set the filter state equal the first measurement
        self.filter_initialized = False

        maxlen = self.config.data_collection.ego_history_length

        self.history_x = deque(maxlen=maxlen)  # filtered states

        # Scaling factors to avoid working with large numbers
        self.start_x = None
        self.start_y = None

    def step(
        self,
        noisy_position: npt.NDArray,
        compass: float,
        speed: float,
        control: carla.VehicleControl,
    ) -> npt.NDArray[np.floating]:
        """Performs one iteration of predict and update of the UKF.

        Args:
            noisy_position: Carla coordinates in meters.
            compass: Unbounded compass angle in radians w.r.t world frame.
            speed: Speed in m/s.
            control: Object containing the last control command.
        Returns:
            npt.NDArray[np.floating]: The filtered state [x, y, angle, speed].
        """
        if self.start_x is None:
            self.start_x = noisy_position[0]
            self.start_y = noisy_position[1]

        # Create scale state
        z = np.array(
            [
                noisy_position[0] - self.start_x,
                noisy_position[1] - self.start_y,
                geometry.normalize_angle(compass),
                speed,
            ],
        )

        if not self.filter_initialized:
            # apply ukf only to x and y coordinates, append z coordinate afterwards
            self.ukf.x = z
            self.filter_initialized = True

        self.ukf.predict(
            steer=control.steer,
            throttle=control.throttle,
            brake=control.brake,
        )
        self.ukf.update(z)

        prediction = self.ukf.x.copy()

        # Rescale back to original coordinates
        prediction[0] += self.start_x
        prediction[1] += self.start_y

        self.history_x.append(prediction)

        return prediction

    def _bicycle_model_forward(
        self,
        x: jt.Float[npt.NDArray, " 4"],
        dt: float,
        steer: float,
        throttle: float,
        brake: float,
    ) -> jt.Float[npt.NDArray, " 4"]:
        """Leaderboard 1.0's kinematic bicycle model. Numbers are the tuned parameters from World on Rails.

        Args:
            x: State vector [x, y, yaw, speed].
            dt: Timestep in seconds.
            steer: Last step's steering command.
            throttle: Last step's throttle command.
            brake: Last step's brake command.

        Returns:
            npt.NDArray[np.floating]: The next predicted state [x, y, yaw, speed].
        """
        front_wb = -0.090769015
        rear_wb = 1.4178275

        steer_gain = 0.36848336
        brake_accel = -4.952399
        throt_accel = 0.5633837

        locs_0 = x[0]
        locs_1 = x[1]
        yaw = x[2]
        speed = x[3]

        if brake:
            accel = brake_accel
        else:
            accel = throt_accel * throttle

        wheel = steer_gain * steer

        beta = math.atan(rear_wb / (front_wb + rear_wb) * math.tan(wheel))
        next_locs_0 = locs_0.item() + speed * math.cos(yaw + beta) * dt
        next_locs_1 = locs_1.item() + speed * math.sin(yaw + beta) * dt
        next_yaws = yaw + speed / rear_wb * math.sin(beta) * dt
        next_speed = speed + accel * dt
        next_speed = next_speed * (next_speed > 0.0)  # Fast ReLU

        next_state_x = np.array([next_locs_0, next_locs_1, next_yaws, next_speed])

        return next_state_x

    def _measurement_function_hx(
        self,
        vehicle_state: jt.Float[npt.NDArray, " 4"],
    ) -> jt.Float[npt.NDArray, " 4"]:
        """
        Identity measurement function.

        Args:
            vehicle_state: Vehicle state variable containing an internal state of the vehicle from the filter

        Returns:
            npt.NDArray: Output.
        """
        return vehicle_state

    def _state_mean(
        self,
        state: jt.Float[npt.NDArray, "N 4"],
        wm: npt.ArrayLike,
    ) -> jt.Float[npt.NDArray, " 4"]:
        """Averaging function.

        Args:
            state: States to be averaged.
            wm: Weights for the mean.
        Returns:
            The averaged state.
        Note: We use the arctan of the average of sin and cos of the angle to calculate the average of orientations.
        """
        x = np.zeros(4)
        sum_sin = np.sum(np.dot(np.sin(state[:, 2]), wm))
        sum_cos = np.sum(np.dot(np.cos(state[:, 2]), wm))
        x[0] = np.sum(np.dot(state[:, 0], wm))
        x[1] = np.sum(np.dot(state[:, 1], wm))
        x[2] = math.atan2(sum_sin, sum_cos)
        x[3] = np.sum(np.dot(state[:, 3], wm))

        return x

    def _measurement_mean(
        self,
        state: jt.Float[npt.NDArray, "N 4"],
        wm: npt.ArrayLike,
    ) -> jt.Float[npt.NDArray, " 4"]:
        """Averaging function.

        Args:
            state: States to be averaged.
            wm: Weights for the mean.
        Returns:
            The averaged state.
        Note: We use the arctan of the average of sin and cos of the angle to calculate the average of orientations.
        """
        x = np.zeros(4)
        sum_sin = np.sum(np.dot(np.sin(state[:, 2]), wm))
        sum_cos = np.sum(np.dot(np.cos(state[:, 2]), wm))
        x[0] = np.sum(np.dot(state[:, 0], wm))
        x[1] = np.sum(np.dot(state[:, 1], wm))
        x[2] = math.atan2(sum_sin, sum_cos)
        x[3] = np.sum(np.dot(state[:, 3], wm))

        return x

    def _residual_state_x(
        self,
        a: npt.NDArray,
        b: npt.NDArray,
    ) -> npt.NDArray[np.floating]:
        """Residual function

        Args:
            a: Predicted state.
            b: State to be subtracted from predicted state.

        Returns:
            The residual.
        """
        y = a - b
        y[2] = geometry.normalize_angle(y[2])
        return y

    def _residual_measurement_h(
        self,
        a: npt.NDArray,
        b: npt.NDArray,
    ) -> npt.NDArray[np.floating]:
        """Residual function

        Args:
            a: Predicted state.
            b: State to be subtracted from predicted state.

        Returns:
            The residual.
        """
        y = a - b
        y[2] = geometry.normalize_angle(y[2])
        return y
