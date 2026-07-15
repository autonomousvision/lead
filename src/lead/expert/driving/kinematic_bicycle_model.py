import jaxtyping as jt
import numpy as np
from numpy.typing import NDArray

from lead.config import ExpertConfig


class KinematicBicycleModel:
    """Kinematic bicycle model describing the motion of a car given its state and action.
    Tuned parameters are taken from World on Rails."""

    def __init__(self, config: ExpertConfig) -> None:
        """Initialize the kinematic bicycle model with configuration parameters.

        Args:
            config: Object of the config for hyperparameters.
        """
        self.config = config

        self.time_step = self.config.bicycle_model.time_step
        self.front_wheel_base = self.config.bicycle_model.front_wheel_base
        self.rear_wheel_base = self.config.bicycle_model.rear_wheel_base
        self.steering_gain = self.config.bicycle_model.steering_gain
        self.brake_acceleration = self.config.bicycle_model.brake_acceleration
        self.throttle_acceleration = self.config.bicycle_model.throttle_acceleration
        self.throttle_values = self.config.pid.throttle_values
        self.brake_values = self.config.pid.brake_values
        self.throttle_threshold_during_forecasting = (
            self.config.bicycle_model.throttle_threshold_during_forecasting
        )

    def forecast_other_vehicles(
        self,
        locations: jt.Float[NDArray, "N 3"],
        headings: jt.Float[NDArray, " N"],
        speeds: jt.Float[NDArray, " N"],
        actions: jt.Float[NDArray, "N 3"],
        num_future_frames: int,
    ) -> tuple[
        jt.Float[NDArray, "T N 3"],
        jt.Float[NDArray, "T N"],
        jt.Float[NDArray, "T N"],
    ]:
        """Forecast the future states of other vehicles over multiple frames at once.

        Actions are held constant, so the per-step recurrence has a closed form:
        speeds are linear in time and headings/positions follow from cumulative
        sums of the per-step increments.

        Args:
            locations: Array of (x, y, z) coordinates representing the locations of other vehicles.
            headings: Array of heading angles (in radians) for other vehicles.
            speeds: Array of speeds (in m/s) for other vehicles.
            actions: Array of actions (steer, throttle, brake) for other vehicles.
            num_future_frames: Number of future frames to forecast.

        Returns:
            A tuple containing the forecasted locations, headings, and speeds for
            other vehicles, each with the frame as leading dimension.
        """
        steers, throttles = actions[:, 0], actions[:, 1]
        with np.errstate(invalid="ignore"):
            brakes = actions[:, 2].astype(np.uint8)
        wheel_angles = self.steering_gain * steers
        slip_angles = np.arctan(
            self.rear_wheel_base
            / (self.front_wheel_base + self.rear_wheel_base)
            * np.tan(wheel_angles),
        )
        accelerations = np.where(
            brakes,
            self.brake_acceleration,
            throttles * self.throttle_acceleration,
        )

        # Speeds before each step k = 0..T-1; the clamp at 0 is final since the
        # acceleration is constant.
        steps = np.arange(num_future_frames)[:, None]
        pre_step_speeds = np.maximum(
            0.0,
            speeds[None] + steps * self.time_step * accelerations[None],
        )
        future_speeds = np.maximum(
            0.0,
            speeds[None] + (steps + 1) * self.time_step * accelerations[None],
        )

        heading_increments = (
            self.time_step * np.sin(slip_angles) / self.rear_wheel_base
        )[None] * pre_step_speeds
        future_headings = headings[None] + np.cumsum(heading_increments, axis=0)
        pre_step_headings = future_headings - heading_increments

        travel_angles = pre_step_headings + slip_angles[None]
        step_distances = pre_step_speeds * self.time_step
        future_x = locations[None, :, 0] + np.cumsum(
            step_distances * np.cos(travel_angles),
            axis=0,
        )
        future_y = locations[None, :, 1] + np.cumsum(
            step_distances * np.sin(travel_angles),
            axis=0,
        )
        future_z = np.broadcast_to(locations[None, :, 2], future_x.shape)
        future_locations = np.stack([future_x, future_y, future_z], axis=-1)

        return future_locations, future_headings, future_speeds

    def forecast_ego_vehicle(
        self,
        location: jt.Float[NDArray, "3"],
        heading: float,
        speed: float,
        action: jt.Float[NDArray, "3"],
    ) -> tuple[jt.Float[NDArray, "3"], float, float]:
        """Forecast the future state of the ego vehicle based on its current state and action.

        Args:
            location: Array of (x, y, z) coordinates representing the location of the ego vehicle.
            heading: Current heading angle (in radians) of the ego vehicle.
            speed: Current speed (in m/s) of the ego vehicle.
            action: Action (steer, throttle, brake) for the ego vehicle.

        Returns:
            A tuple containing the forecasted location, heading, and speed for the ego vehicle.
        """
        steer, throttle, brake = action
        steer = float(steer)
        throttle = float(throttle)
        brake = bool(brake)
        speed = float(speed)
        wheel_angle = self.steering_gain * steer
        slip_angle = np.arctan(
            self.rear_wheel_base
            / (self.front_wheel_base + self.rear_wheel_base)
            * np.tan(wheel_angle),
        )

        next_x = location[0] + speed * np.cos(heading + slip_angle) * self.time_step
        next_y = location[1] + speed * np.sin(heading + slip_angle) * self.time_step
        next_heading = (
            heading + speed / self.rear_wheel_base * np.sin(slip_angle) * self.time_step
        )

        # We use different polynomial models for estimating the speed depending on whether
        # the ego vehicle brakes or not.
        if brake:
            speed_kph = speed * 3.6
            features = speed_kph ** np.arange(1, 8)
            next_speed_kph = features @ self.brake_values
            next_speed = next_speed_kph / 3.6
        else:
            throttle = np.clip(throttle, 0.0, 1.0)

            # Below the throttle threshold the car barely accelerates and the
            # polynomial model below does not hold.
            if throttle < self.throttle_threshold_during_forecasting:
                # If the throttle is low, the car does not accelerate, so we
                # assume constant speed.
                next_speed = speed
            else:
                # For a throttle value > 0.3 the car accelerates and we can
                # use the polynomial model below.
                speed_kph = speed * 3.6
                features = np.array(
                    [
                        speed_kph,
                        speed_kph**2,
                        throttle,
                        throttle**2,
                        speed_kph * throttle,
                        speed_kph * throttle**2,
                        speed_kph**2 * throttle,
                        speed_kph**2 * throttle**2,
                    ],
                ).T

                next_speed_kph = features @ self.throttle_values
                next_speed = next_speed_kph / 3.6

        next_speed = np.maximum(0.0, next_speed)
        next_location = np.array([float(next_x), float(next_y), float(location[2])])

        return next_location, next_heading, next_speed
