"""Numba kernels and array containers for the expert's forecasting hot path."""

import math
import typing

import carla
import jaxtyping as jt
import numpy as np
import numpy.typing as npt
from numba import njit

if typing.TYPE_CHECKING:
    from lead.config import ExpertConfig


class EgoForecast(typing.NamedTuple):
    """Forecasted ego bounding boxes as arrays (rotation is yaw-only)."""

    centers: jt.Float[npt.NDArray, "T 3"]
    yaws_deg: jt.Float[npt.NDArray, " T"]
    extent: jt.Float[npt.NDArray, " 3"]


class ActorForecast(typing.NamedTuple):
    """Forecasted bounding boxes of one vehicle as arrays (rotation is yaw-only)."""

    actor: carla.Actor
    centers: jt.Float[npt.NDArray, "T 3"]
    yaws_deg: jt.Float[npt.NDArray, " T"]
    extents: jt.Float[npt.NDArray, "T 3"]


class WalkerForecast(typing.NamedTuple):
    """Forecasted bounding boxes of one walker; rotation and extent are constant."""

    actor: carla.Actor
    centers: jt.Float[npt.NDArray, "T 3"]
    rotation_deg: jt.Float[npt.NDArray, " 3"]
    extent: jt.Float[npt.NDArray, " 3"]


@njit(cache=True)
def _push_error(
    history: jt.Float[npt.NDArray, " W"],
    count: int,
    window_size: int,
    error: float,
) -> int:
    """Append an error to the rolling PID window, returning the new count."""
    if count < window_size:
        history[count] = error
        return count + 1
    for i in range(window_size - 1):
        history[i] = history[i + 1]
    history[window_size - 1] = error
    return count


@njit(cache=True)
def _lateral_pid_step(
    route_points: jt.Float[npt.NDArray, "N 3"],
    route_index: int,
    x: float,
    y: float,
    heading: float,
    speed: float,
    history: jt.Float[npt.NDArray, " W"],
    count: int,
    window_size: int,
    kp: float,
    kd: float,
    ki: float,
    speed_scale: float,
    speed_offset: float,
    min_lookahead: float,
    max_lookahead: float,
) -> tuple[float, int]:
    """One LateralPIDController.step with the window kept in an array.

    Matches the pure-Python controller bit-for-bit (same operation order).
    """
    speed_kph = speed * 3.6
    lookahead = speed_scale * speed_kph + speed_offset
    if lookahead < min_lookahead:
        lookahead = min_lookahead
    elif lookahead > max_lookahead:
        lookahead = max_lookahead
    remaining = route_points.shape[0] - route_index
    lookahead_index = int(min(lookahead, float(remaining - 1)))

    target_x = route_points[route_index + lookahead_index, 0]
    target_y = route_points[route_index + lookahead_index, 1]
    desired_heading_angle = math.atan2(target_y - y, target_x - x)

    heading_error = (desired_heading_angle - heading) % (2 * np.pi)
    if heading_error >= np.pi:
        heading_error -= 2 * np.pi
    heading_error = heading_error * 180.0 / np.pi / 90.0

    count = _push_error(history, count, window_size, heading_error)
    if count == 1:
        derivative = 0.0
    else:
        derivative = history[count - 1] - history[count - 2]
    integral = 0.0
    for i in range(count):
        integral += history[i]
    integral /= count

    steering = kp * heading_error + kd * derivative + ki * integral
    if steering < -1.0:
        steering = -1.0
    elif steering > 1.0:
        steering = 1.0
    return steering, count


@njit(cache=True)
def _throttle_extrapolation(
    target_speed: float,
    current_speed: float,
    params: jt.Float[npt.NDArray, " 7"],
    maximum_acceleration: float,
    maximum_deceleration: float,
) -> float:
    """LongitudinalController.get_throttle_extrapolation, scalarized."""
    current_speed = current_speed * 3.6
    target_speed = target_speed * 3.6
    speed_error = target_speed - current_speed

    if speed_error > maximum_acceleration:
        return 1.0
    if speed_error < maximum_deceleration:
        return 0.0
    if target_speed < 0.1 or current_speed / target_speed > params[6]:
        return 0.0

    speed_error_cl = max(speed_error, 0.0) / 100.0
    current_speed /= 100.0
    features = np.empty(6, np.float64)
    features[0] = current_speed
    features[1] = current_speed**2
    features[2] = 100 * speed_error_cl
    features[3] = speed_error_cl**2
    features[4] = current_speed * speed_error_cl
    features[5] = current_speed**2 * speed_error_cl
    throttle = np.dot(features, params[:6])
    if throttle < 0.0:
        throttle = 0.0
    elif throttle > 1.0:
        throttle = 1.0
    return throttle


@njit(cache=True)
def forecast_ego_agent_kernel(
    route_points: jt.Float[npt.NDArray, "N 3"],
    route_index: int,
    route_search_distance: int,
    x: float,
    y: float,
    z: float,
    heading: float,
    speed: float,
    target_speed: float,
    error_history: jt.Float[npt.NDArray, " H"],
    window_size: int,
    kp: float,
    kd: float,
    ki: float,
    speed_scale: float,
    speed_offset: float,
    min_lookahead: float,
    max_lookahead: float,
    longitudinal_params: jt.Float[npt.NDArray, " 7"],
    maximum_acceleration: float,
    maximum_deceleration: float,
    time_step: float,
    front_wheel_base: float,
    rear_wheel_base: float,
    steering_gain: float,
    throttle_values: jt.Float[npt.NDArray, " 8"],
    throttle_threshold: float,
    num_future_frames: int,
) -> tuple[jt.Float[npt.NDArray, "T 3"], jt.Float[npt.NDArray, " T"]]:
    """Forecast the ego agent's future locations and headings without hazards.

    Runs PID steering, throttle extrapolation and a kinematic bicycle step on
    local copies of the controller window and route index.

    Args:
        route_points: Full route of the privileged route planner.
        route_index: Current route index of the privileged route planner.
        route_search_distance: Number of route points searched when advancing the index.
        x: Ego x position.
        y: Ego y position.
        z: Ego z position.
        heading: Ego heading in radians.
        speed: Ego speed in m/s.
        target_speed: Target speed the forecast keeps steering towards.
        error_history: Current error window of the lateral PID controller.
        window_size: Maximum size of the lateral PID error window.
        kp: Proportional gain of the lateral PID controller.
        kd: Derivative gain of the lateral PID controller.
        ki: Integral gain of the lateral PID controller.
        speed_scale: Speed factor of the lateral PID lookahead distance.
        speed_offset: Constant offset of the lateral PID lookahead distance.
        min_lookahead: Minimum lateral PID lookahead distance.
        max_lookahead: Maximum lateral PID lookahead distance.
        longitudinal_params: Linear regression parameters of the longitudinal controller.
        maximum_acceleration: Speed error above which full throttle is applied.
        maximum_deceleration: Speed error below which the throttle is released.
        time_step: Simulation time step in seconds.
        front_wheel_base: Distance from the center to the front axle.
        rear_wheel_base: Distance from the center to the rear axle.
        steering_gain: Gain from steering command to wheel angle.
        throttle_values: Regression parameters of the throttle speed model.
        throttle_threshold: Throttle below which the speed is left unchanged.
        num_future_frames: Number of future frames to forecast.

    Returns:
        Future locations and headings (radians), one row per future frame.
    """
    n_points = route_points.shape[0]

    history = np.empty(window_size, np.float64)
    count = min(error_history.shape[0], window_size)
    for i in range(count):
        history[i] = error_history[error_history.shape[0] - count + i]

    throttle = _throttle_extrapolation(
        target_speed,
        speed,
        longitudinal_params,
        maximum_acceleration,
        maximum_deceleration,
    )
    steering, count = _lateral_pid_step(
        route_points,
        route_index,
        x,
        y,
        heading,
        speed,
        history,
        count,
        window_size,
        kp,
        kd,
        ki,
        speed_scale,
        speed_offset,
        min_lookahead,
        max_lookahead,
    )

    locations = np.empty((num_future_frames, 3), np.float64)
    headings = np.empty(num_future_frames, np.float64)
    slip_factor = rear_wheel_base / (front_wheel_base + rear_wheel_base)
    for frame in range(num_future_frames):
        # Kinematic bicycle model step (throttle path only: the forecast never brakes)
        wheel_angle = steering_gain * steering
        slip_angle = math.atan(slip_factor * math.tan(wheel_angle))
        x = x + speed * math.cos(heading + slip_angle) * time_step
        y = y + speed * math.sin(heading + slip_angle) * time_step
        heading = heading + speed / rear_wheel_base * math.sin(slip_angle) * time_step

        if throttle < 0.0:
            throttle = 0.0
        elif throttle > 1.0:
            throttle = 1.0
        if throttle >= throttle_threshold:
            speed_kph = speed * 3.6
            features = np.empty(8, np.float64)
            features[0] = speed_kph
            features[1] = speed_kph**2
            features[2] = throttle
            features[3] = throttle**2
            features[4] = speed_kph * throttle
            features[5] = speed_kph * throttle**2
            features[6] = speed_kph**2 * throttle
            features[7] = speed_kph**2 * throttle**2
            speed = np.dot(features, throttle_values) / 3.6
        if speed < 0.0:
            speed = 0.0

        locations[frame, 0] = x
        locations[frame, 1] = y
        locations[frame, 2] = z
        headings[frame] = heading

        # Privileged route planner run_step: advance to the nearest route point
        search_end = min(route_index + route_search_distance, n_points)
        best_offset = 0
        best_distance = np.inf
        for i in range(route_index, search_end):
            dx = x - route_points[i, 0]
            dy = y - route_points[i, 1]
            distance = math.sqrt(dx * dx + dy * dy)
            if distance < best_distance:
                best_distance = distance
                best_offset = i - route_index
        route_index += best_offset

        steering, count = _lateral_pid_step(
            route_points,
            route_index,
            x,
            y,
            heading,
            speed,
            history,
            count,
            window_size,
            kp,
            kd,
            ki,
            speed_scale,
            speed_offset,
            min_lookahead,
            max_lookahead,
        )
        throttle = _throttle_extrapolation(
            target_speed,
            speed,
            longitudinal_params,
            maximum_acceleration,
            maximum_deceleration,
        )

    return locations, headings


@njit(cache=True)
def _rotation_axes(
    pitch_deg: float,
    yaw_deg: float,
    roll_deg: float,
) -> jt.Float[npt.NDArray, "3 3"]:
    """Forward/right/up unit vectors of a CARLA rotation as matrix rows."""
    pitch = math.radians(pitch_deg)
    yaw = math.radians(yaw_deg)
    roll = math.radians(roll_deg)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cr, sr = math.cos(roll), math.sin(roll)
    axes = np.empty((3, 3), np.float64)
    # forward
    axes[0, 0] = cp * cy
    axes[0, 1] = cp * sy
    axes[0, 2] = sp
    # right
    axes[1, 0] = cy * sp * sr - sy * cr
    axes[1, 1] = sy * sp * sr + cy * cr
    axes[1, 2] = -cp * sr
    # up
    axes[2, 0] = -cy * sp * cr - sy * sr
    axes[2, 1] = -sy * sp * cr + cy * sr
    axes[2, 2] = cp * cr
    return axes


@njit(cache=True)
def _obb_pair_intersects(
    center_a: jt.Float[npt.NDArray, " 3"],
    axes_a: jt.Float[npt.NDArray, "3 3"],
    extent_a: jt.Float[npt.NDArray, " 3"],
    center_b: jt.Float[npt.NDArray, " 3"],
    axes_b: jt.Float[npt.NDArray, "3 3"],
    extent_b: jt.Float[npt.NDArray, " 3"],
) -> bool:
    """Separating-axis test over the 15 candidate axes of two oriented boxes."""
    rel = np.empty(3, np.float64)
    for i in range(3):
        rel[i] = center_b[i] - center_a[i]

    reach_a = math.sqrt(extent_a[0] ** 2 + extent_a[1] ** 2 + extent_a[2] ** 2)
    reach_b = math.sqrt(extent_b[0] ** 2 + extent_b[1] ** 2 + extent_b[2] ** 2)
    max_reach = reach_a + reach_b
    if rel[0] ** 2 + rel[1] ** 2 + rel[2] ** 2 > max_reach * max_reach:
        return False

    candidate = np.empty(3, np.float64)
    for axis_index in range(15):
        if axis_index < 3:
            for k in range(3):
                candidate[k] = axes_a[axis_index, k]
        elif axis_index < 6:
            for k in range(3):
                candidate[k] = axes_b[axis_index - 3, k]
        else:
            i = (axis_index - 6) // 3
            j = (axis_index - 6) % 3
            candidate[0] = axes_a[i, 1] * axes_b[j, 2] - axes_a[i, 2] * axes_b[j, 1]
            candidate[1] = axes_a[i, 2] * axes_b[j, 0] - axes_a[i, 0] * axes_b[j, 2]
            candidate[2] = axes_a[i, 0] * axes_b[j, 1] - axes_a[i, 1] * axes_b[j, 0]

        center_projection = abs(
            candidate[0] * rel[0] + candidate[1] * rel[1] + candidate[2] * rel[2],
        )
        reach_projection = 0.0
        for k in range(3):
            reach_projection += (
                abs(
                    candidate[0] * axes_a[k, 0]
                    + candidate[1] * axes_a[k, 1]
                    + candidate[2] * axes_a[k, 2],
                )
                * extent_a[k]
            )
            reach_projection += (
                abs(
                    candidate[0] * axes_b[k, 0]
                    + candidate[1] * axes_b[k, 1]
                    + candidate[2] * axes_b[k, 2],
                )
                * extent_b[k]
            )
        if center_projection > reach_projection:
            return False
    return True


@njit(cache=True)
def batch_obb_intersection(
    centers_a: jt.Float[npt.NDArray, "T 3"],
    rotations_a: jt.Float[npt.NDArray, "T 3"],
    extents_a: jt.Float[npt.NDArray, "T 3"],
    centers_b: jt.Float[npt.NDArray, "A T 3"],
    rotations_b: jt.Float[npt.NDArray, "A T 3"],
    extents_b: jt.Float[npt.NDArray, "A T 3"],
) -> jt.Bool[npt.NDArray, "A T"]:
    """Frame-wise OBB intersections of one box track against many box tracks.

    Rotations are (pitch, yaw, roll) in degrees, CARLA convention.

    Args:
        centers_a: Box centers of the reference track per frame.
        rotations_a: Box rotations of the reference track per frame.
        extents_a: Box half-extents of the reference track per frame.
        centers_b: Box centers per actor and frame.
        rotations_b: Box rotations per actor and frame.
        extents_b: Box half-extents per actor and frame.

    Returns:
        Boolean matrix where entry (a, t) is True if track ``a`` intersects
        the reference track at future frame ``t``.
    """
    num_actors = centers_b.shape[0]
    num_frames = centers_a.shape[0]
    result = np.zeros((num_actors, num_frames), np.bool_)
    for t in range(num_frames):
        axes_a = _rotation_axes(rotations_a[t, 0], rotations_a[t, 1], rotations_a[t, 2])
        for a in range(num_actors):
            axes_b = _rotation_axes(
                rotations_b[a, t, 0],
                rotations_b[a, t, 1],
                rotations_b[a, t, 2],
            )
            result[a, t] = _obb_pair_intersects(
                centers_a[t],
                axes_a,
                extents_a[t],
                centers_b[a, t],
                axes_b,
                extents_b[a, t],
            )
    return result


def run_ego_forecast(
    config: "ExpertConfig",
    route_points: jt.Float[npt.NDArray, "N 3"],
    route_index: int,
    x: float,
    y: float,
    z: float,
    heading: float,
    speed: float,
    target_speed: float,
    error_history: list[float],
    num_future_frames: int,
) -> tuple[jt.Float[npt.NDArray, "T 3"], jt.Float[npt.NDArray, " T"]]:
    """Run the ego forecast kernel with all parameters taken from the config.

    Args:
        config: Expert configuration providing controller and model parameters.
        route_points: Full route of the privileged route planner.
        route_index: Current route index of the privileged route planner.
        x: Ego x position.
        y: Ego y position.
        z: Ego z position.
        heading: Ego heading in radians.
        speed: Ego speed in m/s.
        target_speed: Target speed the forecast keeps steering towards.
        error_history: Current error window of the lateral PID controller.
        num_future_frames: Number of future frames to forecast.

    Returns:
        Future locations and headings (radians), one row per future frame.
    """
    return forecast_ego_agent_kernel(
        route_points,
        route_index,
        int(config.driving.ego_vehicles_route_point_search_distance),
        x,
        y,
        z,
        heading,
        speed,
        target_speed,
        np.asarray(error_history, dtype=np.float64),
        int(config.pid.lateral_pid_window_size),
        float(config.pid.lateral_pid_kp),
        float(config.pid.lateral_pid_kd),
        float(config.pid.lateral_pid_ki),
        float(config.pid.lateral_pid_speed_scale),
        float(config.pid.lateral_pid_speed_offset),
        float(config.pid.lateral_pid_minimum_lookahead_distance),
        float(config.pid.lateral_pid_maximum_lookahead_distance),
        np.asarray(config.pid.longitudinal_linear_regression_params, dtype=np.float64),
        float(config.pid.longitudinal_linear_regression_maximum_acceleration),
        float(config.pid.longitudinal_linear_regression_maximum_deceleration),
        float(config.bicycle_model.time_step),
        float(config.bicycle_model.front_wheel_base),
        float(config.bicycle_model.rear_wheel_base),
        float(config.bicycle_model.steering_gain),
        np.asarray(config.pid.throttle_values, dtype=np.float64),
        float(config.bicycle_model.throttle_threshold_during_forecasting),
        num_future_frames,
    )


def warmup(config: "ExpertConfig") -> None:
    """Trigger numba compilation of the kernels with tiny dummy inputs.

    Args:
        config: Expert configuration providing controller and model parameters.
    """
    route = np.zeros((300, 3), np.float64)
    route[:, 0] = np.arange(300) * 0.1
    run_ego_forecast(config, route, 0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, [], 2)
    boxes = np.zeros((1, 2, 3), np.float64)
    batch_obb_intersection(
        np.zeros((2, 3), np.float64),
        np.zeros((2, 3), np.float64),
        np.ones((2, 3), np.float64),
        boxes,
        boxes,
        np.ones((1, 2, 3), np.float64),
    )
