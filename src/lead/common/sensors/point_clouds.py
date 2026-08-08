"""Transforms that bring raw LiDAR and radar returns into the ego coordinate system."""

import jaxtyping as jt
import numpy as np
import numpy.typing as npt

from lead.common.geometry import euler_degrees_to_rotation_matrix


def lidar_to_ego_coordinate(
    lidar_rot: list[float],
    lidar_pos: list[float],
    lidar: tuple[int, jt.Float[npt.NDArray, "... 4"]],
) -> jt.Float[npt.NDArray, "... 3"]:
    """Convert LiDAR points from sensor frame to ego vehicle coordinate system.

    Args:
        lidar_rot: LiDAR sensor rotation as [roll, pitch, yaw] in degrees.
        lidar_pos: LiDAR sensor position as [x, y, z] in meters.
        lidar: LiDAR point cloud as provided in the input of run_step, i.e. the
            raw ``(frame_id, points)`` tuple CARLA delivers, where each point is
            ``(x, y, z, intensity)``.

    Returns:
        LiDAR points transformed to ego vehicle coordinate system.
    """
    rotation_matrix = euler_degrees_to_rotation_matrix(
        lidar_rot[0],
        lidar_rot[1],
        lidar_rot[2],
    )
    translation = np.array(lidar_pos)
    lidar_points = (rotation_matrix @ lidar[1][:, :3].T).T + translation
    lidar_points[:, 2] = (
        lidar_points[:, 2] - lidar_pos[-1] / 2
    )  # Not sure why we need this :/
    return lidar_points


def radar_points_to_ego(
    raw_radar: jt.Float[npt.NDArray, "n 4"],
    sensor_pos: list[float],
    sensor_rot: list[float],
) -> jt.Float[npt.NDArray, "n 4"]:
    """Transform radar points from sensor frame to ego vehicle coordinate system.

    Args:
        raw_radar: Radar data of shape (N, 4) with [x, y, z, velocity] in sensor frame.
        sensor_pos: Sensor position [x, y, z] in ego frame (meters).
        sensor_rot: Sensor rotation [roll, pitch, yaw] in degrees (ego frame convention).

    Returns:
        Transformed radar points in ego coordinate system of shape (N, 4).
    """
    sensor_pos = [sensor_pos[0], sensor_pos[1], sensor_pos[2]]
    sensor_rot = [sensor_rot[0], sensor_rot[1], sensor_rot[2]]

    r = raw_radar[:, 0]
    alt = raw_radar[:, 1]
    az = raw_radar[:, 2]
    vel = raw_radar[:, 3]
    x = r * np.cos(az) * np.cos(alt)
    y = r * np.sin(az) * np.cos(alt)
    z = r * np.sin(alt)
    pts = np.stack([x, y, z], axis=1).astype(np.float32)

    R_se = euler_degrees_to_rotation_matrix(*sensor_rot)
    pts_ego = (R_se @ pts.T).T + np.asarray(sensor_pos, dtype=np.float32).reshape(1, 3)

    pts_ego[:, 2] = pts_ego[:, 2] - sensor_pos[-1] / 2  # Not sure why we need this :/

    return np.concatenate([pts_ego, vel.reshape(-1, 1)], axis=1)


def align_lidar(
    lidar: jt.Float[npt.NDArray, "n 3"],
    translation: jt.Float[npt.NDArray, " 3"],
    yaw: float,
) -> jt.Float[npt.NDArray, "n 3"]:
    """
    Translates and rotates a LiDAR into a new coordinate system.
    Rotation is inverse to translation and yaw

    Args:
        lidar: numpy LiDAR point cloud.
        translation: translations in meters.
        yaw: yaw angle in radians.

    Returns:
        numpy LiDAR point cloud in the new coordinate system.
    """
    rotation_matrix = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
    )

    return (rotation_matrix.T @ (lidar - translation).T).T
