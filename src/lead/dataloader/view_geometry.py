"""Re-expresses ego-frame quantities (labels, lidar, routes, waypoints) in the
perturbated sensor rig's virtual ego frame ("view frame"); for the normal view
every transform is the identity."""

import jaxtyping as jt
import numpy as np
import numpy.typing as npt
from py123d.geometry import PoseSE2, PoseSE3


def _view_rotation_matrix(
    perturbation_rotation: float,
) -> jt.Float[npt.NDArray, "2 2"]:
    """Rotation matrix of the rig perturbation in the CARLA ego frame.

    Args:
        perturbation_rotation: Rig yaw perturbation in degrees.

    Returns:
        The 2x2 rotation matrix ``R(rotation)``.
    """
    yaw_rad = np.deg2rad(perturbation_rotation)
    return np.array(
        [
            [np.cos(yaw_rad), -np.sin(yaw_rad)],
            [np.sin(yaw_rad), np.cos(yaw_rad)],
        ],
    )


def to_view_frame_points(
    points: jt.Float[npt.NDArray, "n 2"],
    perturbation_translation: float,
    perturbation_rotation: float,
) -> jt.Float[npt.NDArray, "n 2"]:
    """Re-express CARLA-ego-frame points in the view frame.

    Args:
        points: Points of shape (N, 2) in the CARLA ego frame.
        perturbation_translation: Rig translation perturbation in meters.
        perturbation_rotation: Rig yaw perturbation in degrees.

    Returns:
        The points expressed in the view frame, shape (N, 2).
    """
    rotation_matrix = _view_rotation_matrix(perturbation_rotation)
    translation = np.array([0.0, perturbation_translation])
    return (points - translation) @ rotation_matrix


def to_view_frame_yaws(
    yaws: jt.Float[npt.NDArray, " n"],
    perturbation_rotation: float,
) -> jt.Float[npt.NDArray, " n"]:
    """Re-express CARLA-ego-frame yaw angles in the view frame.

    Args:
        yaws: Yaw angles in radians.
        perturbation_rotation: Rig yaw perturbation in degrees.

    Returns:
        The normalized yaw angles in the view frame.
    """
    yaws = np.asarray(yaws) - np.deg2rad(perturbation_rotation)
    return (yaws + np.pi) % (2.0 * np.pi) - np.pi


def to_view_frame_boxes(
    boxes: list[dict],
    perturbation_translation: float,
    perturbation_rotation: float,
) -> list[dict]:
    """Re-express bounding-box dicts in the view frame.

    Transforms ``position`` (x, y; z is view-independent), ``yaw``, and — when
    present — the grafted ``future_positions``/``future_yaws`` of each box.

    Args:
        boxes: CARLA bounding-box dicts in the ego frame.
        perturbation_translation: Rig translation perturbation in meters.
        perturbation_rotation: Rig yaw perturbation in degrees.

    Returns:
        Shallow copies of the box dicts with view-frame geometry.
    """
    view_boxes = []
    for box in boxes:
        view_box = dict(box)
        position = to_view_frame_points(
            np.array([box["position"][:2]]),
            perturbation_translation,
            perturbation_rotation,
        )[0]
        view_box["position"] = [position[0], position[1], *box["position"][2:]]
        view_box["yaw"] = float(
            to_view_frame_yaws(np.array([box["yaw"]]), perturbation_rotation)[0],
        )
        if box.get("future_positions") is not None:
            view_box["future_positions"] = to_view_frame_points(
                np.array(box["future_positions"]),
                perturbation_translation,
                perturbation_rotation,
            ).tolist()
            view_box["future_yaws"] = to_view_frame_yaws(
                np.array(box["future_yaws"]),
                perturbation_rotation,
            ).tolist()
        view_boxes.append(view_box)
    return view_boxes


def to_view_frame_lidar(
    points: jt.Float[npt.NDArray, "n d"],
    perturbation_translation: float,
    perturbation_rotation: float,
) -> jt.Float[npt.NDArray, "n d"]:
    """Re-express a CARLA-ego-frame point cloud in the view frame.

    Args:
        points: Point cloud with [x, y, ...] leading columns.
        perturbation_translation: Rig translation perturbation in meters.
        perturbation_rotation: Rig yaw perturbation in degrees.

    Returns:
        The point cloud with x/y transformed into the view frame.
    """
    points = points.copy()
    points[:, :2] = to_view_frame_points(
        points[:, :2],
        perturbation_translation,
        perturbation_rotation,
    )
    return points


def view_center_se2(
    center_se2: PoseSE2,
    perturbation_translation: float,
    perturbation_rotation: float,
) -> PoseSE2:
    """Global ISO 8855 pose of the view frame's origin.

    Composes the ego center pose with the rig perturbation (mirrored into the
    ISO convention: y left, yaw counter-clockwise), e.g. for anchoring BEV
    rasters in the view frame.

    Args:
        center_se2: Global ISO ego center pose.
        perturbation_translation: Rig translation perturbation in meters.
        perturbation_rotation: Rig yaw perturbation in degrees.

    Returns:
        The global pose of the view frame.
    """
    yaw = center_se2.yaw
    return PoseSE2(
        x=center_se2.x + perturbation_translation * np.sin(yaw),
        y=center_se2.y - perturbation_translation * np.cos(yaw),
        yaw=yaw - np.deg2rad(perturbation_rotation),
    )


def derive_rig_perturbation(
    normal_camera_to_imu: PoseSE3,
    perturbated_camera_to_imu: PoseSE3,
    rear_axle_to_center_longitudinal: float,
) -> tuple[float, float]:
    """Recover the rig perturbation from the recorded camera extrinsics.

    The perturbation itself is not logged — only the perturbated
    ``camera_to_imu_se3`` (see ``CameraRecorder``); the relative transform
    between the two extrinsics of the same camera is the rig perturbation
    conjugated into the ISO IMU frame — undo the rear-axle shift and the
    CARLA→ISO mirroring to get back the sampled scalars.

    Args:
        normal_camera_to_imu: Camera extrinsic of the normal rig.
        perturbated_camera_to_imu: Camera extrinsic of the perturbated rig.
        rear_axle_to_center_longitudinal: X offset applied by
            ``floor_center_to_rear_axle_translate`` when the extrinsics were
            recorded.

    Returns:
        The perturbation translation in meters and rotation in degrees, in the
        CARLA convention used at collection time.
    """
    relative = perturbated_camera_to_imu.transformation_matrix @ np.linalg.inv(
        normal_camera_to_imu.transformation_matrix,
    )
    # ISO yaw of the perturbation; mirrored, so the CARLA rotation is negated.
    yaw_iso = np.arctan2(relative[1, 0], relative[0, 0])
    perturbation_rotation = -np.rad2deg(yaw_iso)
    # The stored extrinsics are shifted to the rear axle, so the derived
    # translation contains the lever arm of that shift: undo it.
    lever_arm_y = np.sin(yaw_iso) * rear_axle_to_center_longitudinal
    perturbation_translation = -(relative[1, 3] + lever_arm_y)
    return float(perturbation_translation), float(perturbation_rotation)
