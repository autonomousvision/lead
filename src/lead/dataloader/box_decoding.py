"""Derive the expert's CARLA-ego-frame box dicts and ego quantities from the
native 123D modalities; the logs store the geometry only once, natively."""

import jaxtyping as jt
import numpy as np
import numpy.typing as npt
from py123d.datatypes import BoxDetectionsSE3, EgoStateSE3
from py123d.geometry import PoseSE3
from py123d.geometry.transform import translate_se3_along_z

from lead.common import geometry


def carla_forward_speed(ego_state: EgoStateSE3) -> float:
    """The ego's forward speed, as CARLA's speedometer reports it.

    Args:
        ego_state: Native ego state of the tick.

    Returns:
        The velocity projected onto the ego's heading, in m/s.
    """
    rotation = ego_state.center_se3.transformation_matrix[:3, :3]
    dynamic_state = ego_state.dynamic_state_se3
    assert dynamic_state is not None
    return float(dynamic_state.velocity_3d.array @ rotation[:, 0])


def carla_ego_pose(
    ego_state: EgoStateSE3,
) -> tuple[jt.Float[npt.NDArray, " 2"], float]:
    """The ego's CARLA global position and yaw, as the expert recorded them.

    Args:
        ego_state: Native ego state of the tick.

    Returns:
        The actor-origin xy position and the yaw, in CARLA's global frame.
    """
    matrix = _actor_origin(
        ego_state.center_se3,
        ego_state.metadata.half_height,
    ).transformation_matrix
    return np.array([matrix[0, 3], -matrix[1, 3]]), -_yaw_of(matrix)


def box_detections_to_carla_frame(
    box_detections: BoxDetectionsSE3,
    ego_state: EgoStateSE3,
    box_attributes: dict[str, dict],
) -> list[dict]:
    """The expert's ego-relative CARLA box dicts, derived from the native stream.

    Geometry comes from the box detections, the remaining fields from the
    driving-meta box attributes of the same track token.

    Args:
        box_detections: Native box detections of the tick.
        ego_state: Native ego state of the same tick.
        box_attributes: Per-box attribute dicts keyed by track token.

    Returns:
        One dict per box in the expert's format, in the CARLA ego frame.
    """
    ego_matrix = _actor_origin(
        ego_state.center_se3,
        ego_state.metadata.half_height,
    ).transformation_matrix
    inverse_ego = np.linalg.inv(ego_matrix)
    ego_yaw = _yaw_of(ego_matrix)

    boxes: list[dict] = []
    for detection in box_detections.box_detections:
        bounding_box = detection.bounding_box_se3
        floor_matrix = _actor_origin(
            bounding_box.center_se3,
            bounding_box.height / 2.0,
        ).transformation_matrix
        relative = inverse_ego @ floor_matrix[:, 3]
        position = [float(relative[0]), float(-relative[1]), float(relative[2])]
        box_yaw = _yaw_of(floor_matrix)

        token = detection.attributes.track_token
        box: dict = {
            "class": detection.attributes.label.name.lower(),
            "id": int(token),
            "extent": [
                bounding_box.length / 2.0,
                bounding_box.width / 2.0,
                bounding_box.height / 2.0,
            ],
            "position": position,
            "yaw": geometry.normalize_angle(-(box_yaw - ego_yaw)),
            "distance": float(np.linalg.norm(position)),
            **box_attributes.get(token, {}),
        }
        if detection.attributes.num_lidar_points is not None:
            box["num_points"] = detection.attributes.num_lidar_points
        if detection.velocity_3d is not None:
            velocity = detection.velocity_3d.array
            box["speed"] = float(
                velocity[0] * np.cos(box_yaw) + velocity[1] * np.sin(box_yaw),
            )
        boxes.append(box)
    return boxes


def box_detections_to_carla_global(
    box_detections: BoxDetectionsSE3,
) -> dict[int, tuple[jt.Float[npt.NDArray, " 2"], float]]:
    """The CARLA global xy position and yaw of every box, keyed by box id.

    Args:
        box_detections: Native box detections of one tick.

    Returns:
        The global CARLA pose of each box's floor center.
    """
    poses: dict[int, tuple[jt.Float[npt.NDArray, " 2"], float]] = {}
    for detection in box_detections.box_detections:
        bounding_box = detection.bounding_box_se3
        matrix = _actor_origin(
            bounding_box.center_se3,
            bounding_box.height / 2.0,
        ).transformation_matrix
        poses[int(detection.attributes.track_token)] = (
            np.array([matrix[0, 3], -matrix[1, 3]]),
            -_yaw_of(matrix),
        )
    return poses


def _actor_origin(center_se3: PoseSE3, half_height: float) -> PoseSE3:
    """The recording-time actor origin: the box center lowered to the floor.

    Args:
        center_se3: Geometric-center pose in ISO 8855.
        half_height: Half height of the box.

    Returns:
        The pose the expert's CARLA transforms were taken at.
    """
    return translate_se3_along_z(center_se3, -half_height)


def _yaw_of(matrix: jt.Float[npt.NDArray, "4 4"]) -> float:
    """Yaw of a transformation matrix.

    Args:
        matrix: Homogeneous transform.

    Returns:
        The yaw in radians.
    """
    return float(np.arctan2(matrix[1, 0], matrix[0, 0]))
