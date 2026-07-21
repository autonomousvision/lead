"""CARLA → py123d conversion helpers.

Converts between CARLA (Unreal Engine, left-handed, +y right) and ISO 8855
(py123d, right-handed, +y left), and provides the static sensor and vehicle metadata.
"""

import carla
import jaxtyping as jt
import numpy as np
import numpy.typing as npt
from py123d.datatypes import (
    EgoStateSE3Metadata,
    PinholeCameraMetadata,
    PinholeIntrinsics,
    TrafficLightStatus,
)
from py123d.datatypes.sensors.base_camera import CameraID
from py123d.datatypes.sensors.lidar import LidarID, LidarMetadata
from py123d.datatypes.sensors.radar import (
    RadarID,
    RadarMergedMetadata,
    RadarMetadata,
)
from py123d.geometry import (
    BoundingBoxSE3,
    EulerAngles,
    PoseSE3,
    PoseSE3Index,
    Quaternion,
    Vector3D,
)
from py123d.geometry.transform import (
    translate_se3_along_body_frame,
    translate_se3_along_z,
)
from py123d.parser.utils.sensor_utils.camera_conventions import (
    convert_camera_convention,
)

from lead.api import py123d_log_api
from lead.common.sensors.av_sensor_setup import perturbated_pose
from lead.config import ExpertConfig


def quaternion_from_carla_rotation(rotation: carla.Rotation) -> Quaternion:
    """Convert CARLA Euler rotation to py123d quaternion in ISO 8855 coordinates.

    Args:
        rotation: CARLA rotation with roll, pitch, yaw in degrees.

    Returns:
        Quaternion with pitch and yaw inverted for ISO 8855 convention.
    """
    euler_angles = EulerAngles(
        roll=np.deg2rad(rotation.roll),
        pitch=-np.deg2rad(rotation.pitch),  # Invert for ISO 8855
        yaw=-np.deg2rad(rotation.yaw),  # Invert for ISO 8855
    )
    return Quaternion.from_rotation_matrix(euler_angles.rotation_matrix)


def carla_actor_to_se3(
    actor: carla.Actor,
    half_height: float | None = None,
) -> PoseSE3:
    """Get SE3 pose for CARLA actor at geometric center in ISO 8855 coordinates.

    Args:
        actor: CARLA actor with transform and bounding box.
        half_height: Optional custom half-height to override actor's bounding box extent.

    Returns:
        PoseSE3 at actor's geometric center with Y-axis inverted.
    """
    transform = actor.get_transform()
    if half_height is None:
        half_height = actor.bounding_box.extent.z

    quaternion = quaternion_from_carla_rotation(transform.rotation)

    pose = PoseSE3(
        x=transform.location.x,
        y=-transform.location.y,  # Invert Y-axis for ISO 8855
        z=transform.location.z,
        qw=quaternion.qw,
        qx=quaternion.qx,
        qy=quaternion.qy,
        qz=quaternion.qz,
    )

    # Translate from bottom center to geometric center
    return translate_se3_along_z(pose, half_height)


def carla_velocity_to_vector3d(velocity: carla.Vector3D) -> Vector3D:
    """Convert CARLA velocity to py123d Vector3D in ISO 8855 coordinates.

    Args:
        velocity: CARLA velocity vector.

    Returns:
        Vector3D with Y-axis inverted for ISO 8855 convention.
    """
    return Vector3D(
        x=velocity.x,
        y=-velocity.y,  # Invert Y-axis
        z=velocity.z,
    )


def get_bounding_box_se3(bb: dict) -> BoundingBoxSE3:
    """Extract bounding box in SE3 format from a LEAD bounding box dictionary.

    Args:
        bb: Bounding box dict with 'matrix' (4x4 transform) and 'extent' [x, y, z].

    Returns:
        BoundingBoxSE3 in ISO 8855 coordinates.
    """
    matrix = np.array(bb["matrix"])
    x, y, z = matrix[0, 3], matrix[1, 3], matrix[2, 3]

    # Extract yaw from the rotation matrix (assuming primarily yaw rotation)
    rotation_matrix = matrix[:3, :3]
    yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])

    euler_angles = EulerAngles(
        roll=0.0,  # Assume no roll for static objects
        pitch=0.0,  # Assume no pitch for static objects
        yaw=-yaw,  # Invert for ISO 8855
    )
    quaternion = Quaternion.from_rotation_matrix(euler_angles.rotation_matrix)

    extent = bb["extent"]
    half_height = extent[2]

    pose = PoseSE3(
        x=x,
        y=-y,  # Invert Y-axis for ISO 8855
        z=z,
        qw=quaternion.qw,
        qx=quaternion.qx,
        qy=quaternion.qy,
        qz=quaternion.qz,
    )

    # Translate from floor center to geometric center
    pose = translate_se3_along_z(pose, half_height)

    return BoundingBoxSE3(
        center_se3=pose,
        length=extent[0] * 2.0,  # CARLA extent is half-dimensions
        width=extent[1] * 2.0,
        height=extent[2] * 2.0,
    )


def get_actor_velocity(actor: carla.Actor) -> Vector3D:
    """Get velocity from CARLA actor in ISO 8855 coordinates.

    Args:
        actor: CARLA actor.

    Returns:
        Vector3D velocity with Y-axis inverted.
    """
    return carla_velocity_to_vector3d(actor.get_velocity())


def floor_center_to_rear_axle_translate(
    pose: PoseSE3,
    ego_metadata: EgoStateSE3Metadata,
) -> PoseSE3:
    """Translate pose from CARLA floor center to ISO 8855 rear axle frame.

    Args:
        pose: PoseSE3 in floor center reference.
        ego_metadata: Ego vehicle metadata with rear axle offsets.

    Returns:
        PoseSE3 translated to rear axle reference.
    """
    zero_pose = PoseSE3(x=0.0, y=0.0, z=0.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0)
    rear_axle_translate = translate_se3_along_body_frame(
        zero_pose,
        Vector3D(
            x=ego_metadata.rear_axle_to_center_longitudinal,
            y=0.0,
            z=-(ego_metadata.half_height) + ego_metadata.rear_axle_to_center_vertical,
        ),
    )
    pose.array[PoseSE3Index.XYZ] += rear_axle_translate.array[PoseSE3Index.XYZ]
    return pose


def get_camera_extrinsic_as_iso(
    camera_pos: jt.Float[npt.NDArray, "3"] | list[float],
    camera_rot: jt.Float[npt.NDArray, "3"] | list[float],
    ego_metadata: EgoStateSE3Metadata,
) -> PoseSE3:
    """Convert camera extrinsic from Unreal Engine to ISO 8855 format.

    Args:
        camera_pos: Camera position [x, y, z] in Unreal coordinates.
        camera_rot: Camera rotation [roll, pitch, yaw] in degrees.
        ego_metadata: Ego vehicle metadata for rear axle offset.

    Returns:
        PoseSE3 camera extrinsic in ISO 8855/OpenCV convention.
    """
    quaternion = quaternion_from_carla_rotation(
        carla.Rotation(roll=camera_rot[0], pitch=camera_rot[1], yaw=camera_rot[2]),
    )

    camera_extrinsic = PoseSE3(
        x=camera_pos[0],
        y=-camera_pos[1],  # Invert Y for ISO 8855
        z=camera_pos[2],
        qw=quaternion.qw,
        qx=quaternion.qx,
        qy=quaternion.qy,
        qz=quaternion.qz,
    )

    camera_extrinsic = floor_center_to_rear_axle_translate(
        camera_extrinsic,
        ego_metadata,
    )

    # Convert camera convention from pXpZmY (Unreal) to pZmYpX (ISO 8855/OpenCV)
    return convert_camera_convention(
        camera_extrinsic,
        from_convention="pXpZmY",
        to_convention="pZmYpX",
    )


def carla_traffic_light_status_to_py123d(
    state: carla.TrafficLightState,
) -> TrafficLightStatus:
    """Convert CARLA TrafficLightState to py123d TrafficLightStatus.

    Args:
        state: CARLA traffic light state.

    Returns:
        Corresponding py123d TrafficLightStatus.
    """
    _MAP = {
        carla.TrafficLightState.Green: TrafficLightStatus.GREEN,
        carla.TrafficLightState.Yellow: TrafficLightStatus.YELLOW,
        carla.TrafficLightState.Red: TrafficLightStatus.RED,
        carla.TrafficLightState.Off: TrafficLightStatus.OFF,
        carla.TrafficLightState.Unknown: TrafficLightStatus.UNKNOWN,
    }
    return _MAP.get(state, TrafficLightStatus.UNKNOWN)


def build_pinhole_camera_metadatas(
    config_expert: ExpertConfig,
    ego_metadata: EgoStateSE3Metadata,
    perturbation_translation: float = 0.0,
    perturbation_rotation: float = 0.0,
) -> dict[CameraID, PinholeCameraMetadata]:
    """Build 123D pinhole camera metadata for every LEAD camera.

    Args:
        config_expert: Expert configuration with ``sensor_rig.cameras``.
        ego_metadata: Ego vehicle metadata for rear axle offsets.
        perturbation_translation: Rig translation perturbation in meters; zero
            for the normal (unperturbated) rig.
        perturbation_rotation: Rig yaw perturbation in degrees; zero for the
            normal (unperturbated) rig.

    Returns:
        Mapping from camera ID to its pinhole metadata.
    """
    camera_metadatas: dict[CameraID, PinholeCameraMetadata] = {}
    for cam_key in range(1, config_expert.sensor_rig.num_cameras + 1):
        camera_id = py123d_log_api.CAMERA_ID_MAPPING[cam_key]
        calibration = config_expert.sensor_rig.cameras[cam_key - 1]
        width = calibration["width"]
        height = calibration["height"]
        fov = calibration["fov"]
        camera_pos = calibration["pos"]
        camera_rot = calibration["rot"]
        if perturbation_translation != 0.0 or perturbation_rotation != 0.0:
            perturbated = perturbated_pose(
                {
                    "x": camera_pos[0],
                    "y": camera_pos[1],
                    "z": camera_pos[2],
                    "roll": camera_rot[0],
                    "pitch": camera_rot[1],
                    "yaw": camera_rot[2],
                },
                perturbation_translation,
                perturbation_rotation,
            )
            camera_pos = [perturbated["x"], perturbated["y"], perturbated["z"]]
            camera_rot = [perturbated["roll"], perturbated["pitch"], perturbated["yaw"]]

        # https://github.com/carla-simulator/carla/issues/56
        focal_length = width / (2.0 * np.tan(fov * np.pi / 360.0))

        camera_to_imu_se3 = get_camera_extrinsic_as_iso(
            camera_pos=camera_pos,
            camera_rot=camera_rot,
            ego_metadata=ego_metadata,
        )
        camera_metadatas[camera_id] = PinholeCameraMetadata(
            camera_name=str(camera_id),
            camera_id=camera_id,
            intrinsics=PinholeIntrinsics(
                fx=focal_length,
                fy=focal_length,
                cx=width / 2.0,
                cy=height / 2.0,
            ),
            distortion=None,
            width=width,
            height=height,
            camera_to_imu_se3=camera_to_imu_se3,
            is_undistorted=True,
        )
    return camera_metadatas


def build_radar_metadatas(
    config_expert: ExpertConfig,
    perturbation_translation: float = 0.0,
    perturbation_rotation: float = 0.0,
) -> tuple[dict[RadarID, RadarMetadata], RadarMergedMetadata]:
    """Build 123D metadata for every radar of the rig, plus the merged stream.

    Args:
        config_expert: Expert configuration with the radar calibrations.
        perturbation_translation: Rig translation perturbation in meters.
        perturbation_rotation: Rig yaw perturbation in degrees.

    Returns:
        The per-radar metadata keyed by 123D radar ID, and the metadata of the
        merged stream the logs store all radars in.
    """
    ego_metadata = py123d_log_api.get_carla_lincoln_mkz_2020_metadata()
    radar_metadatas: dict[RadarID, RadarMetadata] = {}
    for sensor_index, calibration in enumerate(
        config_expert.sensor_rig.radars,
        start=1,
    ):
        pos = calibration["pos"]
        rot = calibration["rot"]
        if perturbation_translation or perturbation_rotation:
            pose = perturbated_pose(
                {
                    "x": pos[0],
                    "y": pos[1],
                    "z": pos[2],
                    "roll": rot[0],
                    "pitch": rot[1],
                    "yaw": rot[2],
                },
                perturbation_translation,
                perturbation_rotation,
            )
            pos = [pose["x"], pose["y"], pose["z"]]
            rot = [pose["roll"], pose["pitch"], pose["yaw"]]

        quaternion = quaternion_from_carla_rotation(
            carla.Rotation(roll=rot[0], pitch=rot[1], yaw=rot[2]),
        )
        radar_id = py123d_log_api.RADAR_ID_MAPPING[sensor_index]
        radar_metadatas[radar_id] = RadarMetadata(
            radar_name=str(radar_id),
            radar_id=radar_id,
            radar_to_imu_se3=PoseSE3(
                x=pos[0] + ego_metadata.rear_axle_to_center_longitudinal,
                y=-pos[1],  # Invert Y for ISO 8855
                z=pos[2],
                qw=quaternion.qw,
                qx=quaternion.qx,
                qy=quaternion.qy,
                qz=quaternion.qz,
            ),
        )
    return radar_metadatas, RadarMergedMetadata(radar_metadata_dict=radar_metadatas)


def build_lidar_metadata(config_expert: ExpertConfig) -> LidarMetadata:
    """Build 123D metadata for the top LiDAR.

    Args:
        config_expert: Expert configuration with LiDAR mounting position.

    Returns:
        LidarMetadata for the ego's LiDAR.
    """
    lidar_id = LidarID.LIDAR_TOP
    lidar_pos = config_expert.sensor_rig.lidar_pos_1
    lidar_rot = config_expert.sensor_rig.lidar_rot_1

    quaternion = quaternion_from_carla_rotation(
        carla.Rotation(roll=lidar_rot[0], pitch=lidar_rot[1], yaw=lidar_rot[2]),
    )

    lidar_to_imu_se3 = PoseSE3(
        x=lidar_pos[0],
        y=-lidar_pos[1],  # Invert Y
        z=lidar_pos[2],
        qw=quaternion.qw,
        qx=quaternion.qx,
        qy=quaternion.qy,
        qz=quaternion.qz,
    )

    return LidarMetadata(
        lidar_name=str(lidar_id),
        lidar_id=lidar_id,
        lidar_to_imu_se3=lidar_to_imu_se3,
    )
