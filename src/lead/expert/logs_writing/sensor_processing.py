"""Sensor rig specification, pose perturbation and per-tick sensor data processing of the expert."""

import copy
import logging

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from lead.common import constants
from lead.common.sensors import point_clouds
from lead.config import ExpertConfig
from lead.expert.driving import occlusion
from lead.expert.utils import sensor_decoding

LOG = logging.getLogger(__name__)


def av_sensor_setup(
    config: ExpertConfig,
    perturbation_rotation: float,
    perturbation_translation: float,
    lidar: bool,
    perturbate: bool,
    sensor_agent: bool,
    radar: bool = False,
) -> list[dict]:
    """
    Function to set up sensors for an autonomous vehicle (AV) simulation.

    Args:
        config: Configuration object containing sensor parameters
        perturbation_rotation: Rotation perturbation in degrees
        perturbation_translation: Translation perturbation in meters
        lidar: Whether to include the two LiDAR sensors
        perturbate: Whether to create perturbated sensor variants
        sensor_agent: Whether this is for a sensor agent (affects which sensors are created)
        radar: Whether to include Radar sensors
    Returns:
        List of sensor configurations
    """
    result = camera_sensor_setup(
        config,
        perturbation_rotation,
        perturbation_translation,
        perturbate,
        sensor_agent,
    )
    if lidar:
        result.extend(lidar_sensor_setup(config))
    if radar:
        for sensor_index, sensor_cfg in enumerate(config.sensor_rig.radars, start=1):
            result.append(
                {
                    "type": "sensor.other.radar",
                    "x": sensor_cfg["pos"][0],
                    "y": sensor_cfg["pos"][1],
                    "z": sensor_cfg["pos"][2],
                    "roll": sensor_cfg["rot"][0],
                    "pitch": sensor_cfg["rot"][1],
                    "yaw": sensor_cfg["rot"][2],
                    "horizontal_fov": sensor_cfg["horz_fov"],
                    "vertical_fov": sensor_cfg["vert_fov"],
                    "id": f"radar{sensor_index}",
                },
            )
            LOG.info(
                f"Added sensor: {result[-1]['id']} at position ({sensor_cfg['pos'][0]}, {sensor_cfg['pos'][1]}, {sensor_cfg['pos'][2]})",  # noqa: E501
            )
        if perturbate:
            for sensor_index, sensor_cfg in enumerate(
                config.sensor_rig.radars,
                start=1,
            ):
                result.append(
                    perturbated_sensor_cfg(
                        {
                            "type": "sensor.other.radar",
                            "x": sensor_cfg["pos"][0],
                            "y": sensor_cfg["pos"][1],
                            "z": sensor_cfg["pos"][2],
                            "roll": sensor_cfg["rot"][0],
                            "pitch": sensor_cfg["rot"][1],
                            "yaw": sensor_cfg["rot"][2],
                            "horizontal_fov": sensor_cfg["horz_fov"],
                            "vertical_fov": sensor_cfg["vert_fov"],
                            "id": f"radar{sensor_index}_perturbated",
                        },
                        perturbation_translation,
                        perturbation_rotation,
                    ),
                )
                LOG.info(
                    f"Added sensor: {result[-1]['id']} at position ({result[-1]['x']}, {result[-1]['y']}, {result[-1]['z']})",
                )

    result.append(
        {
            "type": "sensor.other.imu",
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": 0.0,
            "sensor_tick": config.simulation.carla_frame_rate,
            "id": "imu",
        },
    )
    LOG.info(f"Added sensor: {result[-1]['id']}")
    result.append(
        {
            "type": "sensor.other.gnss",
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": 0.0,
            "sensor_tick": 0.01,
            "id": "gps",
        },
    )
    LOG.info(f"Added sensor: {result[-1]['id']}")
    result.append(
        {
            "type": "sensor.speedometer",
            "reading_frequency": config.simulation.carla_fps,
            "id": "speed",
        },
    )
    LOG.info(f"Added sensor: {result[-1]['id']}")
    return result


def lidar_sensor_setup(config: ExpertConfig) -> list[dict]:
    """Build the two-LiDAR rig specification.

    The rig always consists of two LiDARs; a single-LiDAR setup is not supported.

    Args:
        config: Configuration object containing the LiDAR mounting poses.

    Returns:
        List with the two LiDAR sensor configurations.
    """
    result = []
    for lidar_id, pos, rot in [
        ("lidar1", config.sensor_rig.lidar_pos_1, config.sensor_rig.lidar_rot_1),
        ("lidar2", config.sensor_rig.lidar_pos_2, config.sensor_rig.lidar_rot_2),
    ]:
        result.append(
            {
                "type": "sensor.lidar.ray_cast",
                "x": pos[0],
                "y": pos[1],
                "z": pos[2],
                "roll": rot[0],
                "pitch": rot[1],
                "yaw": rot[2],
                "id": lidar_id,
            },
        )
        LOG.info(
            f"Added sensor: {lidar_id} at position ({pos[0]}, {pos[1]}, {pos[2]})",
        )
    return result


def camera_sensor_setup(
    config: ExpertConfig,
    perturbation_rotation: float,
    perturbation_translation: float,
    perturbate: bool,
    sensor_agent: bool,
) -> list:
    """Set up camera sensors for the given configuration.

    Args:
        config: Configuration object containing camera parameters
        perturbation_rotation: Rotation perturbation in degrees
        perturbation_translation: Translation perturbation in meters
        perturbate: Whether to create perturbated sensor variants
        sensor_agent: Whether this is for a sensor agent (affects which sensors are created)

    Returns:
        List of camera sensor configurations
    """
    result = []

    # Cameras consumed only on save ticks render at the save rate, while the
    # non-perturbated instance segmentation stays at full rate to feed the
    # expert's per-tick occlusion checks. ``sensor_tick`` must be an exact
    # multiple of the frame time, since UE carries leftover time into the next
    # interval and a fractional value alternates short/long capture intervals.
    save_tick_sensor_tick = (
        config.data_collection.data_save_freq * config.simulation.carla_frame_rate
    )

    for idx, cam_config in enumerate(config.sensor_rig.cameras, start=1):
        cam_pos = cam_config["pos"]
        cam_rot = cam_config["rot"]
        cam_width = cam_config["width"]
        cam_height = cam_config["height"]
        camera_fov = cam_config["fov"]
        suffix = f"_{idx}"

        # RGB camera
        result.append(
            {
                "type": "sensor.camera.rgb",
                "x": cam_pos[0],
                "y": cam_pos[1],
                "z": cam_pos[2],
                "roll": cam_rot[0],
                "pitch": cam_rot[1],
                "yaw": cam_rot[2],
                "width": cam_width,
                "height": cam_height,
                "fov": camera_fov,
                "id": f"rgb{suffix}",
            },
        )
        if not sensor_agent:
            result[-1]["sensor_tick"] = save_tick_sensor_tick
        LOG.info(
            f"Added sensor: {result[-1]['id']} at position ({cam_pos[0]}, {cam_pos[1]}, {cam_pos[2]}), size: {cam_width}x{cam_height}px",  # noqa: E501
        )

        if not sensor_agent:
            # GT sensors
            for base_id, sensor_type in [
                ("depth", "sensor.camera.depth"),
                ("instance", "sensor.camera.instance_segmentation"),
            ]:
                result.append(
                    {
                        "type": sensor_type,
                        "x": cam_pos[0],
                        "y": cam_pos[1],
                        "z": cam_pos[2],
                        "roll": cam_rot[0],
                        "pitch": cam_rot[1],
                        "yaw": cam_rot[2],
                        "width": cam_width,
                        "height": cam_height,
                        "fov": camera_fov,
                        "id": f"{base_id}{suffix}",
                    },
                )
                if base_id == "depth":
                    result[-1]["sensor_tick"] = save_tick_sensor_tick
                LOG.info(
                    f"Added sensor: {result[-1]['id']} at position ({cam_pos[0]}, {cam_pos[1]}, {cam_pos[2]}), size: {cam_width}x{cam_height}px",  # noqa: E501
                )

            # perturbated views
            if perturbate:
                for base_id, sensor_type in [
                    ("rgb", "sensor.camera.rgb"),
                    ("depth", "sensor.camera.depth"),
                    ("instance", "sensor.camera.instance_segmentation"),
                ]:
                    result.append(
                        perturbated_sensor_cfg(
                            {
                                "type": sensor_type,
                                "x": cam_pos[0],
                                "y": cam_pos[1],
                                "z": cam_pos[2],
                                "roll": cam_rot[0],
                                "pitch": cam_rot[1],
                                "yaw": cam_rot[2],
                                "width": cam_width,
                                "height": cam_height,
                                "fov": camera_fov,
                                "id": f"{base_id}{suffix}_perturbated",
                                "sensor_tick": save_tick_sensor_tick,
                            },
                            perturbation_translation,
                            perturbation_rotation,
                        ),
                    )
                    LOG.info(
                        f"Added perturbated sensor: {result[-1]['id']} at position ({result[-1]['x']}, {result[-1]['y']}, {result[-1]['z']}), size: {result[-1]['width']}x{result[-1]['height']}px",  # noqa: E501
                    )

    return result


def perturbated_sensor_cfg(
    sensor_cfg: dict[str, float | str],
    perturbation_translation: float,
    perturbation_rotation: float,
    perturbation_roll: float = 0.0,
    perturbation_pitch: float = 0.0,
) -> dict[str, float | str]:
    """Apply a 3D rigid transformation (rotation + translation) to a sensor pose.

    Args:
        sensor_cfg: Dictionary containing the original sensor configuration
            with keys "x", "y", "z", "roll", "pitch", "yaw".
        perturbation_translation: Translation offset to apply along the Y-axis
            of the perturbation frame, in the same units as the position.
        perturbation_rotation: Rotation angle around the Z-axis (yaw), in degrees.
        perturbation_roll: Additional rotation angle around the X-axis (roll), in degrees.
        perturbation_pitch: Additional rotation angle around the Y-axis (pitch), in degrees.

    Returns:
        dict[str, float | str]: A new sensor configuration dictionary with updated
        "x", "y", "z", "roll", "pitch", and "yaw" values after applying
        the rigid transformation.
    """

    # Original pose
    pos = np.array([sensor_cfg["x"], sensor_cfg["y"], sensor_cfg["z"]])
    R0 = R.from_euler(
        "xyz",
        [sensor_cfg["roll"], sensor_cfg["pitch"], sensor_cfg["yaw"]],
        degrees=True,
    )

    # perturbation transform
    R_aug = R.from_euler(
        "xyz",
        [perturbation_roll, perturbation_pitch, perturbation_rotation],
        degrees=True,
    )
    t_aug = np.array(
        [0, perturbation_translation, 0],
    )  # translate along Y of perturbate frame

    # Apply 3D rigid transform
    pos_new = R_aug.apply(pos) + t_aug
    R_new = R_aug * R0  # compose rotations

    # Back to Euler
    roll, pitch, yaw = R_new.as_euler("xyz", degrees=True)

    sensor_cfg = copy.deepcopy(sensor_cfg)
    sensor_cfg.update(
        {
            "x": float(pos_new[0]),
            "y": float(pos_new[1]),
            "z": float(pos_new[2]),
            "roll": float(roll),
            "pitch": float(pitch),
            "yaw": float(yaw),
        },
    )
    return sensor_cfg


def sample_sensor_perturbation_parameters(
    config: ExpertConfig,
    max_speed_limit_route: float,
    min_lane_width_route: float,
) -> tuple[float, float]:
    """
    Sample sensor perturbation parameters (translation and rotation) based on the route's speed limit and lane width.
    Args:
        config: Configuration object containing perturbation parameters.
        max_speed_limit_route: Maximum speed limit along the route (in m/s).
        min_lane_width_route: Minimum lane width along the route (in meters).
    Returns:
        tuple[float, float]: A tuple containing the perturbation translation (in meters) and perturbation rotation (in degrees).
    """
    safety_translation_perturbation_gap = (
        config.perturbation.default_safety_translation_perturbation_penalty
    )
    if max_speed_limit_route < constants.URBAN_MAX_SPEED_LIMIT:
        safety_translation_perturbation_gap = (
            config.perturbation.urban_safety_translation_perturbation_penalty
        )
    lateral_gap = max(
        min_lane_width_route / 2.0
        - config.simulation.ego_extent_y / 2
        - safety_translation_perturbation_gap,
        0.1,
    )
    tmax = min(config.perturbation.camera_translation_perturbation_max, lateral_gap)

    # Pick perturbation translation shift
    perturbation_translation = np.random.choice([-1, 1]) * np.random.uniform(
        low=config.perturbation.camera_translation_perturbation_min,
        high=tmax,
    )

    # Next, pick rotation perturbation ranges, depending on translation perturbation.
    # Interpolate perturbation rotation depends on translation perturbation to avoid unrealistic configurations.
    neg_range = (
        -config.perturbation.camera_rotation_perturbation_min,
        config.perturbation.camera_rotation_perturbation_max,
    )  # for t <= -1
    pos_range = (
        -config.perturbation.camera_rotation_perturbation_max,
        config.perturbation.camera_rotation_perturbation_min,
    )  # for t >= +1
    if (
        perturbation_translation
        <= -config.perturbation.camera_translation_perturbation_max
    ):
        rmin, rmax = neg_range
    elif (
        perturbation_translation
        >= config.perturbation.camera_translation_perturbation_max
    ):
        rmin, rmax = pos_range
    else:
        alpha = (
            perturbation_translation
            + config.perturbation.camera_translation_perturbation_max
        ) / (
            2.0 * config.perturbation.camera_translation_perturbation_max
        )  # maps to [0,1]
        rmin = -(-neg_range[0] * (1 - alpha) + -pos_range[0] * alpha)
        rmax = neg_range[1] * (1 - alpha) + pos_range[1] * alpha

    beta = tmax / config.perturbation.camera_translation_perturbation_max  # in [0,1]
    rmin *= beta
    rmax *= beta

    # Rejection sampling of rotation perturbation to avoid useless small rotation angles
    perturbation_rotation = np.random.uniform(low=rmin, high=rmax)
    while abs(perturbation_rotation) < config.perturbation.camera_rotation_epsilon:
        perturbation_rotation = np.random.uniform(low=rmin, high=rmax)

    LOG.info(f"Translation perturbation range: [-{tmax:.2f}m, {tmax:.2f}m].")
    LOG.info(f"Sampled translation: {perturbation_translation:.2f}m.")
    LOG.info(
        f"rotation range: [{rmin:.2f}°, {rmax:.2f}°], sampled rotation: {perturbation_rotation:.2f}°",
    )

    return perturbation_translation, perturbation_rotation


class SensorProcessingMixin:
    """Builds the sensor specification and processes raw sensor data per tick."""

    def sensors(self):
        """
        Returns a list of sensor specifications for the ego vehicle.

        Each sensor specification is a dictionary containing the sensor type,
        reading frequency, position, and other relevant parameters.

        Returns:
            list: A list of sensor specification dictionaries.
        """
        result = []
        if not self.config_expert.data_collection.datagen:
            result = [
                {
                    "type": "sensor.opendrive_map",
                    "reading_frequency": 1e-6,
                    "id": "hd_map",
                },
                {
                    "type": "sensor.other.imu",
                    "x": 0.0,
                    "y": 0.0,
                    "z": 0.0,
                    "roll": 0.0,
                    "pitch": 0.0,
                    "yaw": 0.0,
                    "sensor_tick": 0.05,
                    "id": "imu",
                },
                {"type": "sensor.speedometer", "reading_frequency": 20, "id": "speed"},
                {
                    "type": "sensor.other.gnss",
                    "x": 0.0,
                    "y": 0.0,
                    "z": 0.0,
                    "roll": 0.0,
                    "pitch": 0.0,
                    "yaw": 0.0,
                    "sensor_tick": 0.01,
                    "id": "gps",
                },
            ]

        self.perturbation_translation, self.perturbation_rotation = (
            sample_sensor_perturbation_parameters(
                config=self.config_expert,
                max_speed_limit_route=self.max_speed_limit_route,
                min_lane_width_route=self.min_lane_width_route,
            )
        )

        # --- Set up sensor rig ---
        if self.save_path is not None and self.config_expert.data_collection.datagen:
            result += av_sensor_setup(
                self.config_expert,
                perturbation_rotation=self.perturbation_rotation,
                perturbation_translation=self.perturbation_translation,
                lidar=True,
                perturbate=self.config_expert.perturbation.perturbate_sensors,
                sensor_agent=False,
                radar=self.config_expert.sensor_rig.use_radars,
            )
        else:
            result += lidar_sensor_setup(self.config_expert)
        return result

    def tick(self, input_data: dict) -> dict:
        """
        Get the current state of the vehicle from the input data and the vehicle's sensors.

        Args:
            input_data: Input data containing sensor information.

        Returns:
            A dictionary containing the vehicle's position (GPS), speed, and compass heading.
        """
        input_data = super().tick(input_data)
        if (
            self.config_expert.sensor_rig.use_radars
            and self.config_expert.data_collection.datagen
        ):
            radar_arrays = []
            for i in range(1, self.config_expert.sensor_rig.num_radar_sensors + 1):
                radar_arrays.append(input_data[f"radar{i}"])
            input_data["radar"] = np.concatenate(radar_arrays, axis=0)
        # Data that only feeds the recorders is processed on save ticks only.
        # Save-tick cameras render every ``data_save_freq`` ticks via their
        # ``sensor_tick``; their capture phase is not aligned with ``self.step``,
        # so data presence defines the save tick.
        save_tick_keys = []
        for camera_idx in range(1, self.config_expert.sensor_rig.num_cameras + 1):
            save_tick_keys += [f"rgb_{camera_idx}", f"depth_{camera_idx}"]
            if self.config_expert.perturbation.perturbate_sensors:
                save_tick_keys += [
                    f"rgb_{camera_idx}_perturbated",
                    f"depth_{camera_idx}_perturbated",
                    f"instance_{camera_idx}_perturbated",
                ]
        missing_keys = [key for key in save_tick_keys if key not in input_data]
        is_save_tick = len(missing_keys) == 0
        if missing_keys and len(missing_keys) < len(save_tick_keys):
            raise RuntimeError(
                f"Partial save tick at step {self.step}, missing: {missing_keys}",
            )
        self.is_save_tick = is_save_tick
        process_perturbated = (
            self.config_expert.perturbation.perturbate_sensors and is_save_tick
        )
        if self.save_path is not None and self.config_expert.data_collection.datagen:
            if process_perturbated:
                # Process perturbated RGB images for each camera
                for camera_idx in range(
                    1,
                    self.config_expert.sensor_rig.num_cameras + 1,
                ):
                    input_data[f"rgb_{camera_idx}_perturbated"] = input_data[
                        f"rgb_{camera_idx}_perturbated"
                    ][1][:, :, :3]

            if self.config_expert.sensor_rig.use_radars and process_perturbated:
                radar_perturbated_dict = {}
                for i in range(1, self.config_expert.sensor_rig.num_radar_sensors + 1):
                    radar_perturbated = point_clouds.radar_points_to_ego(
                        input_data[f"radar{i}_perturbated"][1],
                        sensor_pos=self.config_expert.sensor_rig.radars[i - 1]["pos"],
                        sensor_rot=self.config_expert.sensor_rig.radars[i - 1]["rot"],
                    )
                    radar_perturbated_dict[f"radar{i}_perturbated"] = radar_perturbated

                input_data.update(radar_perturbated_dict)

            # Instance segmentation - flexible camera processing. Needed every
            # tick: the pixel counts drive the expert's occlusion checks.
            for camera_idx in range(1, self.config_expert.sensor_rig.num_cameras + 1):
                instance = cv2.cvtColor(
                    input_data[f"instance_{camera_idx}"][1][:, :, :3],
                    cv2.COLOR_BGR2RGB,
                )
                input_data[f"instance_{camera_idx}"] = instance
                input_data[f"converted_instance_{camera_idx}"] = (
                    sensor_decoding.convert_instance_segmentation(instance)
                )

            if process_perturbated:
                for camera_idx in range(
                    1,
                    self.config_expert.sensor_rig.num_cameras + 1,
                ):
                    instance_perturbated = cv2.cvtColor(
                        input_data[f"instance_{camera_idx}_perturbated"][1][:, :, :3],
                        cv2.COLOR_BGR2RGB,
                    )
                    input_data[f"instance_{camera_idx}_perturbated"] = (
                        instance_perturbated
                    )
                    input_data[f"converted_instance_{camera_idx}_perturbated"] = (
                        sensor_decoding.convert_instance_segmentation(
                            instance_perturbated,
                        )
                    )

            # Depth - flexible camera processing
            if is_save_tick:
                for camera_idx in range(
                    1,
                    self.config_expert.sensor_rig.num_cameras + 1,
                ):
                    input_data[f"depth_{camera_idx}"] = sensor_decoding.convert_depth(
                        input_data[f"depth_{camera_idx}"][1][:, :, :3],
                    )

            if process_perturbated:
                for camera_idx in range(
                    1,
                    self.config_expert.sensor_rig.num_cameras + 1,
                ):
                    input_data[f"depth_{camera_idx}_perturbated"] = (
                        sensor_decoding.convert_depth(
                            input_data[f"depth_{camera_idx}_perturbated"][1][:, :, :3],
                        )
                    )

            # Per-actor visible pixel counts from the instance segmentation cameras
            input_data["instance_pixel_counts"] = occlusion.instance_pixel_counts(
                [
                    input_data[f"converted_instance_{camera_idx}"]
                    for camera_idx in range(
                        1,
                        self.config_expert.sensor_rig.num_cameras + 1,
                    )
                ],
            )

        # Bounding box (also refreshes ``self.id2actor_map``)
        input_data["bounding_boxes"] = self.get_bounding_boxes(input_data=input_data)
        self.stored_bounding_boxes_of_this_step = input_data["bounding_boxes"]
        self.id2bb_map = {bb["id"]: bb for bb in input_data["bounding_boxes"]}

        self.tick_data = input_data

        return input_data
