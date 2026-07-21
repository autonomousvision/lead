"""Per-tick sensor data processing of the expert."""

import cv2
import numpy as np

from lead.common.sensors import point_clouds
from lead.common.sensors.av_sensor_setup import (
    av_sensor_setup,
    lidar_sensor_setup,
    sample_sensor_perturbation_parameters,
)
from lead.expert.driving import occlusion
from lead.expert.utils import sensor_decoding


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
