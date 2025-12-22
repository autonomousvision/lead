import json
import logging
import os
import random
from copy import copy

import carla
import cv2
import jaxtyping as jt
import laspy
import matplotlib
import numpy as np
import numpy.typing as npt
import torch
from agents.navigation.local_planner import RoadOption
from agents.tools.misc import compute_distance, is_within_distance
from beartype import beartype
from leaderboard.autoagents import autonomous_agent_local
from shapely.geometry import Polygon
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

import lead.common.common_utils as common_utils
import lead.expert.expert_utils as expert_utils
from lead.common import constants, weathers
from lead.common.constants import (
    CameraPointCloudIndex,
    CarlaSemanticSegmentationClass,
    TransfuserSemanticSegmentationClass,
    WeatherVisibility,
)
from lead.common.logging_config import setup_logging
from lead.expert.expert_base import ExpertBase

matplotlib.use("Agg")  # non-GUI backend for headless servers


setup_logging()
LOG = logging.getLogger(__name__)


def get_entry_point() -> str:
    return "Expert"


class Expert(ExpertBase, autonomous_agent_local.AutonomousAgent):
    @beartype
    def setup(
        self, path_to_conf_file: str, route_index: str | None = None, traffic_manager: carla.TrafficManager | None = None
    ):
        super().setup()
        LOG.info("Setup")
        self.expert_setup(path_to_conf_file, route_index, traffic_manager)

    @beartype
    def _init(self, hd_map) -> None:
        LOG.info("Init")
        self.expert_init(hd_map)

    @beartype
    def tick(self, input_data: dict) -> dict:
        """
        Get the current state of the vehicle from the input data and the vehicle's sensors.

        Args:
            input_data (dict): Input data containing sensor information.

        Returns:
            dict: A dictionary containing the vehicle's position (GPS), speed, and compass heading.
        """
        input_data = super().tick(input_data)
        self.transform_queue.append(self._vehicle.get_transform())
        if self.config_expert.use_radars and self.config_expert.datagen:
            radar_arrays = []
            for i in range(1, self.config_expert.num_radar_sensors + 1):
                radar_arrays.append(input_data[f"radar{i}"])
            input_data["radar"] = np.concatenate(radar_arrays, axis=0)
        if self.save_path is not None and self.config_expert.datagen:
            if self.config_expert.perturbate_sensors:
                # Process perturbated RGB images for each camera
                rgb_perturbated_cameras = []
                for camera_idx in range(1, self.config_expert.num_cameras + 1):
                    rgb_perturbated = input_data[f"rgb_{camera_idx}_perturbated"][1][:, :, :3]
                    input_data[f"rgb_{camera_idx}_perturbated"] = rgb_perturbated
                    rgb_perturbated_cameras.append(rgb_perturbated)
                input_data["rgb_perturbated"] = np.concatenate(rgb_perturbated_cameras, axis=1)

            if self.config_expert.use_radars and self.config_expert.perturbate_sensors:
                radar_perturbated_dict = {}
                for i in range(1, self.config_expert.num_radar_sensors + 1):
                    radar_perturbated = common_utils.radar_points_to_ego(
                        input_data[f"radar{i}_perturbated"][1],
                        sensor_pos=self.config_expert.radar_calibration[str(i)]["pos"],
                        sensor_rot=self.config_expert.radar_calibration[str(i)]["rot"],
                    )
                    radar_perturbated_dict[f"radar{i}_perturbated"] = radar_perturbated

                input_data.update(radar_perturbated_dict)

            # Instance segmentation - flexible camera processing
            instances = []
            converted_instances = []
            for camera_idx in range(1, self.config_expert.num_cameras + 1):
                instance = cv2.cvtColor(input_data[f"instance_{camera_idx}"][1][:, :, :3], cv2.COLOR_BGR2RGB)
                converted_instance = expert_utils.convert_instance_segmentation(instance)

                input_data[f"instance_{camera_idx}"] = instance
                input_data[f"converted_instance_{camera_idx}"] = converted_instance
                instances.append(instance)
                converted_instances.append(converted_instance)

            input_data["instance"] = np.concatenate(instances, axis=1)

            if self.config_expert.perturbate_sensors:
                instances_perturbated = []
                converted_instances_perturbated = []
                for camera_idx in range(1, self.config_expert.num_cameras + 1):
                    instance_perturbated = cv2.cvtColor(
                        input_data[f"instance_{camera_idx}_perturbated"][1][:, :, :3], cv2.COLOR_BGR2RGB
                    )
                    converted_instance_perturbated = expert_utils.convert_instance_segmentation(instance_perturbated)

                    input_data[f"instance_{camera_idx}_perturbated"] = instance_perturbated
                    input_data[f"converted_instance_{camera_idx}_perturbated"] = converted_instance_perturbated
                    instances_perturbated.append(instance_perturbated)
                    converted_instances_perturbated.append(converted_instance_perturbated)

                input_data["instance_perturbated"] = np.concatenate(instances_perturbated, axis=1)

            # Standard semantics with some details we don't learn but will be useful to enhance the depth map
            semantics_standard_cameras = []
            for camera_idx in range(1, self.config_expert.num_cameras + 1):
                semantics_standard = input_data[f"semantics_{camera_idx}"][1][:, :, 2]
                input_data[f"semantics_{camera_idx}"] = semantics_standard
                semantics_standard_cameras.append(semantics_standard)
            input_data["semantics"] = np.concatenate(semantics_standard_cameras, axis=1)

            if self.config_expert.perturbate_sensors:
                semantics_perturbated_standard_cameras = []
                for camera_idx in range(1, self.config_expert.num_cameras + 1):
                    semantics_perturbated_standard = input_data[f"semantics_{camera_idx}_perturbated"][1][:, :, 2]
                    input_data[f"semantics_{camera_idx}_perturbated"] = semantics_perturbated_standard
                    semantics_perturbated_standard_cameras.append(semantics_perturbated_standard)
                input_data["semantics_perturbated"] = np.concatenate(semantics_perturbated_standard_cameras, axis=1)

            # Depth - flexible camera processing
            depth_cameras = []
            for camera_idx in range(1, self.config_expert.num_cameras + 1):
                depth = expert_utils.convert_depth(input_data[f"depth_{camera_idx}"][1][:, :, :3])
                depth = self.enhance_depth(
                    depth, input_data[f"semantics_{camera_idx}"], input_data[f"converted_instance_{camera_idx}"]
                )
                input_data[f"depth_{camera_idx}"] = depth
                depth_cameras.append(depth)
            input_data["depth"] = np.concatenate(depth_cameras, axis=1)

            if self.config_expert.perturbate_sensors:
                depth_perturbated_cameras = []
                for camera_idx in range(1, self.config_expert.num_cameras + 1):
                    perturbated_depth = expert_utils.convert_depth(input_data[f"depth_{camera_idx}_perturbated"][1][:, :, :3])
                    perturbated_depth = self.enhance_depth(
                        perturbated_depth,
                        input_data[f"semantics_{camera_idx}_perturbated"],
                        input_data[f"converted_instance_{camera_idx}_perturbated"],
                    )
                    input_data[f"depth_{camera_idx}_perturbated"] = perturbated_depth
                    depth_perturbated_cameras.append(perturbated_depth)
                input_data["depth_perturbated"] = np.concatenate(depth_perturbated_cameras, axis=1)

            # Semantics segmentation using first channel of instance segmentation
            # After enhancing the depth map, we use the first channel of instance segmentation which has cleaner labels
            semantics_cameras = []
            for camera_idx in range(1, self.config_expert.num_cameras + 1):
                semantics = input_data[f"converted_instance_{camera_idx}"][..., 0]
                input_data[f"semantics_{camera_idx}"] = semantics
                semantics_cameras.append(semantics)
            input_data["semantics"] = np.concatenate(semantics_cameras, axis=1)

            if self.config_expert.perturbate_sensors:
                semantics_perturbated_cameras = []
                for camera_idx in range(1, self.config_expert.num_cameras + 1):
                    semantics_perturbated = input_data[f"converted_instance_{camera_idx}_perturbated"][..., 0]
                    input_data[f"semantics_{camera_idx}_perturbated"] = semantics_perturbated
                    semantics_perturbated_cameras.append(semantics_perturbated)
                input_data["semantics_perturbated"] = np.concatenate(semantics_perturbated_cameras, axis=1)

            # Camera point cloud - flexible camera configuration
            camera_pcs = []
            for camera_idx in range(1, self.config_expert.num_cameras + 1):
                cam_config = self.config_expert.camera_calibration[camera_idx]
                input_data[f"semantics_camera_pc_{camera_idx}"] = expert_utils.semantics_camera_pc(
                    input_data[f"depth_{camera_idx}"],
                    instance=input_data[f"converted_instance_{camera_idx}"],
                    camera_fov=cam_config["fov"],
                    camera_width=cam_config["width"],
                    camera_height=cam_config["height"],
                    camera_pos=cam_config["pos"],
                    camera_rot=cam_config["rot"],
                    perturbation_rotation=0.0,
                    perturbation_translation=0.0,
                )
                camera_pcs.append(input_data[f"semantics_camera_pc_{camera_idx}"])

            if self.config_expert.perturbate_sensors:
                for camera_idx in range(1, self.config_expert.num_cameras + 1):
                    cam_config = self.config_expert.camera_calibration[camera_idx]
                    input_data[f"semantics_camera_pc_{camera_idx}_perturbated"] = expert_utils.semantics_camera_pc(
                        input_data[f"depth_{camera_idx}_perturbated"],
                        instance=input_data[f"converted_instance_{camera_idx}_perturbated"],
                        camera_fov=cam_config["fov"],
                        camera_width=cam_config["width"],
                        camera_height=cam_config["height"],
                        camera_pos=cam_config["pos"],
                        camera_rot=cam_config["rot"],
                        perturbation_rotation=self.perturbation_rotation,
                        perturbation_translation=self.perturbation_translation,
                    )
            torch.cuda.synchronize()
            # Concatenate the unprojection together
            input_data["semantics_camera_pc"] = torch.cat(camera_pcs, dim=0).cpu().numpy()

            if self.config_expert.perturbate_sensors:
                camera_pcs_perturbated = [
                    input_data[f"semantics_camera_pc_{i}_perturbated"] for i in range(1, self.config_expert.num_cameras + 1)
                ]
                input_data["semantics_camera_pc_perturbated"] = torch.cat(camera_pcs_perturbated, dim=0).cpu().numpy()

            input_data["semantics_camera_pc_all"] = input_data["semantics_camera_pc"]
            if self.config_expert.perturbate_sensors:
                input_data["semantics_camera_pc_all"] = np.concatenate(
                    (input_data["semantics_camera_pc_all"], input_data["semantics_camera_pc_perturbated"]), axis=0
                )

        # Bounding box
        input_data["bounding_boxes"] = self.get_bounding_boxes(input_data=input_data)
        self.stored_bounding_boxes_of_this_step = input_data["bounding_boxes"]
        self.data_agent_id_to_bb_map = {bb["id"]: bb for bb in input_data["bounding_boxes"]}
        self.data_agent_id_to_actor_map = {
            actor.id: actor for actor in self._world.get_actors() if actor.is_alive and actor.id in self.data_agent_id_to_bb_map
        }

        # BEV Semantic
        self.stop_sign_criteria.tick(self._vehicle)
        input_data["hdmap"] = self.ss_bev_manager.get_observation(self.close_traffic_lights)["hdmap_classes"]
        if self.config_expert.perturbate_sensors:
            input_data["hdmap_perturbated"] = self.ss_bev_manager_perturbated.get_observation(self.close_traffic_lights)[
                "hdmap_classes"
            ]

        # --- Update semantic segmentation to make cones, traffic warning and special vehicles labels ---
        construction_meshes_id_map = expert_utils.match_unreal_engine_ids_to_carla_bounding_boxes_ids(
            self.ego_matrix,
            CarlaSemanticSegmentationClass.Dynamic,
            [box for box in input_data["bounding_boxes"] if box.get("type_id") in constants.CONSTRUCTION_MESHES],
            input_data["semantics_camera_pc_all"],
        )
        emergency_meshes_id_map = expert_utils.match_unreal_engine_ids_to_carla_bounding_boxes_ids(
            self.ego_matrix,
            CarlaSemanticSegmentationClass.Car,
            [box for box in input_data["bounding_boxes"] if box.get("type_id") in constants.EMERGENCY_MESHES],
            input_data["semantics_camera_pc_all"],
            penalize_points_outside=True,
        )
        emergency_meshes_id_map.update(
            expert_utils.match_unreal_engine_ids_to_carla_bounding_boxes_ids(
                self.ego_matrix,
                CarlaSemanticSegmentationClass.Truck,
                [box for box in input_data["bounding_boxes"] if box.get("type_id") in constants.EMERGENCY_MESHES],
                input_data["semantics_camera_pc_all"],
            )
        )
        stop_sign_meshes_id_map = expert_utils.match_unreal_engine_ids_to_carla_actors_ids(
            self.ego_matrix,
            CarlaSemanticSegmentationClass.TrafficSign,
            self.get_nearby_object(
                self.ego_location, self._world.get_actors().filter("*traffic.stop*"), self.config_expert.light_radius
            ),
            input_data["semantics_camera_pc_all"],
        )
        if len(construction_meshes_id_map) > 0 or len(emergency_meshes_id_map) > 0 or len(stop_sign_meshes_id_map) > 0:
            for camera_idx in range(1, self.config_expert.num_cameras + 1):
                input_data[f"semantics_{camera_idx}"] = expert_utils.enhance_semantics_segmentation(
                    input_data[f"converted_instance_{camera_idx}"],
                    input_data.get(f"semantics_{camera_idx}"),
                    construction_meshes_id_map,
                    CarlaSemanticSegmentationClass.ConeAndTrafficWarning,
                )
                input_data[f"semantics_{camera_idx}"] = expert_utils.enhance_semantics_segmentation(
                    input_data[f"converted_instance_{camera_idx}"],
                    input_data.get(f"semantics_{camera_idx}"),
                    emergency_meshes_id_map,
                    CarlaSemanticSegmentationClass.SpecialVehicles,
                )
                input_data[f"semantics_{camera_idx}"] = expert_utils.enhance_semantics_segmentation(
                    input_data[f"converted_instance_{camera_idx}"],
                    input_data.get(f"semantics_{camera_idx}"),
                    stop_sign_meshes_id_map,
                    CarlaSemanticSegmentationClass.StopSign,
                )

            # Concatenate semantics from all cameras
            semantics_cameras = [input_data[f"semantics_{i}"] for i in range(1, self.config_expert.num_cameras + 1)]
            input_data["semantics"] = np.concatenate(semantics_cameras, axis=1)

            if self.config_expert.perturbate_sensors:
                for camera_idx in range(1, self.config_expert.num_cameras + 1):
                    input_data[f"semantics_{camera_idx}_perturbated"] = expert_utils.enhance_semantics_segmentation(
                        input_data[f"converted_instance_{camera_idx}_perturbated"],
                        input_data.get(f"semantics_{camera_idx}_perturbated"),
                        construction_meshes_id_map,
                        CarlaSemanticSegmentationClass.ConeAndTrafficWarning,
                    )
                    input_data[f"semantics_{camera_idx}_perturbated"] = expert_utils.enhance_semantics_segmentation(
                        input_data[f"converted_instance_{camera_idx}_perturbated"],
                        input_data.get(f"semantics_{camera_idx}_perturbated"),
                        emergency_meshes_id_map,
                        CarlaSemanticSegmentationClass.SpecialVehicles,
                    )
                    input_data[f"semantics_{camera_idx}_perturbated"] = expert_utils.enhance_semantics_segmentation(
                        input_data[f"converted_instance_{camera_idx}_perturbated"],
                        input_data.get(f"semantics_{camera_idx}_perturbated"),
                        stop_sign_meshes_id_map,
                        CarlaSemanticSegmentationClass.StopSign,
                    )

                # Concatenate perturbated semantics from all cameras
                semantics_perturbated_cameras = [
                    input_data[f"semantics_{i}_perturbated"] for i in range(1, self.config_expert.num_cameras + 1)
                ]
                input_data["semantics_perturbated"] = np.concatenate(semantics_perturbated_cameras, axis=1)

        self.tick_data = input_data

        return input_data

    @beartype
    def run_step(self, input_data: dict, timestamp: float, world_map) -> carla.VehicleControl:
        """
        Entry main function!
        Run a single step of the agent's control loop.

        Args:
            input_data: Input data for the current step.
            timestamp: Current timestamp.
            world_map: The CARLA world map.

        Returns:
            The control commands (steer, throttle, brake).

        Raises:
            RuntimeError: If the agent is not initialized before calling this method.
        """
        # Initialize the agent if not done yet
        if not self.initialized:
            self._init(None)

        self.step += 1

        if self.config_expert.datagen:
            self.perturbate_camera()

        self.update_3rd_person_camera()

        input_data = self.tick(input_data)

        # Get the control commands and driving data for the current step
        target_speed, control, speed_reduced_by_obj = self._get_control()

        if input_data is not None and "bounding_boxes" in input_data:
            self.bounding_boxes.append(
                (self.step, self.step // self.config_expert.data_save_freq, input_data["bounding_boxes"])
            )

        if self.step % self.config_expert.data_save_freq == 0:
            if self.save_path is not None and self.config_expert.datagen:
                self.save_sensors(input_data)

        self.save_meta(
            control,
            target_speed,
            input_data,
            speed_reduced_by_obj,
        )

        return control

    @beartype
    def perturbate_camera(self) -> None:
        # Update dummy vehicle
        if self.initialized and self.config_expert.perturbate_sensors:
            # We are still rendering the map for the current frame, so we need to use the translation from the last frame.
            last_translation = self.perturbation_translation
            last_rotation = self.perturbation_rotation
            bb_copy = carla.BoundingBox(self._vehicle.bounding_box.location, self._vehicle.bounding_box.extent)
            transform_copy = carla.Transform(
                self._vehicle.get_transform().location,
                self._vehicle.get_transform().rotation,
            )
            perturbated_loc = transform_copy.transform(carla.Location(0.0, last_translation, 0.0))
            transform_copy.location = perturbated_loc
            transform_copy.rotation.yaw = transform_copy.rotation.yaw + last_rotation
            self.perturbated_vehicle_dummy.bounding_box = bb_copy
            self.perturbated_vehicle_dummy.transform = transform_copy

    @beartype
    def shuffle_weather(self) -> None:
        LOG.info("Shuffling weather settings")
        # change weather for visual diversity
        weather = self._world.get_weather()

        if self.config_expert.shuffle_weather or self.config_expert.nice_weather:
            if self.config_expert.nice_weather:
                self.weather_setting = "ClearNoon"
                LOG.info(f"Chose nice weather {self.weather_setting}")
            else:
                self.weather_setting = random.choice(list(weathers.WEATHER_SETTINGS.keys()))
                LOG.info(f"Chose random weather {self.weather_setting}")
            LOG.info(f"Chose weather {self.weather_setting}")
            self.weather_parameters: dict[str, float] = weathers.WEATHER_SETTINGS[self.weather_setting]

            if "Noon" in self.weather_setting:
                self.weather_parameters["sun_altitude_angle"] += np.random.uniform(-45.0, 45.0)
            elif "Custom" not in self.weather_setting:
                self.weather_parameters["sun_altitude_angle"] += np.random.uniform(-15.0, 15.0)

            for randomizing_parameter in ["wind_intensity", "fog_density", "wetness"]:
                if self.weather_parameters[randomizing_parameter] < 30:
                    self.weather_parameters[randomizing_parameter] += np.random.uniform(-5.0, 5.0)
                elif self.weather_parameters[randomizing_parameter] < 80:
                    self.weather_parameters[randomizing_parameter] += np.random.uniform(-10.0, 10.0)
                else:
                    self.weather_parameters[randomizing_parameter] += np.random.uniform(-5.0, 5.0)
                self.weather_parameters[randomizing_parameter] = np.clip(
                    self.weather_parameters[randomizing_parameter], 0.0, 100.0
                )

            weather = carla.WeatherParameters(**self.weather_parameters)

            self._world.set_weather(weather)

            # night mode
            vehicles = self._world.get_actors().filter("*vehicle*")
            if expert_utils.get_night_mode(weather):
                for vehicle in vehicles:
                    vehicle.set_light_state(carla.VehicleLightState(self._vehicle_lights))
            else:
                for vehicle in vehicles:
                    vehicle.set_light_state(carla.VehicleLightState.NONE)
        else:
            self.weather_setting = expert_utils.get_weather_name(weather, self.config_expert)
            self.weather_parameters = expert_utils.weather_parameter_to_dict(weather)

        LOG.info(f"Current weather setting: {self.weather_setting}")
        self.visual_visibility = int(weathers.WEATHER_VISIBILITY_MAPPING[self.weather_setting])

    @beartype
    def encode_depth(self, depth: np.ndarray) -> np.ndarray:
        if self.config_expert.save_depth_bits == 8:
            return common_utils.encode_depth_8bit(depth)
        return common_utils.encode_depth_16bit(depth)

    @beartype
    def save_sensors(self, tick_data: dict) -> None:
        frame = self.step // self.config_expert.data_save_freq

        # Store camera point clouds
        if self.config_expert.save_camera_pc:
            np.savez_compressed(str(self.save_path / "camera_pc" / (f"{frame:04}.npz")), tick_data["semantics_camera_pc"])
            if self.config_expert.perturbate_sensors:
                np.savez_compressed(
                    str(self.save_path / "camera_pc_perturbated" / (f"{frame:04}.npz")),
                    tick_data["semantics_camera_pc_perturbated"],
                )

        # CARLA images are already in opencv's BGR format.
        cv2.imwrite(
            str(self.save_path / "rgb" / (f"{frame:04}.jpg")),
            tick_data["rgb"],
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_storage_quality],
        )
        if self.config_expert.perturbate_sensors:
            cv2.imwrite(
                str(self.save_path / "rgb_perturbated" / (f"{frame:04}.jpg")),
                tick_data["rgb_perturbated"],
                [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_storage_quality],
            )

        semantics = tick_data["semantics"]
        if self.config_expert.save_grouped_semantic:
            semantics = self.semantics_converter[semantics]
        cv2.imwrite(
            str(self.save_path / "semantics" / (f"{frame:04}.png")),
            semantics,
            [int(cv2.IMWRITE_PNG_COMPRESSION), self.config_expert.png_storage_compression_level],
        )

        if self.config_expert.perturbate_sensors:
            semantics_perturbated = tick_data["semantics_perturbated"]
            if self.config_expert.save_grouped_semantic:
                semantics_perturbated = self.semantics_converter[semantics_perturbated]
            cv2.imwrite(
                str(self.save_path / "semantics_perturbated" / (f"{frame:04}.png")),
                semantics_perturbated,
                [int(cv2.IMWRITE_PNG_COMPRESSION), self.config_expert.png_storage_compression_level],
            )

        if self.config_expert.save_depth:
            depth = tick_data["depth"]
            if self.config_expert.save_depth_lower_resolution:
                depth = cv2.resize(
                    depth,
                    (
                        depth.shape[1] // self.config_expert.save_depth_resolution_ratio,
                        depth.shape[0] // self.config_expert.save_depth_resolution_ratio,
                    ),
                    interpolation=cv2.INTER_AREA,
                )
            depth_encoded = self.encode_depth(depth)
            cv2.imwrite(
                str(self.save_path / "depth" / f"{frame:04}.png"),
                depth_encoded,
                [int(cv2.IMWRITE_PNG_COMPRESSION), self.config_expert.png_storage_compression_level],
            )

            if self.config_expert.perturbate_sensors:
                depth_aug = tick_data["depth_perturbated"]
                if self.config_expert.save_depth_lower_resolution:
                    depth_aug = cv2.resize(
                        depth_aug,
                        (
                            depth_aug.shape[1] // self.config_expert.save_depth_resolution_ratio,
                            depth_aug.shape[0] // self.config_expert.save_depth_resolution_ratio,
                        ),
                        interpolation=cv2.INTER_AREA,
                    )
                depth_aug_encoded = self.encode_depth(depth_aug)
                cv2.imwrite(
                    str(self.save_path / "depth_perturbated" / f"{frame:04}.png"),
                    depth_aug_encoded,
                    [int(cv2.IMWRITE_PNG_COMPRESSION), self.config_expert.png_storage_compression_level],
                )

        if self.config_expert.save_instance_segmentation:
            cv2.imwrite(
                str(self.save_path / "instance" / (f"{frame:04}.png")),
                tick_data["instance"],
                [int(cv2.IMWRITE_PNG_COMPRESSION), self.config_expert.png_storage_compression_level],
            )
            if self.config_expert.perturbate_sensors:
                cv2.imwrite(
                    str(self.save_path / "instance_perturbated" / (f"{frame:04}.png")),
                    tick_data["instance_perturbated"],
                    [int(cv2.IMWRITE_PNG_COMPRESSION), self.config_expert.png_storage_compression_level],
                )

        cv2.imwrite(
            str(self.save_path / "hdmap" / (f"{frame:04}.png")),
            tick_data["hdmap"].astype(np.uint8),
            [int(cv2.IMWRITE_PNG_COMPRESSION), self.config_expert.png_storage_compression_level],
        )
        if self.config_expert.perturbate_sensors:
            cv2.imwrite(
                str(self.save_path / "hdmap_perturbated" / (f"{frame:04}.png")),
                tick_data["hdmap_perturbated"].astype(np.uint8),
                [int(cv2.IMWRITE_PNG_COMPRESSION), self.config_expert.png_storage_compression_level],
            )

        if self.config_expert.use_radars:
            # Prepare radar data for saving dynamically
            radar_save_dict = {}
            for i in range(1, self.config_expert.num_radar_sensors + 1):
                radar_save_dict[f"radar{i}"] = tick_data[f"radar{i}"].astype(np.float16)

            np.savez_compressed(self.save_path / "radar" / (f"{frame:04}.npz"), **radar_save_dict)
            if self.config_expert.perturbate_sensors:
                # Prepare perturbated radar data for saving dynamically
                radar_perturbated_save_dict = {}
                for i in range(1, self.config_expert.num_radar_sensors + 1):
                    radar_perturbated_save_dict[f"radar{i}"] = tick_data[f"radar{i}_perturbated"].astype(np.float16)

                np.savez_compressed(self.save_path / "radar_perturbated" / (f"{frame:04}.npz"), **radar_perturbated_save_dict)

        # Specialized LiDAR compression format
        points = self.accumulate_lidar()

        header = laspy.LasHeader(point_format=self.config_expert.point_format)
        header.offsets = np.min(points, axis=0)[:3]
        header.scales = np.array(
            [
                self.config_expert.point_precision_x,
                self.config_expert.point_precision_y,
                self.config_expert.point_precision_z,
            ]
        )
        # Add extra dimension for time
        header.add_extra_dim(laspy.ExtraBytesParams(name="time", type=np.uint8))

        with laspy.open(self.save_path / "lidar" / (f"{frame:04}.laz"), mode="w", header=header) as writer:
            point_record = laspy.ScaleAwarePointRecord.zeros(points.shape[0], header=header)
            point_record.x = points[:, 0]
            point_record.y = points[:, 1]
            point_record.z = points[:, 2]
            point_record["time"] = points[:, 3].astype(np.uint8)
            writer.write_points(point_record)

    @beartype
    def _solve_vulnerable_vehicles_scenarios(self, target_speed: float) -> tuple[float, bool, bool, list | None]:
        """
        Check for vulnerable vehicles (pedestrians, cyclists). If there are any vulnerable vehicles close enough,
        and we are inside certain scenarios, we brake and set the target speed to 0.

        This function checks if there are visible walker or bicycle inside 6m and 30° FOV. If yes, set target_speed to 0.

        Args:
            target_speed: The current target speed of the ego vehicle.
            ego_speed: The current speed of the ego vehicle.

        Returns:
            A tuple containing (adjusted target_speed, is_overridden, should_brake, details or None).
        """
        ego_matrix = self.ego_matrix
        ego_location = ego_matrix[:3, 3]

        if self.current_active_scenario_type in [
            "VehicleTurningRoute",
            "VehicleTurningRoutePedestrian",
            "DynamicObjectCrossing",
            "ParkingCrossingPedestrian",
            "PedestrianCrossing",
        ]:
            for actor in self.walkers_inside_bev + self.bikers_inside_bev:
                num_visible_pixel = (
                    self.data_agent_id_to_bb_map[actor.id]["visible_pixels"] if self.config_expert.datagen else -1
                )
                actor_height = actor.bounding_box.extent.z
                threshold = (
                    50
                    * (self.config_expert.image_height * self.config_expert.image_width)
                    / ((384**2) * 3)
                    * (3 / self.config_expert.num_cameras)
                )
                if 0 < num_visible_pixel / (actor_height**2) < threshold:
                    continue
                if not self.config_expert.pedestrians_and_crossing_brikers_emergency_brake:
                    continue
                actor_velocity = actor.get_velocity().length()
                if (
                    actor_velocity < 0.25
                    and not CarlaDataProvider.memory[self.current_active_scenario_type]["pedestrian_moved"][actor.id]
                ):
                    continue  # Actor did not move yet and is not moving, we come nearer to trigger the scenario
                elif actor_velocity >= 0.25:  # Actor is moving, we mark it as moving and continue with the scenario
                    CarlaDataProvider.memory[self.current_active_scenario_type]["pedestrian_moved"][actor.id] = True

                bb_location = np.array(actor.get_transform().get_matrix())[:3, 3]
                rel_vector = bb_location - ego_location
                distance = np.linalg.norm(rel_vector)

                local_coords = self.inv_ego_matrix @ np.append(bb_location, 1.0)
                x, y = local_coords[0], local_coords[1]
                angle = np.abs(np.degrees(np.arctan2(y, x)))

                if x < 0:
                    continue  # Behind ego vehicle

                LOG.info(f"Found vulnerable vehicle in front: {angle}°, {distance}m, emergency braking.")
                return 0.0, True, True, [0, actor.type_id, actor.id, distance]

        return target_speed, False, False, None

    @beartype
    def _get_control(self) -> tuple[float, carla.VehicleControl, list | None]:
        """
        Compute the control commands for the current frame.

        Returns:
            A tuple containing the target speed, control commands, and speed_reduced_by_obj.
        """
        # Reset hazard flags
        self.stop_sign_close = False
        self.walker_close = False
        self.walker_close_id = None
        self.vehicle_hazard = False
        self.vehicle_affecting_id = None
        self.walker_hazard = False
        self.walker_affecting_id = None
        self.traffic_light_hazard = False
        self.stop_sign_hazard = False

        self.europe_traffic_light = self.over_head_traffic_light = False
        # Waypoint planning and route generation
        (
            route_np,
            route_wp,
            _,
            distance_to_next_traffic_light,
            next_traffic_light,
            distance_to_next_stop_sign,
            next_stop_sign,
        ) = self._waypoint_planner.run_step(self.ego_location_array)

        # Extract relevant route information
        self.remaining_route = route_np[self.config_expert.tf_first_checkpoint_distance :][
            :: self.config_expert.points_per_meter
        ]

        # --- Target speed limit, the highest we can go depending on weather, surrounding, etc ---
        self.target_speed_limit = self.speed_limit
        self.target_speed_limit = max(self.target_speed_limit, self.second_highest_speed_limit)  # We don't to drive too slow
        if self.second_highest_speed > 30.0 / 3.6:  # We don't want to drive too fast either
            self.target_speed_limit = min(self.target_speed_limit, self.second_highest_speed / 0.9)

        # --- Drive slower with bad weather, neight time, junction and clutterness in city ---
        self.weather_setting = expert_utils.get_weather_name(self._world.get_weather(), self.config_expert)
        self.visual_visibility = int(weathers.WEATHER_VISIBILITY_MAPPING[self.weather_setting])
        self.slower_bad_visibility = False
        self.slower_clutterness = False
        self.slower_occluded_junction = False
        if (
            self.current_active_scenario_type
            not in [
                "EnterActorFlow",
                "EnterActorFlowV2",
                "MergerIntoSlowTraffic",
                "MergerIntoSlowTrafficV2",
                "HighwayExit",
                "NonSignalizedJunctionLeftTurn",
                "SignalizedJunctionLeftTurn",
                "SignalizedJunctionLeftTurnEnterFlow",
                "NonSignalizedJunctionLeftTurnEnterFlow",
                "SignalizedJunctionRightTurn",
                "NonSignalizedJunctionRightTurn",
                "InterurbanActorFlow",
                "InterurbanAdvancedActorFlow",
            ]
            and min(self.target_speed_limit, self.speed_limit) < 60 / 3.6  # Assume urban = 60kmh
        ):
            if self.distance_to_next_junction > 0:
                if self.visual_visibility == WeatherVisibility.VERY_LIMITED:
                    self.target_speed_limit -= 4.0
                    self.slower_bad_visibility = True
                if self.num_parking_vehicles_in_proximity >= 2:
                    self.target_speed_limit -= 2.0
                    self.slower_clutterness = True
                    self.slower_occluded_junction = True
                    self.target_speed_limit -= 4.0

            # Reduce target speed if there is a junction ahead
            for i in range(min(self.config_expert.max_lookahead_to_check_for_junction, len(route_wp))):
                if route_wp[i].is_junction:
                    if self.speed_limit < 40 / 3.6:
                        self.target_speed_limit = min(self.target_speed_limit, self.config_expert.max_speed_in_junction_urban)

        else:
            self.target_speed_limit = max(self.target_speed_limit, 40.0 / 3.6)  # We don't want to drive too slow

        self.target_speed_limit = max(self.target_speed_limit, self.config_expert.min_target_speed_limit)
        self.target_speed_limit = min(self.target_speed_limit, 20.0)

        # --- Target speed, begins with target speed limit, reduces further depending on scenarios ---
        target_speed = self.target_speed_limit
        # Get the list of vehicles in the scene
        actors = self._world.get_actors()
        vehicles = self.vehicles_inside_bev

        # Manage route obstacle scenarios and adjust target speed
        target_speed_route_obstacle, obstacle_override, speed_reduced_by_obj = self._solve_obstacle_scenarios(
            target_speed, self.ego_speed, route_wp, vehicles, route_np
        )

        # Manage distance to pedestrians and bikers
        (
            target_speed_vulnerable_vehicles,
            vulnerable_vehicle_override,
            brake_vulnerable_vehicle,
            speed_reduced_by_obj_vulnerable_vehicles,
        ) = self._solve_vulnerable_vehicles_scenarios(target_speed)
        self.does_emergency_brake_for_pedestrians = vulnerable_vehicle_override

        assert int(obstacle_override) + int(vulnerable_vehicle_override) <= 1, "Only one override can be active at a time."

        # Specific cases
        if obstacle_override:
            # In obstacle override, we force the vehicle to drive no matter what.
            brake, target_speed = False, target_speed_route_obstacle
            speed_reduced_by_obj = speed_reduced_by_obj
        elif vulnerable_vehicle_override:
            brake, target_speed = brake_vulnerable_vehicle, min(target_speed, target_speed_vulnerable_vehicles)
            speed_reduced_by_obj = speed_reduced_by_obj_vulnerable_vehicles
        else:  # Generate cases
            brake, target_speed, speed_reduced_by_obj = self._solve_general_scenarios(
                route_np,
                distance_to_next_traffic_light,
                next_traffic_light,
                distance_to_next_stop_sign,
                next_stop_sign,
                vehicles,
                actors,
                target_speed,
                speed_reduced_by_obj,
            )

        target_speed = min(target_speed, target_speed_route_obstacle)
        # Reduce target speed if the ego vehicle is close to road discontinuity
        self.reduce_speed_discontinuous_road = False
        if (
            not obstacle_override
            and self.config_expert.discontinuous_road_reduce_speed
            and self.distance_to_road_discontinuity < self.config_expert.discontinuous_road_max_future_points
        ):
            target_speed = min(self.config_expert.discontinuous_road_max_speed, target_speed)
            self.reduce_speed_discontinuous_road = True
            LOG.info("Future road is discontinuous, reducing target speed.")

        # Reduce target speed if the ego vehicle is close to a very sharp curve
        self.reduce_speed_high_route_curvature = False
        if (
            not obstacle_override
            and self.config_expert.high_road_curvature_reduce_speed
            and self.route_curvature > self.config_expert.high_road_curvature_max_speed
        ):
            target_speed = min(self.config_expert.high_road_curvature_max_speed, target_speed)
            self.reduce_speed_high_route_curvature = True
            LOG.info("Future road has high curvature, reducing target speed.")

        self.emergency_brake_for_special_vehicle = False
        if self.current_active_scenario_type in ["OppositeVehicleTakingPriority", "OppositeVehicleRunningRedLight"]:
            if len(self.adversarial_actors_ids[0]) > 0:
                LOG.info("Reducing target speed, waiting for dangerous adversarial vehicle to pass.")
                target_speed = 0.0
                brake = True
        elif self.current_active_scenario_type in [
            "Accident",
            "ConstructionObstacle",
        ]:
            if self.speed_limit > 25:
                if 25 < self.distance_to_scenario_obstacle < 50:
                    LOG.info("Reducing target speed, driving against visible obstacle.")
                    target_speed = min(target_speed, 10.0)
            elif self.speed_limit > 20:
                if 25 < self.distance_to_scenario_obstacle < 45:
                    LOG.info("Reducing target speed, driving against visible obstacle.")
                    target_speed = min(target_speed, 7.5)
            else:
                if 25 < self.distance_to_scenario_obstacle < 40:
                    LOG.info("Reducing target speed, driving against visible obstacle.")
                    target_speed = min(target_speed, self.config_expert.min_target_speed_limit)
        elif self.current_active_scenario_type in [
            "ParkedObstacle",
        ]:
            if self.speed_limit > 25:
                if 25 < self.distance_to_scenario_obstacle < 45:
                    LOG.info("Reducing target speed, driving against visible obstacle.")
                    target_speed = min(target_speed, 10.0)
            elif self.speed_limit > 20:
                if 25 < self.distance_to_scenario_obstacle < 45:
                    LOG.info("Reducing target speed, driving against visible obstacle.")
                    target_speed = min(target_speed, 7.5)
            else:
                if 25 < self.distance_to_scenario_obstacle < 35:
                    LOG.info("Reducing target speed, driving against visible obstacle.")
                    target_speed = min(target_speed, self.config_expert.min_target_speed_limit)

        # Attempt to try to avoid collision in some cases
        self.rear_danger_8 = self.rear_danger_16 = False
        if self.current_active_scenario_type in [
            "NonSignalizedJunctionRightTurn",
            "SignalizedJunctionRightTurn",
            "EnterActorFlow",
            "EnterActorFlowV2",
            "InterurbanAdvancedActorFlow",
            "OppositeVehicleRunningRedLight",
            "OppositeVehicleTakingPriority",
            "OppositeVehicleTakingPriority",
            "NonSignalizedJunctionLeftTurnEnterFlow",
            "SignalizedJunctionLeftTurnEnterFlow",
        ]:
            rear_adversarial_actor = self.rear_adversarial_actor
            if rear_adversarial_actor:
                vehicle_behind_speed = rear_adversarial_actor.get_velocity().length()
                vehicle_behind_distance = self.ego_location.distance(rear_adversarial_actor.get_location())
                if vehicle_behind_distance < 8 and vehicle_behind_speed > 5 and self.ego_speed > 5:
                    LOG.info("Vehicle behind speed:", vehicle_behind_speed, "Target speed:", target_speed)
                    target_speed = max(target_speed, vehicle_behind_speed + 5)
                    self.rear_danger_8 = True
                    brake = False
                elif vehicle_behind_distance < 16 and vehicle_behind_speed > 5 and self.ego_speed > 5:
                    LOG.info("Vehicle behind speed:", vehicle_behind_speed, "Target speed:", target_speed)
                    target_speed = max(target_speed, vehicle_behind_speed)
                    self.rear_danger_16 = True
                    brake = False

        # Safety brake if some vehicles cutin from side. This make the imitation learning easier.
        self.brake_cutin = False
        if self.current_active_scenario_type in ["ParkingCutIn"]:
            assert len(self.cutin_actors) == 1
            cut_in_vehicle = self.cutin_actors[0]
            if (
                1 < cut_in_vehicle.get_velocity().length() < 4.25
                and not CarlaDataProvider.memory[self.current_active_scenario_type]["stopped"]
            ):
                self.brake_cutin = True
                brake = True
                target_speed = 0.0
            elif (
                not CarlaDataProvider.memory[self.current_active_scenario_type]["stopped"]
                and cut_in_vehicle.get_velocity().length() >= 5
            ):
                CarlaDataProvider.memory[self.current_active_scenario_type]["stopped"] = True
        elif self.current_active_scenario_type in ["StaticCutIn"]:
            assert len(self.cutin_actors) == 1
            cut_in_vehicle = self.cutin_actors[0]
            if (
                2.1 < cut_in_vehicle.get_velocity().length() < 4.25
                and not CarlaDataProvider.memory[self.current_active_scenario_type]["stopped"]
            ):
                self.brake_cutin = True
                brake = True
                target_speed = 0.0
            elif (
                not CarlaDataProvider.memory[self.current_active_scenario_type]["stopped"]
                and cut_in_vehicle.get_velocity().length() >= 5
            ):
                CarlaDataProvider.memory[self.current_active_scenario_type]["stopped"] = True

        # Compute throttle and brake control
        throttle, control_brake = self._longitudinal_controller.get_throttle_and_brake(brake, target_speed, self.ego_speed)

        # Compute steering control
        steer = expert_utils.get_steer(
            self.config_expert,
            self._turn_controller,
            route_np,
            self.ego_location_array,
            self.ego_orientation_rad,
            self.ego_speed,
        )

        # Create the control command
        self.control = carla.VehicleControl()
        self.control.steer = steer + self.config_expert.steer_noise * np.random.randn()
        self.control.throttle = throttle
        self.control.brake = float(brake or control_brake)

        # Apply brake if the vehicle is stopped to prevent rolling back
        if self.control.throttle == 0 and self.ego_speed < self.config_expert.minimum_speed_to_prevent_rolling_back:
            self.control.brake = 1

        # Apply throttle if the vehicle is blocked for too long
        ego_velocity = CarlaDataProvider.get_velocity(self._vehicle)
        if ego_velocity < 0.1:
            self.ego_blocked_for_ticks += 1
        else:
            self.ego_blocked_for_ticks = 0

        if self.ego_blocked_for_ticks >= self.config_expert.max_blocked_ticks:
            self.control.throttle = 1
            self.control.brake = 0

        # Save control commands and target speed
        self.steer = self.control.steer
        self.throttle = self.control.throttle
        self.brake = self.control.brake

        # Run the command planner
        self._command_planner.run_step(self.ego_location_array)
        for v in self._command_planners_dict.values():
            v.run_step(self.ego_location_array)

        return float(target_speed), self.control, speed_reduced_by_obj

    @beartype
    def is_two_ways_overtaking_path_clear(
        self,
        from_index: int,
        to_index: int,
        list_vehicles: list,
        ego_location: carla.Location,
        target_speed: float,
        ego_speed: float,
        previous_lane_ids: list,
        min_speed: float = 50.0 / 3.6,
    ) -> bool:
        """
        Checks if the path between two route indices is clear for the ego vehicle to overtake in two ways scenarios.

        Args:
            from_index: The starting route index.
            to_index: The ending route index.
            list_vehicles: A list of all vehicles in the simulation.
            ego_location: The location of the ego vehicle.
            target_speed: The target speed of the ego vehicle.
            ego_speed: The current speed of the ego vehicle.
            previous_lane_ids: A list of tuples containing previous road IDs and lane IDs.
            min_speed: The minimum speed to consider for overtaking. Defaults to 50/3.6 km/h.

        Returns:
            True if the path is clear for overtaking, False otherwise.
        """
        # 10 m safety distance, overtake with max. 50 km/h
        to_location = self._waypoint_planner.route_points[to_index]
        to_location = carla.Location(to_location[0], to_location[1], to_location[2])

        from_location = self._waypoint_planner.route_points[from_index]
        from_location = carla.Location(from_location[0], from_location[1], from_location[2])

        # Compute the distance and time needed for the ego vehicle to overtake
        ego_distance = (
            to_location.distance(ego_location)
            + self._vehicle.bounding_box.extent.x * 2
            + self.config_expert.check_path_free_safety_distance
        )
        ego_time = expert_utils.compute_min_time_for_distance(
            self.config_expert, ego_distance, min(min_speed, target_speed), ego_speed
        )

        path_clear = True

        if self.config_expert.visualize_internal_data:
            for vehicle in list_vehicles:
                # Sort out ego vehicle
                if vehicle.id == self._vehicle.id:
                    continue

                vehicle_location = vehicle.get_location()
                vehicle_waypoint = self.world_map.get_waypoint(vehicle_location)

                diff_vector = vehicle_location - ego_location
                dot_product = self._vehicle.get_transform().get_forward_vector().dot(diff_vector)
                # Draw dot_product above vehicle
                self._world.debug.draw_string(
                    vehicle_location + carla.Location(x=1, y=0, z=2.5),
                    f"dot1={dot_product:.1f}",
                    color=carla.Color(255, 255, 0),
                    life_time=self.config_expert.draw_life_time,
                )

                # The overtaking path is blocked by vehicle
                diff_vector_2 = to_location - vehicle_location
                dot_product_2 = vehicle.get_transform().get_forward_vector().dot(diff_vector_2)
                self._world.debug.draw_string(
                    vehicle_location + carla.Location(x=2, y=0, z=3.5),
                    f"dot2={dot_product_2:.1f}",
                    color=carla.Color(0, 255, 255),
                    life_time=self.config_expert.draw_life_time,
                )

        for vehicle in list_vehicles:
            # Only consider visible vehicles in the BEV to make TransFuser learn easier
            if not self.is_actor_inside_bev(vehicle):
                continue

            # Sort out ego vehicle
            if vehicle.id == self._vehicle.id:
                continue

            vehicle_location = vehicle.get_location()
            vehicle_waypoint = self.world_map.get_waypoint(vehicle_location)

            # Check if the vehicle is on the previous lane IDs
            if (vehicle_waypoint.road_id, vehicle_waypoint.lane_id) in previous_lane_ids:
                diff_vector = vehicle_location - ego_location
                dot_product = self._vehicle.get_transform().get_forward_vector().dot(diff_vector)

                # One TwoWay scenarios we can skip this vehicle since it's not on the overtaking path and behind
                # the ego vehicle. Otherwise in other scenarios it's coming from behind and is relevant
                if dot_product < 0 and self.current_active_scenario_type in [
                    "ConstructionObstacleTwoWays",
                    "AccidentTwoWays",
                    "ParkedObstacleTwoWays",
                    "VehicleOpensDoorTwoWays",
                    "HazardAtSideLaneTwoWays",
                ]:
                    continue

                # Allow earlier acceleration.
                # We only want predictable earlier acceleration from scratch that is why we hav the ego_speed < 1.0 constraint.
                # Ignore vehicle that are close to the ego vehicle and is already almost out of the way
                if self.ego_speed < 1.0 and vehicle.get_velocity().length() > 3.0:
                    if self.current_active_scenario_type in ["ConstructionObstacleTwoWays"]:
                        threshold = 8
                        if self.ego_lane_width <= 2.76:
                            threshold = 8
                        elif self.ego_lane_width <= 3.01:
                            threshold = 9
                        elif self.ego_lane_width <= 3.51:
                            threshold = 10
                        if dot_product < threshold:
                            continue
                    elif self.current_active_scenario_type in ["AccidentTwoWays"]:
                        threshold = 7
                        if self.ego_lane_width <= 2.76:
                            threshold = 7
                        elif self.ego_lane_width <= 3.01:
                            threshold = 8
                        elif self.ego_lane_width <= 3.51:
                            threshold = 9
                        if dot_product < threshold:
                            continue
                    elif self.current_active_scenario_type in ["ParkedObstacleTwoWays"]:
                        threshold = 7
                        if self.ego_lane_width <= 2.76:
                            threshold = 7
                        elif self.ego_lane_width <= 3.01:
                            threshold = 8
                        elif self.ego_lane_width <= 3.51:
                            threshold = 9
                        if dot_product < threshold:
                            continue
                    elif self.current_active_scenario_type in ["VehicleOpensDoorTwoWays"]:
                        threshold = 8
                        if self.ego_lane_width <= 2.76:
                            threshold = 8
                        elif self.ego_lane_width <= 3.01:
                            threshold = 9
                        elif self.ego_lane_width <= 3.51:
                            threshold = 10
                        if dot_product < threshold:
                            continue
                    elif self.current_active_scenario_type in ["HazardAtSideLaneTwoWays"]:
                        threshold = 8
                        if self.ego_lane_width <= 2.76:
                            threshold = 8
                        elif self.ego_lane_width <= 3.01:
                            threshold = 9
                        elif self.ego_lane_width <= 3.51:
                            threshold = 10
                        if dot_product < threshold:
                            continue

                # The overtaking path is blocked by vehicle
                diff_vector_2 = to_location - vehicle_location
                dot_product_2 = vehicle.get_transform().get_forward_vector().dot(diff_vector_2)

                if dot_product_2 < 0:
                    path_clear = False
                    break

                other_vehicle_distance = to_location.distance(vehicle_location) - vehicle.bounding_box.extent.x
                other_vehicle_time = other_vehicle_distance / max(1.0, vehicle.get_velocity().length())

                # Add 200 ms safety margin
                # Vehicle needs less time to arrive at to_location than the ego vehicle
                if other_vehicle_time < ego_time + self.config_expert.check_path_free_safety_time:
                    path_clear = False
                    break

        return path_clear

    def _solve_general_scenarios(
        self,
        route_points: np.ndarray,
        distance_to_next_traffic_light: float,
        next_traffic_light: carla.TrafficLight | None,
        distance_to_next_stop_sign: float,
        next_stop_sign: carla.Actor | None,
        vehicle_list: list,
        actor_list: carla.ActorList,
        initial_target_speed: float,
        speed_reduced_by_obj: list | None,
    ) -> tuple[bool, float, list | None]:
        """
        Compute the brake command and target speed for the ego vehicle based on various factors.

        Args:
            route_points: An array of waypoints representing the planned route.
            distance_to_next_traffic_light: The distance to the next traffic light.
            next_traffic_light: The next traffic light actor.
            distance_to_next_stop_sign: The distance to the next stop sign.
            next_stop_sign: The next stop sign actor.
            vehicle_list: A list of vehicle actors in the simulation.
            actor_list: A list of all actors (vehicles, pedestrians, etc.) in the simulation.
            initial_target_speed: The initial target speed for the ego vehicle.
            speed_reduced_by_obj: A list containing [reduced speed, object type, object ID, distance]
                    for the object that caused the most speed reduction, or None if no speed reduction.

        Returns:
            A tuple containing the brake command, target speed, and the updated speed_reduced_by_obj list.
        """
        target_speed = initial_target_speed

        ego_vehicle_location = self._vehicle.get_location()
        ego_vehicle_transform = self._vehicle.get_transform()

        # Calculate the global bounding box of the ego vehicle
        center_ego_bb_global = ego_vehicle_transform.transform(self._vehicle.bounding_box.location)
        ego_bb_global = carla.BoundingBox(center_ego_bb_global, self._vehicle.bounding_box.extent)
        ego_bb_global.rotation = ego_vehicle_transform.rotation

        if self.config_expert.visualize_bounding_boxes:
            self._world.debug.draw_box(
                box=ego_bb_global,
                rotation=ego_bb_global.rotation,
                thickness=0.1,
                color=self.config_expert.ego_vehicle_bb_color,
                life_time=self.config_expert.draw_life_time,
            )

        # Compute if there will be a lane change close
        near_lane_change = self._is_near_lane_change(route_points)

        # Compute the number of future frames to consider for collision detection
        num_future_frames = int(
            self.config_expert.bicycle_frame_rate
            * (
                self.config_expert.forecast_length_lane_change
                if near_lane_change
                else self.config_expert.default_forecast_length
            )
        )

        # Get future bounding boxes of pedestrians
        nearby_pedestrians, nearby_pedestrian_ids = self.forecast_walkers(num_future_frames)

        # Forecast the ego vehicle's bounding boxes for the future frames
        ego_bounding_boxes = self.forecast_ego_agent(
            ego_vehicle_transform, num_future_frames, initial_target_speed, route_points
        )

        # Predict bounding boxes of other actors (vehicles, bicycles, etc.)
        predicted_bounding_boxes = self.predict_other_actors_bounding_boxes(vehicle_list, num_future_frames, near_lane_change)

        # Compute the leading and trailing vehicle IDs
        leading_vehicle_ids = self._waypoint_planner.compute_leading_vehicles(vehicle_list, self._vehicle.id)
        trailing_vehicle_ids = self._waypoint_planner.compute_trailing_vehicles(vehicle_list, self._vehicle.id)

        if self.config_expert.visualize_internal_data:
            for vehicle in vehicle_list:
                if vehicle.id in leading_vehicle_ids:
                    self._world.debug.draw_string(
                        vehicle.get_location(),
                        f"Leading Vehicle: {vehicle.get_velocity().length():.2f} m/s",
                        life_time=self.config_expert.draw_life_time,
                        color=self.config_expert.leading_vehicle_color,
                    )
                elif vehicle.id in trailing_vehicle_ids:
                    self._world.debug.draw_string(
                        vehicle.get_location(),
                        f"Trailing Vehicle: {vehicle.get_velocity().length():.2f} m/s",
                        life_time=self.config_expert.draw_life_time,
                        color=self.config_expert.trailing_vehicle_color,
                    )

        # Compute the target speed with respect to the leading vehicle
        target_speed_leading, speed_reduced_by_obj = self.compute_target_speed_wrt_leading_vehicle(
            initial_target_speed,
            predicted_bounding_boxes,
            near_lane_change,
            ego_vehicle_location,
            trailing_vehicle_ids,
            leading_vehicle_ids,
            speed_reduced_by_obj,
        )

        # Compute the target speeds with respect to all actors (vehicles, bicycles, pedestrians)
        target_speed_bicycle, target_speed_pedestrian, target_speed_vehicle, speed_reduced_by_obj = (
            self.compute_target_speeds_wrt_all_actors(
                initial_target_speed,
                ego_bounding_boxes,
                predicted_bounding_boxes,
                near_lane_change,
                leading_vehicle_ids,
                trailing_vehicle_ids,
                speed_reduced_by_obj,
                nearby_pedestrians,
                nearby_pedestrian_ids,
            )
        )

        # Compute the target speed with respect to the red light
        target_speed_red_light = self.ego_agent_affected_by_red_light(
            ego_vehicle_location,
            distance_to_next_traffic_light,
            next_traffic_light,
            initial_target_speed,
        )

        # Update the object causing the most speed reduction
        if speed_reduced_by_obj is None or speed_reduced_by_obj[0] > target_speed_red_light:
            speed_reduced_by_obj = [
                target_speed_red_light,
                None if next_traffic_light is None else next_traffic_light.type_id,
                None if next_traffic_light is None else next_traffic_light.id,
                distance_to_next_traffic_light,
            ]

        # Compute the target speed with respect to the stop sign
        target_speed_stop_sign = self.ego_agent_affected_by_stop_sign(
            ego_vehicle_location, next_stop_sign, initial_target_speed, actor_list
        )
        # Update the object causing the most speed reduction
        if speed_reduced_by_obj is None or speed_reduced_by_obj[0] > target_speed_stop_sign:
            speed_reduced_by_obj = [
                target_speed_stop_sign,
                None if next_stop_sign is None else next_stop_sign.type_id,
                None if next_stop_sign is None else next_stop_sign.id,
                distance_to_next_stop_sign,
            ]

        # Compute the minimum target speed considering all factors
        target_speed = min(
            target_speed_leading,
            target_speed_bicycle,
            target_speed_vehicle,
            target_speed_pedestrian,
            target_speed_red_light,
            target_speed_stop_sign,
        )

        # Set the hazard flags based on the target speed and its cause
        if target_speed == target_speed_pedestrian and target_speed_pedestrian != initial_target_speed:
            self.walker_hazard = True
            self.walker_close = True
        elif target_speed == target_speed_red_light and target_speed_red_light != initial_target_speed:
            self.traffic_light_hazard = True
        elif target_speed == target_speed_stop_sign and target_speed_stop_sign != initial_target_speed:
            self.stop_sign_hazard = True
            self.stop_sign_close = True

        # Determine if the ego vehicle needs to brake based on the target speed
        brake = target_speed == 0
        return brake, target_speed, speed_reduced_by_obj

    def _solve_obstacle_scenarios(
        self, target_speed: float, ego_speed: float, route_waypoints: list, list_vehicles: list, route_points: np.ndarray
    ) -> tuple[float, bool, list]:
        """
        This method handles various obstacle and scenario situations that may arise during navigation.
        It adjusts the target speed, modifies the route, and determines if the ego vehicle should keep driving or wait.
        The method supports different scenario types such as InvadingTurn, Accident, ConstructionObstacle,
        ParkedObstacle, AccidentTwoWays, ConstructionObstacleTwoWays, ParkedObstacleTwoWays, VehicleOpensDoorTwoWays,
        HazardAtSideLaneTwoWays, HazardAtSideLane, and YieldToEmergencyVehicle.

        Args:
            target_speed: The current target speed of the ego vehicle.
            ego_speed: The current speed of the ego vehicle.
            route_waypoints: A list of waypoints representing the current route.
            list_vehicles: A list of all vehicles in the simulation.
            route_points: A numpy array containing the current route points.

        Returns:
            A tuple containing the updated target speed, a boolean indicating whether to keep driving,
                and a list containing information about a potential decreased target speed due to an object.
        """

        keep_driving = False
        speed_reduced_by_obj = [target_speed, None, None, None]  # [target_speed, type, id, distance]

        # Only continue if there are some active scenarios available
        if len(CarlaDataProvider.active_scenarios) != 0:
            ego_location = self._vehicle.get_location()

            # # Sort the scenarios by distance if there is more than one active scenario
            # if len(CarlaDataProvider.active_scenarios) != 1:
            #     sort_scenarios_by_distance(ego_location)

            if self.current_active_scenario_type == "InvadingTurn":
                first_cone, last_cone, offset = [
                    CarlaDataProvider.memory["InvadingTurn"][k] for k in ["first_cone", "last_cone", "offset"]
                ]
                closest_distance = first_cone.get_location().distance(ego_location)
                if closest_distance < self.config_expert.default_max_distance_to_process_scenario:
                    self._waypoint_planner.shift_route_for_invading_turn(first_cone, last_cone, offset)
                    CarlaDataProvider.clean_current_active_scenario()

            elif self.current_active_scenario_type in ["Accident", "ConstructionObstacle", "ParkedObstacle"]:
                first_actor, last_actor, direction = [
                    CarlaDataProvider.memory[self.current_active_scenario_type][k]
                    for k in ["first_actor", "last_actor", "direction"]
                ]

                horizontal_distance = expert_utils.get_horizontal_distance(self._vehicle, first_actor)

                # Shift the route around the obstacles
                if (
                    horizontal_distance < self.config_expert.default_max_distance_to_process_scenario
                    and not CarlaDataProvider.memory[self.current_active_scenario_type]["changed_route"]
                ):
                    add_before_length = {
                        "Accident": self.config_expert.add_before_accident,
                        "ConstructionObstacle": self.config_expert.add_before_construction_obstacle,
                        "ParkedObstacle": self.config_expert.add_before_parked_obstacle,
                    }[self.current_active_scenario_type]
                    add_after_length = {
                        "Accident": self.config_expert.add_after_accident,
                        "ConstructionObstacle": self.config_expert.add_after_construction_obstacle,
                        "ParkedObstacle": self.config_expert.add_after_parked_obstacle,
                    }[self.current_active_scenario_type]
                    transition_length = {
                        "Accident": self.config_expert.transition_smoothness_distance_accident,
                        "ConstructionObstacle": self.config_expert.transition_smoothness_factor_construction_obstacle,
                        "ParkedObstacle": self.config_expert.transition_smoothness_distance_parked_obstacle,
                    }[self.current_active_scenario_type]
                    if self.current_active_scenario_type not in ["ConstructionObstacle"]:
                        if self.visual_visibility == WeatherVisibility.LIMITED:
                            add_before_length -= 5.0
                            transition_length -= int(4.0 * self.config_expert.points_per_meter)
                        elif self.visual_visibility == WeatherVisibility.VERY_LIMITED:
                            add_before_length -= 7.0
                            transition_length -= int(6.0 * self.config_expert.points_per_meter)
                        LOG.info("Later switching because of bad weather")
                    if self.speed_limit < 15:
                        add_after_length = {
                            "Accident": 0.5,
                            "ConstructionObstacle": self.config_expert.add_after_construction_obstacle,
                            "ParkedObstacle": 0.5,
                        }[self.current_active_scenario_type]  # Speed is low and we can switch earlier
                    lane_transition_factor = {
                        "Accident": 1.0,
                        "ConstructionObstacle": self.two_way_obstacle_distance_to_cones_factor,
                        "ParkedObstacle": 1.0,
                    }[self.current_active_scenario_type]
                    from_index, _ = self._waypoint_planner.shift_route_around_actors(
                        first_actor,
                        last_actor,
                        direction,
                        transition_length,
                        lane_transition_factor=lane_transition_factor,
                        extra_length_before=add_before_length,
                        extra_length_after=add_after_length,
                    )
                    CarlaDataProvider.memory[self.current_active_scenario_type].update(
                        {
                            "changed_route": True,
                            "from_index": from_index,
                        }
                    )

                elif CarlaDataProvider.memory[self.current_active_scenario_type]["changed_route"]:
                    first_actor_rel_pos = common_utils.get_relative_transform(
                        self.ego_matrix, np.array(first_actor.get_transform().get_matrix())
                    )
                    if first_actor_rel_pos[0] < 3:
                        CarlaDataProvider.clean_current_active_scenario()
            elif self.current_active_scenario_type in [
                "AccidentTwoWays",
                "ConstructionObstacleTwoWays",
                "ParkedObstacleTwoWays",
                "VehicleOpensDoorTwoWays",
            ]:
                first_actor, last_actor, direction, changed_route, from_index, to_index, path_clear = [
                    CarlaDataProvider.memory[self.current_active_scenario_type][k]
                    for k in ["first_actor", "last_actor", "direction", "changed_route", "from_index", "to_index", "path_clear"]
                ]

                # change the route if the ego is close enough to the obstacle
                horizontal_distance = expert_utils.get_horizontal_distance(self._vehicle, first_actor)

                # Shift the route around the obstacles
                if horizontal_distance < self.config_expert.default_max_distance_to_process_scenario and not changed_route:
                    transition_length = {
                        "AccidentTwoWays": self.config_expert.transition_length_accident_two_ways,
                        "ConstructionObstacleTwoWays": self.config_expert.transition_length_construction_obstacle_two_ways,
                        "ParkedObstacleTwoWays": self.config_expert.transition_length_parked_obstacle_two_ways,
                        "VehicleOpensDoorTwoWays": self.config_expert.transition_length_vehicle_opens_door_two_ways,
                    }[self.current_active_scenario_type]
                    add_before_length = {
                        "AccidentTwoWays": self.add_before_accident_two_ways,
                        "ConstructionObstacleTwoWays": self.add_before_construction_obstacle_two_ways,
                        "ParkedObstacleTwoWays": self.config_expert.add_before_parked_obstacle_two_ways,
                        "VehicleOpensDoorTwoWays": self.config_expert.add_before_vehicle_opens_door_two_ways,
                    }[self.current_active_scenario_type]
                    add_after_length = {
                        "AccidentTwoWays": self.add_after_accident_two_ways,
                        "ConstructionObstacleTwoWays": self.add_after_construction_obstacle_two_ways,
                        "ParkedObstacleTwoWays": self.config_expert.add_after_parked_obstacle_two_ways,
                        "VehicleOpensDoorTwoWays": self.config_expert.add_after_vehicle_opens_door_two_ways,
                    }[self.current_active_scenario_type]
                    factor = {
                        "AccidentTwoWays": self.config_expert.factor_accident_two_ways,
                        "ConstructionObstacleTwoWays": self.two_way_obstacle_distance_to_cones_factor,
                        "ParkedObstacleTwoWays": self.config_expert.factor_parked_obstacle_two_ways,
                        "VehicleOpensDoorTwoWays": self.two_way_vehicle_open_door_distance_to_center_line_factor,
                    }[self.current_active_scenario_type]
                    if self.current_active_scenario_type not in ["ConstructionObstacleTwoWays"]:
                        if self.visual_visibility == WeatherVisibility.LIMITED:
                            add_before_length -= 0.5
                        elif self.visual_visibility == WeatherVisibility.VERY_LIMITED:
                            add_before_length -= 1.0

                    from_index, to_index = self._waypoint_planner.shift_route_around_actors(
                        first_actor,
                        last_actor,
                        direction,
                        transition_length,
                        factor,
                        add_before_length,
                        add_after_length,
                    )

                    changed_route = True
                    CarlaDataProvider.memory[self.current_active_scenario_type].update(
                        {
                            "changed_route": changed_route,
                            "from_index": from_index,
                            "to_index": to_index,
                        }
                    )

                # Check if the ego can overtake the obstacle
                if (
                    changed_route
                    and from_index - self._waypoint_planner.route_index
                    < self.config_expert.max_distance_to_overtake_two_way_scnearios
                    and not path_clear
                ):
                    # Get previous roads and lanes of the target lane
                    target_lane = (
                        route_waypoints[0].get_left_lane() if direction == "right" else route_waypoints[0].get_right_lane()
                    )
                    if target_lane is None:
                        return target_speed, keep_driving, speed_reduced_by_obj
                    prev_road_lane_ids = expert_utils.get_previous_road_lane_ids(self.config_expert, target_lane)
                    path_clear = self.is_two_ways_overtaking_path_clear(
                        int(from_index),
                        int(to_index),
                        list_vehicles,
                        ego_location,
                        target_speed,
                        ego_speed,
                        prev_road_lane_ids,
                        min_speed=self.two_way_overtake_speed,
                    )
                    CarlaDataProvider.memory[self.current_active_scenario_type]["path_clear"] = path_clear

                # If the overtaking path is clear, keep driving; otherwise, wait behind the obstacle
                if path_clear:
                    target_speed = self.two_way_overtake_speed
                    if (
                        self._waypoint_planner.route_index
                        >= to_index - self.config_expert.distance_to_delete_scenario_in_two_ways
                    ):
                        CarlaDataProvider.clean_current_active_scenario()
                    keep_driving = True
                else:
                    offset = {
                        "AccidentTwoWays": 10,
                        "ConstructionObstacleTwoWays": 10,
                        "ParkedObstacleTwoWays": 10,
                        "VehicleOpensDoorTwoWays": 10,
                    }[self.current_active_scenario_type]  # Move a bit to side instead of standing direct behind the obstacle
                    distance_to_merging_point = (
                        float(from_index + offset - self._waypoint_planner.route_index) / self.config_expert.points_per_meter
                    )
                    target_speed = expert_utils.compute_target_speed_idm(
                        config=self.config_expert,
                        desired_speed=target_speed,
                        leading_actor_length=self._vehicle.bounding_box.extent.x,
                        ego_speed=ego_speed,
                        leading_actor_speed=0,
                        distance_to_leading_actor=distance_to_merging_point,
                        s0=self.config_expert.idm_two_way_scenarios_minimum_distance,
                        T=self.config_expert.idm_two_way_scenarios_time_headway,
                    )

                    # Update the object causing the most speed reduction
                    if speed_reduced_by_obj is None or speed_reduced_by_obj[0] > target_speed:
                        speed_reduced_by_obj = [
                            target_speed,
                            first_actor.type_id,
                            first_actor.id,
                            distance_to_merging_point,
                        ]

            elif self.current_active_scenario_type == "HazardAtSideLaneTwoWays":
                first_actor, last_actor, changed_route, from_index, to_index, path_clear = [
                    CarlaDataProvider.memory[self.current_active_scenario_type][k]
                    for k in ["first_actor", "last_actor", "changed_route", "from_index", "to_index", "path_clear"]
                ]

                horizontal_distance = expert_utils.get_horizontal_distance(self._vehicle, first_actor)

                if (
                    horizontal_distance < self.config_expert.max_distance_to_process_hazard_at_side_lane_two_ways
                    and not changed_route
                ):
                    to_index = self._waypoint_planner.get_closest_route_index(
                        int(self._waypoint_planner.route_index), last_actor.get_location()
                    )

                    # Assume the bicycles don't drive too much during the overtaking process
                    to_index += 170
                    from_index = self._waypoint_planner.route_index

                    starting_wp = route_waypoints[0].get_left_lane()
                    prev_road_lane_ids = expert_utils.get_previous_road_lane_ids(self.config_expert, starting_wp)
                    path_clear = self.is_two_ways_overtaking_path_clear(
                        int(from_index),
                        int(to_index),
                        list_vehicles,
                        ego_location,
                        target_speed,
                        ego_speed,
                        prev_road_lane_ids,
                        min_speed=self.config_expert.default_overtake_speed,
                    )

                    if path_clear:
                        transition_length = self.config_expert.transition_smoothness_distance
                        self._waypoint_planner.shift_route_smoothly(from_index, to_index, True, transition_length)
                        changed_route = True
                        CarlaDataProvider.memory[self.current_active_scenario_type].update(
                            {
                                "changed_route": changed_route,
                                "from_index": from_index,
                                "to_index": to_index,
                                "path_clear": path_clear,
                            }
                        )

                # the overtaking path is clear
                if path_clear:
                    # Check if the overtaking is done
                    if self._waypoint_planner.route_index >= to_index:
                        CarlaDataProvider.clean_current_active_scenario()
                    # Overtake with max. 50 km/h
                    target_speed, keep_driving = self.config_expert.default_overtake_speed, True

            elif self.current_active_scenario_type == "HazardAtSideLane":
                first_actor, last_actor, changed_first_part_of_route, from_index, to_index = [
                    CarlaDataProvider.memory[self.current_active_scenario_type][k]
                    for k in ["first_actor", "last_actor", "changed_first_part_of_route", "from_index", "to_index"]
                ]

                horizontal_distance = expert_utils.get_horizontal_distance(self._vehicle, last_actor)

                if (
                    horizontal_distance < self.config_expert.max_distance_to_process_hazard_at_side_lane
                    and not changed_first_part_of_route
                ):
                    transition_length = self.config_expert.transition_smoothness_distance
                    from_index, to_index = self._waypoint_planner.shift_route_around_actors(
                        first_actor, last_actor, "right", transition_length
                    )

                    to_index -= transition_length
                    changed_first_part_of_route = True
                    CarlaDataProvider.memory[self.current_active_scenario_type].update(
                        {
                            "changed_first_part_of_route": changed_first_part_of_route,
                            "from_index": from_index,
                            "to_index": to_index,
                        }
                    )

                if changed_first_part_of_route:
                    to_idx_ = self._waypoint_planner.extend_lane_shift_transition_for_hazard_at_side_lane(last_actor, to_index)
                    to_index = to_idx_
                    CarlaDataProvider.memory[self.current_active_scenario_type]["to_index"] = to_index
                if self._waypoint_planner.route_index > to_index:
                    CarlaDataProvider.clean_current_active_scenario()

            elif self.current_active_scenario_type == "YieldToEmergencyVehicle":
                emergency_veh, changed_route, from_index, to_index, to_left = [
                    CarlaDataProvider.memory[self.current_active_scenario_type][k]
                    for k in ["emergency_vehicle", "changed_route", "from_index", "to_index", "to_left"]
                ]

                horizontal_distance = expert_utils.get_horizontal_distance(self._vehicle, emergency_veh)

                if horizontal_distance < self.config_expert.default_max_distance_to_process_scenario and not changed_route:
                    # Assume the emergency vehicle doesn't drive more than 20 m during the overtaking process
                    from_index = self._waypoint_planner.route_index + 30 * self.config_expert.points_per_meter
                    to_index = from_index + int(2 * self.config_expert.points_per_meter) * self.config_expert.points_per_meter

                    transition_length = self.config_expert.transition_smoothness_distance
                    to_left = self.route_waypoints[from_index].lane_change != carla.LaneChange.Right
                    self._waypoint_planner.shift_route_smoothly(from_index, to_index, to_left, transition_length)

                    changed_route = True
                    to_index -= transition_length
                    CarlaDataProvider.memory[self.current_active_scenario_type].update(
                        {
                            "changed_route": changed_route,
                            "from_index": from_index,
                            "to_index": to_index,
                            "to_left": to_left,
                        }
                    )

                if changed_route:
                    to_idx_ = self._waypoint_planner.extend_lane_shift_transition_for_yield_to_emergency_vehicle(
                        to_left, to_index
                    )
                    to_index = to_idx_
                    CarlaDataProvider.memory[self.current_active_scenario_type]["to_index"] = to_index

                    # Check if the emergency vehicle is in front of the ego vehicle
                    diff = emergency_veh.get_location() - ego_location
                    dot_res = self._vehicle.get_transform().get_forward_vector().dot(diff)
                    if dot_res > 0:
                        CarlaDataProvider.clean_current_active_scenario()

        # expertV1.1 change.
        # Previously in those TwoWays scenarios, we only drive when we are sure that we can not crash.
        # With introduction of expertV1.1, we can not be sure that we can not crash.
        # So following part of the code scans the front of ego for obstacle and if face any, we stop.
        self.construction_obstacle_two_ways_stuck = False
        self.accident_two_ways_stuck = False
        self.parked_obstacle_two_ways_stuck = False
        self.vehicle_opens_door_two_ways_stuck = False
        if self.current_active_scenario_type in ["ConstructionObstacleTwoWays"]:
            speed_constant = 2
            if self.speed_limit > 15:
                speed_constant = 2.75
            max_distance = speed_constant * self.ego_speed + 7
            obstacle_detection = self._vehicle_obstacle_detected(max_distance=max_distance)
            if obstacle_detection[0]:
                _, targ_id, dist = obstacle_detection
                other = self._world.get_actor(targ_id)

                ego_yaw = self._vehicle.get_transform().rotation.yaw
                other_yaw = other.get_transform().rotation.yaw
                relative_yaw = abs(common_utils.normalize_angle_degree(other_yaw - ego_yaw))

                # Only brake if roughly opposing direction (e.g. facing within 45° of us)
                if relative_yaw > 135 and other.get_velocity().length() > 0.0:  # Only brake if the other vehicle is moving
                    self.construction_obstacle_two_ways_stuck = True
                    keep_driving = False
                    target_speed = 0.0
                    speed_reduced_by_obj = [0.0, "vehicle", targ_id, dist]
                elif relative_yaw > 135 and other.get_velocity().length() == 0.0:
                    CarlaDataProvider.clean_current_active_scenario()
                    keep_driving = False
                    target_speed = 0.0
                    speed_reduced_by_obj = [0.0, "vehicle", targ_id, dist]

        elif self.current_active_scenario_type in ["AccidentTwoWays"]:
            speed_constant = 1.75
            if self.speed_limit > 15:
                speed_constant = 2.75
            max_distance = speed_constant * self.ego_speed + 7
            obstacle_detection = self._vehicle_obstacle_detected(max_distance=max_distance)
            if obstacle_detection[0]:
                _, targ_id, dist = obstacle_detection
                other = self._world.get_actor(targ_id)

                ego_yaw = self._vehicle.get_transform().rotation.yaw
                other_yaw = other.get_transform().rotation.yaw
                relative_yaw = abs(common_utils.normalize_angle_degree(other_yaw - ego_yaw))

                # Only brake if roughly opposing direction (e.g. facing within 45° of us)
                if relative_yaw > 135 and other.get_velocity().length() > 0.0:  # Only brake if the other vehicle is moving
                    self.accident_two_ways_stuck = True
                    keep_driving = False
                    target_speed = 0.0
                    speed_reduced_by_obj = [0.0, "vehicle", targ_id, dist]
                elif relative_yaw > 135 and other.get_velocity().length() == 0.0:
                    CarlaDataProvider.clean_current_active_scenario()
                    keep_driving = False
                    target_speed = 0.0
                    speed_reduced_by_obj = [0.0, "vehicle", targ_id, dist]

        elif self.current_active_scenario_type in ["ParkedObstacleTwoWays"]:
            speed_constant = 1.75
            if self.speed_limit > 15:
                speed_constant = 2.75
            max_distance = speed_constant * self.ego_speed + 7
            obstacle_detection = self._vehicle_obstacle_detected(max_distance=max_distance)
            if obstacle_detection[0]:
                _, targ_id, dist = obstacle_detection
                other = self._world.get_actor(targ_id)

                ego_yaw = self._vehicle.get_transform().rotation.yaw
                other_yaw = other.get_transform().rotation.yaw
                relative_yaw = abs(common_utils.normalize_angle_degree(other_yaw - ego_yaw))

                # Only brake if roughly opposing direction (e.g. facing within 45° of us)
                if relative_yaw > 135 and other.get_velocity().length() > 0.0:  # Only brake if the other vehicle is moving
                    self.parked_obstacle_two_ways_stuck = True
                    keep_driving = False
                    target_speed = 0.0
                    speed_reduced_by_obj = [0.0, "vehicle", targ_id, dist]
                elif relative_yaw > 135 and other.get_velocity().length() == 0.0:
                    CarlaDataProvider.clean_current_active_scenario()
                    keep_driving = False
                    target_speed = 0.0
                    speed_reduced_by_obj = [0.0, "vehicle", targ_id, dist]

        elif self.current_active_scenario_type in ["VehicleOpensDoorTwoWays"]:
            speed_constant = 1.75
            if self.speed_limit > 15:
                speed_constant = 2.75
            max_distance = speed_constant * self.ego_speed + 7
            obstacle_detection = self._vehicle_obstacle_detected(max_distance=max_distance)
            if obstacle_detection[0]:
                _, targ_id, dist = obstacle_detection
                other = self._world.get_actor(targ_id)

                ego_yaw = self._vehicle.get_transform().rotation.yaw
                other_yaw = other.get_transform().rotation.yaw
                relative_yaw = abs(common_utils.normalize_angle_degree(other_yaw - ego_yaw))

                # Only brake if roughly opposing direction (e.g. facing within 45° of us)
                if relative_yaw > 135 and other.get_velocity().length() > 0.0:  # Only brake if the other vehicle is moving
                    self.vehicle_opens_door_two_ways_stuck = True
                    keep_driving = False
                    target_speed = 0.0
                    speed_reduced_by_obj = [0.0, "vehicle", targ_id, dist]
                elif relative_yaw > 135 and other.get_velocity().length() == 0.0:
                    CarlaDataProvider.clean_current_active_scenario()
                    keep_driving = False
                    target_speed = 0.0
                    speed_reduced_by_obj = [0.0, "vehicle", targ_id, dist]

        return target_speed, keep_driving, speed_reduced_by_obj

    @beartype
    def save_meta(self, control: carla.VehicleControl, target_speed: float, tick_data: dict, speed_reduced_by_obj):
        """
        Save the driving data for the current frame.

        Args:
            control: The control commands for the current frame.
            target_speed: The target speed for the current frame.
            tick_data: Dictionary containing the current state of the vehicle.
            speed_reduced_by_obj: Tuple containing information about the object that caused speed reduction.

        Returns:
            A dictionary containing the driving data for the current frame.
        """
        frame = self.step // self.config_expert.data_save_freq

        # Extract relevant data from inputs
        previous_target_points = [tp.tolist() for tp in self._command_planner.previous_target_points]
        previous_commands = [int(i) for i in self._command_planner.previous_commands]
        next_target_points = [tp[0].tolist() for tp in self._command_planner.route]
        next_commands = [int(self._command_planner.route[i][1]) for i in range(len(self._command_planner.route))]

        # Get the remaining route points in the local coordinate frame
        dense_route = []
        remaining_route = self.remaining_route[: self.config_expert.num_route_points_saved]

        changed_route = bool(
            (
                self._waypoint_planner.route_points[self._waypoint_planner.route_index]
                != self._waypoint_planner.original_route_points[self._waypoint_planner.route_index]
            ).any()
        )
        for checkpoint in remaining_route:
            dense_route.append(
                common_utils.inverse_conversion_2d(
                    checkpoint[:2], self.ego_location_array[:2], self.ego_orientation_rad
                ).tolist()
            )

        # Extract speed reduction object information
        speed_reduced_by_obj_type, speed_reduced_by_obj_id, speed_reduced_by_obj_distance = None, None, None
        if speed_reduced_by_obj is not None:
            speed_reduced_by_obj_type, speed_reduced_by_obj_id, speed_reduced_by_obj_distance = speed_reduced_by_obj[1:]
            # Convert numpy to float so that it can be saved to json.
            if speed_reduced_by_obj_distance is not None:
                speed_reduced_by_obj_distance = float(speed_reduced_by_obj_distance)

        ego_wp: carla.Waypoint = self.world_map.get_waypoint(
            self._vehicle.get_location(), project_to_road=True, lane_type=carla.libcarla.LaneType.Any
        )
        next_wps = expert_utils.wps_next_until_lane_end(ego_wp)
        try:
            next_lane_wps_ego = next_wps[-1].next(1)
            if len(next_lane_wps_ego) == 0:
                next_lane_wps_ego = [next_wps[-1]]
        except:
            next_lane_wps_ego = []
        if ego_wp.is_junction:
            distance_to_junction_ego = 0.0
            # get distance to ego vehicle
        elif len(next_lane_wps_ego) > 0 and next_lane_wps_ego[0].is_junction:
            distance_to_junction_ego = next_lane_wps_ego[0].transform.location.distance(ego_wp.transform.location)
        else:
            distance_to_junction_ego = None

        # how far is next junction
        next_wps = expert_utils.wps_next_until_lane_end(ego_wp)
        try:
            next_lane_wps_ego = next_wps[-1].next(1)
            if len(next_lane_wps_ego) == 0:
                next_lane_wps_ego = [next_wps[-1]]
        except:
            next_lane_wps_ego = []
        if ego_wp.is_junction:
            distance_to_junction_ego = 0.0
            # get distance to ego vehicle
        elif len(next_lane_wps_ego) > 0 and next_lane_wps_ego[0].is_junction:
            distance_to_junction_ego = next_lane_wps_ego[0].transform.location.distance(ego_wp.transform.location)
        else:
            distance_to_junction_ego = None

        next_road_ids_ego = []
        next_next_road_ids_ego = []
        for wp in next_lane_wps_ego:
            next_road_ids_ego.append(wp.road_id)
            next_next_wps = expert_utils.wps_next_until_lane_end(wp)
            try:
                next_next_lane_wps_ego = next_next_wps[-1].next(1)
                if len(next_next_lane_wps_ego) == 0:
                    next_next_lane_wps_ego = [next_next_wps[-1]]
            except:
                next_next_lane_wps_ego = []
            for wp2 in next_next_lane_wps_ego:
                if wp2.road_id not in next_next_road_ids_ego:
                    next_next_road_ids_ego.append(wp2.road_id)

        tl = self._world.get_traffic_lights_from_waypoint(ego_wp, 50.0)
        if len(tl) == 0:
            tl_state = "None"
        else:
            tl_state = str(tl[0].state)

        privileged_yaw = np.radians(self._vehicle.get_transform().rotation.yaw)  # convert from degrees to radians
        dangerous_adversarial_actors_ids, safe_adversarial_actors_ids, ignored_adversarial_actors_ids = (
            self.adversarial_actors_ids
        )
        data = {
            "target_dataset": int(self.config_expert.target_dataset),
            "num_dangerous_adversarial": len(dangerous_adversarial_actors_ids),
            "num_safe_adversarial": len(safe_adversarial_actors_ids),
            "num_ignored_adversarial": len(ignored_adversarial_actors_ids),
            "rear_adversarial_id": -1 if self.rear_adversarial_actor is None else self.rear_adversarial_actor.id,
            "town": self.town,
            "slower_bad_visibility": self.slower_bad_visibility,
            "slower_clutterness": self.slower_clutterness,
            "slower_occluded_junction": self.slower_occluded_junction,
            "privileged_past_positions": np.array(self.privileged_ego_past_positions, dtype=np.float32)[::-1],
            "past_positions": np.array(self.ego_past_positions, dtype=np.float32)[::-1],
            "past_filtered_state": np.array(self.ego_past_filtered_state, dtype=np.float32)[::-1],
            "past_speeds": np.array(self.speeds_queue, dtype=np.float32)[::-1],
            "past_yaws": np.array(self.ego_past_yaws, dtype=np.float32)[::-1],
            "speed": tick_data["speed"],
            "accel_x": tick_data["accel_x"],
            "accel_y": tick_data["accel_y"],
            "accel_z": tick_data["accel_z"],
            "angular_velocity_x": tick_data["angular_velocity_x"],
            "angular_velocity_y": tick_data["angular_velocity_y"],
            "angular_velocity_z": tick_data["angular_velocity_z"],
            "pos_global": self.ego_location_array.tolist(),
            "noisy_pos_global": tick_data["filtered_state"][:2].tolist(),
            "theta": self.ego_orientation_rad,
            "privileged_yaw": privileged_yaw,
            "target_speed": target_speed,
            "target_speed_limit": self.target_speed_limit,
            "speed_limit": self.speed_limit,
            "last_encountered_speed_limit_sign": self.last_encountered_speed_limit_sign,
            "previous_target_points": previous_target_points,
            "next_target_points": next_target_points,
            "previous_commands": previous_commands,
            "next_commands": next_commands,
            "route": np.array(dense_route, dtype=np.float32),
            "changed_route": changed_route,
            "speed_reduced_by_obj_type": speed_reduced_by_obj_type,
            "speed_reduced_by_obj_id": speed_reduced_by_obj_id,
            "speed_reduced_by_obj_distance": speed_reduced_by_obj_distance,
            "steer": control.steer,
            "throttle": control.throttle,
            "brake": bool(control.brake),
            "vehicle_hazard": bool(self.vehicle_hazard),
            "vehicle_affecting_id": self.vehicle_affecting_id,
            "light_hazard": bool(self.traffic_light_hazard),
            "walker_hazard": bool(self.walker_hazard),
            "walker_affecting_id": self.walker_affecting_id,
            "stop_sign_hazard": bool(self.stop_sign_hazard),
            "stop_sign_close": bool(self.stop_sign_close),
            "walker_close": bool(self.walker_close),
            "walker_close_id": self.walker_close_id,
            "perturbation_translation": self.perturbation_translation,
            "perturbation_rotation": self.perturbation_rotation,
            "ego_matrix": np.array(self._vehicle.get_transform().get_matrix(), dtype=np.float32),
            "scenario": self.scenario_name,
            "traffic_light_state": tl_state,
            "distance_to_next_junction": self.distance_to_next_junction,
            "ego_lane_id": self.ego_lane_id,
            "road_id": ego_wp.road_id,
            "lane_id": ego_wp.lane_id,
            "is_junction": ego_wp.is_junction,
            "is_intersection": ego_wp.is_intersection,
            "junction_id": ego_wp.junction_id,
            "next_road_ids": next_road_ids_ego,
            "next_next_road_ids_ego": next_next_road_ids_ego,
            "lane_change_str": str(ego_wp.lane_change),
            "lane_type_str": str(ego_wp.lane_type),
            "left_lane_marking_color_str": str(ego_wp.left_lane_marking.color),
            "left_lane_marking_type_str": str(ego_wp.left_lane_marking.type),
            "right_lane_marking_color_str": str(ego_wp.right_lane_marking.color),
            "right_lane_marking_type_str": str(ego_wp.right_lane_marking.type),
            "route_curvature": self.route_curvature,
            "dist_to_road_discontinuity": self.distance_to_road_discontinuity,
            "reduce_speed_discontinuous_road": self.reduce_speed_discontinuous_road,
            "reduce_speed_high_route_curvature": self.reduce_speed_high_route_curvature,
            "dist_to_construction_site": self.distance_to_construction_site,
            "dist_to_accident_site": self.distance_to_accident_site,
            "dist_to_parked_obstacle": self.distance_to_parked_obstacle,
            "dist_to_vehicle_opens_door": self.distance_to_vehicle_opens_door,
            "dist_to_cutin_vehicle": self.distance_to_cutin_vehicle,
            "dist_to_pedestrian": self.distance_to_pedestrian,
            "dist_to_biker": self.distance_to_biker,
            "dist_to_junction": distance_to_junction_ego,
            "current_active_scenario_type": self.current_active_scenario_type,
            "previous_active_scenario_type": self.previous_active_scenario_type,
            "does_emergency_brake_for_pedestrians": self.does_emergency_brake_for_pedestrians,
            "construction_obstacle_two_ways_stuck": self.construction_obstacle_two_ways_stuck,
            "accident_two_ways_stuck": self.accident_two_ways_stuck,
            "parked_obstacle_two_ways_stuck": self.parked_obstacle_two_ways_stuck,
            "vehicle_opens_door_two_ways_stuck": self.vehicle_opens_door_two_ways_stuck,
            "scenario_obstacles_ids": self.scenario_obstacles_ids,
            "scenario_actors_ids": self.scenario_actors_ids,
            "vehicle_opened_door": self.vehicle_opened_door,
            "vehicle_door_side": self.vehicle_door_side,
            "scenario_obstacles_convex_hull": self.scenario_obstacles_convex_hull,
            "cut_in_actors_ids": self.cut_in_actors_ids,
            "average_traffic_speed": self.average_traffic_speed,
            "max_traffic_speed": self.max_traffic_speed,
            "max_adversarial_speed": self.max_adversarial_speed,
            "distance_to_intersection_index_ego": self.distance_to_intersection_index_ego,
            "ego_lane_width": self.ego_lane_width,
            "target_lane_width": self.target_lane_width,
            "rear_danger_8": self.rear_danger_8,
            "rear_danger_16": self.rear_danger_16,
            "brake_cutin": self.brake_cutin,
            "weather_setting": self.weather_setting,
            "jpeg_storage_quality": self.jpeg_storage_quality,
            "emergency_brake_for_special_vehicle": self.emergency_brake_for_special_vehicle,
            "route_left_length": self.route_left_length,
            "distance_ego_to_route": self.distance_ego_to_route,
            "weather_parameters": self.weather_parameters,
            "signed_dist_to_lane_change": self.signed_dist_to_lane_change,
            "visual_visibility": int(self.visual_visibility),
            "num_parking_vehicles_in_proximity": self.num_parking_vehicles_in_proximity,
            "europe_traffic_light": self.europe_traffic_light,
            "over_head_traffic_light": self.over_head_traffic_light,
            "second_highest_speed": self.second_highest_speed,
            "second_highest_speed_limit": self.second_highest_speed_limit,
        }

        previous_gps_target_points_dict = {}
        previous_gps_commands_dict = {}
        next_gps_target_points_dict = {}
        next_gps_commands_dict = {}
        for k, v in self.gps_waypoint_planners_dict.items():
            previous_gps_target_points_dict[k] = [tp.tolist() for tp in v.previous_target_points]
            previous_gps_commands_dict[k] = [int(i) for i in v.previous_commands]
            next_gps_target_points_dict[k] = [tp[0].tolist() for tp in v.route]
            next_gps_commands_dict[k] = [int(v.route[i][1]) for i in range(len(v.route))]

        for k, v in previous_gps_target_points_dict.items():
            data[f"previous_gps_target_points_{k}"] = v
        for k, v in previous_gps_commands_dict.items():
            data[f"previous_gps_commands_{k}"] = v
        for k, v in next_gps_target_points_dict.items():
            data[f"next_gps_target_points_{k}"] = v
        for k, v in next_gps_commands_dict.items():
            data[f"next_gps_commands_{k}"] = v

        previous_target_points_dict = {}
        previous_commands_dict = {}
        next_target_points_dict = {}
        next_commands_dict = {}
        for k, v in self._command_planners_dict.items():
            previous_target_points_dict[k] = [tp.tolist() for tp in v.previous_target_points]
            previous_commands_dict[k] = [int(i) for i in v.previous_commands]
            next_target_points_dict[k] = [tp[0].tolist() for tp in v.route]
            next_commands_dict[k] = [int(v.route[i][1]) for i in range(len(v.route))]

        for k, v in previous_target_points_dict.items():
            data[f"previous_target_points_{k}"] = v
        for k, v in previous_commands_dict.items():
            data[f"previous_commands_{k}"] = v
        for k, v in next_target_points_dict.items():
            data[f"next_target_points_{k}"] = v
        for k, v in next_commands_dict.items():
            data[f"next_commands_{k}"] = v

        if not self.config_expert.eval_expert:
            self.metas.append((self.step, frame, data))

        return data

    def destroy(self, results=None) -> None:
        """
        Save the collected data and statistics to files, and clean up the data structures.
        This method should be called at the end of the data collection process.

        Args:
            results: Any additional results to be processed or saved.
        """
        torch.cuda.empty_cache()

        # Re-save metas with privileged information for data filtering later
        # This step is necessary so we can obtain higher qualitative data
        # If any logic is changed, this step should be kept an eye on.
        if (self.save_path is not None) and self.config_expert.datagen:
            metas_dir = self.save_path / "metas"
            delta_t = 1.0 / self.config_expert.fps
            N = len(self.metas)

            # Enhance metas
            for i in range(N):
                step, frame, data = self.metas[i]

                # --- Privileged acceleration and angular velocity, mostly for data filtering ---
                if step % self.config_expert.data_save_freq != 0:  # Very important. Only override data that was saved.
                    continue

                if i < N - 1:
                    _, _, data_next = self.metas[i + 1]

                    speed_now = data.get("speed", 0.0)
                    speed_next = data_next.get("speed", 0.0)
                    accel = (speed_next - speed_now) / delta_t

                    yaw_now = data.get("theta", 0.0)
                    yaw_next = data_next.get("theta", 0.0)
                    dyaw = (yaw_next - yaw_now + np.pi) % (2 * np.pi) - np.pi
                    rot_speed = dyaw / delta_t
                else:
                    # No future data; make final frame zero
                    accel = 0.0
                    rot_speed = 0.0

                data["privileged_acceleration"] = accel
                data["privileged_rotation_speed"] = rot_speed

                # --- Future speeds: from one step after current to furthest future ---
                future_speeds = []
                for offset in range(0, self.config_expert.ego_num_temporal_data_points_saved + 1):
                    idx = i + offset
                    if idx < N:
                        _, _, future_data = self.metas[idx]
                        future_speeds.append(future_data.get("speed", 0.0))
                data["future_speeds"] = np.array(future_speeds, dtype=np.float32)

                # --- Future yaws: from one step after current to furthest future ---
                future_yaws = []
                for offset in range(0, self.config_expert.ego_num_temporal_data_points_saved + 1):
                    idx = i + offset
                    if idx < N:
                        _, _, future_data = self.metas[idx]
                        yaw_future = future_data.get("theta", 0.0)
                        dyaw = (yaw_future - yaw_now + np.pi) % (2 * np.pi) - np.pi
                        future_yaws.append(dyaw)
                data["future_yaws"] = np.array(future_yaws, dtype=np.float32)

                # --- Future positions: from one step after current to furthest future ---
                T_world_to_current_ego = np.linalg.inv(np.array(data["ego_matrix"]))
                future_positions = []
                for offset in range(0, self.config_expert.ego_num_temporal_data_points_saved + 1):
                    idx = i + offset
                    if idx < N:
                        _, _, future_data = self.metas[idx]
                        T_future = np.array(future_data["ego_matrix"])
                        pos_world = np.append(T_future[:3, 3], 1.0)
                        pos_current_ego = T_world_to_current_ego @ pos_world
                        future_positions.append(pos_current_ego[:3].tolist())
                data["future_positions"] = np.array(future_positions, dtype=np.float32)

                # --- Save metas ---
                common_utils.write_pickle(path=metas_dir / f"{frame:04}.pkl", data=data)

            # Enhance bounding boxes with temporal information
            for i in range(N):
                step, frame, bounding_boxes = self.bounding_boxes[i]

                if step % self.config_expert.data_save_freq != 0:
                    continue

                _, _, data = self.metas[i]
                ego_matrix_current = np.array(data["ego_matrix"])
                T_world_to_current_ego = np.linalg.inv(ego_matrix_current)

                for box in bounding_boxes:
                    box_id = box["id"]

                    if box["class"] not in ["car", "walker"]:
                        continue

                    # --- Future positions and yaws: from one step after current to furthest future ---
                    future_positions = []
                    future_yaws = []
                    for offset in range(0, self.config_expert.other_vehicles_num_temporal_data_points_saved + 1):
                        idx = i + offset
                        if idx < N:
                            _, _, future_boxes = self.bounding_boxes[idx]
                            future_box = next((b for b in future_boxes if b["id"] == box_id), None)
                            if future_box:
                                T_future = np.array(future_box["matrix"])
                                pos_world = np.append(T_future[:3, 3], 1.0)
                                pos_current_ego = T_world_to_current_ego @ pos_world
                                future_positions.append(pos_current_ego[:2].tolist())

                                rot_world = T_future[:3, :3]
                                heading_vector_world = rot_world @ np.array([1.0, 0.0, 0.0])
                                heading_vector_world = np.append(heading_vector_world, 0.0)
                                heading_vector_ego = T_world_to_current_ego @ heading_vector_world
                                yaw = np.arctan2(heading_vector_ego[1], heading_vector_ego[0])
                                future_yaws.append(yaw)

                    box["future_positions"] = np.array(future_positions, dtype=np.float16)
                    box["future_yaws"] = np.array(future_yaws, dtype=np.float16)

                common_utils.write_pickle(path=self.save_path / "bboxes" / f"{frame:04}.pkl", data=bounding_boxes)

        if results is not None and self.save_path is not None:
            with open(os.path.join(self.save_path, "results.json"), "w", encoding="utf-8") as f:
                json.dump(results.__dict__, f, indent=2)

        if hasattr(self, "_3rd_person_camera"):
            self._3rd_person_camera.stop()
            self._3rd_person_camera.destroy()

        del self.visible_walker_ids
        del self.walker_past_pos

    @beartype
    def predict_other_actors_bounding_boxes(
        self,
        actor_list: list[carla.Actor],
        num_future_frames: int,
        near_lane_change: bool,
    ) -> dict:
        """
        Predict the future bounding boxes of actors for a given number of frames.

        Args:
            actor_list: A list of actors (e.g., vehicles) in the simulation.
            num_future_frames: The number of future frames to predict.
            near_lane_change: Whether the ego vehicle is near a lane change maneuver.

        Returns:
            A dictionary mapping actor IDs to lists of predicted bounding boxes for each future frame.
        """
        predicted_bounding_boxes = {}

        # --- Determine how wider we should make dangerous adversarial actors' bounding boxes
        if self.current_active_scenario_type in [
            "Accident",
            "ConstructionObstacle",
            "ParkedObstacle",
            "HazardAtSideLane",
        ]:  # Adversarials drive in same direction, we make their bounding boxes not too wide
            adversarial_bb_extra_width = max((self.ego_lane_width - self._vehicle.bounding_box.extent.y) / 2, 0) * 0.6
        elif self.current_active_scenario_type in [
            "SignalizedJunctionLeftTurn",
            "NonSignalizedJunctionLeftTurn",
            "SignalizedJunctionLeftTurnEnterFlow",
            "NonSignalizedJunctionLeftTurnEnterFlow",
            "InterurbanActorFlow",
        ]:  # Adversarials drive in opposite direction, we make their bounding boxes wider
            adversarial_bb_extra_width = max((self.ego_lane_width - self._vehicle.bounding_box.extent.y) / 2, 0) * 0.8
        elif self.current_active_scenario_type in [
            "SignalizedJunctionRightTurn",
            "NonSignalizedJunctionRightTurn",
        ]:  # Adversarials drive in same direction, we make their bounding boxes even wider
            adversarial_bb_extra_width = (
                max((self.ego_lane_width - self._vehicle.bounding_box.extent.y) / 2, 0) * 1.75
            )  # Right turn larger BB to avoid collisions bc of blinding fleck

        minimal_adversarial_bb_width = None
        if self.target_lane_width is not None:
            minimal_adversarial_bb_width = adversarial_bb_extra_width + self.target_lane_width

        # --- Determine which adversarial actors to consider, ignore or make dangerous
        dangerous_adversarial_actors_ids, safe_adversarial_actors_ids, ignored_adversarial_actors_ids = (
            self.adversarial_actors_ids
        )

        # Filter out nearby actors within the detection radius, excluding the ego vehicle
        nearby_actors = [actor for actor in actor_list if actor.id != self._vehicle.id and self.is_actor_inside_bev(actor)]

        # If there are nearby actors, calculate their future bounding boxes
        if nearby_actors:
            # Get the previous control inputs (steering, throttle, brake) for the nearby actors
            previous_controls = [actor.get_control() for actor in nearby_actors]
            previous_actions = np.array([[control.steer, control.throttle, control.brake] for control in previous_controls])

            # Get the current velocities, locations, and headings of the nearby actors
            velocities = []
            for actor in nearby_actors:
                actor_original_velocity = actor.get_velocity().length()
                velocities.append(actor_original_velocity)

            velocities = np.array(velocities)
            locations = np.array(
                [[actor.get_location().x, actor.get_location().y, actor.get_location().z] for actor in nearby_actors]
            )
            headings = np.deg2rad(np.array([actor.get_transform().rotation.yaw for actor in nearby_actors]))

            # Initialize arrays to store future locations, headings, and velocities
            future_locations = np.empty((num_future_frames, len(nearby_actors), 3), dtype="float")
            future_headings = np.empty((num_future_frames, len(nearby_actors)), dtype="float")
            future_velocities = np.empty((num_future_frames, len(nearby_actors)), dtype="float")

            # Forecast the future locations, headings, and velocities for the nearby actors
            for i in range(num_future_frames):
                locations, headings, velocities = self.vehicle_model.forecast_other_vehicles(
                    locations, headings, velocities, previous_actions
                )
                future_locations[i] = locations.copy()
                future_velocities[i] = velocities.copy()
                future_headings[i] = headings.copy()
            # Convert future headings to degrees
            future_headings = np.rad2deg(future_headings)

            # Calculate the predicted bounding boxes for each nearby actor and future frame
            for actor_idx, actor in enumerate(nearby_actors):
                predicted_actor_boxes = []

                for i in range(num_future_frames):
                    # Calculate the future location of the actor
                    location = carla.Location(
                        x=future_locations[i, actor_idx, 0].item(),
                        y=future_locations[i, actor_idx, 1].item(),
                        z=future_locations[i, actor_idx, 2].item(),
                    )

                    # Calculate the future rotation of the actor
                    rotation = carla.Rotation(pitch=0, yaw=future_headings[i, actor_idx], roll=0)

                    # Get the extent (dimensions) of the actor's bounding box
                    extent = actor.bounding_box.extent
                    # Otherwise we would increase the extent of the bounding box of the vehicle
                    extent = carla.Vector3D(x=extent.x, y=extent.y, z=extent.z)

                    # Adjust the bounding box size based on velocity and lane change maneuver to adjust for
                    # uncertainty during forecasting
                    s = (
                        self.config_expert.high_speed_min_extent_x_other_vehicle_lane_change
                        if near_lane_change
                        else self.config_expert.high_speed_min_extent_x_other_vehicle
                    )
                    length_factor = (
                        self.config_expert.slow_speed_extent_factor_ego
                        if future_velocities[i, actor_idx] < self.config_expert.extent_other_vehicles_bbs_speed_threshold
                        else max(
                            s,
                            self.config_expert.high_speed_min_extent_x_other_vehicle * float(i) / float(num_future_frames),
                        )
                    )
                    width_factor = (
                        self.config_expert.slow_speed_extent_factor_ego
                        if future_velocities[i, actor_idx] < self.config_expert.extent_other_vehicles_bbs_speed_threshold
                        else max(
                            self.config_expert.high_speed_min_extent_y_other_vehicle,
                            self.config_expert.high_speed_extent_y_factor_other_vehicle * float(i) / float(num_future_frames),
                        )
                    )

                    if self.current_active_scenario_type in ["CrossingBicycleFlow"]:
                        if actor.type_id in constants.BIKER_MESHES:
                            length_factor = 4.0
                            width_factor = 10.0
                    elif self.current_active_scenario_type in ["NonSignalizedJunctionRightTurn"]:
                        if actor.id in dangerous_adversarial_actors_ids:
                            length_factor = 1.5
                            width_factor = minimal_adversarial_bb_width / (extent.y * 2)
                        elif actor.id in safe_adversarial_actors_ids:
                            length_factor = 1.0
                            width_factor = 1
                        elif actor.id in ignored_adversarial_actors_ids:
                            length_factor = 0
                            width_factor = 0
                    elif self.current_active_scenario_type in ["SignalizedJunctionRightTurn"]:
                        if actor.id in dangerous_adversarial_actors_ids:
                            length_factor = 1.6
                            width_factor = minimal_adversarial_bb_width / (extent.y * 2)
                        elif actor.id in safe_adversarial_actors_ids:
                            length_factor = 1.0
                            width_factor = 1
                        elif actor.id in ignored_adversarial_actors_ids:
                            length_factor = 0
                    elif self.current_active_scenario_type in [
                        "SignalizedJunctionLeftTurnEnterFlow",
                        "NonSignalizedJunctionLeftTurnEnterFlow",
                    ]:
                        if actor.id in dangerous_adversarial_actors_ids:
                            length_factor = 1.25
                            width_factor = minimal_adversarial_bb_width / (extent.y * 2)
                        elif actor.id in safe_adversarial_actors_ids:
                            length_factor = 0.75
                            width_factor = 1
                        elif actor.id in ignored_adversarial_actors_ids:
                            length_factor = 0
                            width_factor = 0

                    extent.x *= length_factor
                    extent.y *= width_factor
                    # Create the bounding box for the future frame
                    bounding_box = carla.BoundingBox(location, extent)
                    bounding_box.rotation = rotation

                    # Append the bounding box to the list of predicted bounding boxes for this actor
                    predicted_actor_boxes.append(bounding_box)

                # Store the predicted bounding boxes for this actor in the dictionary
                predicted_bounding_boxes[actor.id] = predicted_actor_boxes

        if self.config_expert.visualize_bounding_boxes:
            for _actor_idx, actors_forecasted_bounding_boxes in predicted_bounding_boxes.items():
                for bb in actors_forecasted_bounding_boxes:
                    color = self.config_expert.other_vehicles_forecasted_bbs_color
                    if _actor_idx in dangerous_adversarial_actors_ids or _actor_idx in safe_adversarial_actors_ids:
                        color = self.config_expert.adversarial_color
                    self._world.debug.draw_box(
                        box=bb,
                        rotation=bb.rotation,
                        thickness=0.1,
                        color=color,
                        life_time=self.config_expert.draw_life_time,
                    )

        return predicted_bounding_boxes

    @beartype
    def compute_target_speed_wrt_leading_vehicle(
        self,
        initial_target_speed: float,
        predicted_bounding_boxes: dict,
        near_lane_change: bool,
        ego_location: carla.Location,
        rear_vehicle_ids: list,
        leading_vehicle_ids: list,
        speed_reduced_by_obj: list | None,
    ) -> tuple[float, list | None]:
        """
        Compute the target speed for the ego vehicle considering the leading vehicle.

        Args:
            initial_target_speed: The initial target speed for the ego vehicle.
            predicted_bounding_boxes: A dictionary mapping actor IDs to lists of predicted bounding boxes.
            near_lane_change: Whether the ego vehicle is near a lane change maneuver.
            ego_location: The current location of the ego vehicle.
            rear_vehicle_ids: A list of IDs for vehicles behind the ego vehicle.
            leading_vehicle_ids: A list of IDs for vehicles in front of the ego vehicle.
            speed_reduced_by_obj: A list containing [reduced speed, object type, object ID, distance]
                for the object that caused the most speed reduction, or None if no speed reduction.

        Returns:
            A tuple containing the target speed considering the leading vehicle and the updated speed_reduced_by_obj.
        """
        target_speed_wrt_leading_vehicle = initial_target_speed
        # _, _, ignored_adversarial_actors_ids = self.adversarial_actors_ids

        for vehicle_id, _ in predicted_bounding_boxes.items():
            # if vehicle_id in ignored_adversarial_actors_ids:
            #     continue
            if vehicle_id in leading_vehicle_ids and not near_lane_change:
                # Vehicle is in front of the ego vehicle
                ego_speed = self._vehicle.get_velocity().length()
                vehicle = self._world.get_actor(vehicle_id)
                other_speed = vehicle.get_velocity().length()
                distance_to_vehicle = ego_location.distance(vehicle.get_location())

                # Compute the target speed using the IDM
                target_speed_wrt_leading_vehicle = min(
                    target_speed_wrt_leading_vehicle,
                    expert_utils.compute_target_speed_idm(
                        config=self.config_expert,
                        desired_speed=initial_target_speed,
                        leading_actor_length=vehicle.bounding_box.extent.x * 2,
                        ego_speed=ego_speed,
                        leading_actor_speed=other_speed,
                        distance_to_leading_actor=distance_to_vehicle,
                        s0=self.config_expert.idm_leading_vehicle_minimum_distance,
                        T=self.config_expert.idm_leading_vehicle_time_headway,
                    ),
                )

                # Update the object causing the most speed reduction
                if speed_reduced_by_obj is None or speed_reduced_by_obj[0] > target_speed_wrt_leading_vehicle:
                    speed_reduced_by_obj = [
                        target_speed_wrt_leading_vehicle,
                        vehicle.type_id,
                        vehicle.id,
                        distance_to_vehicle,
                    ]

            if self.config_expert.visualize_bounding_boxes:
                for vehicle_id in predicted_bounding_boxes.keys():
                    # check if vehicle is in front of the ego vehicle
                    if vehicle_id in leading_vehicle_ids and not near_lane_change:
                        vehicle = self._world.get_actor(vehicle_id)
                        extent = vehicle.bounding_box.extent
                        bb = carla.BoundingBox(vehicle.get_location(), extent)
                        bb.rotation = carla.Rotation(pitch=0, yaw=vehicle.get_transform().rotation.yaw, roll=0)
                        self._world.debug.draw_box(
                            box=bb,
                            rotation=bb.rotation,
                            thickness=0.5,
                            color=self.config_expert.leading_vehicle_color,
                            life_time=self.config_expert.draw_life_time,
                        )
                    elif vehicle_id in rear_vehicle_ids:
                        vehicle = self._world.get_actor(vehicle_id)
                        extent = vehicle.bounding_box.extent
                        bb = carla.BoundingBox(vehicle.get_location(), extent)
                        bb.rotation = carla.Rotation(pitch=0, yaw=vehicle.get_transform().rotation.yaw, roll=0)
                        self._world.debug.draw_box(
                            box=bb,
                            rotation=bb.rotation,
                            thickness=0.5,
                            color=self.config_expert.trailing_vehicle_color,
                            life_time=self.config_expert.draw_life_time,
                        )

        return target_speed_wrt_leading_vehicle, speed_reduced_by_obj

    @beartype
    def compute_target_speeds_wrt_all_actors(
        self,
        initial_target_speed: float,
        ego_bounding_boxes: list,
        predicted_bounding_boxes: dict,
        near_lane_change: bool,
        leading_vehicle_ids: list,
        rear_vehicle_ids: list,
        speed_reduced_by_obj: list | None,
        nearby_walkers: list,
        nearby_walkers_ids: list,
    ) -> tuple[float, float, float, list | None]:
        """
        Compute the target speeds for the ego vehicle considering all actors (vehicles, bicycles,
        and pedestrians) by checking for intersecting bounding boxes.

        Args:
            initial_target_speed: The initial target speed for the ego vehicle.
            ego_bounding_boxes: A list of bounding boxes for the ego vehicle at different future frames.
            predicted_bounding_boxes: A dictionary mapping actor IDs to lists of predicted bounding boxes.
            near_lane_change: Whether the ego vehicle is near a lane change maneuver.
            leading_vehicle_ids: A list of IDs for vehicles in front of the ego vehicle.
            rear_vehicle_ids: A list of IDs for vehicles behind the ego vehicle.
            speed_reduced_by_obj: A list containing [reduced speed, object type,
                object ID, distance] for the object that caused the most speed reduction, or None if
                no speed reduction.
            nearby_walkers: A list of predicted bounding boxes of nearby pedestrians.
            nearby_walkers_ids: A list of IDs for nearby pedestrians.

        Returns:
            A tuple containing the target speeds for bicycles, pedestrians, vehicles, and the updated
                speed_reduced_by_obj list.
        """
        target_speed_bicycle = initial_target_speed
        target_speed_pedestrian = initial_target_speed
        target_speed_vehicle = initial_target_speed
        ego_vehicle_location = self._vehicle.get_location()
        hazard_color = self.config_expert.ego_vehicle_forecasted_bbs_hazard_color
        normal_color = self.config_expert.ego_vehicle_forecasted_bbs_normal_color
        color = normal_color

        # Iterate over the ego vehicle's bounding boxes and predicted bounding boxes of other actors
        for i, ego_bounding_box in enumerate(ego_bounding_boxes):
            for vehicle_id, bounding_boxes in predicted_bounding_boxes.items():
                # Skip leading and rear vehicles if not near a lane change
                if vehicle_id in leading_vehicle_ids and not near_lane_change:
                    continue
                elif vehicle_id in rear_vehicle_ids and not near_lane_change:
                    continue
                else:
                    # Check if the ego bounding box intersects with the predicted bounding box of the actor
                    intersects_with_ego = expert_utils.check_obb_intersection(ego_bounding_box, bounding_boxes[i])

                    if intersects_with_ego:
                        blocking_actor = self._world.get_actor(vehicle_id)
                        # Handle the case when the blocking actor is a bicycle
                        if "base_type" in blocking_actor.attributes and blocking_actor.attributes["base_type"] == "bicycle":
                            other_speed = blocking_actor.get_velocity().length()
                            distance_to_actor = ego_vehicle_location.distance(blocking_actor.get_location())

                            # Compute the target speed for bicycles using the IDM
                            target_speed_bicycle = min(
                                target_speed_bicycle,
                                expert_utils.compute_target_speed_idm(
                                    config=self.config_expert,
                                    desired_speed=initial_target_speed,
                                    leading_actor_length=blocking_actor.bounding_box.extent.x * 2,
                                    ego_speed=self.ego_speed,
                                    leading_actor_speed=other_speed,
                                    distance_to_leading_actor=distance_to_actor,
                                    s0=self.config_expert.idm_bicycle_minimum_distance,
                                    T=self.config_expert.idm_bicycle_desired_time_headway,
                                ),
                            )

                            # Update the object causing the most speed reduction
                            if speed_reduced_by_obj is None or speed_reduced_by_obj[0] > target_speed_bicycle:
                                speed_reduced_by_obj = [
                                    target_speed_bicycle,
                                    blocking_actor.type_id,
                                    blocking_actor.id,
                                    distance_to_actor,
                                ]

                        # Handle the case when the blocking actor is not a bicycle
                        else:
                            self.vehicle_hazard = True  # Set the vehicle hazard flag
                            self.vehicle_affecting_id = vehicle_id  # Store the ID of the vehicle causing the hazard
                            color = hazard_color  # Change the following colors from green to red (no hazard to hazard)
                            target_speed_vehicle = 0.0  # Set the target speed for vehicles to zero
                            distance_to_actor = blocking_actor.get_location().distance(ego_vehicle_location)

                            # Update the object causing the most speed reduction
                            if speed_reduced_by_obj is None or speed_reduced_by_obj[0] > target_speed_vehicle:
                                speed_reduced_by_obj = [
                                    target_speed_vehicle,
                                    blocking_actor.type_id,
                                    blocking_actor.id,
                                    distance_to_actor,
                                ]

            # Iterate over nearby pedestrians and check for intersections with the ego bounding box
            for pedestrian_bb, pedestrian_id in zip(nearby_walkers, nearby_walkers_ids, strict=False):
                if expert_utils.check_obb_intersection(ego_bounding_box, pedestrian_bb[i]):
                    color = hazard_color
                    blocking_actor = self._world.get_actor(pedestrian_id)
                    distance_to_actor = ego_vehicle_location.distance(blocking_actor.get_location())

                    # Compute the target speed for pedestrians using the IDM
                    target_speed_pedestrian = min(
                        target_speed_pedestrian,
                        expert_utils.compute_target_speed_idm(
                            config=self.config_expert,
                            desired_speed=initial_target_speed,
                            leading_actor_length=0.5 + self._vehicle.bounding_box.extent.x,
                            ego_speed=self.ego_speed,
                            leading_actor_speed=0.0,
                            distance_to_leading_actor=distance_to_actor,
                            s0=self.config_expert.idm_pedestrian_minimum_distance,
                            T=self.config_expert.idm_pedestrian_desired_time_headway,
                        ),
                    )

                    # Update the object causing the most speed reduction
                    if speed_reduced_by_obj is None or speed_reduced_by_obj[0] > target_speed_pedestrian:
                        speed_reduced_by_obj = [
                            target_speed_pedestrian,
                            blocking_actor.type_id,
                            blocking_actor.id,
                            distance_to_actor,
                        ]
            if self.config_expert.visualize_bounding_boxes:
                self._world.debug.draw_box(
                    box=ego_bounding_box,
                    rotation=ego_bounding_box.rotation,
                    thickness=0.1,
                    color=color,
                    life_time=self.config_expert.draw_life_time,
                )
        return float(target_speed_bicycle), float(target_speed_pedestrian), float(target_speed_vehicle), speed_reduced_by_obj

    @beartype
    def forecast_ego_agent(
        self,
        current_ego_transform: carla.Transform,
        num_future_frames: int,
        initial_target_speed: float,
        route_points: np.ndarray,
    ) -> list:
        """
        Forecast the future states of the ego agent using the kinematic bicycle model and assume their is no hazard to
        check subsequently whether the ego vehicle would collide.

        Args:
            current_ego_transform: The current transform of the ego vehicle.
            num_future_frames: The number of future frames to forecast.
            initial_target_speed: The initial target speed for the ego vehicle.
            route_points: An array of waypoints representing the planned route.

        Returns:
            A list of bounding boxes representing the future states of the ego vehicle.
        """
        self._turn_controller.save_state()
        self._waypoint_planner.save()

        # Initialize the initial state without braking
        location = np.array(
            [current_ego_transform.location.x, current_ego_transform.location.y, current_ego_transform.location.z]
        )
        heading_angle = np.array([np.deg2rad(current_ego_transform.rotation.yaw)])
        speed = np.array([self.ego_speed])

        # Calculate the throttle command based on the target speed and current speed
        throttle = self._longitudinal_controller.get_throttle_extrapolation(initial_target_speed, self.ego_speed)
        steering = self._turn_controller.step(route_points, speed, location, heading_angle.item())
        action = np.array([steering, throttle, 0.0]).flatten()

        future_bounding_boxes = []
        # Iterate over the future frames and forecast the ego agent's state
        for _ in range(num_future_frames):
            # Forecast the next state using the kinematic bicycle model
            location, heading_angle, speed = self.ego_model.forecast_ego_vehicle(
                location, float(heading_angle), float(speed), action
            )

            # Update the route and extrapolate steering and throttle commands
            extrapolated_route, _, _, _, _, _, _ = self._waypoint_planner.run_step(location)
            steering = self._turn_controller.step(extrapolated_route, speed, location, heading_angle.item())
            throttle = self._longitudinal_controller.get_throttle_extrapolation(initial_target_speed, speed)
            action = np.array([steering, throttle, 0.0]).flatten()

            heading_angle_degrees = np.rad2deg(heading_angle).item()

            # Decrease the ego vehicles bounding box if it is slow and resolve permanent bounding box
            # intersectinos at collisions.
            # In case of driving increase them for safety.
            extent = self._vehicle.bounding_box.extent
            # Otherwise we would increase the extent of the bounding box of the vehicle
            extent = carla.Vector3D(x=extent.x, y=extent.y, z=extent.z)
            extent.x *= (
                self.config_expert.slow_speed_extent_factor_ego
                if self.ego_speed < self.config_expert.extent_ego_bbs_speed_threshold
                else self.config_expert.high_speed_extent_factor_ego_x
            )
            extent.y *= (
                self.config_expert.slow_speed_extent_factor_ego
                if self.ego_speed < self.config_expert.extent_ego_bbs_speed_threshold
                else self.config_expert.high_speed_extent_factor_ego_y
            )

            transform = carla.Transform(carla.Location(x=location[0].item(), y=location[1].item(), z=location[2].item()))

            ego_bounding_box = carla.BoundingBox(transform.location, extent)
            ego_bounding_box.rotation = carla.Rotation(pitch=0, yaw=heading_angle_degrees, roll=0)

            future_bounding_boxes.append(ego_bounding_box)

        self._turn_controller.load_state()
        self._waypoint_planner.load()

        return future_bounding_boxes

    @beartype
    def forecast_walkers(self, number_of_future_frames: int):
        """
        Forecast the future locations of pedestrians in the vicinity of the ego vehicle assuming they
        keep their velocity and direction

        Args:
            number_of_future_frames (int): The number of future frames to forecast.

        Returns:
            tuple: A tuple containing two lists:
                - list: A list of lists, where each inner list contains the future bounding boxes for a pedestrian.
                - list: A list of IDs for the pedestrians whose locations were forecasted.
        """
        nearby_pedestrians_bbs, nearby_pedestrian_ids = [], []

        # Filter pedestrians within the detection radius
        pedestrians = self.walkers_inside_bev

        # If no pedestrians are found, return empty lists
        if not pedestrians:
            return nearby_pedestrians_bbs, nearby_pedestrian_ids

        # Extract pedestrian locations, speeds, and directions
        pedestrian_locations = np.array(
            [[ped.get_location().x, ped.get_location().y, ped.get_location().z] for ped in pedestrians]
        )
        pedestrian_speeds = np.array([ped.get_velocity().length() for ped in pedestrians])
        pedestrian_speeds = np.maximum(pedestrian_speeds, self.config_expert.min_walker_speed)
        pedestrian_directions = np.array(
            [
                [ped.get_control().direction.x, ped.get_control().direction.y, ped.get_control().direction.z]
                for ped in pedestrians
            ]
        )

        # Calculate future pedestrian locations based on their current locations, speeds, and directions
        future_pedestrian_locations = (
            pedestrian_locations[:, None, :]
            + np.arange(1, number_of_future_frames + 1)[None, :, None]
            * pedestrian_directions[:, None, :]
            * pedestrian_speeds[:, None, None]
            / self.config_expert.bicycle_frame_rate
        )

        # Iterate over pedestrians and calculate their future bounding boxes
        for i, ped in enumerate(pedestrians):
            bb, transform = ped.bounding_box, ped.get_transform()
            rotation = carla.Rotation(
                pitch=bb.rotation.pitch + transform.rotation.pitch,
                yaw=bb.rotation.yaw + transform.rotation.yaw,
                roll=bb.rotation.roll + transform.rotation.roll,
            )
            extent = bb.extent
            extent.x = max(self.config_expert.pedestrian_minimum_extent, extent.x)  # Ensure a minimum width
            extent.y = max(self.config_expert.pedestrian_minimum_extent, extent.y)  # Ensure a minimum length

            pedestrian_future_bboxes = []
            for j in range(number_of_future_frames):
                location = carla.Location(
                    future_pedestrian_locations[i, j, 0],
                    future_pedestrian_locations[i, j, 1],
                    future_pedestrian_locations[i, j, 2],
                )

                bounding_box = carla.BoundingBox(location, extent)
                bounding_box.rotation = rotation
                pedestrian_future_bboxes.append(bounding_box)

            nearby_pedestrian_ids.append(ped.id)
            nearby_pedestrians_bbs.append(pedestrian_future_bboxes)

        # Visualize the future bounding boxes of pedestrians (if enabled)
        if self.config_expert.visualize_bounding_boxes:
            for bbs in nearby_pedestrians_bbs:
                for bbox in bbs:
                    self._world.debug.draw_box(
                        box=bbox,
                        rotation=bbox.rotation,
                        thickness=0.1,
                        color=self.config_expert.pedestrian_forecasted_bbs_color,
                        life_time=self.config_expert.draw_life_time,
                    )

        return nearby_pedestrians_bbs, nearby_pedestrian_ids

    @beartype
    def ego_agent_affected_by_red_light(
        self,
        ego_vehicle_location: carla.Location,
        distance_to_traffic_light_stop_point: float,
        next_traffic_light: carla.TrafficLight | None,
        target_speed: float,
    ) -> float:
        """
        Handles the behavior of the ego vehicle when approaching a traffic light.

        Args:
            ego_vehicle_location: The ego vehicle location.
            ego_vehicle_speed: The current speed of the ego vehicle in m/s.
            distance_to_traffic_light_stop_point: The distance from the ego vehicle to the point we want to stop.
                                                  Default is distance from ego to next traffic light.
            next_traffic_light: The next traffic light in the route.
            target_speed: The target speed for the ego vehicle.

        Returns:
            float: The adjusted target speed for the ego vehicle.
        """

        self.close_traffic_lights.clear()

        for traffic_light, traffic_light_center, traffic_light_waypoints in self.list_traffic_lights:
            center_loc = carla.Location(traffic_light_center)
            if center_loc.distance(ego_vehicle_location) > self.config_expert.light_radius:
                continue

            for wp in traffic_light_waypoints:
                # * 0.9 to make the box slightly smaller than the street to prevent overlapping boxes.
                length_bounding_box = carla.Vector3D(
                    (wp.lane_width / 2.0) * 0.9, traffic_light.trigger_volume.extent.y, traffic_light.trigger_volume.extent.z
                )
                length_bounding_box = carla.Vector3D(1.5, 1.5, 0.5)

                wp_location = wp.transform
                traffic_light_location = traffic_light.get_transform()
                traffic_light_pos_on_street = common_utils.get_relative_transform(
                    ego_matrix=np.array(wp_location.get_matrix()), vehicle_matrix=np.array(traffic_light_location.get_matrix())
                )
                traffic_light_bb_location = carla.Location(
                    x=wp_location.location.x,
                    y=wp_location.location.y,
                    z=wp_location.location.z + traffic_light_pos_on_street[-1],  # z of traffic light is relative to street
                )
                bounding_box = carla.BoundingBox(traffic_light_bb_location, length_bounding_box)

                global_rot = traffic_light.get_transform().rotation
                bounding_box.rotation = carla.Rotation(pitch=global_rot.pitch, yaw=global_rot.yaw, roll=global_rot.roll)

                affects_ego = next_traffic_light is not None and traffic_light.id == next_traffic_light.id

                self.close_traffic_lights.append(
                    [traffic_light, bounding_box, traffic_light.state, traffic_light.id, affects_ego]
                )

                if self.config_expert.visualize_traffic_lights_bounding_boxes:
                    if traffic_light.state == carla.TrafficLightState.Red:
                        color = self.config_expert.red_traffic_light_color
                    elif traffic_light.state == carla.TrafficLightState.Yellow:
                        color = self.config_expert.yellow_traffic_light_color
                    elif traffic_light.state == carla.TrafficLightState.Green:
                        color = self.config_expert.green_traffic_light_color
                    elif traffic_light.state == carla.TrafficLightState.Off:
                        color = self.config_expert.off_traffic_light_color
                    else:  # unknown
                        color = self.config_expert.unknown_traffic_light_color

                    self._world.debug.draw_box(
                        box=bounding_box, rotation=bounding_box.rotation, thickness=0.1, color=color, life_time=0.051
                    )

                    self._world.debug.draw_point(
                        wp.transform.location + carla.Location(z=traffic_light.trigger_volume.location.z),
                        size=0.1,
                        color=color,
                        life_time=(1.0 / self.config_expert.carla_fps) + 1e-6,
                    )

                    self._world.debug.draw_box(
                        box=traffic_light.bounding_box,
                        rotation=traffic_light.bounding_box.rotation,
                        thickness=0.1,
                        color=color,
                        life_time=0.051,
                    )

        # If being a sensory agent, we use the more consistent distance to traffic light using bounding box
        # Here we also perform check if a red light is normal and we should come as near as possible or if
        # the red light is not normal and we should keep a larger
        if (
            self.config_expert.datagen
            and next_traffic_light is not None
            and next_traffic_light.id in self.data_agent_id_to_bb_map
        ):
            next_traffic_light_bb = self.data_agent_id_to_bb_map[
                next_traffic_light.id
            ]  # Red light bounding box of the traffic light projected to ego road, doesn't have to be on same lane as ego
            if next_traffic_light_bb is not None:
                distance_ego_to_traffic_light_bb = next_traffic_light_bb["position"][0]  # Longitudinal distance

                if next_traffic_light_bb["is_over_head_traffic_light"]:  # Overhead traffic light, must stand much further
                    distance_to_traffic_light_stop_point = max(
                        distance_ego_to_traffic_light_bb - self.config_expert.idm_overhead_red_light_minimum_distance, 0
                    )
                    self.over_head_traffic_light = True
                elif next_traffic_light_bb["is_europe_traffic_light"]:  # Overhead traffic light, must stand much further
                    distance_to_traffic_light_stop_point = max(
                        distance_ego_to_traffic_light_bb - self.config_expert.idm_europe_red_light_minimum_distance, 0
                    )
                    self.europe_traffic_light = True
                else:  # Traffic light is on other side of intersection, we can stop normally
                    distance_to_traffic_light_stop_point = distance_ego_to_traffic_light_bb
            else:
                LOG.info("Warning: Could not find traffic light bounding box, using default distance to stop point.")

        if next_traffic_light is not None:
            CarlaDataProvider.memory["next_traffic_light"] = next_traffic_light
        # If green light, just skip
        if next_traffic_light is None or next_traffic_light.state == carla.TrafficLightState.Green:
            # No traffic light or green light, continue with the current target speed
            return target_speed

        # Compute the target speed using the IDM
        target_speed = expert_utils.compute_target_speed_idm(
            config=self.config_expert,
            desired_speed=target_speed,
            leading_actor_length=0.0,
            ego_speed=self.ego_speed,
            leading_actor_speed=0.0,
            distance_to_leading_actor=distance_to_traffic_light_stop_point,
            s0=self.config_expert.idm_red_light_minimum_distance,
            T=self.config_expert.idm_red_light_desired_time_headway,
        )

        return target_speed

    @beartype
    def ego_agent_affected_by_stop_sign(
        self,
        ego_vehicle_location: carla.Location,
        next_stop_sign: carla.Actor | None,
        target_speed: float,
        actor_list: carla.ActorList,
    ) -> float:
        """
        Handles the behavior of the ego vehicle when approaching a stop sign.

        Args:
            ego_vehicle_location: The location of the ego vehicle.
            next_stop_sign: The next stop sign in the route.
            target_speed: The target speed for the ego vehicle.
            actor_list: A list of all actors (vehicles, pedestrians, etc.) in the simulation.

        Returns:
            The adjusted target speed for the ego vehicle.
        """
        self.close_stop_signs.clear()
        stop_signs = self.get_nearby_object(
            ego_vehicle_location, actor_list.filter("*traffic.stop*"), self.config_expert.light_radius
        )

        for stop_sign in stop_signs:
            center_bb_stop_sign = stop_sign.get_transform().transform(stop_sign.trigger_volume.location)
            stop_sign_extent = carla.Vector3D(1.5, 1.5, 0.5)
            bounding_box_stop_sign = carla.BoundingBox(center_bb_stop_sign, stop_sign_extent)
            rotation_stop_sign = stop_sign.get_transform().rotation
            bounding_box_stop_sign.rotation = carla.Rotation(
                pitch=rotation_stop_sign.pitch, yaw=rotation_stop_sign.yaw, roll=rotation_stop_sign.roll
            )

            affects_ego = next_stop_sign is not None and next_stop_sign.id == stop_sign.id and not self.cleared_stop_sign
            self.close_stop_signs.append([bounding_box_stop_sign, stop_sign.id, affects_ego])

            if self.config_expert.visualize_bounding_boxes:
                color = carla.Color(0, 1, 0) if affects_ego else carla.Color(1, 0, 0)
                self._world.debug.draw_box(
                    box=bounding_box_stop_sign,
                    rotation=bounding_box_stop_sign.rotation,
                    thickness=0.1,
                    color=color,
                    life_time=(1.0 / self.config_expert.carla_fps) + 1e-6,
                )

        if next_stop_sign is None:
            # No stop sign, continue with the current target speed
            return target_speed

        # Calculate the accurate distance to the stop sign
        distance_to_stop_sign = (
            next_stop_sign.get_transform().transform(next_stop_sign.trigger_volume.location).distance(ego_vehicle_location)
        )

        # Reset the stop sign flag if we are farther than 10m away
        if distance_to_stop_sign > self.config_expert.unclearing_distance_to_stop_sign:
            self.cleared_stop_sign = False
            self.waiting_ticks_at_stop_sign = 0
        else:
            # Set the stop sign flag if we are closer than 3m and speed is low enough
            if self.ego_speed < 0.1 and distance_to_stop_sign < self.config_expert.clearing_distance_to_stop_sign:
                self.waiting_ticks_at_stop_sign += 1
                if self.waiting_ticks_at_stop_sign > 25:
                    self.cleared_stop_sign = True
            else:
                self.waiting_ticks_at_stop_sign = 0

        # Set the distance to stop sign as infinity if the stop sign has been cleared
        distance_to_stop_sign = np.inf if self.cleared_stop_sign else distance_to_stop_sign

        # If being a sensory agent, we use the more consistent distance to stop sign using bounding box
        if self.config_expert.datagen and distance_to_stop_sign < np.inf and next_stop_sign.id in self.data_agent_id_to_bb_map:
            next_traffic_light_bb = self.data_agent_id_to_bb_map[next_stop_sign.id]
            distance_to_stop_sign = next_traffic_light_bb["distance"]

        # Compute the target speed using the IDM
        target_speed = expert_utils.compute_target_speed_idm(
            config=self.config_expert,
            desired_speed=target_speed,
            leading_actor_length=0,
            ego_speed=self.ego_speed,
            leading_actor_speed=0.0,
            distance_to_leading_actor=distance_to_stop_sign,
            s0=self.config_expert.idm_stop_sign_minimum_distance,
            T=self.config_expert.idm_stop_sign_desired_time_headway,
        )

        # Return whether the ego vehicle is affected by the stop sign and the adjusted target speed
        return target_speed

    @beartype
    def get_nearby_object(
        self, ego_vehicle_position: carla.Location, all_actors: carla.ActorList, search_radius: float
    ) -> list:
        """
        Find actors, who's trigger boxes are within a specified radius around the ego vehicle.

        Args:
            ego_vehicle_position: The position of the ego vehicle.
            all_actors: A list of all actors.
            search_radius: The radius (in meters) around the ego vehicle to search for nearby actors.

        Returns:
            A list of actors within the specified search radius.
        """
        nearby_objects = []
        for actor in all_actors:
            try:
                trigger_box_global_pos = actor.get_transform().transform(actor.trigger_volume.location)
            except:
                LOG.info("Warning! Error caught in get_nearby_objects. (probably AttributeError: actor.trigger_volume)")
                LOG.info("Skipping this object.")
                continue

            # Convert the vector to a carla.Location for distance calculation
            trigger_box_global_pos = carla.Location(
                x=trigger_box_global_pos.x, y=trigger_box_global_pos.y, z=trigger_box_global_pos.z
            )

            # Check if the actor's trigger volume is within the search radius
            if trigger_box_global_pos.distance(ego_vehicle_position) < search_radius:
                nearby_objects.append(actor)

        return nearby_objects

    @beartype
    def _get_actor_forward_speed(self, actor: carla.Actor) -> float:
        return self._get_forward_speed(transform=actor.get_transform(), velocity=actor.get_velocity())

    @beartype
    def _get_forward_speed(self, transform: carla.Transform | None = None, velocity: carla.Vector3D | None = None) -> float:
        """
        Calculate the forward speed of the vehicle based on its transform and velocity.

        Args:
            transform: The transform of the vehicle. If not provided, it will be obtained from ego.
            velocity: The velocity of the vehicle. If not provided, it will be obtained from ego.

        Returns:
            The forward speed of the vehicle in m/s.
        """
        if not velocity:
            velocity = self._vehicle.get_velocity()

        if not transform:
            transform = self._vehicle.get_transform()

        # Convert the velocity vector to a NumPy array
        velocity_np = np.array([velocity.x, velocity.y, velocity.z])

        # Convert rotation angles from degrees to radians
        pitch_rad = np.deg2rad(transform.rotation.pitch)
        yaw_rad = np.deg2rad(transform.rotation.yaw)

        # Calculate the orientation vector based on pitch and yaw angles
        orientation_vector = np.array(
            [
                np.cos(pitch_rad) * np.cos(yaw_rad),
                np.cos(pitch_rad) * np.sin(yaw_rad),
                np.sin(pitch_rad),
            ]
        )

        # Calculate the forward speed by taking the dot product of velocity and orientation vectors
        forward_speed = np.dot(velocity_np, orientation_vector)

        return float(forward_speed)

    @beartype
    def _is_near_lane_change(self, route_points: np.ndarray, look_ahead_points: int | None = None) -> bool:
        """
        Computes if the ego agent is/was close to a lane change maneuver.

        Args:
            route_points: An array of locations representing the planned route.
            look_ahead_points: Number of waypoints to look ahead.

        Returns:
            True if the ego agent is/was close to a lane change maneuver, False otherwise.
        """
        # Calculate the braking distance based on the ego velocity
        braking_distance = (
            ((self.ego_speed * 3.6) / 10.0) ** 2 / 2.0
        ) + self.config_expert.braking_distance_calculation_safety_distance

        # Determine the number of waypoints to look ahead based on the braking distance
        if look_ahead_points is None:
            look_ahead_points = max(
                self.config_expert.minimum_lookahead_distance_to_compute_near_lane_change,
                min(route_points.shape[0], self.config_expert.points_per_meter * int(braking_distance)),
            )
        current_route_index = self._waypoint_planner.route_index
        max_route_length = len(self._waypoint_planner.commands)

        from_index = max(0, current_route_index - self.config_expert.check_previous_distance_for_lane_change)
        to_index = min(max_route_length - 1, current_route_index + look_ahead_points)
        # Iterate over the points around the current position, checking for lane change commands
        for i in range(from_index, to_index, 1):
            if self._waypoint_planner.commands[i] in (RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT):
                return True

        return False

    @beartype
    def _vehicle_obstacle_detected(
        self,
        vehicle_list: list | None = None,
        max_distance: float | None = None,
        up_angle_th: float = 90,
        low_angle_th: float = 0,
        lane_offset: float = 0,
    ) -> tuple[bool, int | None, float | None]:
        """
        Check if there is a vehicle in front of the agent blocking its path.

        Args:
            vehicle_list: List containing vehicle objects. If None, all vehicles in the scene are used.
            max_distance: Maximum freespace to check for obstacles. If None, the base threshold value is used.
            up_angle_th: Upper angle threshold in degrees.
            low_angle_th: Lower angle threshold in degrees.
            lane_offset: Lane offset for checking adjacent lanes.

        Returns:
            A tuple containing (obstacle_detected, vehicle_id, distance_to_obstacle).
        """
        self._use_bbs_detection = False
        self._offset = 0

        def get_route_polygon():
            route_bb = []
            extent_y = self._vehicle.bounding_box.extent.y
            r_ext = extent_y + self._offset
            l_ext = -extent_y + self._offset
            r_vec = ego_transform.get_right_vector()
            p1 = ego_location + carla.Location(r_ext * r_vec.x, r_ext * r_vec.y)
            p2 = ego_location + carla.Location(l_ext * r_vec.x, l_ext * r_vec.y)
            route_bb.extend([[p1.x, p1.y, p1.z], [p2.x, p2.y, p2.z]])

            for wp, _ in self._local_planner.get_plan():
                if ego_location.distance(wp.transform.location) > max_distance:
                    break

                r_vec = wp.transform.get_right_vector()
                p1 = wp.transform.location + carla.Location(r_ext * r_vec.x, r_ext * r_vec.y)
                p2 = wp.transform.location + carla.Location(l_ext * r_vec.x, l_ext * r_vec.y)
                route_bb.extend([[p1.x, p1.y, p1.z], [p2.x, p2.y, p2.z]])

            # Two points don't create a polygon, nothing to check
            if len(route_bb) < 3:
                return None

            return Polygon(route_bb)

        if not vehicle_list:
            vehicle_list = self.vehicles_inside_bev

        ego_transform = self._vehicle.get_transform()
        ego_location = ego_transform.location
        ego_wpt = self.world_map.get_waypoint(ego_location, lane_type=carla.libcarla.LaneType.Any)

        # Get the right offset
        if ego_wpt.lane_id < 0 and lane_offset != 0:
            lane_offset *= -1

        # Get the transform of the front of the ego
        ego_front_transform = ego_transform
        ego_front_transform.location += carla.Location(self._vehicle.bounding_box.extent.x * ego_transform.get_forward_vector())

        opposite_invasion = abs(self._offset) + self._vehicle.bounding_box.extent.y > ego_wpt.lane_width / 2
        use_bbs = self._use_bbs_detection or opposite_invasion or ego_wpt.is_junction

        # Get the route bounding box
        route_polygon = get_route_polygon()

        for target_vehicle in vehicle_list:
            if target_vehicle.id == self._vehicle.id:
                continue

            target_transform = target_vehicle.get_transform()
            if target_transform.location.distance(ego_location) > max_distance:
                continue

            target_wpt = self.world_map.get_waypoint(target_transform.location, lane_type=carla.LaneType.Any)

            # General approach for junctions and vehicles invading other lanes due to the offset
            if (use_bbs or target_wpt.is_junction) and route_polygon:
                target_bb = target_vehicle.bounding_box
                target_vertices = target_bb.get_world_vertices(target_vehicle.get_transform())
                target_list = [[v.x, v.y, v.z] for v in target_vertices]
                target_polygon = Polygon(target_list)

                if route_polygon.intersects(target_polygon):
                    return (
                        True,
                        target_vehicle.id,
                        float(compute_distance(target_vehicle.get_location(), ego_location)),
                    )

            # Simplified approach, using only the plan waypoints (similar to TM)
            else:
                if target_wpt.road_id != ego_wpt.road_id or target_wpt.lane_id != ego_wpt.lane_id + lane_offset:
                    next_wpt = self._local_planner.get_incoming_waypoint_and_direction(steps=3)[0]
                    if not next_wpt:
                        continue
                    if target_wpt.road_id != next_wpt.road_id or target_wpt.lane_id != next_wpt.lane_id + lane_offset:
                        continue

                target_forward_vector = target_transform.get_forward_vector()
                target_extent = target_vehicle.bounding_box.extent.x
                target_rear_transform = target_transform
                target_rear_transform.location -= carla.Location(
                    x=target_extent * target_forward_vector.x,
                    y=target_extent * target_forward_vector.y,
                )

                if is_within_distance(
                    target_rear_transform,
                    ego_front_transform,
                    max_distance,
                    [low_angle_th, up_angle_th],
                ):
                    return (
                        True,
                        target_vehicle.id,
                        float(compute_distance(target_transform.location, ego_transform.location)),
                    )

        return False, None, -1.0

    @beartype
    def perturbate_bounding_boxes(
        self, bounding_boxes: list[dict], perturbate_translation: float, perturbate_rotation: float
    ) -> list[dict]:
        """
        Return the bounding boxes dict but in perturbated sensor space.
        Args:
            bounding_boxes: list of bounding boxes
        Return:
            perturbated_bounding_boxes: list of bounding boxes in perturbated sensor space
        """
        aug_yaw_rad = np.deg2rad(perturbate_rotation)

        rotation_matrix = np.array(
            [
                [np.cos(aug_yaw_rad), -np.sin(aug_yaw_rad)],
                [np.sin(aug_yaw_rad), np.cos(aug_yaw_rad)],
            ]
        )

        ret = []
        for bbox_dict in bounding_boxes:
            position = np.array([[bbox_dict["position"][0]], [bbox_dict["position"][1]]])
            translation = np.array([[0.0], [perturbate_translation]])

            position_aug = rotation_matrix.T @ (position - translation)

            x, y = position_aug[:2, 0]

            bbox_dict_perturbated = copy.deepcopy(bbox_dict)
            bbox_dict_perturbated["position"] = [x, y, bbox_dict["position"][2]]
            bbox_dict_perturbated["yaw"] = common_utils.normalize_angle(bbox_dict["yaw"] - aug_yaw_rad)
            ret.append(bbox_dict_perturbated)
        return ret

    @beartype
    def enhance_depth(
        self,
        depth: jt.Float[npt.NDArray, "H W"],
        semantic_segmentation: jt.UInt8[npt.NDArray, "H W"],
        instance_semantic_segmentation: jt.Int32[npt.NDArray, "H W 2"],
    ) -> jt.Float[npt.NDArray, "H W"]:
        """
        Make car windows not transparent in depth images.
        This function could be extended further for less noisy depth map but it's good for now.

        Args:
            instance: First channel is semantic id and second channel is unreal engine instance id
            depth: Metric depth
            semantic_segmentation: Semantic segmentation of the image
        Return:
            repaired_depth: Repaired depth image where car windows are not transparent
        """
        CAR_SEMANTIC_ID = 14
        instance_id = instance_semantic_segmentation[..., 1]
        semantic_id = instance_semantic_segmentation[..., 0]
        instance_ids = np.unique(instance_id[semantic_id == CAR_SEMANTIC_ID])

        depth_repaired = depth.copy()
        for inst_id in instance_ids:
            inst_mask = instance_id == inst_id
            window_mask = inst_mask & (semantic_segmentation != CAR_SEMANTIC_ID)
            depth_repaired[window_mask] = depth_repaired[inst_mask].min()

        return depth_repaired

    @beartype
    def get_bounding_boxes(self, input_data: dict) -> list[dict]:
        boxes = []

        ego_transform = self._vehicle.get_transform()
        ego_control = self._vehicle.get_control()
        ego_velocity = self._vehicle.get_velocity()
        ego_matrix = np.array(ego_transform.get_matrix())
        ego_rotation = ego_transform.rotation
        ego_extent = self._vehicle.bounding_box.extent
        ego_speed = self._get_forward_speed(transform=ego_transform, velocity=ego_velocity)
        ego_dx = np.array([ego_extent.x, ego_extent.y, ego_extent.z])
        ego_yaw = np.deg2rad(ego_rotation.yaw)
        ego_brake = ego_control.brake

        relative_yaw = 0.0
        relative_pos = common_utils.get_relative_transform(ego_matrix, ego_matrix)

        ego_wp = self.world_map.get_waypoint(
            self._vehicle.get_location(), project_to_road=True, lane_type=carla.libcarla.LaneType.Any
        )

        # how far is next junction
        next_wps = expert_utils.wps_next_until_lane_end(ego_wp)
        try:
            next_lane_wps_ego = next_wps[-1].next(1)
            if len(next_lane_wps_ego) == 0:
                next_lane_wps_ego = [next_wps[-1]]
        except:
            next_lane_wps_ego = []

        next_road_ids_ego = []
        next_next_road_ids_ego = []
        for wp in next_lane_wps_ego:
            next_road_ids_ego.append(wp.road_id)
            next_next_wps = expert_utils.wps_next_until_lane_end(wp)
            try:
                next_next_lane_wps_ego = next_next_wps[-1].next(1)
                if len(next_next_lane_wps_ego) == 0:
                    next_next_lane_wps_ego = [next_next_wps[-1]]
            except:
                next_next_lane_wps_ego = []
            for wp2 in next_next_lane_wps_ego:
                if wp2.road_id not in next_next_road_ids_ego:
                    next_next_road_ids_ego.append(wp2.road_id)

        # Check for possible vehicle obstacles
        # Retrieve all relevant actors
        self._actors = self._world.get_actors()
        vehicle_list = self._actors.filter("*vehicle*")

        try:
            next_action = self.tm.get_next_action(self._vehicle)[0]
        except:
            next_action = None

        # --- Start iterating through actors ---
        result = {
            "class": "ego_car",
            "transfuser_semantics_id": int(TransfuserSemanticSegmentationClass.UNLABELED),
            "extent": [ego_dx[0], ego_dx[1], ego_dx[2]],
            "position": [0, 0, 0],
            "yaw": 0.0,
            "num_points": -1,
            "distance": 0,
            "speed": ego_speed,
            "brake": ego_brake,
            "id": int(self._vehicle.id),
            "matrix": ego_transform.get_matrix(),
            "visible_pixels": -1,
        }
        boxes.append(result)

        transfuser_camera_semantics_pc = input_data["semantics_camera_pc"].copy()
        transfuser_camera_semantics_pc_semantics_id = np.array(list(constants.SEMANTIC_SEGMENTATION_CONVERTER.values()))[
            transfuser_camera_semantics_pc[:, CameraPointCloudIndex.UNREAL_SEMANTICS_ID].astype(np.int32)
        ]
        global_camera_pc = {
            TransfuserSemanticSegmentationClass.VEHICLE: transfuser_camera_semantics_pc[
                (transfuser_camera_semantics_pc_semantics_id == TransfuserSemanticSegmentationClass.VEHICLE)
                | (transfuser_camera_semantics_pc_semantics_id == TransfuserSemanticSegmentationClass.SPECIAL_VEHICLE)
                | (transfuser_camera_semantics_pc_semantics_id == TransfuserSemanticSegmentationClass.BIKER)
            ][:, : CameraPointCloudIndex.Z + 1],
            TransfuserSemanticSegmentationClass.PEDESTRIAN: transfuser_camera_semantics_pc[
                transfuser_camera_semantics_pc_semantics_id == TransfuserSemanticSegmentationClass.PEDESTRIAN
            ][:, : CameraPointCloudIndex.Z + 1],
            TransfuserSemanticSegmentationClass.OBSTACLE: transfuser_camera_semantics_pc[
                transfuser_camera_semantics_pc_semantics_id == TransfuserSemanticSegmentationClass.OBSTACLE
            ][:, : CameraPointCloudIndex.Z + 1],
        }

        for vehicle in vehicle_list:
            if vehicle.get_location().distance(self._vehicle.get_location()) < self.config_expert.bb_save_radius:
                if vehicle.id != self._vehicle.id:
                    vehicle_transform = vehicle.get_transform()
                    vehicle_rotation = vehicle_transform.rotation
                    vehicle_matrix = np.array(vehicle_transform.get_matrix())
                    vehicle_control = vehicle.get_control()
                    vehicle_velocity = vehicle.get_velocity()
                    vehicle_extent = vehicle.bounding_box.extent
                    vehicle_id = vehicle.id
                    vehicle_wp = self.world_map.get_waypoint(
                        vehicle.get_location(), project_to_road=True, lane_type=carla.libcarla.LaneType.Any
                    )

                    next_wps = expert_utils.wps_next_until_lane_end(vehicle_wp)
                    next_lane_wps = next_wps[-1].next(1)
                    if len(next_lane_wps) == 0:
                        next_lane_wps = [next_wps[-1]]

                    next_next_wps = []
                    for wp in next_lane_wps:
                        next_next_wps = expert_utils.wps_next_until_lane_end(wp)

                    try:
                        next_next_lane_wps = next_next_wps[-1].next(1)
                        if len(next_next_lane_wps) == 0:
                            next_next_lane_wps = [next_next_wps[-1]]
                    except:
                        next_next_lane_wps = []

                    next_road_ids = []
                    for wp in next_lane_wps:
                        if wp.road_id not in next_road_ids:
                            next_road_ids.append(wp.road_id)

                    next_next_road_ids = []
                    for wp in next_next_lane_wps:
                        if wp.road_id not in next_next_road_ids:
                            next_next_road_ids.append(wp.road_id)

                    vehicle_extent_list = [vehicle_extent.x, vehicle_extent.y, vehicle_extent.z]
                    yaw = np.deg2rad(vehicle_rotation.yaw)

                    relative_yaw = common_utils.normalize_angle(yaw - ego_yaw)
                    relative_pos = common_utils.get_relative_transform(ego_matrix, vehicle_matrix)
                    vehicle_speed = self._get_forward_speed(transform=vehicle_transform, velocity=vehicle_velocity)
                    vehicle_brake = vehicle_control.brake
                    vehicle_steer = vehicle_control.steer
                    vehicle_throttle = vehicle_control.throttle

                    distance = np.linalg.norm(relative_pos)

                    try:
                        next_action = self.tm.get_next_action(vehicle)[0]
                    except:
                        next_action = None

                    vehicle_cuts_in = False
                    if (self.scenario_name == "ParkingCutIn") and vehicle.attributes["role_name"] == "scenario":
                        if self.cutin_vehicle_starting_position is None:
                            self.cutin_vehicle_starting_position = vehicle.get_location()

                        if (
                            vehicle.get_location().distance(self.cutin_vehicle_starting_position) > 0.2
                            and vehicle.get_location().distance(self.cutin_vehicle_starting_position) < 8
                        ):  # to make sure the vehicle drives
                            vehicle_cuts_in = True

                    elif (self.scenario_name == "StaticCutIn") and vehicle.attributes["role_name"] == "scenario":
                        if vehicle_speed > 1.0 and abs(vehicle_steer) > 0.2:
                            vehicle_cuts_in = True

                    vehicle_extent_list = [
                        vehicle_extent.x,
                        vehicle_extent.y,
                        vehicle_extent.z,
                    ]
                    yaw = np.deg2rad(vehicle_rotation.yaw)

                    relative_yaw = common_utils.normalize_angle(yaw - ego_yaw)
                    relative_pos = common_utils.get_relative_transform(ego_matrix, vehicle_matrix)
                    vehicle_speed = self._get_forward_speed(transform=vehicle_transform, velocity=vehicle_velocity)
                    vehicle_brake = vehicle_control.brake
                    vehicle_steer = vehicle_control.steer
                    vehicle_throttle = vehicle_control.throttle

                    # Computes how many LiDAR hits are on a bounding box. Used to filter invisible boxes during data loading.
                    if input_data.get("lidar") is not None:
                        num_in_bbox_points = expert_utils.get_num_points_in_actor(
                            self._vehicle, vehicle, input_data["lidar"], pad=True
                        )
                    else:
                        num_in_bbox_points = -1

                    if input_data.get("radar") is not None:
                        num_in_bb_radar_points = expert_utils.get_num_points_in_actor(
                            self._vehicle, vehicle, input_data["radar"], pad=True
                        )
                    else:
                        num_in_bb_radar_points = -1

                    distance = np.linalg.norm(relative_pos)

                    result = {
                        "class": "car",
                        "ego_velocity": expert_utils.get_vehicle_velocity_in_ego_frame(self._vehicle, vehicle),
                        "next_action": next_action,
                        "vehicle_cuts_in": vehicle_cuts_in,
                        "road_id": vehicle_wp.road_id,
                        "lane_id": vehicle_wp.lane_id,
                        "lane_type_str": str(vehicle_wp.lane_type),
                        "base_type": vehicle.attributes["base_type"],
                        "transfuser_semantics_id": int(TransfuserSemanticSegmentationClass.VEHICLE),
                        "extent": vehicle_extent_list,
                        "position": [relative_pos[0], relative_pos[1], relative_pos[2]],
                        "yaw": relative_yaw,
                        "num_points": int(num_in_bbox_points),
                        "num_radar_points": int(num_in_bb_radar_points),
                        "distance": distance,
                        "speed": vehicle_speed,
                        "brake": vehicle_brake,
                        "steer": vehicle_steer,
                        "throttle": vehicle_throttle,
                        "id": int(vehicle_id),
                        "role_name": vehicle.attributes["role_name"],
                        "type_id": vehicle.type_id,
                        "matrix": vehicle_transform.get_matrix(),
                        "speed_limit": vehicle.get_speed_limit(),
                        "visible_pixels": expert_utils.get_num_points_in_actor(
                            self._vehicle, vehicle, global_camera_pc[TransfuserSemanticSegmentationClass.VEHICLE], pad=True
                        ),
                    }
                    boxes.append(result)

        walkers = self._actors.filter("*walker*")
        for walker in walkers:
            if walker.get_location().distance(self._vehicle.get_location()) < self.config_expert.bb_save_radius:
                walker_transform = walker.get_transform()
                walker_velocity = walker.get_velocity()
                walker_rotation = walker.get_transform().rotation
                walker_matrix = np.array(walker_transform.get_matrix())
                walker_id = walker.id
                walker_extent = walker.bounding_box.extent
                walker_extent = [walker_extent.x, walker_extent.y, walker_extent.z]
                yaw = np.deg2rad(walker_rotation.yaw)

                relative_yaw = common_utils.normalize_angle(yaw - ego_yaw)
                relative_pos = common_utils.get_relative_transform(ego_matrix, walker_matrix)

                walker_speed = self._get_forward_speed(transform=walker_transform, velocity=walker_velocity)

                # Computes how many LiDAR hits are on a bounding box. Used to filter invisible boxes during data loading.
                if input_data.get("lidar") is not None:
                    num_in_bbox_points = expert_utils.get_num_points_in_actor(
                        self._vehicle, walker, input_data["lidar"], pad=True
                    )
                else:
                    num_in_bbox_points = -1

                if input_data.get("radar") is not None:
                    num_in_bb_radar_points = expert_utils.get_num_points_in_actor(
                        self._vehicle, walker, input_data["radar"], pad=True
                    )
                else:
                    num_in_bb_radar_points = -1

                distance = np.linalg.norm(relative_pos)

                result = {
                    "class": "walker",
                    "ego_velocity": expert_utils.get_vehicle_velocity_in_ego_frame(self._vehicle, walker),
                    "role_name": walker.attributes["role_name"],
                    "transfuser_semantics_id": int(TransfuserSemanticSegmentationClass.PEDESTRIAN),
                    "extent": walker_extent,
                    "position": [relative_pos[0], relative_pos[1], relative_pos[2]],
                    "yaw": relative_yaw,
                    "num_points": int(num_in_bbox_points),
                    "num_radar_points": int(num_in_bb_radar_points),
                    "distance": distance,
                    "speed": walker_speed,
                    "id": int(walker_id),
                    "matrix": walker_transform.get_matrix(),
                    "visible_pixels": expert_utils.get_num_points_in_actor(
                        self._vehicle, walker, global_camera_pc[TransfuserSemanticSegmentationClass.PEDESTRIAN], pad=True
                    ),
                }
                boxes.append(result)

        # Note this only saves static actors, which does not include static background objects
        static_list = self._actors.filter("*static*")
        for static in static_list:
            if static.get_location().distance(self._vehicle.get_location()) < self.config_expert.bb_save_radius:
                static_transform = static.get_transform()
                static_velocity = static.get_velocity()
                static_rotation = static.get_transform().rotation
                static_matrix = np.array(static_transform.get_matrix())
                static_id = static.id
                type_id = static.type_id
                mesh_path = static.attributes.get("mesh_path", None)
                static_extent = static.bounding_box.extent
                static_extent = [static_extent.x, static_extent.y, static_extent.z]
                if mesh_path is not None and mesh_path in constants.LOOKUP_TABLE:
                    static_extent = constants.LOOKUP_TABLE[mesh_path]
                if type_id == "static.propr.trafficwarning":
                    static_extent[0], static_extent[1] = (
                        self.config_expert.traffic_warning_bb_size[0],
                        self.config_expert.traffic_warning_bb_size[1],
                    )
                elif type_id == "static.prop.constructioncone":
                    static_extent[0], static_extent[1] = (
                        self.config_expert.construction_cone_bb_size[0],
                        self.config_expert.construction_cone_bb_size[1],
                    )

                yaw = np.deg2rad(static_rotation.yaw)

                relative_yaw = common_utils.normalize_angle(yaw - ego_yaw)
                relative_pos = common_utils.get_relative_transform(ego_matrix, static_matrix)

                static_speed = self._get_forward_speed(transform=static_transform, velocity=static_velocity)

                # Computes how many LiDAR hits are on a bounding box. Used to filter invisible boxes during data loading.
                if input_data.get("lidar") is not None:
                    num_in_bbox_points = expert_utils.get_num_points_in_actor(
                        self._vehicle, static, input_data["lidar"], pad=True
                    )
                else:
                    num_in_bbox_points = -1

                distance = np.linalg.norm(relative_pos)

                result = {
                    "class": "static",
                    "transfuser_semantics_id": int(TransfuserSemanticSegmentationClass.UNLABELED),
                    "extent": static_extent,
                    "position": [relative_pos[0], relative_pos[1], relative_pos[2]],
                    "yaw": relative_yaw,
                    "num_points": int(num_in_bbox_points),
                    "distance": distance,
                    "speed": static_speed,
                    "id": int(static_id),
                    "matrix": static_transform.get_matrix(),
                    "type_id": type_id,
                    "mesh_path": mesh_path,
                }
                if result["mesh_path"] is not None and "Car" in result["mesh_path"]:
                    result["transfuser_semantics_id"] = int(TransfuserSemanticSegmentationClass.VEHICLE)
                    result["visible_pixels"] = expert_utils.get_num_points_in_bbox(
                        self._vehicle, result, global_camera_pc[TransfuserSemanticSegmentationClass.VEHICLE], pad=True
                    )
                else:
                    result["transfuser_semantics_id"] = int(TransfuserSemanticSegmentationClass.OBSTACLE)
                    result["visible_pixels"] = expert_utils.get_num_points_in_bbox(
                        self._vehicle, result, global_camera_pc[TransfuserSemanticSegmentationClass.OBSTACLE], pad=True
                    )

                boxes.append(result)

        # --- Traffic light ---
        # New logic needed to handle some weird traffic lights in Town12 and Town13
        for (
            traffic_light,
            original_traffic_light_bounding_box,
            traffic_light_state,
            traffic_light_id,
            traffic_light_affects_ego,
        ) in self.close_traffic_lights:
            original_waypoint = self.world_map.get_waypoint(original_traffic_light_bounding_box.location)
            waypoint_transform_matrix = np.array(original_waypoint.transform.get_matrix())
            traffic_light_transform_matrix = np.array(traffic_light.get_transform().get_matrix())

            traffic_light_in_waypoint = common_utils.get_relative_transform(
                ego_matrix=waypoint_transform_matrix,
                vehicle_matrix=traffic_light_transform_matrix,
            )

            is_over_head_traffic_light = (
                self.town in ["Town11", "Town12", "Town13", "Town15"] and abs(traffic_light_in_waypoint[0]) < 4.0
            )
            is_europe_traffic_light = (
                self.town in ["Town01", "Town02", "Town03", "Town04", "Town05", "Town06", "Town07", "Town10HD"]
                and abs(traffic_light_in_waypoint[0]) < 4.0
            )
            if traffic_light_affects_ego:
                red_light_stop_waypoints = expert_utils.get_stop_waypoints(self.ego_wp, traffic_light)

                # Create bounding boxes for each additional lane
                for i, red_light_stop_waypoint in enumerate(red_light_stop_waypoints):
                    # Create bounding box for this waypoint
                    duplicated_traffic_light_bounding_box = expert_utils.create_bounding_box_for_waypoint(
                        original_traffic_light_bounding_box, red_light_stop_waypoint
                    )

                    traffic_light_extent = [
                        duplicated_traffic_light_bounding_box.extent.x,
                        duplicated_traffic_light_bounding_box.extent.y,
                        duplicated_traffic_light_bounding_box.extent.z,
                    ]

                    # Keep original rotation/yaw, only change position
                    traffic_light_transform = carla.Transform(
                        duplicated_traffic_light_bounding_box.location, original_traffic_light_bounding_box.rotation
                    )
                    traffic_light_rotation = traffic_light_transform.rotation
                    traffic_light_matrix = np.array(traffic_light_transform.get_matrix())
                    yaw = np.deg2rad(traffic_light_rotation.yaw)

                    relative_yaw = common_utils.normalize_angle(yaw - ego_yaw)
                    relative_pos = common_utils.get_relative_transform(ego_matrix, traffic_light_matrix)

                    distance = np.linalg.norm(relative_pos)

                    # Keep the original distance_to_physical_traffic_light
                    distance_traffic_light_to_bounding_box = np.linalg.norm(
                        np.array(
                            [
                                traffic_light.get_transform().location.x - duplicated_traffic_light_bounding_box.location.x,
                                traffic_light.get_transform().location.y - duplicated_traffic_light_bounding_box.location.y,
                            ]
                        )
                    )

                    # Duplicated bounding box result
                    if i == 0:
                        result = {
                            "class": "traffic_light",
                            "transfuser_semantics_id": int(TransfuserSemanticSegmentationClass.TRAFFIC_LIGHT),
                            "extent": traffic_light_extent,
                            "position": [relative_pos[0], relative_pos[1], relative_pos[2]],
                            "yaw": relative_yaw,
                            "distance": distance,
                            "state": str(traffic_light_state),
                            "id": int(traffic_light_id),  # Keep original ID
                            "affects_ego": traffic_light_affects_ego,
                            "matrix": traffic_light_transform.get_matrix(),
                            "distance_to_physical_traffic_light": distance_traffic_light_to_bounding_box,
                            "dummy_traffic_light_bounding_box": False,
                            "same_lane_as_ego": red_light_stop_waypoint.lane_id == ego_wp.lane_id,
                            "is_over_head_traffic_light": is_over_head_traffic_light,
                            "is_europe_traffic_light": is_europe_traffic_light,
                        }
                    else:
                        result = {
                            "class": "traffic_light",
                            "transfuser_semantics_id": int(TransfuserSemanticSegmentationClass.TRAFFIC_LIGHT),
                            "extent": traffic_light_extent,
                            "position": [relative_pos[0], relative_pos[1], relative_pos[2]],
                            "yaw": relative_yaw,
                            "distance": distance,
                            "state": str(traffic_light_state),
                            "id": self.pdm_lite_id(traffic_light.id),  # Use negative ID counter for duplicates
                            "affects_ego": traffic_light_affects_ego,
                            "matrix": traffic_light_transform.get_matrix(),
                            "distance_to_physical_traffic_light": distance_traffic_light_to_bounding_box,
                            "dummy_traffic_light_bounding_box": True,
                            "same_lane_as_ego": red_light_stop_waypoint.lane_id == ego_wp.lane_id,
                            "is_over_head_traffic_light": is_over_head_traffic_light,
                            "is_europe_traffic_light": is_europe_traffic_light,
                        }
                    boxes.append(result)

        for stop_sign in self.close_stop_signs:
            stop_sign_extent = [
                stop_sign[0].extent.x,
                stop_sign[0].extent.y,
                stop_sign[0].extent.z,
            ]

            stop_sign_transform = carla.Transform(stop_sign[0].location, stop_sign[0].rotation)
            stop_sign_rotation = stop_sign_transform.rotation
            stop_sign_matrix = np.array(stop_sign_transform.get_matrix())
            yaw = np.deg2rad(stop_sign_rotation.yaw)

            relative_yaw = common_utils.normalize_angle(yaw - ego_yaw)
            relative_pos = common_utils.get_relative_transform(ego_matrix, stop_sign_matrix)

            distance = np.linalg.norm(relative_pos)

            result = {
                "class": "stop_sign",
                "transfuser_semantics_id": int(TransfuserSemanticSegmentationClass.UNLABELED),
                "extent": stop_sign_extent,
                "position": [relative_pos[0], relative_pos[1], relative_pos[2]],
                "yaw": relative_yaw,
                "distance": distance,
                "id": int(stop_sign[1]),
                "affects_ego": stop_sign[2],
                "matrix": stop_sign_transform.get_matrix(),
            }
            boxes.append(result)

        # Others static meshes that dont belong to an actors but are still relevant for perception
        static_parking_id_start = 1e8
        found = 0
        existed_bboxes_ids = set([box["id"] for box in boxes])
        for bb_class in [carla.CityObjectLabel.Car]:
            for i, vehicle_bounding_box in enumerate(self._world.get_level_bbs(bb_class)):
                extent = vehicle_bounding_box.extent
                location = vehicle_bounding_box.location
                rotation = vehicle_bounding_box.rotation
                matrix = carla.Transform(location, rotation).get_matrix()
                relative_pos = common_utils.get_relative_transform(ego_matrix, np.array(matrix))
                distance = np.linalg.norm(relative_pos)

                if distance > self.config_expert.bb_save_radius:
                    continue
                relative_yaw = common_utils.normalize_angle(np.deg2rad(rotation.yaw) - ego_yaw)
                result = {
                    "class": "static_prop_car",
                    "transfuser_semantics_id": int(TransfuserSemanticSegmentationClass.VEHICLE),
                    "extent": [extent.x, extent.y, extent.z],
                    "position": [relative_pos[0], relative_pos[1], relative_pos[2]],
                    "yaw": relative_yaw,
                    "distance": distance,
                    "id": int(static_parking_id_start + i),
                    "matrix": matrix,
                }

                # Since we iterate every bounding box in the world, we need to make sure that we dont duplicate anything
                too_close = False
                for other in list(self._actors.filter("*vehicle*")) + list(self._actors.filter("*static*")):
                    if other.id not in existed_bboxes_ids:
                        continue
                    bb = other.bounding_box  # local-space BB (relative to actor origin)
                    global_location = other.get_transform().transform(bb.location)
                    global_location = carla.Location(x=global_location.x, y=global_location.y, z=global_location.z)
                    # copy into a new BoundingBox
                    other_bb = carla.BoundingBox(global_location, bb.extent)
                    other_bb.rotation = bb.rotation
                    if expert_utils.check_obb_intersection(vehicle_bounding_box, other_bb):
                        too_close = True
                        break

                if not too_close:
                    if input_data.get("lidar") is not None:
                        result["num_points"] = expert_utils.get_num_points_in_bbox(
                            self._vehicle, result, input_data["lidar"], pad=True
                        )
                    else:
                        result["num_points"] = -1

                    result["visible_pixels"] = expert_utils.get_num_points_in_bbox(
                        self._vehicle, result, global_camera_pc[TransfuserSemanticSegmentationClass.VEHICLE], pad=True
                    )

                    boxes.append(result)
                    found += 1

        # Filter bounding boxes with duplicate id
        bounding_box_ids = set()
        filtered_bounding_boxes = []
        for box in boxes:
            if box["id"] not in bounding_box_ids:
                bounding_box_ids.add(box["id"])
                filtered_bounding_boxes.append(box)
        # Sort by distances to ego
        filtered_bounding_boxes = sorted(filtered_bounding_boxes, key=lambda x: x["distance"])

        return filtered_bounding_boxes
