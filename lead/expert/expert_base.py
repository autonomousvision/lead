import copy
import logging
import os
import pathlib
from collections import deque
from functools import cached_property

import carla
import cv2
import numpy as np
from agents.navigation.local_planner import LocalPlanner, RoadOption
from beartype import beartype
from leaderboard.autoagents import autonomous_agent
from privileged_route_planner import PrivilegedRoutePlanner
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

import lead.common.common_utils as common_utils
import lead.expert.expert_utils as pdm_lite_utils
from lead.common import constants, ransac
from lead.common.base_agent import BaseAgent
from lead.common.constants import TransfuserSemanticSegmentationClass
from lead.common.kinematic_bicycle_model import KinematicBicycleModel
from lead.common.pid_controller import ExpertLateralPIDController, ExpertLongitudinalController
from lead.common.route_planner import RoutePlanner
from lead.common.sensor_setup import av_sensor_setup
from lead.expert.expert_utils import step_cached_property
from lead.expert.hdmap.chauffeurnet import ObsManager
from lead.expert.hdmap.run_stop_sign import RunStopSign

LOG = logging.getLogger(__name__)


class ExpertBase(BaseAgent):
    def expert_setup(
        self, path_to_conf_file: str, route_index: str | None = None, traffic_manager: carla.TrafficManager | None = None
    ):
        """
        Set up the autonomous agent for the CARLA simulation.

        Args:
            path_to_conf_file: Path to the configuration file.
            route_index: Index of the route to follow.
            traffic_manager: The traffic manager object.
        """
        LOG.info("Setup")

        self.recording = False
        self.track = autonomous_agent.Track.MAP
        self.config_path = path_to_conf_file
        self.step = -1
        self.initialized = False
        self.save_path = None
        self.route_index = route_index
        # Dynamics models
        self.ego_model = KinematicBicycleModel(self.config_expert)
        self.vehicle_model = KinematicBicycleModel(self.config_expert)

        # To avoid failing the ActorBlockedTest, the agent has to move at least 0.1 m/s every 179 ticks
        self.waiting_ticks_at_stop_sign = 0
        self.ego_blocked_for_ticks = 0

        # Controllers
        self._turn_controller = ExpertLateralPIDController(self.config_expert)

        self.list_traffic_lights: list[tuple[carla.TrafficLight, carla.Location, list[carla.Waypoint]]] = []

        # Initialize controls
        self.steer = 0.0
        self.throttle = 0.0
        self.brake = 0.0

        self.perturbation_translation = 0
        self.perturbation_rotation = 0

        # Angle to the next waypoint, normalized in [-1, 1] corresponding to [-90, 90]
        self.remaining_route = None  # Remaining route
        self.close_traffic_lights: list[tuple[carla.TrafficLight, carla.BoundingBox, carla.TrafficLightState, int, bool]] = []
        self.close_stop_signs = []
        self.was_at_stop_sign = False
        self.cleared_stop_sign = False
        self.visible_walker_ids = []
        self.walker_past_pos = {}  # Position of walker in the last frame

        self._vehicle_lights = carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam

        # Get the world map and the ego vehicle
        self.world_map = CarlaDataProvider.get_map()

        # Set up the save path if specified
        if os.environ.get("SAVE_PATH", None) is not None:
            string = os.environ.get("TOWN", "999")
            string += "_Rep" + os.environ.get("REPETITION", "-1")
            string += f"_{self.route_index}"

            self.save_path = pathlib.Path(os.environ["SAVE_PATH"]) / string
            self.save_path.mkdir(parents=True, exist_ok=False)

            if self.config_expert.datagen:
                (self.save_path / "metas").mkdir()

        # Store metas to update acceleration with forward difference after finishing route
        self.metas = []
        self.transform_queue = deque(maxlen=self.config_expert.ego_num_temporal_data_points_saved + 1)
        self.pdm_lite_id = pdm_lite_utils.NegativeIdCounter()
        self.tm = traffic_manager

        self.scenario_name = pathlib.Path(path_to_conf_file).parent.name
        self.cutin_vehicle_starting_position = None

        if self.save_path is not None and self.config_expert.datagen:
            (self.save_path / "lidar").mkdir()
            (self.save_path / "rgb").mkdir()
            if self.config_expert.save_camera_pc:
                (self.save_path / "camera_pc").mkdir()
                if self.config_expert.perturbate_sensors:
                    (self.save_path / "camera_pc_perturbated").mkdir()
            if self.config_expert.perturbate_sensors:
                (self.save_path / "rgb_perturbated").mkdir()
            (self.save_path / "semantics").mkdir()
            if self.config_expert.perturbate_sensors:
                (self.save_path / "semantics_perturbated").mkdir()
            if self.config_expert.save_depth:
                (self.save_path / "depth").mkdir()
                if self.config_expert.perturbate_sensors:
                    (self.save_path / "depth_perturbated").mkdir()
            (self.save_path / "hdmap").mkdir()
            if self.config_expert.perturbate_sensors:
                (self.save_path / "hdmap_perturbated").mkdir()
            (self.save_path / "bboxes").mkdir()
            if self.config_expert.save_instance_segmentation:
                (self.save_path / "instance").mkdir()
                if self.config_expert.perturbate_sensors:
                    (self.save_path / "instance_perturbated").mkdir()
            if self.config_expert.use_radars:
                (self.save_path / "radar").mkdir()
                if self.config_expert.perturbate_sensors:
                    (self.save_path / "radar_perturbated").mkdir()

        self._active_traffic_light = None
        self.weather_setting = "ClearNoon"
        self.semantics_converter = np.uint8(list(constants.SEMANTIC_SEGMENTATION_CONVERTER.values()))

    def expert_init(self, hd_map):
        """
        Initialize the agent by setting up the route planner, longitudinal controller,
        command planner, and other necessary components.

        Args:
            hd_map (carla.Map): The map object of the CARLA world.
        """
        LOG.info("Init")
        client = CarlaDataProvider.get_client()
        if hd_map is None:
            hd_map = client.get_world().get_map()
        # Get the hero vehicle and the CARLA world
        self._vehicle: carla.Actor = CarlaDataProvider.get_hero_actor()
        self._world: carla.World = self._vehicle.get_world()

        # Check if the vehicle starts from a parking spot
        distance_to_road = self.org_dense_route_world_coord[0][0].location.distance(self._vehicle.get_location())
        # The first waypoint starts at the lane center, hence it's more than 2 m away from the center of the
        # ego vehicle at the beginning.
        starts_with_parking_exit = distance_to_road > 2
        LOG.info(f"Vehicle starts {'with' if starts_with_parking_exit else 'without'} parking exit.")

        # Set up the route planner and extrapolation
        self._waypoint_planner = PrivilegedRoutePlanner(self.config_expert)
        self._waypoint_planner.setup_route(
            self.org_dense_route_world_coord,
            self._world,
            self.world_map,
            starts_with_parking_exit,
            self._vehicle.get_location(),
        )
        self._waypoint_planner.save()
        LOG.info(f"Route setup with {len(self._waypoint_planner.route_waypoints)} waypoints.")

        # Set up the longitudinal controller and command planner
        self._longitudinal_controller = ExpertLongitudinalController(self.config_expert)
        self._command_planner = RoutePlanner(
            self.config_expert.route_planner_min_distance, self.config_expert.route_planner_max_distance
        )
        self._command_planner.set_route(self._global_plan_world_coord)

        self._command_planners_dict = {}
        for dist in self.config_expert.tp_distances:
            planner = RoutePlanner(dist, self.config_expert.route_planner_max_distance)
            planner.set_route(self._global_plan_world_coord)
            self._command_planners_dict[dist] = planner

        # Preprocess traffic lights
        all_actors = self._world.get_actors()
        for actor in all_actors:
            if "traffic_light" in actor.type_id:
                center, waypoints = pdm_lite_utils.get_traffic_light_waypoints(actor, self.world_map)
                self.list_traffic_lights.append((actor, center, waypoints))

        # Remove bugged 2-wheelers
        # https://github.com/carla-simulator/carla/issues/3670
        for actor in all_actors:
            if "vehicle" in actor.type_id:
                extent = actor.bounding_box.extent
                if extent.x < 0.001 or extent.y < 0.001 or extent.z < 0.001:
                    actor.destroy()

        # Use camera
        if (
            self.config_expert.save_3rd_person_camera
            and (not self.config_expert.is_on_slurm or self.save_path is not None)
            and not self.config_expert.eval_expert
        ):
            bp_lib = self._world.get_blueprint_library()
            camera_bp = bp_lib.find("sensor.camera.rgb")
            camera_bp.set_attribute("image_size_x", self.config_expert.camera_3rd_person_calibration["image_size_x"])
            camera_bp.set_attribute("image_size_y", self.config_expert.camera_3rd_person_calibration["image_size_y"])
            camera_bp.set_attribute("fov", self.config_expert.camera_3rd_person_calibration["fov"])
            self._3rd_person_camera = self._world.spawn_actor(camera_bp, self.transform_3rd_person_camera)

            def _save_image(image):
                frame = self.step // self.config_expert.data_save_freq

                def _save(img, path):
                    array = np.frombuffer(img.raw_data, dtype=np.uint8)
                    array = copy.deepcopy(array)
                    array = np.reshape(array, (img.height, img.width, 4))
                    bgr = array[:, :, :3]
                    cv2.imwrite(path, bgr)

                if self.config_expert.is_on_slurm or self.save_path is not None:
                    save_path_3rd_person = str(self.save_path / "3rd_person")
                    os.makedirs(save_path_3rd_person, exist_ok=True)
                    _save(image, os.path.join(save_path_3rd_person, f"{str(frame).zfill(4)}.jpg"))

                if not self.config_expert.is_on_slurm:
                    save_path_3rd_person = self.config_expert.path_3rd_person_camera_save
                    os.makedirs(save_path_3rd_person, exist_ok=True)
                    _save(image, os.path.join(save_path_3rd_person, f"{str(self.step).zfill(4)}.png"))

            self._3rd_person_camera.listen(_save_image)
        if self.config_expert.datagen:
            self.shuffle_weather()
        jpeg_storage_quality_distribution = self.config_expert.weather_jpeg_compression_quality[
            self.weather_setting
        ]  # key value: quality maps to probability
        if self.config_expert.jpeg_compression:
            self.jpeg_storage_quality = int(
                np.random.choice(
                    list(jpeg_storage_quality_distribution.keys()), p=list(jpeg_storage_quality_distribution.values())
                )
            )
        else:
            self.jpeg_storage_quality = 90
        LOG.info(f"[DataAgent] Chose JPEG storage quality {self.jpeg_storage_quality}")

        obs_config = {
            "width_in_pixels": 256,  # self.config.lidar_resolution_width,
            "pixels_ev_to_bottom": 32 * self.config_expert.pixels_per_meter,
            "pixels_per_meter": self.config_expert.pixels_per_meter_collection,
            "history_idx": [-1],
            "scale_bbox": True,
            "scale_mask_col": 1.0,
            "map_folder": "maps_2ppm_cv",
        }
        if obs_config["width_in_pixels"] != self.config_expert.lidar_width_pixel:
            LOG.warning("The BEV resolution is not the same as the LiDAR resolution. This might lead to unexpected results")

        self.stop_sign_criteria = RunStopSign(self._world)
        self.ss_bev_manager = ObsManager(obs_config, self.config_expert)
        self.ss_bev_manager.attach_ego_vehicle(self._vehicle, criteria_stop=self.stop_sign_criteria)

        if self.config_expert.perturbate_sensors:
            self.ss_bev_manager_perturbated = ObsManager(obs_config, self.config_expert)
            bb_copy = carla.BoundingBox(self._vehicle.bounding_box.location, self._vehicle.bounding_box.extent)
            transform_copy = carla.Transform(
                self._vehicle.get_transform().location,
                self._vehicle.get_transform().rotation,
            )
            # Can't clone the carla vehicle object, so I use a dummy class with similar attributes.
            self.perturbated_vehicle_dummy = pdm_lite_utils.CarlaActorDummy(
                self._vehicle.get_world(), bb_copy, transform_copy, self._vehicle.id
            )
            self.ss_bev_manager_perturbated.attach_ego_vehicle(
                self.perturbated_vehicle_dummy, criteria_stop=self.stop_sign_criteria
            )

        self._local_planner = LocalPlanner(self._vehicle, opt_dict={}, map_inst=self.world_map)
        self.bounding_boxes = []
        ransac.remove_ground(np.random.rand(1000, 3), self.config_expert, parallel=True)  # Pre-compile numba code

        self.initialized = True

    @beartype
    def is_actor_inside_bev(self, actor: carla.Actor) -> bool:
        """
        Check if actor is visible in TransFuser++'s planning visible range.
        This is used to filter out actors that are not visible to TransFuser++'s
        planning module even though they might be visible in the camera.
        """
        actor_in_ego = common_utils.get_relative_transform(self.ego_matrix, np.array(actor.get_transform().get_matrix()))
        x_ego, y_ego, _ = actor_in_ego
        return bool(
            self.config_expert.min_x_meter - 2 < x_ego < self.config_expert.max_x_meter + 2
            and self.config_expert.min_y_meter - 2 < y_ego < self.config_expert.max_y_meter + 2
            and np.linalg.norm(actor_in_ego) < self.config_expert.bb_save_radius
        )

    def update_3rd_person_camera(self):
        """
        Track ego with 3rd person camera.
        """
        if hasattr(self, "_3rd_person_camera") and self._3rd_person_camera.is_alive:
            self._3rd_person_camera.set_transform(self.transform_3rd_person_camera)

    def sensors(self):
        """
        Returns a list of sensor specifications for the ego vehicle.

        Each sensor specification is a dictionary containing the sensor type,
        reading frequency, position, and other relevant parameters.

        Returns:
            list: A list of sensor specification dictionaries.
        """
        result = []
        if not self.config_expert.datagen:
            result = [
                {"type": "sensor.opendrive_map", "reading_frequency": 1e-6, "id": "hd_map"},
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

        self.perturbation_translation, self.perturbation_rotation = pdm_lite_utils.sample_sensor_perturbation_parameters(
            config=self.config_expert,
            max_speed_limit_route=self.max_speed_limit_route,
            min_lane_width_route=self.min_lane_width_route,
        )

        # --- Set up sensor rig ---
        if self.save_path is not None and self.config_expert.datagen:
            result += av_sensor_setup(
                self.config_expert,
                perturbation_rotation=self.perturbation_rotation,
                perturbation_translation=self.perturbation_translation,
                lidar=True,
                perturbate=self.config_expert.perturbate_sensors,
                sensor_agent=False,
                radar=self.config_expert.use_radars,
            )
        else:
            result.append(
                {
                    "type": "sensor.lidar.ray_cast",
                    "x": self.config_expert.lidar_pos_1[0],
                    "y": self.config_expert.lidar_pos_1[1],
                    "z": self.config_expert.lidar_pos_1[2],
                    "roll": self.config_expert.lidar_rot_1[0],
                    "pitch": self.config_expert.lidar_rot_1[1],
                    "yaw": self.config_expert.lidar_rot_1[2],
                    "id": "lidar1",
                },
            )
            if self.config_expert.use_two_lidars:
                result.append(
                    {
                        "type": "sensor.lidar.ray_cast",
                        "x": self.config_expert.lidar_pos_2[0],
                        "y": self.config_expert.lidar_pos_2[1],
                        "z": self.config_expert.lidar_pos_2[2],
                        "roll": self.config_expert.lidar_rot_2[0],
                        "pitch": self.config_expert.lidar_rot_2[1],
                        "yaw": self.config_expert.lidar_rot_2[2],
                        "id": "lidar2",
                    },
                )
        return result

    @cached_property
    def min_lane_width_route(self):
        self._vehicle = CarlaDataProvider.get_hero_actor()
        self._world = self._vehicle.get_world()

        global_plan = self.org_dense_route_world_coord
        carla_map = self._world.get_map()
        route_waypoints = [transform.location for transform, _ in global_plan]
        route_waypoints = [carla_map.get_waypoint(loc) for loc in route_waypoints]
        widths = []
        for waypoint in route_waypoints:
            if waypoint is not None and not waypoint.is_junction:
                widths.append(waypoint.lane_width)
        return max(min(widths), 2.75)

    @cached_property
    def max_speed_limit_route(self):
        self._vehicle = CarlaDataProvider.get_hero_actor()
        self._world = self._vehicle.get_world()
        # Check if the vehicle starts from a parking spot
        distance_to_road = self.org_dense_route_world_coord[0][0].location.distance(self._vehicle.get_location())
        # The first waypoint starts at the lane center, hence it's more than 2 m away from the center of the
        # ego vehicle at the beginning.
        starts_with_parking_exit = distance_to_road > 2
        waypoint_planner = PrivilegedRoutePlanner(self.config_expert)
        waypoint_planner.setup_route(
            self.org_dense_route_world_coord,
            self._world,
            self.world_map,
            starts_with_parking_exit,
            self._vehicle.get_location(),
        )
        way_point_planner = waypoint_planner
        waypoints = way_point_planner.route_waypoints
        if len(waypoints) == 0:
            return 30 / 3.6
        speed_limits = []
        for wp in waypoints:
            if wp is not None:
                # Get speed limit landmarks within a reasonable distance ahead
                landmarks = wp.get_landmarks(200.0, True)
                for landmark in landmarks:
                    # Check if the landmark is a MaximumSpeed sign
                    if landmark.type == carla.LandmarkType.MaximumSpeed:
                        # Extract speed limit value from landmark
                        try:
                            speed_limit = float(landmark.value)
                            if speed_limit > 0:
                                speed_limits.append(speed_limit)
                        except (ValueError, AttributeError):
                            pass

        # Return the maximum speed limit found, or default to 30 km/h
        if len(speed_limits) > 0:
            return max(speed_limits) / 3.6  # Convert km/h to m/s
        return 30 / 3.6

    @property
    def town(self):
        return self._world.get_map().name.split("/")[-1]

    @step_cached_property
    def privileged_ego_past_positions(self):
        ego_matrix_current = self.transform_queue[-1].get_matrix()
        T_world_to_current_ego = np.linalg.inv(ego_matrix_current)
        past_positions = []
        for transform in self.transform_queue:
            T_past = np.array(transform.get_matrix())
            pos_world = np.append(T_past[:3, 3], 1.0)
            pos_current_ego = T_world_to_current_ego @ pos_world
            past_positions.append(pos_current_ego[:2].tolist())
        return past_positions

    @step_cached_property
    def average_traffic_speed(self):
        """
        Average speed of traffic in the BEV.
        """
        if not self.initialized:
            raise RuntimeError("Agent is not initialized. Call setup() first.")
        speeds = []
        for actor in self.vehicles_inside_bev:
            if actor.id == self._vehicle.id:
                continue
            speed = self._get_actor_forward_speed(actor)
            speeds.append(speed)
        if len(speeds) > 0:
            return np.mean(speeds)
        return 0.0

    @step_cached_property
    def max_traffic_speed(self):
        """
        Maximum speed of traffic in the BEV.
        """
        if not self.initialized:
            raise RuntimeError("Agent is not initialized. Call setup() first.")
        speeds = []
        for actor in self.vehicles_inside_bev:
            if actor.id == self._vehicle.id:
                continue
            speed = self._get_actor_forward_speed(actor)
            speeds.append(speed)
        if len(speeds) > 0:
            return np.max(speeds)
        return 0.0

    @step_cached_property
    def max_adversarial_speed(self):
        """
        Maximum speed of adversarial actors in the BEV.
        """
        if not self.initialized:
            raise RuntimeError("Agent is not initialized. Call setup() first.")
        speeds = []
        dangerous, safe, ignored = self.adversarial_actors_ids
        for vehicle in self.vehicles_inside_bev:
            if vehicle.id == self._vehicle.id:
                continue
            if vehicle.id in dangerous or vehicle.id in safe or vehicle.id in ignored:
                speed = self._get_actor_forward_speed(vehicle)
                speeds.append(speed)
        if len(speeds) > 0:
            return np.max(speeds)
        return 0.0

    @step_cached_property
    def distance_to_intersection_index_ego(self):
        """
        Returns the index of the intersection point in the route waypoints.
        If no intersection point is found, returns None.
        """
        if self.current_active_scenario_type in [
            "SignalizedJunctionLeftTurn",
            "NonSignalizedJunctionLeftTurn",
            "SignalizedJunctionRightTurn",
            "NonSignalizedJunctionRightTurn",
            "SignalizedJunctionLeftTurnEnterFlow",
            "NonSignalizedJunctionLeftTurnEnterFlow",
            "InterurbanActorFlow",
        ]:
            intersection_index_ego = CarlaDataProvider.memory[self.current_active_scenario_type].get(
                "intersection_index_ego", None
            )
            if intersection_index_ego is not None:
                return (intersection_index_ego - self._waypoint_planner.route_index) / self.config_expert.points_per_meter
        return float("inf")

    @step_cached_property
    def last_encountered_speed_limit_sign(self):
        ret = self._vehicle.get_speed_limit()
        if ret is not None:
            ret /= 3.6
        return ret

    @step_cached_property
    def speed_limit(self):
        if self.last_encountered_speed_limit_sign is not None:
            return self.last_encountered_speed_limit_sign
        return 30.0 / 3.6

    @step_cached_property
    def adversarial_actors_ids(self) -> None:
        """
        Return a list of:
            - dangerous adversarial actors IDs: we should be very waried of them
            - safe adversarial actors IDs: we can treat their bounding boxes a bit smaller
            - ignored adversarial actors IDs: we can ignore them completely
        """
        # Obstacle scenarios: compute source and target lane once
        if self.current_active_scenario_type in ["Accident", "ConstructionObstacle", "ParkedObstacle"]:
            obstacle, direction = [
                CarlaDataProvider.memory[self.current_active_scenario_type][key] for key in ["first_actor", "direction"]
            ]
            source_lane = self.world_map.get_waypoint(
                obstacle.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving
            )
            target_lane = source_lane.get_right_lane() if direction == "left" else source_lane.get_left_lane()
            if source_lane and target_lane:
                CarlaDataProvider.memory[self.current_active_scenario_type]["source_lane"] = source_lane
                CarlaDataProvider.memory[self.current_active_scenario_type]["target_lane"] = target_lane

        if self.current_active_scenario_type in ["HazardAtSideLane"]:
            if CarlaDataProvider.memory[self.current_active_scenario_type]["bicycle_1"] is not None:
                target_lane = CarlaDataProvider.memory[self.current_active_scenario_type]["target_lane"]
                source_lane = CarlaDataProvider.memory[self.current_active_scenario_type]["source_lane"]
                if target_lane is None or source_lane is None:
                    bicycle_1 = CarlaDataProvider.memory[self.current_active_scenario_type]["bicycle_1"]
                    source_lane = self.world_map.get_waypoint(
                        bicycle_1.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving
                    )
                    target_lane = source_lane.get_left_lane()
                    CarlaDataProvider.memory[self.current_active_scenario_type]["target_lane"] = target_lane
                    CarlaDataProvider.memory[self.current_active_scenario_type]["soure_lane"] = source_lane

        # One way obstacle scenarios: adversarial actors are those on the target lane
        if (
            min([self.distance_to_accident_site, self.distance_to_construction_site, self.distance_to_parked_obstacle]) <= 40
            or self.current_active_scenario_type == "HazardAtSideLane"
        ):
            for scenario in ["Accident", "ConstructionObstacle", "ParkedObstacle", "HazardAtSideLane"]:
                dangerous_adversarial_actors_ids = []
                safe_adversarial_actors_ids = []
                ignored_adversarial_actors_ids = []
                if self.current_active_scenario_type != scenario:
                    continue
                if (
                    CarlaDataProvider.memory[scenario]["source_lane"] is not None
                    and CarlaDataProvider.memory[scenario]["target_lane"] is not None
                ):
                    target_lane = CarlaDataProvider.memory[scenario]["target_lane"]
                    for actor in self.vehicles_inside_bev:
                        if actor.id == self._vehicle.id:
                            continue
                        actor_lane = self.world_map.get_waypoint(
                            actor.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving
                        )
                        if actor_lane and actor_lane.lane_id == target_lane.lane_id:
                            rel_loc = common_utils.get_relative_transform(
                                self.ego_matrix,
                                np.array(actor.get_transform().get_matrix()),
                            )
                            if self.speed_limit > 25:
                                dangerous_adversarial_actors_ids.append(actor.id)
                            else:
                                if 0.5 <= rel_loc[0]:  # actor in front, is safe
                                    safe_adversarial_actors_ids.append(actor.id)
                                else:  # Normal speed, be more careful
                                    dangerous_adversarial_actors_ids.append(actor.id)
                CarlaDataProvider.memory[scenario]["dangerous_adversarial_actors_ids"] = dangerous_adversarial_actors_ids
                CarlaDataProvider.memory[scenario]["safe_adversarial_actors_ids"] = safe_adversarial_actors_ids
                CarlaDataProvider.memory[scenario]["ignored_adversarial_actors_ids"] = ignored_adversarial_actors_ids

        # High speed merging scenarios
        for scenario in ["EnterActorFlow", "EnterActorFlowV2", "InterurbanAdvancedActorFlow"]:
            if self.current_active_scenario_type != scenario:
                continue
            safe_adversarial_actors_ids = []
            ignored_adversarial_actors_ids = []
            dangerous_adversarial_actors_ids = []
            for adversarial_actor in CarlaDataProvider.memory[scenario]["adversarial_actors"]:
                try:
                    if not self.is_actor_inside_bev(adversarial_actor):
                        continue
                    adversarial_lane = self.world_map.get_waypoint(
                        adversarial_actor.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving
                    )
                    if self.ego_lane_id != adversarial_lane.lane_id:
                        continue
                    dangerous_adversarial_actors_ids.append(adversarial_actor.id)
                except:
                    pass

            CarlaDataProvider.memory[scenario]["dangerous_adversarial_actors_ids"] = dangerous_adversarial_actors_ids
            CarlaDataProvider.memory[scenario]["safe_adversarial_actors_ids"] = safe_adversarial_actors_ids
            CarlaDataProvider.memory[scenario]["ignored_adversarial_actors_ids"] = ignored_adversarial_actors_ids

        # Priority scenarios
        for scenario in ["OppositeVehicleRunningRedLight", "OppositeVehicleTakingPriority"]:
            if self.current_active_scenario_type != scenario:
                continue
            safe_adversarial_actors_ids = []
            ignored_adversarial_actors_ids = []
            dangerous_adversarial_actors_ids = []
            for adversarial_actor in CarlaDataProvider.memory[scenario]["adversarial_actors"]:
                try:
                    if (
                        not self.is_actor_inside_bev(adversarial_actor)
                        or adversarial_actor.get_velocity().length() < 0.1
                        or (
                            self.data_agent_id_to_bb_map[adversarial_actor.id]["visible_pixels"] < 10
                            and self.data_agent_id_to_bb_map[adversarial_actor.id]["num_points"] < 10
                        )
                    ):
                        continue
                    dangerous_adversarial_actors_ids.append(adversarial_actor.id)
                except:
                    pass

            CarlaDataProvider.memory[scenario]["dangerous_adversarial_actors_ids"] = dangerous_adversarial_actors_ids
            CarlaDataProvider.memory[scenario]["safe_adversarial_actors_ids"] = safe_adversarial_actors_ids
            CarlaDataProvider.memory[scenario]["ignored_adversarial_actors_ids"] = ignored_adversarial_actors_ids

        # Unprotected left and right turns scenarios
        for scenario in [
            "SignalizedJunctionLeftTurn",
            "NonSignalizedJunctionLeftTurn",
            "SignalizedJunctionRightTurn",
            "NonSignalizedJunctionRightTurn",
            "SignalizedJunctionLeftTurnEnterFlow",
            "NonSignalizedJunctionLeftTurnEnterFlow",
            "InterurbanActorFlow",
        ]:
            if self.current_active_scenario_type != scenario:
                continue

            # Computer intersection point of ego route and adversarial route
            source_wp: carla.Waypoint = CarlaDataProvider.memory[scenario]["source_wp"]
            sink_wp: carla.Waypoint = CarlaDataProvider.memory[scenario]["sink_wp"]
            opponent_traffic_route = CarlaDataProvider.memory[scenario]["opponent_traffic_route"]
            if opponent_traffic_route is None:
                opponent_traffic_route = pdm_lite_utils.compute_global_route(
                    world=self._world,
                    source_location=source_wp.transform.location,
                    sink_location=sink_wp.transform.location,
                )
                CarlaDataProvider.memory[scenario]["opponent_traffic_route"] = opponent_traffic_route

            intersection_point = CarlaDataProvider.memory[scenario]["intersection_point"]
            if opponent_traffic_route is not None and intersection_point is None:
                intersection_point, intersection_index_ego = pdm_lite_utils.intersection_of_routes(
                    points_a=self.route_waypoints_np[
                        : self.config_expert.draw_future_route_till_distance
                    ],  # Don't use full route otherwise too expensive
                    points_b=opponent_traffic_route,
                )
                if intersection_index_ego is not None:
                    intersection_index_ego += self._waypoint_planner.route_index
                CarlaDataProvider.memory[scenario]["intersection_index_ego"] = intersection_index_ego
                CarlaDataProvider.memory[scenario]["intersection_point"] = intersection_point

            # Filter adversarial actors for unprotected left turns
            intersection_point = CarlaDataProvider.memory[scenario]["intersection_point"]
            if intersection_point is not None:
                safe_adversarial_actors_ids = CarlaDataProvider.memory[scenario][
                    "safe_adversarial_actors_ids"
                ]  # We keep track safe adversarial actors over time, once they are safe, they won't be dangerous anymore
                ignored_adversarial_actors_ids = []
                dangerous_adversarial_actors_ids = []
                for adversarial_actor in CarlaDataProvider.memory[scenario]["adversarial_actors"]:
                    if adversarial_actor.id == self._vehicle.id:
                        continue
                    if adversarial_actor.id in safe_adversarial_actors_ids:
                        continue
                    try:
                        if not self.is_actor_inside_bev(adversarial_actor):
                            continue
                        if (
                            self.distance_to_next_junction < 10
                            and (
                                (
                                    self.distance_to_intersection_index_ego < 13
                                    and scenario
                                    in ["SignalizedJunctionLeftTurnEnterFlow", "NonSignalizedJunctionLeftTurnEnterFlow"]
                                )
                                or (
                                    self.distance_to_intersection_index_ego < 13
                                    and scenario in ["NonSignalizedJunctionLeftTurn", "InterurbanActorFlow"]
                                )
                                or (
                                    self.distance_to_intersection_index_ego < 18
                                    and scenario in ["SignalizedJunctionRightTurn", "NonSignalizedJunctionRightTurn"]
                                )
                                or (self.distance_to_intersection_index_ego < 23 and scenario in ["SignalizedJunctionLeftTurn"])
                            )
                            and not self.stop_sign_hazard
                            and not self.traffic_light_hazard
                        ):  # If only we are not near enough to the junction, we ignore adversarial actors. Smoother stopping
                            if scenario in [
                                "SignalizedJunctionLeftTurn",
                                "NonSignalizedJunctionLeftTurn",
                                "InterurbanActorFlow",
                            ]:
                                safe_threshold = (
                                    self.distance_to_intersection_index_ego * 1.1
                                )  # Safe threshold, the lower, the earlier we ignore an adversarial actor
                                if scenario in [
                                    "SignalizedJunctionLeftTurn"
                                ]:  # Urban scenarios, we need to treat them a bit differently
                                    if self.distance_to_intersection_index_ego < 13:
                                        safe_threshold = (
                                            self.distance_to_intersection_index_ego * 1.2
                                        )  # Urban, we need more time to reach intersection. Only empirical observations.
                                    else:
                                        safe_threshold = (
                                            self.distance_to_intersection_index_ego * 1.6
                                        )  # Urban, we need more time to reach intersection. Only empirical observations.
                                safe_threshold = min(safe_threshold, 22)  # Don't go too far, otherwise we ignore all actors
                                if (
                                    adversarial_actor.get_location().distance(intersection_point) < safe_threshold
                                ):  # If actor is/was near enough to the intersection point, we can safely ignore it
                                    safe_adversarial_actors_ids.append(adversarial_actor.id)
                                else:
                                    dangerous_adversarial_actors_ids.append(adversarial_actor.id)
                            elif scenario in ["SignalizedJunctionRightTurn", "NonSignalizedJunctionRightTurn"]:
                                safe_threshold = self.distance_to_intersection_index_ego * 1.1
                                if scenario in [
                                    "SignalizedJunctionRightTurn"
                                ]:  # Urban scenarios, we need to treat them a bit differently
                                    if self.distance_to_intersection_index_ego < 13:
                                        safe_threshold = (
                                            self.distance_to_intersection_index_ego * 1.2
                                        )  # Urban, we need more time to reach intersection. Only empirical observations.
                                    else:
                                        safe_threshold = (
                                            self.distance_to_intersection_index_ego * 1.6
                                        )  # Urban, we need more time to reach intersection. Only empirical observations.
                                safe_threshold = min(safe_threshold, 22)  # Don't go too far, otherwise we ignore all actors
                                if self.distance_to_intersection_index_ego < 5:
                                    safe_adversarial_actors_ids.append(
                                        adversarial_actor.id
                                    )  # If we are very close to the intersection, we really want to commit
                                elif (
                                    adversarial_actor.get_location().distance(intersection_point) < safe_threshold
                                ):  # If actor is/was near enough to the intersection point, we can safely ignore it
                                    safe_adversarial_actors_ids.append(adversarial_actor.id)
                                else:
                                    dangerous_adversarial_actors_ids.append(adversarial_actor.id)
                            elif scenario in ["SignalizedJunctionLeftTurnEnterFlow", "NonSignalizedJunctionLeftTurnEnterFlow"]:
                                adversarial_actor_location = adversarial_actor.get_location()
                                if (
                                    pdm_lite_utils.distance_location_to_route(
                                        route=CarlaDataProvider.memory[scenario]["opponent_traffic_route"],
                                        location=np.array(
                                            [
                                                adversarial_actor_location.x,
                                                adversarial_actor_location.y,
                                                adversarial_actor_location.z,
                                            ]
                                        ),
                                    )
                                    > 1.0
                                ):
                                    # If actor is further than the intersection point in the route, we can safely ignore it
                                    safe_adversarial_actors_ids.append(adversarial_actor.id)
                                    LOG.info("Adversarial actor went out of route. ignore")
                                else:
                                    dangerous_adversarial_actors_ids.append(adversarial_actor.id)
                        else:
                            ignored_adversarial_actors_ids = [
                                actor.id
                                for actor in CarlaDataProvider.memory[self.current_active_scenario_type]["adversarial_actors"]
                            ]

                    except RuntimeError as e:
                        if "trying to operate on a destroyed actor" in str(e):
                            LOG.info(f"Error processing adversarial actor {adversarial_actor.id} in scenario {scenario}.")
                            ignored_adversarial_actors_ids.append(adversarial_actor.id)
                            continue
                        else:
                            raise e

                CarlaDataProvider.memory[scenario]["dangerous_adversarial_actors_ids"] = dangerous_adversarial_actors_ids
                CarlaDataProvider.memory[scenario]["safe_adversarial_actors_ids"] = safe_adversarial_actors_ids
                CarlaDataProvider.memory[scenario]["ignored_adversarial_actors_ids"] = ignored_adversarial_actors_ids

        if (
            self.current_active_scenario_type in CarlaDataProvider.memory
            and "dangerous_adversarial_actors_ids" in CarlaDataProvider.memory[self.current_active_scenario_type]
        ):
            return (
                CarlaDataProvider.memory[self.current_active_scenario_type]["dangerous_adversarial_actors_ids"],
                CarlaDataProvider.memory[self.current_active_scenario_type]["safe_adversarial_actors_ids"],
                CarlaDataProvider.memory[self.current_active_scenario_type]["ignored_adversarial_actors_ids"],
            )
        return [], [], []

    @step_cached_property
    def rear_adversarial_actor(self):
        rear_adversarial_vehicle = None
        if (
            self.current_active_scenario_type
            in [
                "SignalizedJunctionRightTurn",
                "NonSignalizedJunctionRightTurn",
                "SignalizedJunctionLeftTurnEnterFlow",
                "NonSignalizedJunctionLeftTurnEnterFlow",
            ]
            and self.distance_to_intersection_index_ego < 2
        ):
            min_distance = float("inf")
            rear_adversarial_vehicle = None
            dangerous_adversarial_actors_ids, safe_adversarial_actors_ids, ignored_adversarial_actors_ids = (
                self.adversarial_actors_ids
            )
            for vehicle in self.vehicles_inside_bev:
                if vehicle.id == self._vehicle.id:
                    continue
                if vehicle.id not in dangerous_adversarial_actors_ids and vehicle.id not in safe_adversarial_actors_ids:
                    continue
                rel_loc = common_utils.get_relative_transform(
                    self.ego_matrix,
                    np.array(vehicle.get_transform().get_matrix()),
                )
                if rel_loc[0] < -1.0:  # Vehicle is behind the ego vehicle
                    distance = np.linalg.norm(rel_loc[:2])
                    if distance < min_distance:
                        min_distance = distance
                        rear_adversarial_vehicle = vehicle
        elif self.current_active_scenario_type in ["EnterActorFlow", "EnterActorFlowV2", "InterurbanAdvancedActorFlow"]:
            min_distance = float("inf")
            rear_adversarial_vehicle = None
            dangerous_adversarial_actors_ids, safe_adversarial_actors_ids, ignored_adversarial_actors_ids = (
                self.adversarial_actors_ids
            )
            for vehicle in self.vehicles_inside_bev:
                if vehicle.id == self._vehicle.id:
                    continue
                if vehicle.id not in dangerous_adversarial_actors_ids:
                    continue
                rel_loc = common_utils.get_relative_transform(
                    self.ego_matrix,
                    np.array(vehicle.get_transform().get_matrix()),
                )
                if rel_loc[0] < -1.0:  # Vehicle is behind the ego vehicle
                    distance = np.linalg.norm(rel_loc[:2])
                    if distance < min_distance:
                        min_distance = distance
                        rear_adversarial_vehicle = vehicle
        elif self.current_active_scenario_type in ["OppositeVehicleRunningRedLight", "OppositeVehicleTakingPriority"]:
            min_distance = float("inf")
            rear_adversarial_vehicle = None
            dangerous_adversarial_actors_ids, safe_adversarial_actors_ids, ignored_adversarial_actors_ids = (
                self.adversarial_actors_ids
            )
            for vehicle in self.vehicles_inside_bev:
                if vehicle.id == self._vehicle.id:
                    continue
                if vehicle.id not in dangerous_adversarial_actors_ids:
                    continue
                rel_loc = common_utils.get_relative_transform(
                    self.ego_matrix,
                    np.array(vehicle.get_transform().get_matrix()),
                )
                if rel_loc[0] < -1.0:  # Vehicle is behind the ego vehicle
                    distance = np.linalg.norm(rel_loc[:2])
                    if distance < min_distance:
                        min_distance = distance
                        rear_adversarial_vehicle = vehicle
        return rear_adversarial_vehicle

    @step_cached_property
    def target_lane_width(self):
        if self.current_active_scenario_type in ["Accident", "ConstructionObstacle", "ParkedObstacle", "HazardAtSideLane"]:
            target_lane = CarlaDataProvider.memory[self.current_active_scenario_type]["target_lane"]
            if target_lane is not None:
                return target_lane.lane_width

        if self.current_active_scenario_type in [
            "SignalizedJunctionLeftTurn",
            "NonSignalizedJunctionLeftTurn",
            "SignalizedJunctionRightTurn",
            "NonSignalizedJunctionRightTurn",
            "SignalizedJunctionLeftTurnEnterFlow",
            "NonSignalizedJunctionLeftTurnEnterFlow",
            "InterurbanActorFlow",
        ]:
            sink_wp = CarlaDataProvider.memory[self.current_active_scenario_type]["sink_wp"]
            if sink_wp is not None:
                return sink_wp.lane_width

        return None

    @step_cached_property
    def ego_lane_width(self):
        """
        Returns the width of the lane the ego vehicle is currently on.
        """
        if not self.initialized:
            raise RuntimeError("Agent is not initialized. Call setup() first.")
        return self.ego_lane.lane_width

    @step_cached_property
    def transform_3rd_person_camera(self):
        ego_camera_location = carla.Location(
            x=self.config_expert.camera_3rd_person_calibration["x"],
            y=self.config_expert.camera_3rd_person_calibration["y"],
            z=self.config_expert.camera_3rd_person_calibration["z"],
        )
        world_camera_location = common_utils.get_world_coordinate_2d(self._vehicle.get_transform(), ego_camera_location)
        return carla.Transform(
            world_camera_location,
            carla.Rotation(
                pitch=self.config_expert.camera_3rd_person_calibration["pitch"],
                yaw=self._vehicle.get_transform().rotation.yaw + self.config_expert.camera_3rd_person_calibration["yaw"],
            ),
        )

    @step_cached_property
    def ego_lane(self):
        """
        Returns the current lane of the ego vehicle as a CARLA waypoint.
        """
        if not self.initialized:
            raise RuntimeError("Agent is not initialized. Call setup() first.")
        return self.world_map.get_waypoint(self._vehicle.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving)

    @step_cached_property
    def ego_lane_id(self):
        """
        Returns the current lane ID of the ego vehicle.
        """
        return self.ego_lane.lane_id

    @step_cached_property
    def ego_location(self):
        return self._vehicle.get_location()

    @property
    def ego_location_array(self):
        """
        Returns the ego vehicle's location as a numpy array.
        """
        location = self.ego_location
        return np.array([location.x, location.y, location.z])

    @step_cached_property
    def ego_speed(self):
        return self._vehicle.get_velocity().length()

    @step_cached_property
    def ego_yaw_degree(self):
        return self._vehicle.get_transform().rotation.yaw

    @step_cached_property
    def ego_orientation_rad(self):
        return self.compass

    @property
    def route_waypoints(self):
        return self._waypoint_planner.route_waypoints[self._waypoint_planner.route_index :]

    @property
    def route_waypoints_np(self):
        return self._waypoint_planner.route_points[self._waypoint_planner.route_index :]

    @property
    def original_route_waypoints_np(self):
        return self._waypoint_planner.original_route_points[self._waypoint_planner.route_index :]

    @step_cached_property
    def signed_dist_to_lane_change(self) -> float:
        """
        Compute the signed distance to the next or previous lane change command in the route.

        Returns:
            float: Signed distance to the next lane change command. Positive if ahead, negative if behind.
            Inf if no lane change command found in proximity.
        """
        route_points = self._waypoint_planner.route_points
        current_index = self._waypoint_planner.route_index
        from_index = max(0, current_index - 250)
        to_index = min(len(route_points) - 1, current_index + 250)
        # Iterate over the points around the current position, checking for lane change commands

        def dist(index_a, index_b):
            index_min = min(index_a, index_b)
            index_max = max(index_a, index_b)
            d = 0
            for i in range(index_min, index_max):
                p1 = route_points[i]
                p2 = route_points[i + 1]
                d += np.linalg.norm(p2 - p1)
            if index_a < index_b:
                return d
            return -d

        min_dist = np.inf
        for i in range(from_index, to_index, 1):
            if self._waypoint_planner.commands[i] in (RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT):
                considered_dist = dist(current_index, i)
                if abs(considered_dist) < abs(min_dist):
                    min_dist = considered_dist

        return min_dist / self.config_expert.points_per_meter

    @property
    def ego_wp(self):
        return self.route_waypoints[0]

    @step_cached_property
    def ego_matrix(self):
        return np.array(self._vehicle.get_transform().get_matrix())

    @step_cached_property
    def inv_ego_matrix(self):
        return np.linalg.inv(self.ego_matrix)

    @property
    def current_active_scenario_type(self):
        if len(CarlaDataProvider.active_scenarios) == 0:
            return None
        return CarlaDataProvider.active_scenarios[0][0]

    @property
    def previous_active_scenario_type(self):
        return CarlaDataProvider.previous_active_scenario

    @step_cached_property
    def distance_to_construction_site(self):
        if self.current_active_scenario_type in [
            "ConstructionObstacle",
            "ConstructionObstacleTwoWays",
        ] or self.previous_active_scenario_type in ["ConstructionObstacle", "ConstructionObstacleTwoWays"]:
            num_cones = 0
            num_warning_traffic_signs = 0
            distances = []
            for static in self.static_inside_bev:
                if static.type_id == "static.prop.constructioncone":
                    num_cones += 1
                    distances.append(static.get_location().distance(self.ego_location))
                elif static.type_id == "static.prop.trafficwarning":
                    num_warning_traffic_signs += 1
                    distances.append(static.get_location().distance(self.ego_location))
            if num_cones > 0 and num_warning_traffic_signs > 0:
                distances = np.array(distances)
                distance = distances.mean()
                return distance
        return float("inf")

    @step_cached_property
    def distance_to_scenario_obstacle(self):
        return min(
            [
                self.distance_to_accident_site,
                self.distance_to_construction_site,
                self.distance_to_parked_obstacle,
                self.distance_to_vehicle_opens_door,
            ]
        )

    @step_cached_property
    def distance_to_accident_site(self):
        if self.current_active_scenario_type in ["Accident", "AccidentTwoWays"] or self.previous_active_scenario_type in [
            "Accident",
            "AccidentTwoWays",
        ]:
            distances = []
            num_scenario_cars = 0
            for actor in self.scenario_obstacles:
                if "scenario" in actor.attributes["role_name"] and self._get_actor_forward_speed(actor) == 0.0:
                    num_scenario_cars += 1
                    distances.append(actor.get_location().distance(self.ego_location))
            if num_scenario_cars > 0:
                return np.mean(distances)
        return float("inf")

    @step_cached_property
    def distance_to_parked_obstacle(self):
        if self.current_active_scenario_type in [
            "ParkedObstacle",
            "ParkedObstacleTwoWays",
        ] or self.previous_active_scenario_type in ["ParkedObstacle", "ParkedObstacleTwoWays"]:
            distances = []
            num_scenario_cars = 0
            for actor in self.scenario_obstacles:
                if "scenario" in actor.attributes["role_name"] and self._get_actor_forward_speed(actor) == 0.0:
                    num_scenario_cars += 1
                    distances.append(actor.get_location().distance(self.ego_location))
            if num_scenario_cars > 0:
                return np.mean(distances)
        return float("inf")

    @step_cached_property
    def distance_to_vehicle_opens_door(self):
        if self.current_active_scenario_type in ["VehicleOpensDoorTwoWays"] or self.previous_active_scenario_type in [
            "VehicleOpensDoorTwoWays"
        ]:
            distances = []
            num_scenario_cars = 0
            for actor in self.scenario_obstacles:
                if "scenario" in actor.attributes["role_name"] and self._get_actor_forward_speed(actor) == 0.0:
                    num_scenario_cars += 1
                    distances.append(actor.get_location().distance(self.ego_location))
            if num_scenario_cars > 0:
                return np.mean(distances)
        return float("inf")

    @step_cached_property
    def distance_to_cutin_vehicle(self):
        if not self.config_expert.datagen:
            return float("inf")
        if self.current_active_scenario_type in ["ParkingCutIn", "StaticCutIn", "HighwayCutIn"]:
            distances = []
            num_scenario_cars = 0
            for actor in self.cutin_actors:
                if self.is_actor_inside_bev(actor):
                    num_scenario_cars += 1
                    distances.append(actor.get_location().distance(self.ego_location))
            if num_scenario_cars > 0:
                return np.mean(distances)
        return float("inf")

    @step_cached_property
    def distance_to_pedestrian(self):
        """
        Calculate the distance to the closest pedestrian within the BEV (Bird's Eye View) range.

        Returns:
            float: The distance to the closest pedestrian, or infinity if no pedestrians are inside the BEV.
        """
        pedestrians = self.walkers_inside_bev
        if not pedestrians:
            return float("inf")

        # Get the location of the ego vehicle
        ego_location = self._vehicle.get_location()

        # Find the closest pedestrian
        closest_pedestrian = min(pedestrians, key=lambda p: ego_location.distance(p.get_location()))
        return ego_location.distance(closest_pedestrian.get_location())

    @step_cached_property
    def distance_to_biker(self):
        """
        Calculate the distance to the closest biker within the BEV (Bird's Eye View) range.

        Returns:
            float: The distance to the closest biker, or infinity if no bikers are inside the BEV.
        """
        bikers = self.bikers_inside_bev
        if not bikers:
            return float("inf")

        # Get the location of the ego vehicle
        ego_location = self._vehicle.get_location()

        # Find the closest biker
        closest_biker = min(bikers, key=lambda b: ego_location.distance(b.get_location()))
        return ego_location.distance(closest_biker.get_location())

    @step_cached_property
    def distance_to_road_discontinuity(self):
        route_points = self.route_waypoints_np
        cumulative_dist = 0.0
        for i in range(min(len(route_points) - 1, self.config_expert.discontinuous_road_max_future_check) - 1):
            loc_current = route_points[i]
            loc_next = route_points[i + 1]
            dist = ((loc_current[0] - loc_next[0]) ** 2 + (loc_current[1] - loc_next[1]) ** 2) ** 0.5
            cumulative_dist += dist
            if dist > self.config_expert.max_distance_between_future_route_points:
                return cumulative_dist / self.config_expert.points_per_meter
        return np.inf

    @step_cached_property
    def route_left_length(self):
        route_points = self.route_waypoints_np
        dist_diff = np.diff(route_points[:, :2], axis=0)
        segment_lengths = np.linalg.norm(dist_diff, axis=1)
        return np.sum(segment_lengths)

    @step_cached_property
    def distance_ego_to_route(self):
        ego_wp = self.ego_wp
        route_wp = self.route_waypoints[0]
        return ego_wp.transform.location.distance(route_wp.transform.location)

    @step_cached_property
    def route_curvature(self):
        route_points = self.route_waypoints_np
        total_curvature = 0.0
        for i in range(1, min(len(route_points) - 1, self.config_expert.high_road_curvature_max_future_points) - 1):
            loc_prev = route_points[i - 1]
            loc_current = route_points[i]
            loc_next = route_points[i + 1]
            curv = (
                (loc_current[0] - loc_prev[0]) * (loc_next[1] - loc_current[1])
                - (loc_current[1] - loc_prev[1]) * (loc_next[0] - loc_current[0])
            ) / (
                ((loc_current[0] - loc_prev[0]) ** 2 + (loc_current[1] - loc_prev[1]) ** 2) ** 0.5
                * ((loc_next[0] - loc_current[0]) ** 2 + (loc_next[1] - loc_current[1]) ** 2) ** 0.5
            )
            total_curvature += abs(curv)
        return total_curvature

    @step_cached_property
    def vehicles_inside_bev(self):
        vehicles = self._world.get_actors().filter("*vehicle*")
        vehicles = [vehicle for vehicle in vehicles if self.is_actor_inside_bev(vehicle)]
        if (
            self.config_expert.datagen and self.config_expert.vehicle_occlusion_check
        ):  # Can only perform occlusion check if we have sensor data
            vehicles = [
                vehicle
                for vehicle in vehicles
                if not (
                    0
                    <= self.data_agent_id_to_bb_map[vehicle.id]["num_points"]
                    < self.config_expert.vehicle_occlusion_check_min_num_points
                    and 0
                    <= self.data_agent_id_to_bb_map[vehicle.id]["visible_pixels"]
                    < self.config_expert.vehicle_min_num_visible_pixels
                )
            ]
        return vehicles

    @step_cached_property
    def walkers_inside_bev(self):
        walkers = self._world.get_actors().filter("*walker*")
        walkers = [walker for walker in walkers if self.is_actor_inside_bev(walker)]
        if self.config_expert.datagen:  # Can only perform occlusion check if we have sensor data
            walkers = [
                walker
                for walker in walkers
                if not (
                    0
                    <= self.data_agent_id_to_bb_map[walker.id]["visible_pixels"]
                    < self.config_expert.pedestrian_min_num_visible_pixels
                )
            ]
        return walkers

    @step_cached_property
    def bikers_inside_bev(self):
        bikers = self._world.get_actors().filter("*vehicle*")
        bikers = [
            b
            for b in bikers
            if b.type_id in ["vehicle.diamondback.century", "vehicle.gazelle.omafiets", "vehicle.bh.crossbike"]
        ]
        bikers = [biker for biker in bikers if self.is_actor_inside_bev(biker)]
        if (
            self.config_expert.datagen and self.config_expert.bikers_occlusion_check
        ):  # Can only perform occlusion check if we have sensor data
            bikers = [
                biker
                for biker in bikers
                if not (
                    0
                    <= self.data_agent_id_to_bb_map[biker.id]["visible_pixels"]
                    < self.config_expert.bikers_occlusion_check_min_visible_pixels
                )
            ]
        return bikers

    @step_cached_property
    def static_inside_bev(self):
        """
        Get static actors inside the BEV (Bird's Eye View) range.
        This includes traffic lights and other static objects that are not vehicles or walkers.

        Returns:
            list: A list of static actors inside the BEV.
        """
        static_actors = self._world.get_actors().filter("*static*")
        static_actors = [actor for actor in static_actors if self.is_actor_inside_bev(actor)]
        return static_actors

    @step_cached_property
    def distance_to_next_junction(self):
        ego_wp = self.world_map.get_waypoint(
            self._vehicle.get_location(), project_to_road=True, lane_type=carla.libcarla.LaneType.Any
        )
        next_wps = pdm_lite_utils.wps_next_until_lane_end(ego_wp)
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
            distance_to_junction_ego = np.inf

        return distance_to_junction_ego

    @step_cached_property
    def scenario_actors(self):
        ret = []
        for actor in self.vehicles_inside_bev + self.walkers_inside_bev + self.bikers_inside_bev:
            if "scenario" in actor.attributes["role_name"]:
                ret.append(actor)
        return ret

    @step_cached_property
    def scenario_actors_ids(self):
        """
        Get the IDs of the scenario actors that are currently inside the BEV (Bird's Eye View) range.

        Returns:
            list: A list of IDs of the scenario actors.
        """
        return [actor.id for actor in self.scenario_actors]

    @step_cached_property
    def scenario_obstacles(self):
        ret = []
        scenarios = [
            "Accident",
            "ConstructionObstacle",
            "ParkedObstacle",
            "AccidentTwoWays",
            "ConstructionObstacleTwoWays",
            "ParkedObstacleTwoWays",
            "VehicleOpensDoorTwoWays",
            "InvadingTurn",
            "BlockedIntersection",
        ]
        if self.current_active_scenario_type in scenarios:
            ret = CarlaDataProvider.memory[self.current_active_scenario_type]["obstacles"]
        elif self.previous_active_scenario_type in scenarios:
            obstacles = CarlaDataProvider.previous_memory[self.previous_active_scenario_type]["obstacles"]
            try:
                obstacles = [actor for actor in obstacles if self.is_actor_inside_bev(actor)]
                ret = obstacles
            except RuntimeError as e:
                if "trying to operate on a destroyed actor" in str(e):
                    # If the scenario obstacles were destroyed, return an empty list
                    ret = []
                else:
                    raise e
        ret = [actor for actor in ret if self.is_actor_inside_bev(actor)]
        return ret

    @step_cached_property
    def scenario_obstacles_convex_hull(self):
        """
        Get the convex hull of the scenario obstacles' bounding box corners that are currently inside the BEV range.

        Returns:
            list: A list of (x, y) points representing the convex hull of the obstacles.
        """
        if not self.scenario_obstacles:
            return []

        points = []
        for actor in self.scenario_obstacles:
            bbox = actor.bounding_box
            actor_transform = actor.get_transform()

            # Get bounding box center in world coordinates
            bbox_center_world = actor_transform.transform(bbox.location)

            # Rotation matrix from actor yaw
            yaw = np.radians(actor_transform.rotation.yaw)
            cos_yaw = np.cos(yaw)
            sin_yaw = np.sin(yaw)

            extent = bbox.extent

            # Define corners in local coordinates
            local_corners = [
                (extent.x, extent.y),
                (-extent.x, extent.y),
                (-extent.x, -extent.y),
                (extent.x, -extent.y),
            ]

            # Transform corners to world coordinates
            for lx, ly in local_corners:
                x = bbox_center_world.x + lx * cos_yaw - ly * sin_yaw
                y = bbox_center_world.y + lx * sin_yaw + ly * cos_yaw
                points.append((x, y))

        if len(points) < 3:
            return points

        points_np = np.array(points, dtype=np.float32)
        hull = cv2.convexHull(points_np)
        return hull.squeeze().tolist()

    @step_cached_property
    def scenario_obstacles_ids(self):
        """
        Get the IDs of the scenario obstacles that are currently inside the BEV (Bird's Eye View) range.

        Returns:
            list: A list of IDs of the scenario obstacles.
        """
        return [actor.id for actor in self.scenario_obstacles]

    @step_cached_property
    def vehicle_opened_door(self):
        """
        Check if the vehicle opened its door in the current scenario.
        This is used to determine if the agent should react to a vehicle opening its door.
        """
        if self.current_active_scenario_type == "VehicleOpensDoorTwoWays":
            return CarlaDataProvider.memory["VehicleOpensDoorTwoWays"]["vehicle_opened_door"]
        elif self.previous_active_scenario_type == "VehicleOpensDoorTwoWays":
            try:
                CarlaDataProvider.previous_memory["VehicleOpensDoorTwoWays"]["obstacles"][0].get_location()
                return CarlaDataProvider.previous_memory["VehicleOpensDoorTwoWays"]["vehicle_opened_door"]
            except RuntimeError as e:
                if "trying to operate on a destroyed actor" in str(e):
                    return False
                else:
                    raise e
        return False

    @step_cached_property
    def vehicle_door_side(self):
        """
        Get the side of the vehicle that opened its door in the current scenario.
        This is used to determine if the agent should react to a vehicle opening its door.
        """
        if self.current_active_scenario_type == "VehicleOpensDoorTwoWays":
            return CarlaDataProvider.memory["VehicleOpensDoorTwoWays"]["vehicle_door_side"]
        elif self.previous_active_scenario_type == "VehicleOpensDoorTwoWays":
            return CarlaDataProvider.previous_memory["VehicleOpensDoorTwoWays"]["vehicle_door_side"]
        return None

    @step_cached_property
    def cutin_actors(self):
        if self.current_active_scenario_type in ["ParkingCutIn", "StaticCutIn", "HighwayCutIn"]:
            return [CarlaDataProvider.memory[self.current_active_scenario_type]["cut_in_vehicle"]]
        return []

    @step_cached_property
    def cut_in_actors_ids(self):
        return [actor.id for actor in self.cutin_actors]

    @step_cached_property
    def two_way_obstacle_distance_to_cones_factor(self):
        if self.ego_lane_width <= 2.76:
            return 1.13
        elif self.ego_lane_width <= 3.01:
            return 1.12
        return 1.12

    @step_cached_property
    def two_way_vehicle_open_door_distance_to_center_line_factor(self):
        if self.ego_lane_width <= 2.76:
            return 1.0
        elif self.ego_lane_width <= 3.01:
            return 0.875
        return 0.75

    @step_cached_property
    def add_after_construction_obstacle_two_ways(self):
        if self.ego_lane_width <= 2.76:
            return self.config_expert.add_after_construction_obstacle_two_ways + 0.5
        return self.config_expert.add_after_construction_obstacle_two_ways

    @step_cached_property
    def add_before_construction_obstacle_two_ways(self):
        if self.ego_lane_width <= 2.76:
            return self.config_expert.add_before_construction_obstacle_two_ways + 0.5
        return self.config_expert.add_before_construction_obstacle_two_ways

    @step_cached_property
    def two_way_overtake_speed(self):
        return {
            "AccidentTwoWays": self.config_expert.default_overtake_speed,
            "ConstructionObstacleTwoWays": self.config_expert.default_overtake_speed,
            "ParkedObstacleTwoWays": self.config_expert.default_overtake_speed,
            "VehicleOpensDoorTwoWays": self.config_expert.default_overtake_speed
            if self.ego_lane_width > 3.01
            else self.config_expert.overtake_speed_vehicle_opens_door_two_ways,
        }[self.current_active_scenario_type]

    @step_cached_property
    def add_after_accident_two_ways(self):
        if self.ego_lane_width <= 2.76:
            return self.config_expert.add_after_accident_two_ways + 0.5
        return self.config_expert.add_after_accident_two_ways

    @step_cached_property
    def add_before_accident_two_ways(self):
        if self.ego_lane_width <= 2.76:
            return self.config_expert.add_before_accident_two_ways + 0.5
        return self.config_expert.add_before_accident_two_ways

    @step_cached_property
    def num_parking_vehicles_in_proximity(self):
        count = 0
        for bb in self.stored_bounding_boxes_of_this_step:
            if not (-8 <= bb["position"][0] <= 32 and abs(bb["position"][1]) <= 10):
                continue
            if bb["class"] == "static" and bb["transfuser_semantics_id"] == TransfuserSemanticSegmentationClass.VEHICLE:
                count += 1
            if bb["class"] == "static_prop_car":
                count += 1
        return count

    @step_cached_property
    def second_highest_speed(self):
        speeds = []
        for vehicle in self.vehicles_inside_bev:
            if vehicle.id == self._vehicle.id:
                continue
            speeds.append(vehicle.get_velocity().length())
        if len(speeds) == 0:
            return 0.0
        elif len(speeds) == 1:
            return speeds[0]
        speeds = sorted(speeds, reverse=True)
        return speeds[1]

    @step_cached_property
    def second_highest_speed_limit(self):
        speed_limits = []
        for vehicle in self.vehicles_inside_bev:
            if vehicle.id == self._vehicle.id:
                continue
            speed_limits.append(vehicle.get_speed_limit() / 3.6)
        if len(speed_limits) == 0:
            return 0.0
        elif len(speed_limits) == 1:
            return speed_limits[0]
        speed_limits = sorted(speed_limits, reverse=True)
        return speed_limits[1]
