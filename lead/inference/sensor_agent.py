import copy
import json
import logging
import os
import shutil
from collections import deque
from copy import deepcopy

import carla
import cv2
import jaxtyping as jt
import numpy as np
import numpy.typing as npt
import PIL
import torch
from beartype import beartype
from leaderboard.autoagents import autonomous_agent
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from lead.common import common_utils
from lead.common.base_agent import BaseAgent
from lead.common.logging_config import setup_logging
from lead.common.route_planner import RoutePlanner
from lead.common.sensor_setup import av_sensor_setup
from lead.common.visualizer import Visualizer
from lead.expert import expert_utils
from lead.inference.closed_loop_inference import (
    ClosedLoopInference,
    ClosedLoopPrediction,
)
from lead.inference.config_closed_loop import ClosedLoopConfig
from lead.training.config_training import TrainingConfig
from lead.training.data_loader import carla_dataset_utils
from lead.training.data_loader.carla_dataset_utils import rasterize_lidar

setup_logging()
LOG = logging.getLogger(__name__)

# Configure pytorch for maximum performance
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.allow_tf32 = True

DEMO_CAMERAS = [
    {
        "name": "cinematic_camera",
        "draw_points": False,
        "image_size_x": "960",
        "image_size_y": "1080",
        "fov": "100",
        "x": -6.5,
        "y": -0.0,
        "z": 6.0,
        "pitch": -30.0,
        "yaw": 0.0,
    },
    {
        "name": "bev_camera",
        "draw_points": True,
        "image_size_x": "960",
        "image_size_y": "1080",
        "fov": "100",
        "x": 0.0,
        "y": 0.0,
        "z": 22.0,
        "pitch": -90.0,
        "yaw": 0.0,
    },
]

assert len(DEMO_CAMERAS) == 2, "Expected exactly two demo cameras."


def get_entry_point():
    return "SensorAgent"


class SensorAgent(BaseAgent, autonomous_agent.AutonomousAgent):
    @beartype
    def setup(self, path_to_conf_file: str, _=None, __=None):
        # Set up test time training default parameters
        self.config_closed_loop = ClosedLoopConfig()
        super().setup(sensor_agent=True)
        self.config_path = path_to_conf_file
        self.step = -1
        self.initialized = False
        self.device = torch.device("cuda:0")

        # Load the config saved during training
        if self.config_closed_loop.is_bench2drive:
            path_to_conf_file = path_to_conf_file.split("+")[0]
        with open(os.path.join(path_to_conf_file, "config.json"), encoding="utf-8") as f:
            json_config = f.read()
            json_config = json.loads(json_config)

        # Generate new config for the case that it has new variables.
        self.training_config = TrainingConfig(json_config)

        # Load model files
        self.closed_loop_inference = ClosedLoopInference(
            config_training=self.training_config,
            config_closed_loop=self.config_closed_loop,
            config_expert=self.config_expert,
            model_path=path_to_conf_file,
            device=self.device,
            prefix="model",
        )

        # Post-processing heuristics
        self.bb_buffer = deque(maxlen=1)
        self.force_move_post_processor = ForceMovePostProcessor(
            config=self.training_config, config_test_time=self.config_closed_loop, lidar_queue=self.lidar_pc_queue
        )
        self.metric_info = {}

        self.track = autonomous_agent.Track.SENSORS

        if not shutil.which("ffmpeg"):
            raise RuntimeError("ffmpeg is not installed or not found in PATH. Please install ffmpeg to use video compression.")

    def _init(self):
        # Get the hero vehicle and the CARLA world
        self._vehicle: carla.Actor = CarlaDataProvider.get_hero_actor()
        self._world: carla.World = self._vehicle.get_world()

        # Set up cameras
        if self.config_closed_loop.save_path is not None:
            if self.config_closed_loop.produce_debug_video:
                self.debug_video_writer = None

            if self.config_closed_loop.produce_demo_video or self.config_closed_loop.produce_demo_image:
                self.demo_video_writer = None
                self._demo_cameras = []
                self._demo_camera_images = {}  # Store latest images from demo cameras
                bp_lib = self._world.get_blueprint_library()
                for idx, camera_config in enumerate(DEMO_CAMERAS, start=1):
                    camera_bp = bp_lib.find("sensor.camera.rgb")
                    camera_bp.set_attribute("image_size_x", camera_config["image_size_x"])
                    camera_bp.set_attribute("image_size_y", camera_config["image_size_y"])
                    camera_bp.set_attribute("fov", camera_config["fov"])
                    camera_bp.set_attribute("motion_blur_intensity", "0.0")

                    # Create transform for this demo camera
                    demo_camera_location = carla.Location(
                        x=camera_config["x"],
                        y=camera_config["y"],
                        z=camera_config["z"],
                    )
                    world_camera_location = common_utils.get_world_coordinate_2d(
                        self._vehicle.get_transform(), demo_camera_location
                    )
                    demo_camera_transform = carla.Transform(
                        world_camera_location,
                        carla.Rotation(
                            pitch=camera_config["pitch"],
                            yaw=self._vehicle.get_transform().rotation.yaw + camera_config["yaw"],
                        ),
                    )

                    demo_camera = self._world.spawn_actor(camera_bp, demo_camera_transform)

                    # Create callback to store image in buffer
                    def _make_image_callback(camera_idx):
                        def _store_image(image):
                            array = np.frombuffer(image.raw_data, dtype=np.uint8)
                            array = copy.deepcopy(array)
                            array = np.reshape(array, (image.height, image.width, 4))
                            bgr = array[:, :, :3]
                            self._demo_camera_images[camera_idx] = bgr

                        return _store_image

                    demo_camera.listen(_make_image_callback(idx))
                    self._demo_cameras.append(
                        {
                            "camera": demo_camera,
                            "config": camera_config,
                            "index": idx,
                        }
                    )

        self.set_weather()
        self.initialized = True

    def set_weather(self):
        weather_name = None

        if self.config_closed_loop.random_weather:
            weathers = self.config_expert.weather_settings.keys()
            weather_name = np.random.choice(list(weathers))

        if self.config_closed_loop.custom_weather is not None:
            weather_name = self.config_closed_loop.custom_weather

        if weather_name is not None:
            weather = carla.WeatherParameters(**self.config_expert.weather_settings[weather_name])
            self._world.set_weather(weather)
            LOG.info(f"Set weather to: {weather_name}")
            # night mode
            vehicles = self._world.get_actors().filter("*vehicle*")
            if expert_utils.get_night_mode(weather):
                for vehicle in vehicles:
                    vehicle.set_light_state(
                        carla.VehicleLightState(carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam)
                    )
            else:
                for vehicle in vehicles:
                    vehicle.set_light_state(carla.VehicleLightState.NONE)

    @beartype
    def sensors(self) -> list[dict]:
        return av_sensor_setup(
            config=self.training_config,
            lidar=True,
            radar=True,
            sensor_agent=True,
            perturbate=False,
            perturbation_rotation=0.0,
            perturbation_translation=0.0,
        )

    @beartype
    def move_demo_cameras_with_ego(self) -> None:
        """Update demo camera transforms to follow ego vehicle position and orientation."""
        if self.config_closed_loop.save_path is None or not (
            self.config_closed_loop.produce_demo_video or self.config_closed_loop.produce_demo_image
        ):
            return

        for demo_cam_info in self._demo_cameras:
            if demo_cam_info["camera"].is_alive:
                camera_config = demo_cam_info["config"]
                demo_camera_location = carla.Location(
                    x=camera_config["x"],
                    y=camera_config["y"],
                    z=camera_config["z"],
                )
                world_camera_location = common_utils.get_world_coordinate_2d(
                    self._vehicle.get_transform(), demo_camera_location
                )
                demo_camera_transform = carla.Transform(
                    world_camera_location,
                    carla.Rotation(
                        pitch=camera_config["pitch"],
                        yaw=self._vehicle.get_transform().rotation.yaw + camera_config["yaw"],
                    ),
                )
                demo_cam_info["camera"].set_transform(demo_camera_transform)

    @beartype
    def save_demo_cameras(
        self,
        pred_waypoints: jt.Float[torch.Tensor, "n_waypoints 2"] | None = None,
        target_points: dict[str, jt.Float[npt.NDArray, " 2"] | None] | None = None,
    ) -> None:
        """Save concatenated demo cameras (cinematic + BEV) as single JPG/video.

        Args:
            pred_waypoints: Waypoints in vehicle coords, shape (n_waypoints, 2) with (x, y) in meters.
            target_points: Route targets {'previous': (x,y), 'current': (x,y), 'next': (x,y)}.
        """
        if self.config_closed_loop.save_path is None or not (
            self.config_closed_loop.produce_demo_video or self.config_closed_loop.produce_demo_image
        ):
            return

        processed_images = []
        for camera_idx in sorted(self._demo_camera_images.keys()):
            image = self._demo_camera_images[camera_idx]
            camera_config = DEMO_CAMERAS[camera_idx - 1]  # camera_idx is 1-based
            camera_name = camera_config.get("name", f"demo_{camera_idx}")
            draw_points = camera_config.get("draw_points", False)

            processed_image = image.copy()

            # Add visualizations if enabled
            if draw_points and pred_waypoints is not None:
                processed_image = self.draw_waypoints(processed_image, pred_waypoints, camera_config)
            if draw_points and target_points is not None and camera_name == "bev_camera":
                processed_image = self.draw_target_points(processed_image, target_points, camera_config)

            processed_images.append(processed_image)

        # Concatenate horizontally: [cinematic | BEV]
        concatenated = np.hstack(processed_images)

        # Save as PNG for demo (higher quality for presentation)
        if self.config_closed_loop.produce_demo_image:
            save_path_demo = str(self.config_closed_loop.save_path / "demo_images")
            os.makedirs(save_path_demo, exist_ok=True)
            PIL.Image.fromarray(cv2.cvtColor(concatenated, cv2.COLOR_BGR2RGB)).save(
                f"{save_path_demo}/{str(self.step).zfill(5)}.png",
                optimize=False,
                compress_level=0,  # Really space expensive, do this local only.
            )

        # Add to demo video if enabled
        if self.config_closed_loop.produce_demo_video:
            if self.demo_video_writer is None:
                os.makedirs(os.path.dirname(self.config_closed_loop.demo_video_path), exist_ok=True)
                self.demo_video_writer = cv2.VideoWriter(
                    self.config_closed_loop.demo_video_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    self.config_closed_loop.video_fps,
                    (concatenated.shape[1], concatenated.shape[0]),
                )
            self.demo_video_writer.write(concatenated)

    @beartype
    def draw_waypoints(
        self,
        image: jt.UInt8[npt.NDArray, "height width 3"],
        pred_waypoints: jt.Float[torch.Tensor, "n_waypoints 2"],
        camera_config: dict[str, str | float | bool],
    ) -> jt.UInt8[npt.NDArray, "height width 3"]:
        """Project and draw waypoints from vehicle to image coordinates.

        Args:
            image: BGR image, shape (height, width, 3).
            pred_waypoints: Waypoints in vehicle coords, shape (n_waypoints, 2) with (x, y) in meters.
            camera_config: Camera params {'x','y','z','pitch','yaw','fov'}.

        Returns:
            Image copy with yellow waypoints and connecting lines.
        """
        img_with_viz = image.copy()
        camera_height = image.shape[0]
        camera_width = image.shape[1]

        # Extract camera parameters from config
        camera_fov = float(camera_config["fov"])
        camera_pos = [camera_config["x"], camera_config["y"], camera_config["z"]]
        camera_rot = [0.0, camera_config["pitch"], camera_config["yaw"]]  # roll, pitch, yaw

        # Draw route in blue
        if pred_waypoints is not None and len(pred_waypoints) > 0:
            route_points = pred_waypoints.detach().cpu().float().numpy()
            projected_route, points_inside_image = common_utils.project_points_to_image(
                camera_rot, camera_pos, camera_fov, camera_width, camera_height, route_points
            )

            # Draw circles for waypoints
            for pt, inside in zip(projected_route, points_inside_image, strict=True):
                if inside:
                    cv2.circle(
                        img_with_viz,
                        (int(pt[0]), int(pt[1])),
                        radius=3,
                        color=(255, 255, 0),
                        thickness=-1,  # Red in BGR
                        lineType=cv2.LINE_AA,
                    )
            # # Draw connected line for route
            for i in range(len(projected_route) - 1):
                pt1, inside1 = projected_route[i], points_inside_image[i]
                pt2, inside2 = projected_route[i + 1], points_inside_image[i + 1]
                if inside1 and inside2:
                    cv2.line(
                        img_with_viz,
                        (int(pt1[0]), int(pt1[1])),
                        (int(pt2[0]), int(pt2[1])),
                        (255, 255, 0),  # Blue in BGR
                        thickness=2,
                        lineType=cv2.LINE_AA,
                    )

        return img_with_viz

    @beartype
    def draw_target_points(
        self,
        image: jt.UInt8[npt.NDArray, "height width 3"],
        target_points: dict[str, jt.Float[npt.NDArray, " 2"] | None],
        camera_config: dict[str, str | float | bool],
    ) -> jt.UInt8[npt.NDArray, "height width 3"]:
        """Project and draw route target points (previous/current/next) as red circles.

        Args:
            image: BGR image, shape (height, width, 3).
            target_points: Route targets {'previous': (x,y), 'current': (x,y), 'next': (x,y)} in vehicle coords (meters).
            camera_config: Camera params {'x','y','z','pitch','yaw','fov'}.

        Returns:
            Image copy with red target point circles.
        """
        img_with_targets = image.copy()
        camera_height = image.shape[0]
        camera_width = image.shape[1]

        # Extract camera parameters from config
        camera_fov = float(camera_config["fov"])
        camera_pos = [camera_config["x"], camera_config["y"], camera_config["z"]]
        camera_rot = [0.0, camera_config["pitch"], camera_config["yaw"]]

        # Define colors and sizes for each target point (BGR format)
        targets_config = [
            ("previous", (0, 0, 255), 3),  # Gray, smaller square
            ("current", (0, 0, 255), 3),  # Green, bigger square
            ("next", (0, 0, 255), 3),  # Cyan, smaller square
        ]

        for key, color, size in targets_config:
            if key in target_points and target_points[key] is not None:
                # Get target point in vehicle coordinates
                target_point = np.array([[target_points[key][0], target_points[key][1]]])

                # Project to image
                projected, points_inside_image = common_utils.project_points_to_image(
                    camera_rot, camera_pos, camera_fov, camera_width, camera_height, target_point
                )

                if len(projected) > 0:
                    pt, inside = projected[0], points_inside_image[0]
                    if inside:
                        # Draw square (rectangle with equal width and height)
                        x, y = int(pt[0]), int(pt[1])
                        cv2.circle(
                            img_with_targets,
                            (x, y),
                            size + 1,
                            (255, 255, 255),
                            thickness=-1,
                            lineType=cv2.LINE_AA,
                        )

                        # Filled colored circle
                        cv2.circle(
                            img_with_targets,
                            (x, y),
                            size,
                            color,
                            thickness=-1,
                            lineType=cv2.LINE_AA,
                        )

        return img_with_targets

    @beartype
    def set_target_points(self, input_data: dict, pop_distance: float):
        """Defines local planning signals based on the input data.

        Args:
            input_data: The input data containing sensor information and state. Will be fed into model.
        """
        planner: RoutePlanner = self.gps_waypoint_planners_dict[pop_distance]

        @beartype
        def transform(point: list[float]) -> jt.Float[npt.NDArray, " 2"]:
            return common_utils.inverse_conversion_2d(np.array(point), np.array(self.filtered_state[:2]), self.compass)

        previous_target_points = [tp.tolist() for tp in planner.previous_target_points]
        next_target_points = [tp[0].tolist() for tp in planner.route]
        next_commands = [int(planner.route[i][1]) for i in range(len(planner.route))]

        # Merge duplicate consecutive target points
        filtered_tp_list = []
        filtered_command_list = []
        for pt, cmd in zip(next_target_points, next_commands, strict=False):
            if len(next_target_points) == 2 or not filtered_tp_list or not np.allclose(pt[:2], filtered_tp_list[-1][:2]):
                filtered_tp_list.append(pt)
                filtered_command_list.append(cmd)
        next_target_points = filtered_tp_list
        next_commands = filtered_command_list

        if len(next_target_points) > 2:
            input_data["target_point_next"] = transform(next_target_points[2][:2])
            input_data["target_point"] = transform(next_target_points[1][:2])
            input_data["target_point_previous"] = transform(next_target_points[0][:2])
        else:
            assert len(next_target_points) == 2
            input_data["target_point"] = transform(next_target_points[1][:2])
            input_data["target_point_next"] = transform(next_target_points[1][:2])
            if len(previous_target_points) > 0:
                input_data["target_point_previous"] = transform(previous_target_points[0][:2])
            else:
                input_data["target_point_previous"] = transform(next_target_points[0][:2])

        input_data["command"] = carla_dataset_utils.command_to_one_hot(next_commands[0])
        input_data["next_command"] = carla_dataset_utils.command_to_one_hot(next_commands[1])

    @beartype
    @torch.inference_mode()
    def tick(self, input_data: dict) -> dict:
        """Pre-processes sensor data"""
        input_data = super().tick(input_data)

        # Store RGB for later visualization (before JPEG compression)
        self._rgb_for_visualization = input_data["rgb"].copy()

        # Simulate JPEG compression to avoid train-test mismatch
        rgb = input_data["rgb"]
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        _, rgb = cv2.imencode(".jpg", rgb, [int(cv2.IMWRITE_JPEG_QUALITY), self.config_closed_loop.jpeg_quality])
        rgb = cv2.imdecode(rgb, cv2.IMREAD_UNCHANGED)
        rgb = np.transpose(rgb, (2, 0, 1))
        input_data["rgb"] = rgb

        # Cut cameras down to only used cameras
        if self.training_config.num_used_cameras != self.training_config.num_available_cameras:
            n = self.training_config.num_available_cameras
            w = input_data["rgb"].shape[2] // n

            rgb_slices = []
            for i, use in enumerate(self.training_config.used_cameras):
                if use:
                    s, e = i * w, (i + 1) * w
                    rgb_slices.append(input_data["rgb"][:, :, s:e])

            input_data["rgb"] = np.concatenate(rgb_slices, axis=2)

        # Plan next target point and command.
        self.set_target_points(input_data, pop_distance=self.config_closed_loop.route_planner_min_distance)
        if self.config_closed_loop.sensor_agent_pop_distance_adaptive:
            dense_points = (
                np.linalg.norm(input_data["target_point"] - input_data["target_point_next"]) < 10.0
                and min(np.linalg.norm(input_data["target_point_previous"]), np.linalg.norm(input_data["target_point"])) < 10.0
            )
            dense_points = dense_points or (
                np.linalg.norm(input_data["target_point_previous"] - input_data["target_point"]) < 10.0
                and min(np.linalg.norm(input_data["target_point_previous"]), np.linalg.norm(input_data["target_point"])) < 10.0
            )
            if dense_points:
                self.set_target_points(input_data, pop_distance=4.0)

        # Ignore the next target point if it's too far away
        if (
            self.config_closed_loop.sensor_agent_skip_distant_target_point
            and np.linalg.norm(input_data["target_point_next"])
            > self.config_closed_loop.sensor_agent_skip_distant_target_point_threshold
        ):
            # Skip the next target point if it's too far away
            input_data["target_point_next"] = input_data["target_point"]

        # Lidar input
        lidar = self.accumulate_lidar()
        # Use only part of the lidar history we trained on
        lidar = lidar[lidar[:, -1] < self.training_config.training_used_lidar_steps]

        # At inference time, simulate laspy quantization to avoid train-test mismatch
        lidar[:, 0] = np.round(lidar[:, 0] / self.config_expert.point_precision_x) * self.config_expert.point_precision_x
        lidar[:, 1] = np.round(lidar[:, 1] / self.config_expert.point_precision_y) * self.config_expert.point_precision_y
        lidar[:, 2] = np.round(lidar[:, 2] / self.config_expert.point_precision_z) * self.config_expert.point_precision_z

        # Convert to pseudo image
        input_data["rasterized_lidar"] = rasterize_lidar(config=self.training_config, lidar=lidar[:, :3])[..., None]

        # Simulate training time compression to avoid train-test mismatch
        input_data["rasterized_lidar"] = carla_dataset_utils.compress_float_image(
            input_data["rasterized_lidar"], self.training_config
        )
        input_data["rasterized_lidar"] = carla_dataset_utils.decompress_float_image(input_data["rasterized_lidar"]).squeeze()[
            None, None
        ]

        # Radar input preprocessing
        if self.training_config.use_radars:
            # Preprocess radar input using the same function as during training
            input_data["radar"] = np.concatenate(
                carla_dataset_utils.preprocess_radar_input(self.training_config, input_data), axis=0
            )

        return input_data

    @beartype
    @torch.inference_mode()
    def run_step(self, input_data: dict, _, __=None) -> carla.VehicleControl:
        self.step += 1

        if not self.initialized:
            self._init()
            self.control = carla.VehicleControl(steer=0.0, throttle=0.0, brake=1.0)
            input_data = self.tick(input_data)
            return self.control

        # Update demo cameras
        if self.config_closed_loop.save_path is not None and (
            self.config_closed_loop.produce_demo_video or self.config_closed_loop.produce_demo_image
        ):
            self.move_demo_cameras_with_ego()

        # Need to run this every step for GPS filtering
        input_data = self.tick(input_data)

        # Transform the data into torch tensor comforting with data loader's format.
        input_data_tensors = {
            "rgb": torch.Tensor(input_data["rgb"]).to(self.device, dtype=torch.float32)[None],
            "rasterized_lidar": torch.Tensor(input_data["rasterized_lidar"]).to(self.device, dtype=torch.float32),
            "target_point_previous": torch.Tensor(input_data["target_point_previous"])
            .to(self.device, dtype=torch.float32)
            .view(1, 2),
            "target_point": torch.Tensor(input_data["target_point"]).to(self.device, dtype=torch.float32).view(1, 2),
            "target_point_next": (torch.Tensor(input_data["target_point_next"]).to(self.device, dtype=torch.float32)).view(
                1, 2
            ),
            "speed": torch.Tensor([input_data["speed"]]).to(self.device, dtype=torch.float32).view(1),
            "command": torch.Tensor(input_data["command"]).to(self.device, dtype=torch.float32).view(1, 6),
        }

        # Add radar data if available
        if self.training_config.use_radars and "radar" in input_data:
            input_data_tensors["radar"] = torch.Tensor(input_data["radar"]).to(self.device, dtype=torch.float32)[None]

        # Save input log if need
        if self.config_closed_loop.save_path is not None and self.config_closed_loop.produce_input_log:
            torch.save(
                {k: v.to(torch.device("cpu")) if isinstance(v, torch.Tensor) else v for k, v in input_data_tensors.items()},
                os.path.join(self.config_closed_loop.input_log_path, str(self.step).zfill(5)) + ".pth",
            )

        # Forward pass
        closed_loop_prediction: ClosedLoopPrediction = self.closed_loop_inference.forward(data=input_data_tensors)
        # Update bounding boxes
        if (
            closed_loop_prediction.pred_bounding_box_vehicle_system is not None
            and len(closed_loop_prediction.pred_bounding_box_vehicle_system) > 0
        ):
            self.bb_buffer.append(closed_loop_prediction.pred_bounding_box_vehicle_system)

        # Post-processing heuristic
        closed_loop_prediction.throttle, closed_loop_prediction.brake = self.force_move_post_processor.adjust(
            input_data["speed"].item(), closed_loop_prediction.throttle, closed_loop_prediction.brake
        )

        self.control = carla.VehicleControl(
            steer=float(closed_loop_prediction.steer),
            throttle=float(closed_loop_prediction.throttle),
            brake=float(closed_loop_prediction.brake),
        )

        # CARLA will not let the car drive in the initial frames. This help the filter not get confused.
        if self.step < self.training_config.inital_frames_delay:
            self.control = carla.VehicleControl(0.0, 0.0, 1.0)

        # Visualization of prediction for debugging and video recording
        input_data_tensors.update(
            {
                "steer": torch.Tensor([self.control.steer]),
                "throttle": torch.Tensor([self.control.throttle]),
                "brake": torch.Tensor([self.control.brake]),
                "stuck_detector": torch.Tensor([self.force_move_post_processor.stuck_detector]),
                "force_move": torch.Tensor([self.force_move_post_processor.force_move]),
                "route_curvature": torch.Tensor(
                    [common_utils.waypoints_curvature(closed_loop_prediction.pred_route.squeeze())]
                ),
            }
        )

        # Save demo images
        if self.config_closed_loop.save_path is not None and self.step >= 0:
            # Get predicted route and waypoints (if available)
            pred_waypoints = (
                closed_loop_prediction.pred_future_waypoints[0]
                if closed_loop_prediction.pred_future_waypoints is not None
                else None
            )

            # Prepare target points dictionary for BEV visualization
            target_points = {
                "previous": input_data.get("target_point_previous"),
                "current": input_data.get("target_point"),
                "next": input_data.get("target_point_next"),
            }

            # Save demo cameras with visualization
            self.save_demo_cameras(pred_waypoints, target_points)

        # Save abstract debug images
        if (
            self.config_closed_loop.save_path is not None
            and (self.config_closed_loop.produce_debug_video or self.config_closed_loop.produce_debug_image)
            and self.step % self.config_closed_loop.produce_frame_frequency == 0
        ):
            # Produce image
            image = Visualizer(
                config=self.training_config,
                data=input_data_tensors,
                prediction=closed_loop_prediction,
                config_test_time=self.config_closed_loop,
                test_time=True,
            ).visualize_inference_prediction()
            image = np.array(image).astype(np.uint8)

            # Save image as video
            if self.config_closed_loop.produce_debug_video:
                if self.debug_video_writer is None:
                    os.makedirs(os.path.dirname(self.config_closed_loop.debug_video_path), exist_ok=True)
                    self.debug_video_writer = cv2.VideoWriter(
                        str(self.config_closed_loop.debug_video_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        self.config_closed_loop.video_fps,
                        (image.shape[1], image.shape[0]),
                    )

                self.debug_video_writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

            # Save image as png
            if self.config_closed_loop.produce_debug_image:
                save_dir = self.config_closed_loop.save_path / "debug_images"
                os.makedirs(save_dir, exist_ok=True)
                PIL.Image.fromarray(image).save(
                    f"{save_dir}/{str(self.step).zfill(5)}.png",
                    optimize=False,
                    compress_level=0,  # Really space expensive, do this local only.
                )

        # Save metric info if in Bench2Drive mode
        if self.config_closed_loop.is_bench2drive and hasattr(self, "get_metric_info"):
            metric = self.get_metric_info()
            self.metric_info[self.step] = metric
            with open(f"{self.config_closed_loop.save_path}/metric_info.json", "w") as outfile:
                json.dump(self.metric_info, outfile, indent=4)
        return self.control

    def destroy(self, _=None):
        # Clean up demo cameras
        if hasattr(self, "_demo_cameras"):
            for demo_cam_info in self._demo_cameras:
                if demo_cam_info["camera"].is_alive:
                    demo_cam_info["camera"].stop()
                    demo_cam_info["camera"].destroy()
                    LOG.info(f"[SensorAgent] Destroyed demo camera {demo_cam_info['index']}")

        # Clean up video writers
        if self.config_closed_loop.save_path is not None:
            if self.config_closed_loop.produce_debug_video:
                # Debug video - low quality for disk space
                self.debug_video_writer.release()
                self.compress_video(
                    temp_path=self.config_closed_loop.temp_debug_video_path,
                    final_path=self.config_closed_loop.debug_video_path,
                    crf=38,
                    preset="veryslow",
                )

            # Demo video - high quality for presentation
            if self.config_closed_loop.produce_demo_video:
                self.demo_video_writer.release()
                self.compress_video(
                    temp_path=self.config_closed_loop.temp_demo_video_path,
                    final_path=self.config_closed_loop.demo_video_path,
                    crf=18,
                    preset="slow",
                )

    @beartype
    def compress_video(self, temp_path: str, final_path: str, crf: int, preset: str):
        """Compresses a video using ffmpeg.

        Args:
            temp_path: Path to the uncompressed video.
            final_path: Path to save the compressed video.
            crf: Constant Rate Factor for ffmpeg compression (lower is better quality).
        """
        # Check if ffmpeg is installed
        command = f"ffmpeg -i {final_path} -c:v libx264 -crf {crf} -preset {preset} -an {temp_path} -y"
        os.system(command)
        os.replace(temp_path, final_path)


class ForceMovePostProcessor:
    """Forces the agent to move after a certain time of being stuck."""

    @beartype
    def __init__(self, config: TrainingConfig, config_test_time: ClosedLoopConfig, lidar_queue: deque):
        self.config = config
        self.config_test_time = config_test_time
        self.stuck_detector = 0
        self.force_move = 0
        self.lidar_buffer = lidar_queue

    @beartype
    def adjust(self, ego_speed: float, current_throttle: float, current_brake: float) -> tuple[float, float]:
        if ego_speed < 0.1:  # 0.1 is just an arbitrary low number to threshold when the car is stopped
            self.stuck_detector += 1
        else:
            self.stuck_detector = 0

        # If last red light was encountered a long time ago, we can assume it was cleared
        stuck_threshold = self.config_test_time.sensor_agent_stuck_threshold

        if self.stuck_detector > stuck_threshold:
            self.force_move = self.config_test_time.sensor_agent_stuck_move_duration

        if self.force_move > 0:
            emergency_stop = False
            # safety check
            safety_box = deepcopy(self.lidar_buffer[-1])

            # z-axis
            safety_box = safety_box[safety_box[..., 2] > self.config.safety_box_z_min]
            safety_box = safety_box[safety_box[..., 2] < self.config.safety_box_z_max]

            # y-axis
            safety_box = safety_box[safety_box[..., 1] > self.config.safety_box_y_min]
            safety_box = safety_box[safety_box[..., 1] < self.config.safety_box_y_max]

            # x-axis
            safety_box = safety_box[safety_box[..., 0] > self.config.safety_box_x_min]
            safety_box = safety_box[safety_box[..., 0] < self.config.safety_box_x_max]
            if len(safety_box) > 0:  # Checks if the List is empty
                emergency_stop = True
                LOG.info("Creeping overriden by safety box.")
            if not emergency_stop:
                LOG.info("Detected agent being stuck.")
                current_throttle = max(self.config_test_time.sensor_agent_stuck_throttle, current_throttle)
                current_brake = 0.0
                self.force_move -= 1
            else:
                LOG.info("Forced moving stopped by safety box.")
                current_throttle = 0.0
                current_brake = 1.0
                self.force_move = self.config_test_time.sensor_agent_stuck_move_duration
        return current_throttle, current_brake


if __name__ == "__main__":
    sensor_agent = SensorAgent()
