"""Ground-truth visualizer: LiDAR BEV with label overlays, camera perspectives and a meta panel."""

import os
import typing

import cv2
import jaxtyping as jt
import numpy as np
import numpy.typing as npt
import torch
from PIL import Image, ImageDraw, ImageFont

import lead.common.common_utils as common_utils
import lead.common.constants as constants
from lead.common.constants import RadarLabels, TransfuserBoundingBoxIndex
from lead.config import LeadConfig
from lead.policy.transfuser.dataloader import dataset_utils as carla_dataset_utils
from lead.policy.transfuser.visualization import viz_utils

# Bundled fonts live alongside this module; resolve relative to the file so the
# paths hold regardless of the current working directory.
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
FONT_BOLD = os.path.join(ASSETS_DIR, "Roboto-Bold.ttf")
FONT_REGULAR = os.path.join(ASSETS_DIR, "Roboto-Regular.ttf")

# Units shown next to numeric meta-panel attributes.
_META_UNITS: dict[str, str] = {
    # Velocities
    "speed": "m/s",
    "target_speed": "m/s",
    "second_highest_speed": "m/s",
    "speed_limit": "m/s",
    "target_speed_limit": "m/s",
    "second_highest_speed_limit": "m/s",
    # Accelerations
    "accel_x": "m/s²",
    "accel_y": "m/s²",
    "accel_z": "m/s²",
    "privileged_acceleration": "m/s²",
    # Distances
    "distance_to_next_junction": "m",
    "distance_to_junction": "m",
    "signed_dist_to_lane_change": "m",
    "distance_to_stop_sign": "m",
    "meters_travelled": "m",
    "ego_lane_width": "m",
    "perturbation_translation": "m",
    # Angles
    "theta": "rad",
    "privileged_yaw": "rad",
    "perturbation_rotation": "rad",
    # Angular velocities
    "angular_velocity_x": "rad/s",
    "angular_velocity_y": "rad/s",
    "angular_velocity_z": "rad/s",
}

# Meta-panel attributes formatted with two decimal places and a unit.
_META_FLOAT_KEYS: list[str] = [
    "steer",
    "throttle",
    "brake",
    "signed_dist_to_lane_change",
    "distance_to_next_junction",
    "speed_limit",
    "target_speed_limit",
    "route_curvature",
    "distance_to_junction",
    "ego_lane_width",
    "perturbation_translation",
    "perturbation_rotation",
    "speed",
    "accel_x",
    "accel_y",
    "accel_z",
    "angular_velocity_x",
    "angular_velocity_y",
    "angular_velocity_z",
    "target_speed",
    "theta",
    "privileged_yaw",
    "second_highest_speed",
    "second_highest_speed_limit",
    "distance_to_stop_sign",
    "privileged_acceleration",
    "meters_travelled",
]

# Meta-panel attributes printed verbatim.
_META_VERBATIM_KEYS: list[str] = [
    "vehicle_hazard",
    "light_hazard",
    "walker_hazard",
    "stop_sign_hazard",
    "stop_sign_close",
    "num_lanes_opposite_direction",
    "changed_route",
    "does_emergency_brake_for_pedestrians",
    "weather_setting",
    "emergency_brake_for_special_vehicle",
    "visual_visibility",
    "num_parking_vehicles_in_proximity",
    "slower_bad_visibility",
    "slower_clutterness",
    "used_traffic_bounding_boxes",
    "europe_traffic_light",
    "over_head_traffic_light",
    "jpeg_storage_quality",
    "augment_sample",
    "rear_danger_8",
    "rear_danger_16",
    "town",
    "frame_number",
    "num_dangerous_adversarial",
    "num_safe_adversarial",
    "num_ignored_adversarial",
    "rear_adversarial_id",
    "stuck_detector",
    "force_move",
    "bucket_identity",
    "perturbate_sensor",
    "route_number",
    "current_active_scenario_type",
    "previous_active_scenario_type",
    "scenario_type",
]


class GroundTruthVisualizer:
    """Visualizes the ground-truth labels of one batched sample.

    Composes the LiDAR BEV with label overlays, the camera perspectives and a
    meta-information panel into one image. Subclasses overwrite the drawing
    hooks to visualize predictions instead of labels.
    """

    # Height of the meta panel in pixels (before resizing to the image width).
    meta_panel_height: typing.ClassVar[int] = 639

    # Radar blob rendering: radius scaling with radial velocity.
    radar_velocity_max: typing.ClassVar[float] = 20.0
    radar_min_radius_pixel: typing.ClassVar[float] = 1.0
    radar_max_radius_pixel: typing.ClassVar[float] = 20.0

    def __init__(self, lead_config: LeadConfig, data: dict[str, typing.Any]) -> None:
        """Initialize the visualizer from one batched sample.

        Args:
            lead_config: Root config tree.
            data: Dictionary containing batched input data tensors.
        """
        self.lead_config = lead_config
        self.config = lead_config.agent.transfuser
        data_config = lead_config.expert.data_collection
        self.data_config = data_config
        self.data: dict[str, typing.Any] = data

        self.scale_factor: int = 4
        self.size_width: int = int(
            (data_config.max_y_meter - data_config.min_y_meter)
            * data_config.pixels_per_meter,
        )
        self.size_height: int = int(
            (data_config.max_x_meter - data_config.min_x_meter)
            * data_config.pixels_per_meter,
        )
        self.origin: tuple[float, float] = (
            (self.size_height * self.scale_factor)
            // (
                (data_config.max_x_meter - data_config.min_x_meter)
                / max((-data_config.min_x_meter), 1)
            ),
            (self.size_width * self.scale_factor) // 2,
        )
        self.loc_pixels_per_meter: float = (
            data_config.pixels_per_meter * self.scale_factor
        )

        start_color = np.array([255, 255, 255], dtype=np.float32)
        end_color = np.array(constants.LIDAR_COLOR, dtype=np.float32)

        rasterized_lidar: torch.Tensor = self.data["rasterized_lidar"]
        bev: jt.Float32[npt.NDArray, "h w"] = (
            rasterized_lidar.detach().cpu().numpy()[0][0]
        )
        bev = (bev / (bev.max() + 1e-6)).astype(np.float32)

        bev_img = np.zeros((*bev.shape, 3), dtype=np.float32)
        for c in range(3):
            bev_img[..., c] = start_color[c] + (end_color[c] - start_color[c]) * bev

        # Drawing blends label colors in, so the BEV stays a float image until
        # the final composition casts it back to uint8.
        self.bev_image: jt.Shaped[npt.NDArray, "h w 3"] = cv2.resize(
            bev_img.astype(np.uint8),
            dsize=(
                bev_img.shape[1] * self.scale_factor,
                bev_img.shape[0] * self.scale_factor,
            ),
            interpolation=cv2.INTER_NEAREST,
        )

        self.meta_panel: jt.UInt8[npt.NDArray, "h w 3"] = 255 * np.ones(
            (self.meta_panel_height, 1492, 3),
            dtype=np.uint8,
        )
        self.perspectives: dict[str, jt.UInt8[npt.NDArray, "h w 3"]] = {}

    def visualize(self) -> jt.UInt8[npt.NDArray, "h w 3"]:
        """Render the sample.

        Returns:
            The composed visualization image.
        """
        self._process_all_perspectives()
        self._draw_bev()
        self.bev_image = np.ascontiguousarray(np.rot90(self.bev_image, k=1))
        self._meta()
        return self._concatenate_all_perspectives_and_bev()

    def _draw_bev(self) -> None:
        """Draw all BEV overlays for the ground-truth view."""
        self._bev_semantic()
        self._route()
        self._future_waypoints()
        self._ego_bounding_box()
        self._bounding_boxes()
        self._target_point()
        self._radars()

    # ------------------------------------------------------------------
    # Perspectives
    # ------------------------------------------------------------------

    def _process_all_perspectives(self) -> None:
        """Fill ``self.perspectives`` with the available perspective images."""
        for modality in ["rgb", "semantic"]:
            image = self._perspective_image(modality)
            if image is not None:
                self.perspectives[modality] = image

    def _perspective_image(
        self,
        modality: str,
    ) -> jt.UInt8[npt.NDArray, "h w 3"] | None:
        """Build one perspective image from the ground-truth data.

        Args:
            modality: Perspective modality, ``"rgb"`` or ``"semantic"``.

        Returns:
            The perspective image, or None when the modality is unavailable.
        """
        perspective = self.data.get(modality)
        if perspective is None:
            return None
        perspective = perspective[0]
        if modality == "semantic":
            perspective = perspective.unsqueeze(0)
        image = (
            perspective.permute(1, 2, 0).detach().cpu().float().numpy().astype(np.uint8)
        )
        image = np.ascontiguousarray(image)
        if modality == "semantic":
            image = self._semantic_to_color(image[..., 0])
        return image

    def _semantic_to_color(
        self,
        semantic: jt.UInt8[npt.NDArray, "h w"],
    ) -> jt.UInt8[npt.NDArray, "h w 3"]:
        """Map semantic class labels to their visualization colors.

        Args:
            semantic: Semantic class label image.

        Returns:
            The colored semantic image.
        """
        converter = np.array(
            list(constants.TRANSFUSER_SEMANTIC_COLORS.values()),
            dtype=np.uint8,
        )
        return converter[semantic]

    # ------------------------------------------------------------------
    # BEV semantic
    # ------------------------------------------------------------------

    def _bev_semantic(self) -> None:
        """Overlay the ground-truth BEV semantic map."""
        bev_semantic = self.data.get("bev_semantic")
        if bev_semantic is None:
            return
        labels = bev_semantic[0].detach().cpu().float().numpy().astype(np.int32)
        self._overlay_bev_semantic(labels)

    def _overlay_bev_semantic(
        self,
        bev_semantic: jt.Int[npt.NDArray, "h w"],
    ) -> None:
        """Alpha-blend a BEV semantic class map onto the BEV image.

        Args:
            bev_semantic: BEV semantic class labels.
        """
        converter = np.array(
            list(constants.CARLA_TRANSFUSER_BEV_SEMANTIC_COLOR_CONVERTER.values()),
        )
        converter[1][0:3] = 40
        bev_semantic_image = converter[bev_semantic, ...].astype("uint8")
        alpha = (np.ones_like(bev_semantic) * 0.33).astype(np.float32)
        alpha[bev_semantic == 0] = 0.0
        alpha[bev_semantic == 1] = 0.15

        alpha = cv2.resize(
            alpha,
            dsize=(
                alpha.shape[1] * self.scale_factor,
                alpha.shape[0] * self.scale_factor,
            ),
            interpolation=cv2.INTER_NEAREST,
        )
        alpha = np.expand_dims(alpha, 2)
        bev_semantic_image = cv2.resize(
            bev_semantic_image,
            dsize=(self.bev_image.shape[1], self.bev_image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        self.bev_image = bev_semantic_image * alpha + (1 - alpha) * self.bev_image

    # ------------------------------------------------------------------
    # Route, waypoints and target points
    # ------------------------------------------------------------------

    def _route(self) -> None:
        """Draw the ground-truth route as connected line segments."""
        route = self.data.get("route")
        if route is None or not (
            self.config.use_planning_decoder
            or self.lead_config.training.experiment.visualize_dataset
        ):
            return
        wps = route.detach().cpu().numpy()[0]
        for i in range(len(wps) - 1):
            x1 = int(wps[i][0] * self.loc_pixels_per_meter + self.origin[0])
            y1 = int(wps[i][1] * self.loc_pixels_per_meter + self.origin[1])
            x2 = int(wps[i + 1][0] * self.loc_pixels_per_meter + self.origin[0])
            y2 = int(wps[i + 1][1] * self.loc_pixels_per_meter + self.origin[1])
            color = viz_utils.lighter_shade(
                constants.PREDICTION_ROUTE_COLOR,
                i,
                len(wps),
            )
            cv2.line(
                self.bev_image,
                (x1, y1),
                (x2, y2),
                color=color,
                thickness=constants.PREDICTION_ROUTE_RADIUS,
                lineType=cv2.LINE_AA,
            )

    def _future_waypoints(self) -> None:
        """Draw the ground-truth future and past ego waypoints."""
        if not (
            self.config.use_planning_decoder
            or self.lead_config.training.experiment.visualize_dataset
        ):
            return
        for key, base_color in [
            ("future_waypoints", constants.GROUNDTRUTH_FUTURE_WAYPOINT_COLOR),
            ("past_waypoints", constants.GROUND_TRUTH_PAST_WAYPOINT_COLOR),
        ]:
            waypoints = self.data.get(key)
            if waypoints is None or waypoints[0] is None:
                continue
            wps = waypoints.detach().cpu().numpy()[0]
            for i, wp in enumerate(wps):
                wp_x = wp[0] * self.loc_pixels_per_meter + self.origin[0]
                wp_y = wp[1] * self.loc_pixels_per_meter + self.origin[1]
                color = viz_utils.lighter_shade(base_color, i, len(wps))
                cv2.circle(
                    self.bev_image,
                    (int(wp_x), int(wp_y)),
                    radius=constants.PREDICTION_WAYPOINT_RADIUS,
                    color=color,
                    thickness=-1,
                )

    def _target_point(self) -> None:
        """Draw the previous, current and next target points."""
        for key, radius, number in [
            ("target_point_previous", 14, 0),
            ("target_point_next", 14, 2),
            ("target_point", 18, 1),
        ]:
            target_point = self.data.get(key)
            if target_point is None:
                continue
            x_tp = target_point[0][0] * self.loc_pixels_per_meter + self.origin[0]
            y_tp = target_point[0][1] * self.loc_pixels_per_meter + self.origin[1]
            viz_utils.draw_circle_with_number(
                self.bev_image,
                int(x_tp),
                int(y_tp),
                constants.TP_DEFAULT_COLOR,
                radius=radius,
                number=number,
            )

    # ------------------------------------------------------------------
    # Bounding boxes
    # ------------------------------------------------------------------

    def _ego_bounding_box(self) -> None:
        """Draw the ego bounding box with its current speed."""
        ego_box = np.array(
            [
                int(
                    self.bev_image.shape[1]
                    * (
                        -self.data_config.min_x_meter
                        / (self.data_config.max_x_meter - self.data_config.min_x_meter)
                    ),
                ),
                int(self.bev_image.shape[0] / 2),
                self.lead_config.expert.simulation.ego_extent_x
                * self.loc_pixels_per_meter,
                self.lead_config.expert.simulation.ego_extent_y
                * self.loc_pixels_per_meter,
                np.deg2rad(0.0),
                self.data["speed"][0].item(),
            ],
        )
        self.bev_image = viz_utils.draw_box(
            self.bev_image,
            ego_box,
            color=constants.EGO_BB_COLOR,
            thickness=4,
        )

    def _bounding_boxes(self) -> None:
        """Draw the ground-truth bounding boxes and their future footprints."""
        bounding_boxes = self.data.get("center_net_bounding_boxes")
        if bounding_boxes is not None:
            boxes = bounding_boxes.detach().cpu().numpy()[0]
            boxes = boxes[boxes.sum(axis=-1) != 0.0]
            for box in boxes:
                box = box.copy()
                box[:4] = box[:4] * self.scale_factor
                color_box = list(constants.TRANSFUSER_BOUNDING_BOX_COLORS.values())[
                    int(box[TransfuserBoundingBoxIndex.CLASS])
                ]
                self.bev_image = viz_utils.draw_box(
                    self.bev_image,
                    box,
                    color=color_box,
                )

        vehicles_future_waypoints = self.data.get("vehicles_future_waypoints")
        vehicle_future_yaws = self.data.get("vehicles_future_yaws")

        if vehicles_future_waypoints is not None:
            for future_box_waypoints in vehicles_future_waypoints:
                wps = future_box_waypoints[0]
                for i, wp in enumerate(wps):
                    wp_x = wp[0] * self.loc_pixels_per_meter + self.origin[0]
                    wp_y = wp[1] * self.loc_pixels_per_meter + self.origin[1]
                    color = viz_utils.lighter_shade(
                        constants.GROUNDTRUTH_BB_WP_COLOR,
                        i,
                        len(wps),
                    )
                    cv2.circle(
                        self.bev_image,
                        (int(wp_x), int(wp_y)),
                        radius=constants.PREDICTION_WAYPOINT_RADIUS // 4,
                        color=color,
                        thickness=-1,
                    )

        if vehicles_future_waypoints is not None and vehicle_future_yaws is not None:
            for future_box_waypoints, future_box_yaws in zip(
                vehicles_future_waypoints,
                vehicle_future_yaws,
                strict=True,
            ):
                wps = future_box_waypoints[0]
                yaws = future_box_yaws[0]
                for i, wp in enumerate(wps):
                    color = viz_utils.lighter_shade(
                        constants.GROUNDTRUTH_BB_WP_COLOR,
                        i,
                        len(wps),
                    )
                    self._draw_bounding_box_from_waypoint(wp[0], wp[1], yaws[i], color)

    def _draw_bounding_box_from_waypoint(
        self,
        x: float,
        y: float,
        yaw: float,
        color: tuple[int, int, int],
    ) -> None:
        """Draw an ego-sized bounding box footprint at a future waypoint.

        Args:
            x: X coordinate in the ego frame in meters.
            y: Y coordinate in the ego frame in meters.
            yaw: Yaw in the ego frame in radians.
            color: Box color (RGB).
        """
        bb = np.array(
            [
                x,
                y,
                self.lead_config.expert.simulation.ego_extent_x,
                self.lead_config.expert.simulation.ego_extent_y,
                yaw,
            ],
        )
        bb = carla_dataset_utils.bb_vehicle_to_image_system(
            bb[None],
            pixels_per_meter=self.data_config.pixels_per_meter,
            min_x=self.data_config.min_x_meter,
            min_y=self.data_config.min_y_meter,
        )[0]
        bb[:4] = bb[:4] * self.scale_factor
        self.bev_image = viz_utils.draw_box(
            self.bev_image,
            bb,
            color=color,
            thickness=1,
        )

    # ------------------------------------------------------------------
    # Radar
    # ------------------------------------------------------------------

    def _radars(self) -> None:
        """Draw the raw radar returns and the radar detections."""
        if not self.lead_config.expert.sensor_rig.use_radars:
            return

        for i in range(
            1,
            self.lead_config.expert.sensor_rig.num_radar_sensors + 1,
        ):
            radar_i = self.data.get(f"radar{i}")
            if radar_i is None:
                continue
            arr = radar_i[0].detach().cpu().float().numpy()
            points = arr[(arr[:, :3] != 0).any(axis=1)]  # drop zero-padded
            if points.shape[0] == 0:
                continue
            self._draw_radar_returns(
                points[:, 0],
                points[:, 1],
                np.nan_to_num(points[:, 3], nan=0.0),
                color=constants.RADAR_COLOR,
            )

        self._radar_detections()

    def _draw_radar_returns(
        self,
        x: jt.Float[npt.NDArray, " n"],
        y: jt.Float[npt.NDArray, " n"],
        velocity: jt.Float[npt.NDArray, " n"],
        color: tuple[int, int, int],
        radius_offset: int = 0,
    ) -> None:
        """Draw radar returns as Gaussian blobs sized by radial velocity.

        Args:
            x: X coordinates in meters.
            y: Y coordinates in meters.
            velocity: Radial velocities in m/s; positive means approaching.
            color: Blob color (RGB).
            radius_offset: Extra blob radius in pixels.
        """
        min_r = self.radar_min_radius_pixel
        max_r = self.radar_max_radius_pixel
        for xm, ym, vk in zip(x, y, velocity, strict=True):
            px = int(self.origin[0] + xm * self.loc_pixels_per_meter)
            py = int(self.origin[1] + ym * self.loc_pixels_per_meter)
            rpx = radius_offset + int(
                np.clip(
                    min_r + (abs(vk) / self.radar_velocity_max) * (max_r - min_r),
                    min_r,
                    max_r,
                ),
            )
            # Filled = approaching, outline = receding.
            viz_utils.draw_gaussian_blob(
                self.bev_image,
                px,
                py,
                rpx,
                color,
                vk > 0,
            )

    def _radar_detections(self) -> None:
        """Draw the ground-truth radar detections and their waypoints."""
        radar_detections = self.data.get("radar_detections")
        if radar_detections is None:
            return
        detections = radar_detections[0].cpu().numpy()
        valid_mask = detections[:, RadarLabels.VALID].astype(bool)
        valid_detections = detections[valid_mask]

        if valid_detections.shape[0] > 0:
            self._draw_radar_returns(
                valid_detections[:, 0],
                valid_detections[:, 1],
                np.nan_to_num(valid_detections[:, 2], nan=0.0),
                color=constants.RADAR_DETECTION_COLOR,
                radius_offset=1,
            )

        detection_waypoints = self.data.get("radar_detection_waypoints")
        detection_num_waypoints = self.data.get("radar_detection_num_waypoints")
        if detection_waypoints is None or detection_num_waypoints is None:
            return
        valid_waypoints = detection_waypoints[0].cpu().numpy()[valid_mask]
        valid_num_waypoints = detection_num_waypoints[0].cpu().numpy()[valid_mask]

        for waypoints, num_wps in zip(
            valid_waypoints,
            valid_num_waypoints,
            strict=True,
        ):
            pixel_points = [
                (
                    int(self.origin[0] + waypoints[i, 0] * self.loc_pixels_per_meter),
                    int(self.origin[1] + waypoints[i, 1] * self.loc_pixels_per_meter),
                )
                for i in range(int(num_wps))
            ]
            for point in pixel_points:
                cv2.circle(
                    self.bev_image,
                    point,
                    radius=3,
                    color=constants.RADAR_DETECTION_COLOR,
                    thickness=-1,
                )
            for point, next_point in zip(
                pixel_points[:-1],
                pixel_points[1:],
                strict=True,
            ):
                cv2.line(
                    self.bev_image,
                    point,
                    next_point,
                    color=constants.RADAR_DETECTION_COLOR,
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )

    # ------------------------------------------------------------------
    # Perspective projection helper
    # ------------------------------------------------------------------

    def _draw_points_in_perspective(
        self,
        points: jt.Float[npt.NDArray, "n 2"],
        size: int,
        connected: bool,
        color: tuple[int, int, int],
    ) -> None:
        """Project ego-frame points into the concatenated RGB perspectives.

        Args:
            points: Points in the ego frame in meters.
            size: Circle radius or line thickness in pixels.
            connected: Draw connected lines instead of circles.
            color: Drawing color (RGB).
        """
        if not (len(points) > 0 and points.shape[0] > 0):
            return
        rgb = self.perspectives["rgb"]

        sensor_rig = self.lead_config.expert.sensor_rig
        camera_rots = []
        camera_poses = []
        camera_fovs = []
        camera_widths = []
        for i in range(1, min(4, sensor_rig.num_cameras + 1)):
            camera_config = sensor_rig.cameras[i - 1]
            camera_rots.append(camera_config["rot"])
            camera_poses.append(camera_config["pos"])
            camera_fovs.append(camera_config["fov"])
            camera_widths.append(camera_config["width"])

        # Reverse rotation and position lists to match the stored image order.
        camera_rots = camera_rots[:3][::-1]
        camera_poses = camera_poses[:3][::-1]
        camera_height = sensor_rig.camera_height

        x_offset = 0
        for i in range(3):
            projected_points, inside_image = common_utils.project_points_to_image(
                camera_rots[i],
                camera_poses[i],
                camera_fovs[i],
                camera_widths[i],
                camera_height,
                points,
            )

            if connected:
                for j in range(len(projected_points) - 1):
                    if not inside_image[j + 1]:
                        continue
                    x1, y1 = projected_points[j]
                    x2, y2 = projected_points[j + 1]
                    cv2.line(
                        rgb,
                        (int(x1 + x_offset), int(y1)),
                        (int(x2 + x_offset), int(y2)),
                        color,
                        size,
                    )
            else:
                for (x, y), inside in zip(
                    projected_points,
                    inside_image,
                    strict=True,
                ):
                    if not inside:
                        continue
                    img_x = int(x + x_offset)
                    img_y = int(y)
                    if 0 <= img_x < rgb.shape[1] and 0 <= img_y < rgb.shape[0]:
                        cv2.circle(rgb, (img_x, img_y), size, color, -1)

            x_offset += camera_widths[i]

    # ------------------------------------------------------------------
    # Meta panel
    # ------------------------------------------------------------------

    def _extra_meta_lines(self) -> list[str]:
        """Additional meta-panel lines contributed by subclasses.

        Returns:
            Extra text lines in ``"<name> <value>"`` format.
        """
        return []

    def _meta(self) -> None:
        """Render the meta-information panel."""
        text_lines: list[str] = []

        for attr_name in _META_FLOAT_KEYS:
            attr_data = self.data.get(attr_name)
            if attr_data is None:
                continue
            attr_data = attr_data[0]
            if isinstance(attr_data, torch.Tensor):
                attr_data = attr_data.item()
            unit = _META_UNITS.get(attr_name, "")
            unit_str = f" {unit}" if unit else ""
            text_lines.append(f"{attr_name} {attr_data:.2f}{unit_str}")

        for attr_name in _META_VERBATIM_KEYS:
            attr_data = self.data.get(attr_name)
            if attr_data is None:
                continue
            attr_data = attr_data[0]
            if isinstance(attr_data, torch.Tensor):
                attr_data = attr_data.item()
            text_lines.append(f"{attr_name} {attr_data}")

        for attr_name in ["target_point_previous", "target_point", "target_point_next"]:
            attr_data = self.data.get(attr_name)
            if attr_data is None:
                continue
            attr_data = attr_data[0]
            if isinstance(attr_data, torch.Tensor):
                attr_data = attr_data.detach().cpu().numpy()
            text_lines.append(
                f"{attr_name} (x={attr_data[0]:.1f}m, y={attr_data[1]:.1f}m)",
            )

        text_lines.extend(self._extra_meta_lines())
        text_lines = sorted(text_lines)

        font_regular = ImageFont.truetype(FONT_REGULAR, 17)
        font_bold = ImageFont.truetype(FONT_BOLD, 17)

        img_pil = Image.fromarray(self.meta_panel)
        draw = ImageDraw.Draw(img_pil)

        column_width = 350
        num_columns = 4
        start_x = 10
        start_y = 10
        line_height = 20

        for item_index, text in enumerate(text_lines):
            row = item_index // num_columns
            col = item_index % num_columns
            x = start_x + col * column_width
            y = start_y + row * line_height

            if " " not in text:
                draw.text((x, y), text, font=font_regular, fill=(0, 0, 0))
                continue

            # Attribute name in bold, value right-aligned within the column.
            split_idx = text.find(" ")
            name_part = text[:split_idx]
            value_part = text[split_idx:].strip()
            value_bbox = draw.textbbox((0, 0), value_part, font=font_regular)
            value_width = value_bbox[2] - value_bbox[0]

            draw.text((x, y), name_part, font=font_bold, fill=(0, 0, 0))
            draw.text(
                (x + column_width - value_width - 20, y),
                value_part,
                font=font_regular,
                fill=(0, 0, 0),
            )

        self.meta_panel = np.array(img_pil)

    # ------------------------------------------------------------------
    # Composition
    # ------------------------------------------------------------------

    def _concatenate_all_perspectives_and_bev(
        self,
        border_size: int = 10,
        border_color: tuple[int, int, int] = (255, 255, 255),
    ) -> jt.UInt8[npt.NDArray, "h w 3"]:
        """Compose the BEV, the stacked perspectives and the meta panel.

        The perspectives are stacked vertically on the right, the BEV sits on
        the left, and the meta panel is appended at the bottom, all separated
        by borders.

        Args:
            border_size: Size of the border between components in pixels.
            border_color: Color of the border (RGB).

        Returns:
            The composed image.
        """
        lidar_image = np.ascontiguousarray(self.bev_image, dtype=np.uint8)

        if not self.perspectives:
            raise ValueError("No perspectives available")

        # Resize all perspectives to the BEV width.
        perspective_images: list[jt.UInt8[npt.NDArray, "h w 3"]] = []
        target_width = lidar_image.shape[1]
        for modality in ["rgb", "semantic"]:
            if modality not in self.perspectives:
                continue
            img = np.ascontiguousarray(self.perspectives[modality], dtype=np.uint8)
            target_height = int(img.shape[0] * (target_width / img.shape[1]))
            perspective_images.append(cv2.resize(img, (target_width, target_height)))

        def horizontal_border(width: int) -> jt.UInt8[npt.NDArray, "h w 3"]:
            return np.full((border_size, width, 3), border_color, dtype=np.uint8)

        def vertical_border(height: int) -> jt.UInt8[npt.NDArray, "h w 3"]:
            return np.full((height, border_size, 3), border_color, dtype=np.uint8)

        # Stack perspectives vertically with borders in between.
        bordered_perspectives: list[jt.UInt8[npt.NDArray, "h w 3"]] = []
        for i, img in enumerate(perspective_images):
            if i > 0:
                bordered_perspectives.append(horizontal_border(img.shape[1]))
            bordered_perspectives.append(img)
        stacked_perspectives = np.concatenate(bordered_perspectives, axis=0)
        stacked_perspectives = np.concatenate(
            (stacked_perspectives, vertical_border(stacked_perspectives.shape[0])),
            axis=1,
        )

        # Resize the BEV to the stacked perspectives height and border it.
        target_height = stacked_perspectives.shape[0]
        bev_width = int(lidar_image.shape[1] * (target_height / lidar_image.shape[0]))
        lidar_resized = cv2.resize(lidar_image, (bev_width, target_height))
        lidar_bordered = np.concatenate(
            (vertical_border(target_height), lidar_resized),
            axis=1,
        )

        ret = np.concatenate(
            (lidar_bordered, vertical_border(target_height), stacked_perspectives),
            axis=1,
        )
        ret = np.concatenate(
            (horizontal_border(ret.shape[1]), ret, horizontal_border(ret.shape[1])),
            axis=0,
        )

        # Append the meta panel at the bottom.
        meta_panel = np.ascontiguousarray(self.meta_panel, dtype=np.uint8)
        target_width = ret.shape[1]
        target_height = int(meta_panel.shape[0] * (target_width / meta_panel.shape[1]))
        meta_panel_resized = cv2.resize(meta_panel, (target_width, target_height))
        ret = np.concatenate(
            (ret, horizontal_border(target_width), meta_panel_resized),
            axis=0,
        )

        return ret
