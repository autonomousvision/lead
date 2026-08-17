"""Ground-truth visualizer: LiDAR BEV with label overlays, camera perspectives and a meta panel."""

import os
import typing

import cv2
import jaxtyping as jt
import numpy as np
import numpy.typing as npt
import torch
from PIL import Image, ImageDraw, ImageFont

from lead.common.constants import RadarLabels
from lead.config import LeadConfig
from lead.config.policy.transfuser.label_classes import BoundingBoxIndex
from lead.policy.transfuser.dataloader.sample import TransfuserForwardBatch
from lead.policy.transfuser.utils import ops
from lead.policy.transfuser.visualization import colors, drawing

# One coordinate out of a batch, whichever way it was collated: a numpy scalar
# and a 0-d tensor are neither of them Python floats.
_Scalar = float | np.floating | torch.Tensor

# Bundled fonts live alongside this module; resolve relative to the file so the
# paths hold regardless of the current working directory.
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
FONT_BOLD = os.path.join(ASSETS_DIR, "Roboto-Bold.ttf")
FONT_REGULAR = os.path.join(ASSETS_DIR, "Roboto-Regular.ttf")

# Units shown next to numeric meta-panel attributes.
_META_UNITS: dict[str, str] = {
    "target_speed": "m/s",
    "perturbation_translation": "m",
    "perturbation_rotation": "rad",
}

# Meta-panel attributes formatted with two decimal places and a unit.
_META_FLOAT_KEYS: list[str] = [
    "throttle",
    "brake",
    "target_speed",
    "perturbation_translation",
    "perturbation_rotation",
]

# Number of grid columns spanned by a meta-panel entry too wide for one column.
WIDE_COLUMN_SPAN = 2

# Meta-panel attributes printed as-is.
_META_VERBATIM_KEYS: list[str] = [
    "route_number",
    "frame_number",
    "town",
    "weather_setting",
    "current_active_scenario_type",
    "previous_active_scenario_type",
    "over_head_traffic_light",
    "slower_bad_visibility",
    "jpeg_storage_quality",
    "perturbate_sensor",
]


class GroundTruthVisualizer:
    """Visualizes the ground-truth labels of one batched sample.

    Composes the LiDAR BEV with label overlays, the camera perspectives and a
    meta-information panel into one image; subclasses overwrite the drawing
    hooks to visualize predictions rather than labels.
    """

    # Height of the meta panel in pixels (before resizing to the image width).
    meta_panel_height: typing.ClassVar[int] = 639

    # Perspective modalities, in the order they are stacked in the composition.
    perspective_modalities: typing.ClassVar[list[str]] = ["rgb", "semantic", "depth"]

    # Radar marker rendering: radius scaling with radial velocity.
    radar_velocity_max: typing.ClassVar[float] = 20.0
    radar_min_radius_pixel: typing.ClassVar[float] = 3.0
    radar_max_radius_pixel: typing.ClassVar[float] = 43.0

    def __init__(self, lead_config: LeadConfig, data: TransfuserForwardBatch) -> None:
        """Initialize the visualizer from one batched sample.

        Args:
            lead_config: Root config tree.
            data: Dictionary containing batched input data tensors.
        """
        self.lead_config = lead_config
        self.config = lead_config.policy.transfuser
        config = self.config
        self.data: TransfuserForwardBatch = data

        self.scale_factor: int = 4
        self.size_width: int = int(
            (config.bev_max_y_meter - config.bev_min_y_meter)
            * config.bev_pixels_per_meter,
        )
        self.size_height: int = int(
            (config.bev_max_x_meter - config.bev_min_x_meter)
            * config.bev_pixels_per_meter,
        )
        self.origin: tuple[float, float] = (
            (self.size_height * self.scale_factor)
            // (
                (config.bev_max_x_meter - config.bev_min_x_meter)
                / max((-config.bev_min_x_meter), 1)
            ),
            (self.size_width * self.scale_factor) // 2,
        )
        self.loc_pixels_per_meter: float = (
            config.bev_pixels_per_meter * self.scale_factor
        )

        start_color = np.array([255, 255, 255], dtype=np.float32)
        end_color = np.array(colors.LIDAR_COLOR, dtype=np.float32)

        # A camera-only model (LTF) has no LiDAR raster; its overlays are drawn
        # on the empty BEV grid instead.
        rasterized_lidar: torch.Tensor | None = self.data.get("rasterized_lidar")
        bev: jt.Float32[npt.NDArray, "h w"] = (
            np.zeros(
                (config.lidar_height_pixel, config.lidar_width_pixel),
                dtype=np.float32,
            )
            if rasterized_lidar is None
            else rasterized_lidar.detach().cpu().numpy()[0][0]
        )
        bev = (bev / (bev.max() + 1e-6)).astype(np.float32)

        bev_img = np.zeros((*bev.shape, 3), dtype=np.float32)
        for c in range(3):
            bev_img[..., c] = start_color[c] + (end_color[c] - start_color[c]) * bev

        self.bev_image: jt.UInt8[npt.NDArray, "h w 3"] = cv2.resize(
            bev_img.astype(np.uint8),
            dsize=(
                bev_img.shape[1] * self.scale_factor,
                bev_img.shape[0] * self.scale_factor,
            ),
            interpolation=cv2.INTER_NEAREST,
        )

        self.meta_panel: jt.UInt8[npt.NDArray, "h w 3"] = np.full(
            (self.meta_panel_height, 1492, 3),
            255,
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
        self.bev_image = cv2.rotate(self.bev_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
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

    # --- Perspectives ---

    def _process_all_perspectives(self) -> None:
        """Fill ``self.perspectives`` with the available perspective images."""
        for modality in self.perspective_modalities:
            image = self._perspective_image(modality)
            if image is not None:
                self.perspectives[modality] = image

    def _perspective_image(
        self,
        modality: str,
    ) -> jt.UInt8[npt.NDArray, "h w 3"] | None:
        """Build one perspective image from the ground-truth data."""
        perspective = self.data.get(modality)
        if perspective is None:
            return None
        perspective = perspective[0]
        if modality == "depth":
            metric_depth = ops.dequantize_depth(
                perspective.detach().cpu(),
                self.lead_config.expert.storage.save_depth_max_meters,
            )
            return self._depth_to_color(metric_depth.numpy())
        if modality == "semantic":
            perspective = perspective.unsqueeze(0)
        image = perspective.permute(1, 2, 0).detach().cpu().numpy()
        image = np.ascontiguousarray(image, dtype=np.uint8)
        if modality == "semantic":
            image = self._semantic_to_color(image[..., 0])
        return image

    def _depth_to_color(
        self,
        depth: jt.Float32[npt.NDArray, "h w"],
    ) -> jt.UInt8[npt.NDArray, "h w 3"]:
        """Colorize a metric depth map against the far plane it was stored with."""
        max_depth_meter = self.lead_config.expert.storage.save_depth_max_meters
        return drawing.depth_to_color(depth, max_depth_meter)

    def _semantic_to_color(
        self,
        semantic: jt.UInt8[npt.NDArray, "h w"],
    ) -> jt.UInt8[npt.NDArray, "h w 3"]:
        """Map semantic class labels to their visualization colors."""
        converter = np.array(
            list(colors.TRANSFUSER_SEMANTIC_COLORS.values()),
            dtype=np.uint8,
        )
        return drawing.color_by_class(semantic, converter)

    # --- BEV semantic ---

    def _bev_semantic(self) -> None:
        """Overlay the ground-truth BEV semantic map."""
        bev_semantic = self.data.get("bev_semantic")
        if bev_semantic is None:
            return
        labels = bev_semantic[0].detach().cpu().numpy().astype(np.uint8)
        self._overlay_bev_semantic(labels)

    def _overlay_bev_semantic(
        self,
        bev_semantic: jt.UInt8[npt.NDArray, "h w"],
    ) -> None:
        """Alpha-blend a BEV semantic class map onto the BEV image.

        Blending on the raster the BEV image was scaled up from and scaling the
        result up again is equivalent to blending on the scaled-up image, at a
        scale factor squared less work.
        """
        converter = np.array(
            list(colors.CARLA_TRANSFUSER_BEV_SEMANTIC_COLOR_CONVERTER.values()),
            dtype=np.uint8,
        )
        converter[1][0:3] = 40
        bev_semantic_image = drawing.color_by_class(bev_semantic, converter)
        alpha = np.full(bev_semantic.shape, 0.33, dtype=np.float32)
        alpha[bev_semantic == 0] = 0.0
        alpha[bev_semantic == 1] = 0.15

        scaled_size = (self.bev_image.shape[1], self.bev_image.shape[0])
        raster_size = (
            scaled_size[0] // self.scale_factor,
            scaled_size[1] // self.scale_factor,
        )
        bev_semantic_image = cv2.resize(
            bev_semantic_image,
            dsize=raster_size,
            interpolation=cv2.INTER_NEAREST,
        )
        alpha = cv2.resize(
            alpha,
            dsize=raster_size,
            interpolation=cv2.INTER_NEAREST,
        )[..., np.newaxis]
        bev_raster = cv2.resize(
            self.bev_image,
            dsize=raster_size,
            interpolation=cv2.INTER_NEAREST,
        )
        blended = (bev_semantic_image * alpha + (1 - alpha) * bev_raster).astype(
            np.uint8,
        )
        self.bev_image = cv2.resize(
            blended,
            dsize=scaled_size,
            interpolation=cv2.INTER_NEAREST,
        )

    # --- Shared drawing primitives ---

    def _to_pixel(self, x: _Scalar, y: _Scalar) -> tuple[int, int]:
        """Convert an ego-frame position to BEV pixel coordinates."""
        return (
            int(float(x) * self.loc_pixels_per_meter + self.origin[0]),
            int(float(y) * self.loc_pixels_per_meter + self.origin[1]),
        )

    def _to_pixels(
        self,
        points: jt.Float[npt.NDArray, "n 2"],
    ) -> jt.Int32[npt.NDArray, "n 2"]:
        """Convert ego-frame positions to BEV pixel coordinates."""
        return (
            points[:, :2] * self.loc_pixels_per_meter + np.asarray(self.origin)
        ).astype(np.int32)

    def _draw_waypoints(
        self,
        waypoints: jt.Float[npt.NDArray, "n 2"],
        base_color: tuple[int, int, int],
        radius: int,
    ) -> None:
        """Draw a scatter of ego-frame waypoints, lightening along the sequence."""
        for i, (x_pixel, y_pixel) in enumerate(self._to_pixels(waypoints)):
            cv2.circle(
                self.bev_image,
                (int(x_pixel), int(y_pixel)),
                radius=radius,
                color=drawing.lighter_shade(base_color, i, len(waypoints)),
                thickness=-1,
                lineType=cv2.LINE_AA,
            )

    @property
    def _planning_visible(self) -> bool:
        """Whether the route and ego waypoints are drawn."""
        return (
            self.config.use_planning_decoder
            or self.lead_config.training.experiment.visualize_dataset
        )

    # --- Route, waypoints and target points ---

    def _route(self) -> None:
        """Draw the ground-truth route as a scatter of waypoints."""
        route = self.data.get("route")
        if route is None or not self._planning_visible:
            return
        self._draw_waypoints(
            route.detach().cpu().numpy()[0],
            colors.PREDICTION_ROUTE_COLOR,
            colors.PREDICTION_ROUTE_RADIUS,
        )

    def _future_waypoints(self) -> None:
        """Draw the ground-truth future and past ego waypoints."""
        if not self._planning_visible:
            return
        for key, base_color in [
            ("future_waypoints", colors.GROUNDTRUTH_FUTURE_WAYPOINT_COLOR),
            ("past_waypoints", colors.GROUND_TRUTH_PAST_WAYPOINT_COLOR),
        ]:
            waypoints = self.data.get(key)
            if waypoints is None or waypoints[0] is None:
                continue
            self._draw_waypoints(
                waypoints.detach().cpu().numpy()[0],
                base_color,
                colors.PREDICTION_WAYPOINT_RADIUS,
            )

    def _target_point(self) -> None:
        """Draw the previous, current and next target points."""
        for key, radius, number in [
            ("previous_target_point", 14, 0),
            ("next_target_point", 14, 2),
            ("target_point", 18, 1),
        ]:
            target_point = self.data.get(key)
            if target_point is None:
                continue
            x_tp, y_tp = self._to_pixel(target_point[0][0], target_point[0][1])
            drawing.draw_circle_with_number(
                self.bev_image,
                x_tp,
                y_tp,
                colors.TP_DEFAULT_COLOR,
                radius=radius,
                number=number,
            )

    # --- Bounding boxes ---

    def _ego_bounding_box(self) -> None:
        """Draw the ego bounding box with its current speed."""
        ego_box = np.array(
            [
                int(
                    self.bev_image.shape[1]
                    * (
                        -self.config.bev_min_x_meter
                        / (self.config.bev_max_x_meter - self.config.bev_min_x_meter)
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
        self.bev_image = drawing.draw_box(
            self.bev_image,
            ego_box,
            color=colors.EGO_BB_COLOR,
            thickness=4,
        )

    def _class_color(
        self,
        class_index: int,
        brake: float | None = None,
    ) -> list[float]:
        """Color of a bounding box of the given class."""
        color = list(
            list(colors.TRANSFUSER_BOUNDING_BOX_COLORS.values())[class_index],
        )
        if brake is not None:
            color[1] = color[1] * (1.0 - brake)
        return color

    def _draw_boxes(
        self,
        boxes: jt.Float[npt.NDArray, "n d"],
        brake_shading: bool,
    ) -> None:
        """Draw bounding boxes given in the image system, colored by class."""
        for box in boxes:
            box = box.copy()
            box[:4] = box[:4] * self.scale_factor
            self.bev_image = drawing.draw_box(
                self.bev_image,
                box,
                color=self._class_color(
                    int(box[BoundingBoxIndex.CLASS]),
                    float(box[BoundingBoxIndex.BRAKE]) if brake_shading else None,
                ),
            )

    def _bounding_boxes(self) -> None:
        """Draw the ground-truth bounding boxes and their future footprints."""
        bounding_boxes = self.data.get("center_net_bounding_boxes")
        if bounding_boxes is not None:
            boxes = bounding_boxes.detach().cpu().numpy()[0]
            self._draw_boxes(
                boxes[boxes.sum(axis=-1) != 0.0],
                brake_shading=False,
            )

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
                color=colors.RADAR_COLOR,
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
        """Draw radar returns as markers sized by radial velocity."""
        min_r = self.radar_min_radius_pixel
        max_r = self.radar_max_radius_pixel
        pixels = self._to_pixels(np.stack((x, y), axis=1))
        radii = radius_offset + np.clip(
            min_r + (np.abs(velocity) / self.radar_velocity_max) * (max_r - min_r),
            min_r,
            max_r,
        ).astype(np.int32)
        for (px, py), rpx, vk in zip(pixels, radii, velocity, strict=True):
            # Ring = approaching, cross = receding.
            draw = drawing.draw_ring if vk > 0 else drawing.draw_cross
            draw(self.bev_image, int(px), int(py), int(rpx), color)

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
                color=colors.RADAR_DETECTION_COLOR,
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
            pixel_points = self._to_pixels(waypoints[: int(num_wps)])
            for point in pixel_points:
                cv2.circle(
                    self.bev_image,
                    (int(point[0]), int(point[1])),
                    radius=3,
                    color=colors.RADAR_DETECTION_COLOR,
                    thickness=-1,
                )
            cv2.polylines(
                self.bev_image,
                [pixel_points],
                isClosed=False,
                color=colors.RADAR_DETECTION_COLOR,
                thickness=1,
                lineType=cv2.LINE_AA,
            )

    # --- Meta panel ---

    def _extra_meta_lines(self) -> list[str]:
        """Additional meta-panel lines contributed by subclasses."""
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

        for attr_name in ["previous_target_point", "target_point", "next_target_point"]:
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

        start_x = 10
        start_y = 10
        line_height = 20
        gutter = 20
        num_columns = 4
        column_width = (self.meta_panel.shape[1] - 2 * start_x) // num_columns

        def name_and_value(text: str) -> tuple[str, str]:
            split_idx = text.find(" ")
            return text[:split_idx], text[split_idx:].strip()

        # Measuring a string costs as much as drawing it, and the layout below
        # measures each of them more than once.
        measured: dict[tuple[str, bool], int] = {}

        def text_width(text: str, font: ImageFont.FreeTypeFont) -> int:
            key = (text, font is font_bold)
            if key not in measured:
                bbox = draw.textbbox((0, 0), text, font=font)
                measured[key] = bbox[2] - bbox[0]
            return measured[key]

        def fits(text: str, width: int) -> bool:
            """Whether a name/value pair fits side by side in the given width."""
            if " " not in text:
                return text_width(text, font_regular) <= width - gutter
            name_part, value_part = name_and_value(text)
            name_width = text_width(name_part, font_bold)
            value_width = text_width(value_part, font_regular)
            return name_width + value_width <= width - gutter

        # Entries too wide for one column are laid out in double-width columns.
        wide_column_width = column_width * WIDE_COLUMN_SPAN
        grid_lines: list[str] = []
        wide_lines: list[str] = []
        for text in text_lines:
            target = grid_lines if fits(text, column_width) else wide_lines
            target.append(text)

        def draw_line(text: str, x: int, y: int, width: int) -> None:
            if " " not in text:
                draw.text((x, y), text, font=font_regular, fill=(0, 0, 0))
                return
            # Attribute name in bold, value right-aligned within the width.
            name_part, value_part = name_and_value(text)
            value_width = text_width(value_part, font_regular)
            draw.text((x, y), name_part, font=font_bold, fill=(0, 0, 0))
            draw.text(
                (x + width - value_width - gutter, y),
                value_part,
                font=font_regular,
                fill=(0, 0, 0),
            )

        for item_index, text in enumerate(grid_lines):
            row = item_index // num_columns
            col = item_index % num_columns
            draw_line(
                text,
                start_x + col * column_width,
                start_y + row * line_height,
                column_width,
            )

        wide_num_columns = num_columns // WIDE_COLUMN_SPAN
        wide_start_row = -(-len(grid_lines) // num_columns)
        for wide_index, text in enumerate(wide_lines):
            row = wide_start_row + wide_index // wide_num_columns
            col = wide_index % wide_num_columns
            draw_line(
                text,
                start_x + col * wide_column_width,
                start_y + row * line_height,
                wide_column_width,
            )

        self.meta_panel = np.array(img_pil)

    # --- Composition ---

    def _concatenate_all_perspectives_and_bev(
        self,
        border_size: int = 10,
        border_color: tuple[int, int, int] = (255, 255, 255),
    ) -> jt.UInt8[npt.NDArray, "h w 3"]:
        """Compose the BEV, the stacked perspectives and the meta panel.

        The perspectives are stacked vertically on the right, the BEV sits on
        the left, and the meta panel is appended at the bottom, all separated
        by borders.
        """
        lidar_image = np.ascontiguousarray(self.bev_image, dtype=np.uint8)

        if not self.perspectives:
            raise ValueError("No perspectives available")

        # Resize all perspectives to the BEV width.
        perspective_images: list[jt.UInt8[npt.NDArray, "h w 3"]] = []
        perspective_width = lidar_image.shape[1]
        for modality in self.perspective_modalities:
            if modality not in self.perspectives:
                continue
            img = np.ascontiguousarray(self.perspectives[modality], dtype=np.uint8)
            target_height = int(img.shape[0] * (perspective_width / img.shape[1]))
            perspective_images.append(
                cv2.resize(img, (perspective_width, target_height)),
            )

        stack_height = sum(img.shape[0] for img in perspective_images) + border_size * (
            len(perspective_images) - 1
        )
        bev_width = int(lidar_image.shape[1] * (stack_height / lidar_image.shape[0]))
        meta_panel = np.ascontiguousarray(self.meta_panel, dtype=np.uint8)
        width = 3 * border_size + bev_width + perspective_width
        meta_height = int(meta_panel.shape[0] * (width / meta_panel.shape[1]))

        # The borders are what the canvas is not painted over with.
        height = stack_height + 3 * border_size + meta_height
        canvas = np.empty((height, width, 3), dtype=np.uint8)
        cv2.rectangle(canvas, (0, 0), (width, height), border_color, thickness=-1)
        canvas[
            border_size : border_size + stack_height,
            border_size : border_size + bev_width,
        ] = cv2.resize(lidar_image, (bev_width, stack_height))

        perspective_left = 2 * border_size + bev_width
        row = border_size
        for img in perspective_images:
            canvas[
                row : row + img.shape[0],
                perspective_left : perspective_left + perspective_width,
            ] = img
            row += img.shape[0] + border_size

        canvas[stack_height + 3 * border_size :] = cv2.resize(
            meta_panel,
            (width, meta_height),
        )
        return canvas
