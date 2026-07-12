"""Visualizer for the raw predictions of a single TransFuser network."""

import typing

import cv2
import jaxtyping as jt
import numpy as np
import numpy.typing as npt
import torch
from PIL import Image, ImageDraw, ImageFont

import lead.common.constants as constants
from lead.common.constants import RadarLabels, TransfuserBoundingBoxIndex
from lead.config import LeadConfig
from lead.evaluation.inference.ensemble import EnsemblePrediction
from lead.policy.transfuser.transfuser import Prediction
from lead.policy.transfuser.visualization import viz_utils
from lead.policy.transfuser.visualization.ground_truth_visualizer import (
    FONT_BOLD,
    GroundTruthVisualizer,
)


class PredictionVisualizer(GroundTruthVisualizer):
    """Visualizes model predictions instead of ground-truth labels.

    Overwrites the BEV semantic, route, waypoint, bounding-box and radar
    drawing hooks to read from the prediction, and renders the predicted
    semantic segmentation as the second perspective.
    """

    def __init__(
        self,
        lead_config: LeadConfig,
        data: dict[str, typing.Any],
        prediction: Prediction | EnsemblePrediction,
    ) -> None:
        """Initialize the visualizer from one batched sample and its prediction.

        Args:
            lead_config: Root config tree.
            data: Dictionary containing batched input data tensors.
            prediction: Model outputs to visualize.
        """
        super().__init__(lead_config=lead_config, data=data)
        self.prediction: Prediction | EnsemblePrediction = prediction

    def _draw_bev(self) -> None:
        """Draw all BEV overlays for the prediction view."""
        self._bev_semantic()
        self._target_point()
        if self.config.use_planning_decoder:
            self._future_waypoints()
            self._route()
        self._ego_bounding_box()
        self._bounding_boxes()
        self._target_point()
        self._radars()

    def _perspective_image(
        self,
        modality: str,
    ) -> jt.UInt8[npt.NDArray, "h w 3"] | None:
        """Build one perspective image; the semantic view uses the prediction.

        Args:
            modality: Perspective modality, ``"rgb"`` or ``"semantic"``.

        Returns:
            The perspective image, or None when the modality is unavailable.
        """
        if modality == "rgb":
            image = super()._perspective_image("rgb")
            if image is not None:
                image = self._draw_target_speed(image)
            return image

        pred_semantic = self.prediction.pred_semantic
        if pred_semantic is None:
            return None
        semantic = pred_semantic[0].argmax(dim=0).detach().cpu().numpy()
        return self._semantic_to_color(
            np.ascontiguousarray(semantic).astype(np.uint8),
        )

    def _draw_target_speed(
        self,
        image: jt.UInt8[npt.NDArray, "h w 3"],
    ) -> jt.UInt8[npt.NDArray, "h w 3"]:
        """Overlay the predicted target speed at the lower middle of the image.

        Args:
            image: RGB perspective image.

        Returns:
            The image with the target speed text drawn.
        """
        if self.prediction.pred_target_speed_scalar is None:
            return image
        pred_speed = float(
            self.prediction.pred_target_speed_scalar.detach().cpu().float()[0],
        )
        img_height, img_width = image.shape[:2]
        text = f"Target Speed: {pred_speed:.1f} m/s"

        img_pil = Image.fromarray(image)
        draw = ImageDraw.Draw(img_pil)
        try:
            font = ImageFont.truetype(FONT_BOLD, 20)
        except OSError:
            font = ImageFont.load_default()

        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = (img_width - text_width) // 2
        y = img_height - text_height - 20

        # Background rectangle for better visibility.
        padding_horizontal = 6
        padding_top = 3
        padding_bottom = 6
        draw.rectangle(
            [
                (x - padding_horizontal, y - padding_top),
                (x + text_width + padding_horizontal, y + text_height + padding_bottom),
            ],
            fill=(0, 0, 0, 180),
        )
        draw.text((x, y), text, font=font, fill=(255, 255, 255))
        return np.array(img_pil)

    def _bev_semantic(self) -> None:
        """Overlay the predicted BEV semantic map."""
        pred_bev_semantic = self.prediction.pred_bev_semantic
        if pred_bev_semantic is None:
            return
        labels = pred_bev_semantic.argmax(dim=1)[0].detach().cpu().numpy()
        self._overlay_bev_semantic(labels)

    def _route(self) -> None:
        """Draw the predicted route as circles."""
        pred_route = self.prediction.pred_route
        if pred_route is None:
            return
        wps = pred_route.detach().cpu().float().numpy()[0]
        for i in range(len(wps) - 1):
            x1 = int(wps[i][0] * self.loc_pixels_per_meter + self.origin[0])
            y1 = int(wps[i][1] * self.loc_pixels_per_meter + self.origin[1])
            color = viz_utils.lighter_shade(
                constants.PREDICTION_ROUTE_COLOR,
                i,
                len(wps),
            )
            cv2.circle(
                self.bev_image,
                (x1, y1),
                radius=constants.PREDICTION_ROUTE_RADIUS,
                color=color,
                thickness=-1,
            )

    def _future_waypoints(self) -> None:
        """Draw the predicted future waypoints."""
        pred_waypoints = self.prediction.pred_future_waypoints
        if pred_waypoints is None:
            return
        wps = pred_waypoints.detach().cpu().float().numpy()[0]
        for i, wp in enumerate(wps):
            wp_x = wp[0] * self.loc_pixels_per_meter + self.origin[0]
            wp_y = wp[1] * self.loc_pixels_per_meter + self.origin[1]
            color = viz_utils.lighter_shade(
                constants.PREDICTION_WAYPOINT_COLOR,
                i,
                len(wps),
            )
            cv2.circle(
                self.bev_image,
                (int(wp_x), int(wp_y)),
                radius=constants.PREDICTION_WAYPOINT_RADIUS,
                color=color,
                thickness=-1,
            )

    def _bounding_boxes(self) -> None:
        """Draw the predicted bounding boxes above the confidence threshold."""
        if not self.config.detect_boxes:
            return
        prediction = self.prediction
        assert isinstance(prediction, Prediction)
        if prediction.pred_bounding_box is None:
            return
        boxes = prediction.pred_bounding_box.pred_bounding_box_image_system[0]
        if boxes is None:
            return
        for box in boxes:
            if (
                box[TransfuserBoundingBoxIndex.SCORE]
                < self.config.bb_confidence_threshold
            ):
                continue
            box = box.copy()
            inv_brake = 1.0 - box[TransfuserBoundingBoxIndex.BRAKE]
            color_box = list(
                list(constants.TRANSFUSER_BOUNDING_BOX_COLORS.values())[
                    int(box[TransfuserBoundingBoxIndex.CLASS])
                ],
            )
            color_box[1] = color_box[1] * inv_brake
            box[:4] = box[:4] * self.scale_factor
            self.bev_image = viz_utils.draw_box(
                self.bev_image,
                box,
                color=color_box,
            )

    def _radar_detections(self) -> None:
        """Draw the predicted radar detections."""
        pred_radar_predictions = self.prediction.pred_radar_predictions
        if pred_radar_predictions is None:
            return
        radar_predictions = pred_radar_predictions[0].cpu()
        valid_mask = torch.sigmoid(radar_predictions[:, RadarLabels.VALID]) > 0.5
        valid_predictions = radar_predictions[valid_mask].detach().float().numpy()
        if valid_predictions.shape[0] == 0:
            return
        self._draw_radar_returns(
            valid_predictions[:, 0],
            valid_predictions[:, 1],
            np.nan_to_num(valid_predictions[:, 2], nan=0.0),
            color=constants.RADAR_DETECTION_COLOR,
            radius_offset=1,
        )

    def _extra_meta_lines(self) -> list[str]:
        """Add the predicted target speed to the meta panel.

        Returns:
            Extra text lines in ``"<name> <value>"`` format.
        """
        lines = super()._extra_meta_lines()
        if self.prediction.pred_target_speed_scalar is not None:
            pred_speed = float(
                self.prediction.pred_target_speed_scalar.detach().cpu().float()[0],
            )
            lines.append(f"pred_target_speed {pred_speed:.2f} m/s")
        return lines
