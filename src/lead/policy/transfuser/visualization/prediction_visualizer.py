"""Visualizer for the raw predictions of a single TransFuser network."""

import jaxtyping as jt
import numpy as np
import numpy.typing as npt
import torch
from PIL import Image, ImageDraw, ImageFont

from lead.common.constants import RadarLabels
from lead.config import LeadConfig
from lead.config.policy.transfuser.label_classes import BoundingBoxIndex
from lead.policy.transfuser.dataloader.sample import TransfuserForwardBatch
from lead.policy.transfuser.transfuser import Prediction
from lead.policy.transfuser.visualization import colors
from lead.policy.transfuser.visualization.ground_truth_visualizer import (
    FONT_BOLD,
    GroundTruthVisualizer,
)


class PredictionVisualizer(GroundTruthVisualizer):
    """Visualizes model predictions rather than ground-truth labels.

    Overwrites the BEV semantic, route, waypoint, bounding-box and radar
    drawing hooks to read from the prediction, and renders the predicted
    semantic segmentation as the second perspective.
    """

    def __init__(
        self,
        lead_config: LeadConfig,
        data: TransfuserForwardBatch,
        prediction: Prediction,
    ) -> None:
        """Initialize the visualizer from one batched sample and its prediction.

        Args:
            lead_config: Root config tree.
            data: Dictionary containing batched input data tensors.
            prediction: Model outputs to visualize.
        """
        super().__init__(lead_config=lead_config, data=data)
        self.prediction: Prediction = prediction

    def _draw_bev(self) -> None:
        """Draw all BEV overlays for the prediction view."""
        self._bev_semantic()
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
        """Build one perspective image; the semantic and depth views use the prediction."""
        if modality == "rgb":
            image = super()._perspective_image("rgb")
            if image is not None:
                image = self._draw_target_speed(image)
            return image

        if modality == "depth":
            pred_depth = self.prediction.depth
            if pred_depth is None:
                return None
            return self._depth_to_color(
                pred_depth[0].detach().cpu().float().numpy(),
            )

        pred_semantic = self.prediction.semantic
        if pred_semantic is None:
            return None
        semantic = pred_semantic[0].argmax(dim=0).detach().cpu().numpy()
        return self._semantic_to_color(
            np.ascontiguousarray(semantic).astype(np.uint8),
        )

    def _target_speed(self) -> float | None:
        """The predicted target speed to display."""
        if self.prediction.target_speed_scalar is None:
            return None
        return float(
            self.prediction.target_speed_scalar.detach().cpu().float().flatten()[0],
        )

    def _draw_target_speed(
        self,
        image: jt.UInt8[npt.NDArray, "h w 3"],
    ) -> jt.UInt8[npt.NDArray, "h w 3"]:
        """Overlay the predicted target speed at the lower middle of the image."""
        pred_speed = self._target_speed()
        if pred_speed is None:
            return image
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
        pred_bev_semantic = self.prediction.bev_semantic
        if pred_bev_semantic is None:
            return
        labels = pred_bev_semantic.argmax(dim=1)[0].detach().cpu().numpy()
        self._overlay_bev_semantic(labels.astype(np.uint8))

    def _route(self) -> None:
        """Draw the predicted route as circles."""
        pred_route = self.prediction.route
        if pred_route is None:
            return
        self._draw_waypoints(
            pred_route.detach().cpu().float().numpy()[0],
            colors.PREDICTION_ROUTE_COLOR,
            colors.PREDICTION_ROUTE_RADIUS,
        )

    def _future_waypoints(self) -> None:
        """Draw the predicted future waypoints."""
        pred_waypoints = self.prediction.future_waypoints
        if pred_waypoints is None:
            return
        self._draw_waypoints(
            pred_waypoints.detach().cpu().float().numpy()[0],
            colors.PREDICTION_WAYPOINT_COLOR,
            colors.PREDICTION_WAYPOINT_RADIUS,
        )

    def _bounding_boxes(self) -> None:
        """Draw the predicted bounding boxes above the confidence threshold."""
        if not self.config.detect_boxes:
            return
        if self.prediction.bounding_box is None:
            return
        boxes = self.prediction.bounding_box.bounding_boxes_in_image_frame[0]
        if boxes is None:
            return
        boxes = np.asarray(boxes)
        confident = boxes[
            boxes[:, BoundingBoxIndex.SCORE] >= self.config.box_confidence_threshold
        ]
        self._draw_boxes(confident, brake_shading=True)

    def _radar_detections(self) -> None:
        """Draw the predicted radar detections."""
        pred_radar_predictions = self.prediction.radar_predictions
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
            color=colors.RADAR_DETECTION_COLOR,
            radius_offset=1,
        )

    def _extra_meta_lines(self) -> list[str]:
        """Add the predicted target speed to the meta panel."""
        lines = super()._extra_meta_lines()
        pred_speed = self._target_speed()
        if pred_speed is not None:
            lines.append(f"pred_target_speed {pred_speed:.2f} m/s")
        return lines
