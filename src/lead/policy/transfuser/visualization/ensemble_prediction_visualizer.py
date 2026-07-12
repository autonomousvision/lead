"""Visualizer for the aggregated predictions of a model ensemble at test time."""

import typing

import lead.common.constants as constants
from lead.config import LeadConfig
from lead.evaluation.inference.ensemble import EnsemblePrediction
from lead.policy.transfuser.visualization import viz_utils
from lead.policy.transfuser.visualization.prediction_visualizer import (
    PredictionVisualizer,
)


class EnsemblePredictionVisualizer(PredictionVisualizer):
    """Visualizes an :class:`EnsemblePrediction` during closed-loop evaluation.

    Overwrites the bounding-box hook for the ensemble's NMS-aggregated
    :class:`PredictedBoundingBox` list and uses the compact test-time meta
    panel.
    """

    meta_panel_height: typing.ClassVar[int] = 115

    def __init__(
        self,
        lead_config: LeadConfig,
        data: dict[str, typing.Any],
        prediction: EnsemblePrediction,
    ) -> None:
        """Initialize the visualizer from one batched sample and its prediction.

        Args:
            lead_config: Root config tree.
            data: Dictionary containing batched input data tensors.
            prediction: Aggregated ensemble outputs to visualize.
        """
        super().__init__(lead_config=lead_config, data=data, prediction=prediction)

    def _draw_bev(self) -> None:
        """Draw all BEV overlays for the test-time inference view."""
        self._bev_semantic()
        self._route()
        self._target_point()
        self._radars()
        self._ego_bounding_box()
        self._bounding_boxes()

    def _bounding_boxes(self) -> None:
        """Draw the ensembled bounding boxes above the confidence threshold."""
        if not self.config.detect_boxes:
            return
        prediction = self.prediction
        assert isinstance(prediction, EnsemblePrediction)
        if prediction.pred_bounding_box_image_system is None:
            return
        for box in prediction.pred_bounding_box_image_system:
            if box.score < self.config.bb_confidence_threshold:
                continue
            inv_brake = 1.0 - box.brake
            color_box = list(
                list(constants.TRANSFUSER_BOUNDING_BOX_COLORS.values())[box.clazz],
            )
            color_box[1] = color_box[1] * inv_brake
            self.bev_image = viz_utils.draw_box(
                self.bev_image,
                box.scale(self.scale_factor),
                color=color_box,
            )
