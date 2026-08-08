"""Visualizer for agent predictions, adding the predicted vehicle controls."""

import typing

from lead.config import LeadConfig
from lead.policy.transfuser.dataloader.sample import TransfuserForwardBatch
from lead.policy.transfuser.transfuser import AgentPrediction
from lead.policy.transfuser.visualization.prediction_visualizer import (
    PredictionVisualizer,
)


class AgentPredictionVisualizer(PredictionVisualizer):
    """Visualizes an :class:`AgentPrediction` during closed-loop evaluation.

    Renders the model predictions and additionally lists the predicted steer,
    throttle and brake commands on the compact test-time meta panel.
    """

    meta_panel_height: typing.ClassVar[int] = 115

    def __init__(
        self,
        lead_config: LeadConfig,
        data: TransfuserForwardBatch,
        prediction: AgentPrediction,
    ) -> None:
        """Initialize the visualizer from one batched sample and its prediction.

        Args:
            lead_config: Root config tree.
            data: Dictionary containing batched input data tensors.
            prediction: Agent outputs to visualize.
        """
        super().__init__(
            lead_config=lead_config,
            data=data,
            prediction=prediction.prediction,
        )
        self.agent_prediction: AgentPrediction = prediction

    def _draw_bev(self) -> None:
        """Draw all BEV overlays for the test-time inference view."""
        self._bev_semantic()
        self._route()
        self._target_point()
        self._radars()
        self._ego_bounding_box()
        self._bounding_boxes()

    def _target_speed(self) -> float | None:
        """The post-processed target speed the agent actually tracks."""
        if self.agent_prediction.target_speed is None:
            return None
        return float(
            self.agent_prediction.target_speed.detach().cpu().float().flatten()[0],
        )

    def _extra_meta_lines(self) -> list[str]:
        """Add the predicted vehicle controls to the meta panel."""
        lines = super()._extra_meta_lines()
        lines.append(f"pred_steer {self.agent_prediction.steer:.2f}")
        lines.append(f"pred_throttle {self.agent_prediction.throttle:.2f}")
        lines.append(f"pred_brake {self.agent_prediction.brake:.2f}")
        return lines
