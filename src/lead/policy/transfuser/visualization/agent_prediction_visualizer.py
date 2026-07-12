"""Visualizer for agent predictions, adding the predicted vehicle controls."""

import typing

from lead.config import LeadConfig
from lead.evaluation.inference.ensemble import AgentPrediction
from lead.policy.transfuser.visualization.ensemble_prediction_visualizer import (
    EnsemblePredictionVisualizer,
)


class AgentPredictionVisualizer(EnsemblePredictionVisualizer):
    """Visualizes an :class:`AgentPrediction` during closed-loop evaluation.

    Renders like the ensemble visualizer and additionally lists the predicted
    steer, throttle and brake commands on the meta panel.
    """

    def __init__(
        self,
        lead_config: LeadConfig,
        data: dict[str, typing.Any],
        prediction: AgentPrediction,
    ) -> None:
        """Initialize the visualizer from one batched sample and its prediction.

        Args:
            lead_config: Root config tree.
            data: Dictionary containing batched input data tensors.
            prediction: Agent outputs to visualize.
        """
        super().__init__(lead_config=lead_config, data=data, prediction=prediction)
        self.agent_prediction: AgentPrediction = prediction

    def _extra_meta_lines(self) -> list[str]:
        """Add the predicted vehicle controls to the meta panel.

        Returns:
            Extra text lines in ``"<name> <value>"`` format.
        """
        lines = super()._extra_meta_lines()
        lines.append(f"pred_steer {self.agent_prediction.steer:.2f}")
        lines.append(f"pred_throttle {self.agent_prediction.throttle:.2f}")
        lines.append(f"pred_brake {self.agent_prediction.brake:.2f}")
        return lines
