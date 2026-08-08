import typing

import carla

from lead.api.abstract_driving_agent import AbstractDrivingAgent
from lead.evaluation.inference.trackers import WaypointTracker
from lead.policy.ego_status.dataloader.sample import EgoStatusPrediction


def get_entry_point():  # dead: disable
    return "EgoStatusAgent"


class EgoStatusAgent(AbstractDrivingAgent):
    """Driving agent wrapping the ego-status baseline policy.

    The policy predicts only waypoints; the waypoint tracker turns them into
    vehicle controls.
    """

    def setup_policy(self, checkpoint_dir: str) -> None:
        """Build the tracker turning the predicted waypoints into controls.

        Args:
            checkpoint_dir: Directory holding the trained model checkpoint.
        """
        del checkpoint_dir
        self.waypoint_tracker = WaypointTracker(self.lead_config)

    def compute_control(
        self,
        prediction: EgoStatusPrediction,
        features: dict[str, typing.Any],
    ) -> carla.VehicleControl:
        """Track the predicted waypoints into a vehicle control.

        Args:
            prediction: Raw prediction of the ego-status policy.
            features: The batched model inputs the prediction was computed on.

        Returns:
            The vehicle control to apply this step.
        """
        steer, throttle, brake = self.waypoint_tracker.step(
            prediction["waypoints"],
            features["speed"].unsqueeze(1),
        )
        return carla.VehicleControl(
            steer=float(steer),
            throttle=float(throttle),
            brake=float(brake),
        )
