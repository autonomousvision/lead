"""Recorder for the ego-state modality stream."""

import typing

from py123d.datatypes import DynamicStateSE3, EgoStateSE3, Timestamp
from py123d.geometry import Vector3D

from lead.api import py123d_log_api
from lead.common import carla_to_123d

if typing.TYPE_CHECKING:
    from lead.expert.logs_writing import ExpertData


class EgoStateRecorder:
    """Extracts the ego vehicle state in ISO 8855 coordinates.

    Unlike the other recorders, the ego state is extracted first and passed to
    them, since camera/lidar poses are composed against the ego pose.
    """

    def __init__(self, expert: "ExpertData") -> None:
        """Initialize recorder with a handle to the running expert.

        Args:
            expert: The expert agent owning the CARLA state to record.
        """
        self.expert = expert
        self.ego_metadata = py123d_log_api.CARLA_LINCOLN_MKZ_2020_METADATA

    def extract(self, timestamp: Timestamp) -> EgoStateSE3:
        """Extract the current ego state from CARLA.

        Args:
            timestamp: Current simulation timestamp.

        Returns:
            EgoStateSE3 with position, velocity, and acceleration.
        """
        ego_vehicle = self.expert.ego_vehicle
        ego_velocity = ego_vehicle.get_velocity()
        ego_acceleration = ego_vehicle.get_acceleration()
        ego_angular_velocity = ego_vehicle.get_angular_velocity()

        # Velocity and acceleration have Y-axis inverted for ISO 8855.
        # Angular velocity does NOT have Y-axis inverted.
        dynamic_state = DynamicStateSE3(
            velocity=Vector3D.from_list(
                [ego_velocity.x, -ego_velocity.y, ego_velocity.z],
            ),
            acceleration=Vector3D.from_list(
                [ego_acceleration.x, -ego_acceleration.y, ego_acceleration.z],
            ),
            angular_velocity=Vector3D.from_list(
                [
                    ego_angular_velocity.x,
                    ego_angular_velocity.y,
                    ego_angular_velocity.z,
                ],
            ),
        )

        return EgoStateSE3.from_center(
            center_se3=carla_to_123d.carla_actor_to_se3(ego_vehicle),
            metadata=self.ego_metadata,
            timestamp=timestamp,
            dynamic_state_se3=dynamic_state,
        )
