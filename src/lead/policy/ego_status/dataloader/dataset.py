"""Ego-status training dataset: navigation features and planning targets only.

No sensor stream is ever read; the scene-data spine (ego state, driving meta,
target points) plus the future iterations cover everything this policy needs.
"""

import typing

import numpy as np

from lead.api.abstract_dataset import AbstractPolicyDataset
from lead.api.py123d_log_api import (
    LOCALIZED_EGO_STATE_KEY,
    se3_matrix_to_localized_pose,
)
from lead.api.training_sample import SamplePart
from lead.common import geometry
from lead.config import LeadConfig
from lead.log_reader import SceneData, SceneLoadingSpec, carla_decoding
from lead.log_reader.scene_loader import SceneLoader
from lead.policy.ego_status.dataloader.sample import EgoStatusTrainingSample


class EgoStatusDataset(AbstractPolicyDataset):
    """Training dataset of the ego-status baseline."""

    sample_class = EgoStatusTrainingSample

    def __init__(self, lead_config: LeadConfig, scene_loader: SceneLoader) -> None:
        """Construct the dataset, opening the cache store if configured.

        Args:
            lead_config: Root config tree.
            scene_loader: Loader over the scenes to train on, built by the
                policy (see ``EgoStatus.build_scene_loader``).
        """
        super().__init__(lead_config, scene_loader, lead_config.policy.ego_status)

    @property
    def cache_finger_print(self) -> dict[str, str]:
        """Inherited, see superclass; empty, as this policy caches nothing."""
        return {}

    def get_sample_parts(self) -> dict[str, SamplePart]:
        config = self.lead_config.policy.ego_status
        return {
            "ego_features": SamplePart(
                reads=SceneLoadingSpec(
                    ego_pose_tick_ages=config.past_ego_pose_tick_ages,
                ),
                builds=self._build_ego_features,
            ),
            "planning_targets": SamplePart(
                reads=SceneLoadingSpec(
                    future_iterations=config.future_ego_pose_iterations,
                ),
                # Never cached: positions are cheap to read live.
                builds=self._build_planning_targets,
            ),
        }

    # --- Builders ---
    def _build_ego_features(self, scene_data: SceneData) -> dict[str, typing.Any]:
        """The navigation inputs of one scene."""
        assert scene_data.target_point is not None
        assert scene_data.ego_state is not None
        return {
            "target_point": scene_data.target_point.astype(np.float32),
            "speed": carla_decoding.carla_forward_speed(scene_data.ego_state),
        }

    def _build_planning_targets(self, scene_data: SceneData) -> dict[str, typing.Any]:
        """The ego's future waypoints."""
        driving_meta = scene_data.driving_meta
        assert driving_meta is not None
        assert scene_data.ego_state is not None

        ego_position, _ = carla_decoding.carla_ego_pose(scene_data.ego_state)
        _, ego_yaw = se3_matrix_to_localized_pose(
            np.asarray(driving_meta[LOCALIZED_EGO_STATE_KEY], dtype=np.float64),
        )

        future_iterations = (
            self.lead_config.policy.ego_status.future_ego_pose_iterations
        )
        waypoints = []
        future_ego_states = scene_data.future_ego_states or {}
        for future_iteration in future_iterations:
            future_state = future_ego_states.get(future_iteration)
            # Truncating would yield a shorter label than the loss expects.
            if future_state is None:
                raise ValueError(
                    f"Missing future iteration {future_iteration} of "
                    f"{future_iterations}; the scene filter must "
                    f"only enumerate scenes whose full future is available.",
                )
            future_position, _ = carla_decoding.carla_ego_pose(future_state)
            waypoints.append(
                geometry.to_local_frame_2d(future_position, ego_position, ego_yaw),
            )
        return {
            "future_waypoints": np.array(waypoints, dtype=np.float32).reshape(-1, 2),
        }
