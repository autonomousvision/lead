"""Base class for per-modality 123D recorders."""

import abc
import typing

from py123d.datatypes import BaseModality, EgoStateSE3, Timestamp

if typing.TYPE_CHECKING:
    from lead.expert.logs_writing import ExpertData


class BaseRecorder(abc.ABC):
    """Converts one CARLA modality stream into py123d modalities.

    A recorder builds its stream's static metadata once at construction and is
    called whenever its storage frequency is met to convert the live CARLA
    state into py123d modality objects.
    """

    def __init__(
        self,
        expert: "ExpertData",
        perturbated: bool = False,
        store_freq: int = 1,
    ) -> None:
        """Initialize recorder with a handle to the running expert.

        Args:
            expert: The expert agent owning the CARLA state to record.
            perturbated: If true record the perturbated sensor views instead
                of the normal ones.
            store_freq: Storage period of the stream in simulator steps.
        """
        self.expert = expert
        self.perturbated = perturbated
        self.store_freq = store_freq
        # Suffix of the perturbated sensor keys in the post-tick input data.
        self.key_suffix = "_perturbated" if perturbated else ""
        # Rig pose perturbation of the recorded views; zero for the normal rig.
        self.perturbation_translation = (
            expert.perturbation_translation if perturbated else 0.0
        )
        self.perturbation_rotation = (
            expert.perturbation_rotation if perturbated else 0.0
        )

    @abc.abstractmethod
    def record(
        self,
        input_data: dict,
        timestamp: Timestamp,
        ego_state: EgoStateSE3,
    ) -> list[BaseModality]:
        """Convert the current CARLA state into py123d modalities.

        Args:
            input_data: Post-tick sensor data from CARLA.
            timestamp: Current simulation timestamp.
            ego_state: Ego state of the current tick, for global-frame poses.

        Returns:
            The modalities to write for this tick (may be several, e.g. one
            camera frame per camera ID).
        """
        raise NotImplementedError
