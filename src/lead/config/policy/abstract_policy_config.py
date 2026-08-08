"""The config contract every policy's section of the tree fulfills."""

import abc
from pathlib import Path

from py123d.datatypes.sensors.base_camera import CameraID

from lead.config.node import ConfigNode, overridable_property


class AbstractPolicyConfig(ConfigNode, abc.ABC):
    """Contract of a policy's config section, the ``policy.<name>`` node.

    A policy's section subclasses this, and
    :meth:`~lead.api.abstract_policy.AbstractPolicy.get_policy_config` returns
    it; policy-agnostic code reads these values only through that method.

    The temporal window a policy reads is declared per modality: a
    ``_length_s`` in seconds away from the anchor, and a ``_frequency`` in Hz
    to sample that span at. Both are physical, so a config keeps its meaning
    when the simulator's tick rate changes or another dataset is logged at a
    different rate; the tick counts fall out of ``carla_fps``. Past spans
    include the anchor, so a past length of 0 means the anchor alone; future
    spans exclude it, since the anchor is the present rather than a
    prediction. A frequency must divide the simulator's tick rate evenly, and
    the stride it implies must be a multiple of the modality's storage period
    in the dataset -- the expert renders the camera streams only on save
    ticks, so RGB cannot be sampled faster than ``camera_store_freq`` allows
    however fast the simulator ran.
    """

    # --- Required of every policy ---
    @property
    @abc.abstractmethod
    def cache_store_dir_name(self) -> str:
        """Directory name of this policy's cache store under the dataset root.

        Which tensors the store holds and the codec each is written with are
        declared by the dataset's cacheable builders.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def input_cameras(self) -> list[CameraID]:
        """The cameras this policy ingests, in left-to-right stitch order.

        Returns:
            The camera ids, empty for a policy that reads no cameras.
        """
        raise NotImplementedError

    @overridable_property
    def cache_store_root(self) -> str:
        """Cache-store root; holds one subdir per view (``normal_view/``, ``perturbated_view/``)."""
        return str(
            Path(self._root.training.data.py123d_data_root) / self.cache_store_dir_name,
        )

    @property
    def required_scene_modalities(self) -> list[str]:
        """py123d modalities a scene must hold to be enumerated for this policy.

        The ego-state and driving-meta spine by default; a policy reading
        sensors extends this with the streams its sample parts index.
        """
        return ["ego_state_se3", "custom.driving_meta@initial"]

    # --- Temporal window ---
    # The per-modality frequencies default to the fastest the dataset offers,
    # i.e. every stored sample, so the defaults hold at any tick rate. Only
    # the future rate is a model choice rather than a dataset limit.

    # --- Past: ego pose (from the per-tick driving-meta stream) ---
    # The sweep alignment already widens the pose read to cover whatever sweep
    # ages it needs, so this is the *extra* history a model input wants on top.
    past_ego_pose_length_s: float = 0.0

    @overridable_property
    def past_ego_pose_frequency(self) -> int:
        """Rate in Hz to sample the ego-pose history at; every tick by default."""
        return self._fastest_frequency(1)

    # --- Past: lidar sweeps ---
    past_lidar_length_s: float = 0.2

    @overridable_property
    def past_lidar_frequency(self) -> int:
        """Rate in Hz to sample the lidar history at; every stored sweep by default."""
        return self._fastest_frequency(
            self._root.expert.data_collection.lidar_store_freq,
        )

    # --- Past: radar sweeps ---
    past_radar_length_s: float = 0.2

    @overridable_property
    def past_radar_frequency(self) -> int:
        """Rate in Hz to sample the radar history at; every stored sweep by default."""
        return self._fastest_frequency(
            self._root.expert.data_collection.radar_store_freq,
        )

    # --- Past: RGB cameras; anchor-locked to the save ticks ---
    past_rgb_length_s: float = 0.0

    @overridable_property
    def past_rgb_frequency(self) -> int:
        """Rate in Hz to sample the camera history at; every stored frame by default."""
        return self._fastest_frequency(
            self._root.expert.data_collection.camera_store_freq,
        )

    # --- Future: ego pose, the planning labels ---
    future_ego_pose_length_s: float = 2.0
    # A model choice, not a dataset limit: how densely to predict the plan.
    future_ego_pose_frequency: int = 4

    def _fastest_frequency(self, store_freq: int) -> int:
        """The highest rate in Hz the dataset stores one modality at.

        Args:
            store_freq: Storage period of the modality in the dataset, in ticks.

        Returns:
            The rate in Hz.
        """
        return self._root.expert.simulation.carla_fps // store_freq

    @property
    def past_window_num_iterations(self) -> int:
        """Ticks of past the widest modality window spans, sizing the runtime queues."""
        return max(
            self._num_iterations("past_ego_pose", self.past_ego_pose_length_s),
            self._num_iterations("past_lidar", self.past_lidar_length_s),
            self._num_iterations("past_radar", self.past_radar_length_s),
            self._num_iterations("past_rgb", self.past_rgb_length_s),
        )

    @property
    def required_history_num_iterations(self) -> int:
        """Ticks of past a scene must provide to be usable: the camera history.

        Only the cameras are exact; sweeps and poses degrade to whatever a
        route's start offers, the same way driving from the first tick does.
        """
        return self._num_iterations("past_rgb", self.past_rgb_length_s)

    @property
    def future_window_num_iterations(self) -> int:
        """Ticks of future a scene must provide, the planning labels by default."""
        return self.future_ego_pose_num_iterations

    @property
    def past_lidar_num_iterations(self) -> int:
        """Ticks of lidar sweep history, the anchor excluded."""
        return self._num_iterations("past_lidar", self.past_lidar_length_s)

    @property
    def past_radar_num_iterations(self) -> int:
        """Ticks of radar sweep history, the anchor excluded."""
        return self._num_iterations("past_radar", self.past_radar_length_s)

    @property
    def past_ego_pose_tick_ages(self) -> tuple[int, ...]:
        """Tick ages of the ego-pose history, 0 being the anchor."""
        return self._past_tick_ages(
            "past_ego_pose",
            self.past_ego_pose_length_s,
            self.past_ego_pose_frequency,
            store_freq=1,
        )

    @property
    def past_lidar_tick_ages(self) -> tuple[int, ...]:
        """Tick ages of the lidar sweep window, 0 being the anchor."""
        return self._past_tick_ages(
            "past_lidar",
            self.past_lidar_length_s,
            self.past_lidar_frequency,
            store_freq=self._root.expert.data_collection.lidar_store_freq,
        )

    @property
    def past_radar_tick_ages(self) -> tuple[int, ...]:
        """Tick ages of the radar sweep window, 0 being the anchor."""
        return self._past_tick_ages(
            "past_radar",
            self.past_radar_length_s,
            self.past_radar_frequency,
            store_freq=self._root.expert.data_collection.radar_store_freq,
        )

    @property
    def past_rgb_tick_ages(self) -> tuple[int, ...]:
        """Tick ages of the RGB camera history, 0 being the anchor."""
        return self._past_tick_ages(
            "past_rgb",
            self.past_rgb_length_s,
            self.past_rgb_frequency,
            store_freq=self._root.expert.data_collection.camera_store_freq,
        )

    @property
    def future_ego_pose_num_iterations(self) -> int:
        """Ticks of future ego-pose every training scene must provide."""
        return self._num_iterations("future_ego_pose", self.future_ego_pose_length_s)

    @property
    def future_ego_pose_iterations(self) -> tuple[int, ...]:
        """Scene iterations ahead of the anchor the planning labels are read at."""
        stride = self._stride("future_ego_pose", self.future_ego_pose_frequency, 1)
        return tuple(range(stride, self.future_ego_pose_num_iterations + 1, stride))

    @property
    def num_ego_pose_prediction(self) -> int:
        """Number of ego-pose waypoints the policy predicts.

        Derived from the future span and its stride, never independently
        configurable, so the model's output width and the scene filter's
        future fetch can never disagree.
        """
        return len(self.future_ego_pose_iterations)

    def _num_iterations(self, field: str, length_s: float) -> int:
        """The tick count one span covers.

        Args:
            field: Field-name stem of the span, for the error message.
            length_s: Length of the span in seconds.

        Returns:
            The span in simulator ticks.

        Raises:
            ValueError: If the span is not a whole number of ticks.
        """
        carla_fps = self._root.expert.simulation.carla_fps
        num_iterations = length_s * carla_fps
        if num_iterations < 0 or abs(num_iterations - round(num_iterations)) > 1e-9:
            raise ValueError(
                f"{field}_length_s={length_s} s is not a whole number of ticks "
                f"at the simulator's {carla_fps} Hz tick rate.",
            )
        return round(num_iterations)

    def _past_tick_ages(
        self,
        field: str,
        length_s: float,
        frequency: int,
        store_freq: int,
    ) -> tuple[int, ...]:
        """The tick ages one modality's past window covers, the anchor included.

        Args:
            field: Field-name stem of the modality, for the error messages.
            length_s: Length of the window in seconds.
            frequency: Rate in Hz to sample the window at.
            store_freq: Storage period of the modality in the dataset, in ticks.

        Returns:
            The tick ages to read, ascending, always starting at the anchor.
        """
        stride = self._stride(field, frequency, store_freq)
        return tuple(range(0, self._num_iterations(field, length_s) + 1, stride))

    def _stride(self, field: str, frequency: int, store_freq: int) -> int:
        """Simulator ticks between samples of one modality at a target frequency.

        Args:
            field: Field-name stem of the modality, for the error messages.
            frequency: Rate in Hz to sample at.
            store_freq: Storage period of the modality in the dataset, in ticks.

        Returns:
            The tick stride.

        Raises:
            ValueError: If the frequency does not divide the simulator's tick
                rate, or asks for samples the dataset does not store.
        """
        carla_fps = self._root.expert.simulation.carla_fps
        if frequency <= 0 or carla_fps % frequency:
            raise ValueError(
                f"{field}_frequency={frequency} Hz must divide the simulator's "
                f"{carla_fps} Hz tick rate evenly.",
            )
        stride = carla_fps // frequency
        if stride % store_freq:
            raise ValueError(
                f"{field}_frequency={frequency} Hz needs a stride of {stride} "
                f"ticks, but the dataset stores this modality every "
                f"{store_freq} ticks; the fastest available rate is "
                f"{carla_fps // store_freq} Hz.",
            )
        return stride
