"""Data loading, caching, filtering, and augmentation configuration for training."""

from py123d.common.runtime.dataset_paths import get_dataset_paths

from lead.config.node import ConfigNode, overridable_property


class TrainingDataConfig(ConfigNode):
    """Dataset paths, caches, split selection, filtering, and augmentation."""

    # --- 123D data selection ---
    @overridable_property
    def py123d_dataset(self) -> str:
        """123D dataset name; defaults to the one the expert collected."""
        return self._root.expert.data_collection.py123d_dataset

    @overridable_property
    def py123d_split(self) -> str:
        """123D split to train on; defaults to the expert's normal-view split."""
        return self._root.expert.data_collection.py123d_split

    @overridable_property
    def py123d_perturbated_split(self) -> str:
        """123D split holding the perturbated sensor views paired with the normal split."""
        return self._root.expert.data_collection.py123d_perturbated_split

    @property
    def py123d_split_name(self) -> str:
        """Full 123D split directory name."""
        return self.py123d_split

    @overridable_property
    def py123d_logs_root(self) -> str:
        """Root directory of the 123D logs (defaults to $PY123D_DATA_ROOT/logs)."""
        return str(get_dataset_paths().py123d_logs_root)

    @overridable_property
    def py123d_maps_root(self) -> str:
        """Root directory of the 123D maps (defaults to $PY123D_DATA_ROOT/maps)."""
        return str(get_dataset_paths().py123d_maps_root)

    # --- Modalities loaded from disk ---
    # We stack lidar frames for motion cues. Number of past frames we stack for the model input.
    training_used_lidar_steps: int = 10
    # Minimum Z coordinate for LiDAR points.
    min_z: float = -4
    # Maximum Z coordinate for LiDAR points.
    max_z: float = 4
    # Max number of LiDAR points per pixel in voxelized LiDAR.
    hist_max_per_pixel: int = 5

    # --- Sequence sampling ---
    @property
    def skip_first(self) -> int:
        """Number of frames to skip at the beginning of sequences."""
        if self._root.training.is_pretraining:
            return 1
        return self._root.agent.transfuser.num_way_points_prediction

    @property
    def skip_last(self) -> int:
        """Number of frames to skip at the end of sequences."""
        if self._root.training.is_pretraining:
            return 1
        return self._root.agent.transfuser.num_way_points_prediction

    # --- Data loader ---
    # Number of data loader workers to prefetch batches.
    prefetch_factor: int = 16
    # Number of data loader workers per CPU core.
    workers_per_cpu_cores: int = 1

    # --- Data filtering ---
    # If true then we skip Town13 routes during training
    hold_out_town13_routes: bool = False

    # --- Augmentation ---
    # If true use rotation and translation perburtation.
    use_sensor_perburtation: bool = True

    @overridable_property
    def use_sensor_perburtation_prob(self) -> float:
        """Probability of the perburtated sample being used."""
        if not self.use_sensor_perburtation:
            return 0.0
        return 0.5

    @property
    def use_color_aug(self) -> bool:
        """If true apply image color based augmentations."""
        return not self._root.training.experiment.visualize_dataset

    # Probability to apply the different image color augmentations.
    use_color_aug_prob: float = 0.2
