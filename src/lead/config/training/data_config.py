"""Data loading, caching, filtering, and augmentation configuration for training."""

from py123d.common.runtime.dataset_paths import get_dataset_paths

from lead.config.node import ConfigNode, overridable_property


class TrainingDataConfig(ConfigNode):
    """Dataset paths, caches, split selection, filtering, and augmentation."""

    # --- 123D data selection ---
    @overridable_property
    def py123d_split(self) -> str:
        """123D split to train on; defaults to the expert's normal-view split."""
        return self._root.expert.data_collection.py123d_split

    @overridable_property
    def py123d_data_root(self) -> str:
        """Dataset root holding ``logs/`` and ``maps/`` (defaults to $PY123D_DATA_ROOT)."""
        return str(get_dataset_paths().py123d_data_root)

    # Logs of the split to use, by directory name; empty uses every log. Scoping
    # a dataset to single logs is what lets a process work on one log alone.
    py123d_log_names: list[str] = []
    # Towns to train on, by the location the log metadata records; empty uses
    # every town. Filtering here reads log metadata only, never a scene.
    towns: list[str] = []
    # Scenes to keep after filtering; 0 keeps every scene.
    max_num_scenes: int = 0
    # Partition of the scene list this run trains on; 1 chunk is every scene.
    num_chunks: int = 1
    chunk_index: int = 0
    # Whether to shuffle the scene order, applied after every other selection.
    shuffle_scenes: bool = False

    # --- Data loader ---
    # Batches each worker holds ready. Every one of them is a full batch of
    # decoded samples in host memory, so this multiplies with the worker count.
    prefetch_batches_per_worker: int = 2
    # Number of data loader workers per CPU core.
    workers_per_cpu_core: int = 1
    # If false let DataLoader workers return batches as they finish rather than
    # in submission order, so one slow sample does not stall the queue behind it.
    loader_in_order: bool = True
    # If true copy every batch into page-locked memory, which is what lets the
    # host-to-device transfer overlap with compute. Worth about a third of the
    # step rate here, far more than the copy costs.
    pin_memory: bool = True
    # If true upload batches on a dedicated CUDA stream so the host-to-device
    # copy overlaps the previous step's compute instead of queueing behind it.
    copy_batch_on_side_stream: bool = True

    # --- Augmentation ---
    # If true use rotation and translation perburtation.
    use_sensor_perturbation: bool = True

    @overridable_property
    def sensor_perturbation_probability(self) -> float:
        """Probability of the perburtated sample being used."""
        if not self.use_sensor_perturbation:
            return 0.0
        return 0.5

    @overridable_property
    def use_color_augmentation(self) -> bool:
        """If true apply batched color augmentation on the device after collation."""
        return not self._root.training.experiment.visualize_dataset

    # Probability of each color augmentation op applying, per sample.
    color_augmentation_probability: float = 0.2

    # --- Cache store ---
    # The store's location is per policy: ``policy.<name>.cache_store_root``.
    read_from_cache_store: bool = False

    # If true, recompute and overwrite stored part outputs even where present.
    # Also needed to deliberately rebuild after a config change that affects
    # cached content (e.g. BEV geometry): build_cache otherwise refuses a
    # store whose cache_finger_print no longer matches the current config.
    force_cache_rebuild: bool = False

    # Partition of the cache build this process computes, sharded by log so
    # concurrent shards never write the same store file. Launchers map their
    # task index onto these; a sharded run leaves the store unsealed until a
    # final unsharded run verifies and writes the manifest.
    cache_build_shard_index: int = 0
    cache_build_shard_count: int = 1
