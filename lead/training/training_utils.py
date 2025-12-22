from __future__ import annotations

import abc
import datetime
import json
import logging
import os
import pathlib
import random
import typing

import diskcache
import numpy as np
import torch
import torch.multiprocessing as mp
from beartype import beartype
from diskcache import Cache
from torch import optim
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, LambdaLR
from torch.utils.data import DataLoader

from lead.training.config_training import TrainingConfig
from lead.training.data_loader.carla_dataset import CARLAData
from lead.training.data_loader.navsim_dataset import NavsimData
from lead.training.data_loader.waymo_e2e_dataset import WODE2EData
from lead.training.tfv6 import fn

LOG = logging.getLogger(__name__)


@beartype
def increase_limit_file_descriptors(n: int = 4096):
    # On some systems it is necessary to increase the limit on open file descriptors.
    try:
        import resource

        rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
    except (ModuleNotFoundError, ImportError) as e:
        LOG.error(str(e))


@beartype
def initialize_config() -> TrainingConfig:
    config = TrainingConfig()
    if config.load_file is not None:
        with open(os.path.join("/".join(config.load_file.split("/")[:-1]), "config.json")) as f:
            loaded_config = json.load(f)
        config = TrainingConfig(loaded_config, raise_error_on_missing_key=False)
    return config


@beartype
def initialize_cache(config: TrainingConfig) -> Cache | None:
    ssd_cache = None
    if config.use_training_session_cache:
        ssd_cache = Cache(directory=config.ssd_cache_path, size_limit=int(768 * 1024**3))
    return ssd_cache


@beartype
def initialize_torch(config: TrainingConfig) -> int:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    ngpus_per_node = torch.cuda.device_count()
    ncpus_per_node = config.assigned_cpu_cores
    num_workers = int(ncpus_per_node / ngpus_per_node) * config.workers_per_cpu_cores

    if torch.cuda.device_count() > 1:
        torch.distributed.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=config.world_size,
            rank=config.rank,
            timeout=datetime.timedelta(minutes=120),
        )

    torch.cuda.device(config.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = True
    return num_workers


@beartype
def initialize_model(config: TrainingConfig) -> tuple[typing.Any | torch.nn.parallel.distributed.DistributedDataParallel, int]:
    from lead.training.tfv6.tfv6 import TFv6

    model = TFv6(config.device, config)

    model.cuda(device=config.device)
    if config.sync_batchnorm:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        LOG.info("Using sync_batch_norm")

    # Convert all norm layers to use fp32
    fn.patch_norm_fp32(model)

    start_epoch = 0  # Epoch to continue training from
    if config.load_file is not None:
        LOG.info(f"Loading model from {config.load_file}")
        # Add +1 because the epoch before that was already trained
        load_name = str(pathlib.Path(config.load_file).stem)
        if config.continue_failed_training:
            start_epoch = int("".join(filter(str.isdigit, load_name))) + 1
        model.load_state_dict(
            torch.load(config.load_file, map_location=config.device, weights_only=True), strict=config.continue_failed_training
        )

    model.backbone.requires_grad_(not config.freeze_backbone)
    LOG.info(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")
    if config.channel_last:
        model = model.to(memory_format=torch.channels_last)
        LOG.info("Using channel last memory format")
    if torch.cuda.device_count() > 1:
        model_wrapper = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=None, output_device=None, broadcast_buffers=False
        )
    else:
        model_wrapper = model
    if config.compile:
        model = torch.compile(
            model,
            fullgraph=True,  # require entire model to be compiled, fail if not
            dynamic=False,  # aggressively specialize to current input shapes
            backend="inductor",
            mode="max-autotune",  # highest autotune + CUDA graph
        )
    return model_wrapper, start_epoch


@beartype
def initialize_optimizer(
    model_wrapper: typing.Any | torch.nn.parallel.DistributedDataParallel,
    model: torch.nn.Module,
    config: TrainingConfig,
    gradient_steps_per_epoch: int,
) -> tuple[
    ZeroRedundancyOptimizer | torch.optim.AdamW,
    CosineAnnealingWarmRestarts | LambdaLR | CosineAnnealingLR,
    torch.amp.GradScaler,
    int,
]:
    params = model_wrapper.parameters()
    if config.use_zero_redundancy and torch.cuda.device_count() > 1:
        optimizer = ZeroRedundancyOptimizer(
            params, optimizer_class=torch.optim.AdamW, lr=config.lr, amsgrad=True, weight_decay=config.weight_decay
        )
    else:
        optimizer = optim.AdamW(params, lr=config.lr, amsgrad=True, weight_decay=config.weight_decay)

    if config.use_cosine_annealing_with_restarts:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=gradient_steps_per_epoch, T_mult=2)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=gradient_steps_per_epoch * config.epochs)

    if config.load_file is not None and config.continue_failed_training:
        scheduler.load_state_dict(
            torch.load(
                config.load_file.replace("model_", "scheduler_"),
                map_location=config.device,
                weights_only=True,
            )
        )

    if config.load_file is not None and config.continue_failed_training:
        optimizer.load_state_dict(
            torch.load(
                config.load_file.replace("model_", "optimizer_"),
                map_location=config.device,
                weights_only=True,
            )
        )

    scaler = torch.amp.GradScaler(
        init_scale=config.grad_scaler_init_scale,
        growth_factor=config.grad_scaler_growth_factor,
        backoff_factor=config.grad_scaler_backoff_factor,
        growth_interval=config.grad_scaler_growth_interval,
        enabled=config.need_grad_scaler,
    )
    if config.load_file is not None and config.continue_failed_training:
        scaler.load_state_dict(
            torch.load(
                config.load_file.replace("model_", "scaler_"),
                map_location=config.device,
                weights_only=True,
            )
        )

    gradient_steps_skipped = 0
    if config.load_file is not None and config.continue_failed_training:
        gradient_steps_skipped_path = config.load_file.replace("model_", "gradient_steps_skipped_").replace(".pth", ".txt")
        if os.path.exists(gradient_steps_skipped_path):
            with open(gradient_steps_skipped_path) as f:
                gradient_steps_skipped = int(f.read().strip())

    return optimizer, scheduler, scaler, gradient_steps_skipped


@beartype
def initialize_dataloader(
    config: TrainingConfig,
    ssd_cache: dict | diskcache.core.Cache | None,
    num_workers: int,
):
    g_cuda = torch.Generator(device="cpu")
    g_cuda.manual_seed(config.seed)

    datasets, samplers = [], []
    if config.use_carla_data:
        datasets.append(
            CARLAData(
                root=config.carla_data,
                config=config,
                training_session_cache=ssd_cache,
            )
        )
        assert not datasets[-1].build_cache and not datasets[-1].build_buckets
        samplers.append(
            torch.utils.data.DistributedSampler(
                datasets[-1], shuffle=True, num_replicas=config.world_size, rank=config.rank, drop_last=True
            )
        )
    if config.use_navsim_data:
        datasets.append(
            NavsimData(
                root=config.navsim_data_root,
                config=config,
                training_session_cache=ssd_cache,
            )
        )
        samplers.append(
            torch.utils.data.DistributedSampler(
                datasets[-1], shuffle=True, num_replicas=config.world_size, rank=config.rank, drop_last=True
            )
        )
    if config.use_waymo_e2e_data:
        datasets.append(
            WODE2EData(
                root=config.waymo_e2e_training_data_root,
                config=config,
                training_session_cache=ssd_cache,
                training=True,
            )
        )
        samplers.append(
            torch.utils.data.DistributedSampler(
                datasets[-1], shuffle=True, num_replicas=config.world_size, rank=config.rank, drop_last=True
            )
        )

    assert len(datasets) > 0, "No datasets selected for training!"

    for ds in datasets:
        LOG.info(f"Dataset size: {len(ds)} samples")

    if config.schedule_carla_num_samples:
        assert config.use_carla_data and config.mixed_data_training
        sample_scheduler = Sim2RealSampleScheduler(config, datasets)
    else:
        sample_scheduler = UniformSampleScheduler(config, datasets)

    train_dataset = MixedDataset(
        config=config,
        datasets=datasets,
    )

    mixed_sampler = MixedSampler(
        samplers=samplers,
        sample_scheduler=sample_scheduler,
        config=config,
    )

    dataloader_train = DataLoader(
        train_dataset,
        batch_sampler=mixed_sampler,
        worker_init_fn=seed_worker,
        generator=g_cuda,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=config.prefetch_factor,
        persistent_workers=True,
        collate_fn=mixed_data_collate_fn,
    )
    return dataloader_train, mixed_sampler


def mixed_data_collate_fn(batch):
    all_keys = set()
    for b in batch:
        all_keys.update(b.keys())

    collated = {}
    for key in all_keys:
        vals = [b.get(key, None) for b in batch]

        if all(v is None for v in vals):
            collated[key] = None
            continue

        ref = next(v for v in vals if v is not None)

        new_vals = []
        for v in vals:
            if v is None:
                if torch.is_tensor(ref):
                    v = torch.zeros_like(ref)
                elif isinstance(ref, np.ndarray):
                    v = np.zeros_like(ref)
                elif isinstance(ref, str):
                    v = ""
                else:
                    try:
                        v = ref.__class__()
                    except:
                        v = None
            new_vals.append(v)
        try:
            collated[key] = torch.utils.data._utils.collate.default_collate(new_vals)
        except:
            pass

    return collated


@beartype
def save_config(config: TrainingConfig, rank: int):
    def is_json_serializable(v):
        try:
            json.dumps(v)
            return True
        except (TypeError, OverflowError):
            return False

    if rank == 0 and config.logdir is not None:
        os.makedirs(config.logdir, exist_ok=True)
        json_config = {
            k: v
            for k, v in config.training_dict().items()
            if is_json_serializable(v) and not k.startswith("_") and not k.endswith("__")
        }
        json_config = json.dumps(json_config, indent=4)
        # LOG.info(json_config)
        with open(os.path.join(config.logdir, "config.json"), "w") as f2:
            f2.write(json_config)


def seed_worker(_):
    # We need to seed the workers individually otherwise random processes in the
    # dataloader return the same values across workers!
    worker_seed = (torch.initial_seed()) % 2**32  # this is different across workers, but not gpus when setting config.seed
    rank = int(os.environ.get("RANK", "0"))
    worker_seed = worker_seed + rank * 1000
    # if config.seed is not None, torch.initial_seed is the same across different gpus,
    # so we need to combine it with the rank to get different rng seeds on different gpus.
    # multiply with 1000 because the last digit is already incremented across workers
    torch.manual_seed(worker_seed)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.cuda.manual_seed(worker_seed)
    torch.cuda.manual_seed_all(worker_seed)


def set_start_method():
    # Select how the threads in the data loader are spawned
    # See this: https://stackoverflow.com/a/66113051
    # To edit code while processes run, we generally prefer fork.
    available_start_methods = mp.get_all_start_methods()
    if "fork" in available_start_methods:
        mp.set_start_method("fork")
    # Available on all OS.
    elif "spawn" in available_start_methods:
        mp.set_start_method("spawn")
    elif "forkserver" in available_start_methods:
        mp.set_start_method("forkserver")


class AbstractMixedDatasetSampleScheduler(abc.ABC):
    @beartype
    def __init__(self, datasets: list[torch.utils.data.Dataset]):
        self.datasets = datasets
        self.num_datasets = len(datasets)

    @beartype
    @abc.abstractmethod
    def get_batches_schedule(self, epoch: int) -> list[float]:
        """Get batch ratio for each dataset for the given epoch. Should sum to to batch size per GPU."""
        pass


# Uniform sample scheduler
class UniformSampleScheduler(AbstractMixedDatasetSampleScheduler):
    @beartype
    def __init__(self, config: TrainingConfig, datasets: list[torch.utils.data.Dataset]):
        self.config = config
        self.datasets = datasets
        self.num_datasets = len(datasets)

    def get_batches_schedule(self, _: int) -> list[float]:
        """Return equal ratios for all datasets."""
        return [int(1.0 / self.num_datasets * self.config.batch_size / torch.cuda.device_count())] * self.num_datasets


# Sim2Real sample annealing scheduler
class Sim2RealSampleScheduler(AbstractMixedDatasetSampleScheduler):
    @beartype
    def __init__(self, config: TrainingConfig, datasets: list[torch.utils.data.Dataset]):
        from data_loader.carla_dataset import CARLAData

        self.config = config
        self.datasets = datasets
        self.num_datasets = len(datasets)
        assert self.num_datasets == 2, "Sim2RealSampleScheduler only supports 2 datasets."
        assert isinstance(datasets[0], CARLAData), "First dataset must be CARLAData."

    def get_batches_schedule(self, epoch: int) -> list[float]:
        """Return batch ratios for sim and real datasets based on epoch."""
        # TODO: make this work with any BS number
        anchors = {
            0: {0: 40, 1: 24},
            1: {0: 32, 1: 32},
            3: {0: 24, 1: 40},
            7: {0: 16, 1: 48},
            15: {0: 8, 1: 56},
            31: {0: 0, 1: 64},
        }
        for anchor_epoch in sorted(anchors.keys(), reverse=True):
            if epoch >= anchor_epoch:
                return [anchors[anchor_epoch][i] // torch.cuda.device_count() for i in range(self.num_datasets)]


class MixedDataset(torch.utils.data.Dataset):
    @beartype
    def __init__(
        self,
        config: TrainingConfig,
        datasets: list[torch.utils.data.Dataset],
    ):
        self.datasets = datasets
        self.config = config
        self.num_datasets = len(datasets)

        # Store sub-dataset sizes
        self.dataset_sizes = [len(ds) for ds in datasets]
        self.size = sum(self.dataset_sizes)

        assert all(len(ds) == self.dataset_sizes[0] for ds in datasets), (
            "Assumption failed: All datasets must have the same size."
        )

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        """
        Get item from the mixed dataset.

        The index comes from MixedSampler in the form:
            index = dataset_idx + dataset_sample_idx * num_datasets

        Where:
            - dataset_idx is which dataset (0, 1, 2, ...)
            - dataset_sample_idx is the actual index in that underlying dataset
        """
        dataset_idx = index % self.num_datasets
        dataset_sample_idx = index // self.num_datasets

        # Access the underlying dataset directly
        return self.datasets[dataset_idx][dataset_sample_idx]

    def shuffle(self, epoch):
        """Shuffle the underlying datasets with custom implemented shuffle function."""
        for dataset in self.datasets:
            dataset.shuffle(epoch)


class MixedSampler(torch.utils.data.BatchSampler):
    @beartype
    def __init__(
        self,
        config: TrainingConfig,
        samplers: list[torch.utils.data.Sampler],
        sample_scheduler: AbstractMixedDatasetSampleScheduler,
    ):
        """
        Sampler for MixedDataset that samples from each dataset according to sample_scheduler.

        Args:
            samplers: List of samplers, one for each underlying dataset
            sample_scheduler: Scheduler that determines the ratio of samples from each dataset
            batch_size: Total batch size across all datasets
            epoch: Current epoch (used by sample_scheduler to determine ratios)
        """
        self.samplers = samplers
        LOG.info(f"MixedSampler using {len(samplers)} samplers. Each sampler size: {[len(s) for s in samplers]}")
        assert all(len(samplers[0]) == len(s) for s in samplers), "All samplers must have the same length."
        self.sample_scheduler = sample_scheduler
        self.drop_last = True
        self.num_datasets = len(samplers)
        self.config = config

        # Calculate batch sizes per dataset based on scheduler ratios
        self.update_batch_sizes(0)

    def update_batch_sizes(self, epoch):
        """Calculate how many samples to take from each dataset per batch."""
        self.batch_sizes = self.sample_scheduler.get_batches_schedule(epoch)
        assert sum(self.batch_sizes) * torch.cuda.device_count() == self.config.batch_size, (
            "Batch sizes must sum to total batch size"
        )

    def __iter__(self):
        """
        Yields batches where each batch contains samples from each dataset according to ratios.

        The MixedDataset interleaves samples: [ds0_s0, ds1_s0, ds0_s1, ds1_s1, ...]
        So we need to generate indices that respect this interleaving pattern.
        """
        iterators = [iter(s) for s in self.samplers]

        try:
            while True:
                batch = []

                # Step 1: Collect indices from each dataset's sampler
                dataset_indices = []
                for dataset_idx, it in enumerate(iterators):
                    indices = []
                    for _ in range(self.batch_sizes[dataset_idx]):
                        indices.append(next(it))
                    dataset_indices.append(indices)

                # Step 2: Convert to MixedDataset global indices
                # MixedDataset structure: index % num_datasets gives dataset_idx
                #                        index // num_datasets gives sample_idx within that dataset
                for dataset_idx in range(self.num_datasets):
                    for local_idx in dataset_indices[dataset_idx]:
                        # Map from (dataset_idx, dataset_sample_idx) to MixedDataset global index
                        mixed_dataset_idx = dataset_idx + local_idx * self.num_datasets
                        batch.append(mixed_dataset_idx)

                yield batch

        except StopIteration:
            return

    def __len__(self):
        """Return the number of batches per epoch."""
        return (len(self.samplers[0]) // self.config.batch_size) * torch.cuda.device_count()
