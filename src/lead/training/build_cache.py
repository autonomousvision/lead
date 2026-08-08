"""Build the policy's cache store: every cacheable builder, both sensor views.

Re-runnable: samples whose cached tensors are all stored are skipped unless
force_cache_rebuild is set, and the manifest is written only once a run ends
with nothing missing.

A store's cache_finger_print (see AbstractPolicyDataset) is checked against
the dataset's current one before any work starts; a mismatch refuses the run
unless force_cache_rebuild is set, since it means a config change affecting
cached content (e.g. BEV geometry) has made the store stale.

Run:

    python -m lead.training.build_cache training.data.py123d_split=<split>
"""

import logging
import multiprocessing
import os
import typing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, NamedTuple

import cv2
import pyarrow
import torch
import yaml
from tqdm import tqdm

from lead.api.abstract_dataset import AbstractPolicyDataset
from lead.api.abstract_policy import AbstractPolicy, build_policy
from lead.cache.cache_store import (
    CacheManifest,
    CacheStoreWriter,
    check_cache_finger_print,
    encode_tensor,
    read_manifest,
    write_manifest,
)
from lead.cache.lmdb_backend import read_cached_tensor_addresses
from lead.common.logging_setup import setup_logging
from lead.training import run_config

setup_logging()
LOG = logging.getLogger(__name__)

_WORKER: "_Worker | None" = None

# Store subdir holding one record per finished shard; the manifest is written
# once they are all there.
_SHARD_RECORD_DIR = "shards"


class _LogResult(NamedTuple):
    """What one log contributed to the store."""

    sample_view_pairs: int
    written_bytes: int
    sensor_views: list[str]
    tensor_codecs: dict[str, Any]


class _Worker(NamedTuple):
    """What a worker sets up once and reuses for every log it gets."""

    config: Any
    policy: AbstractPolicy
    writer: CacheStoreWriter


def _find_log_names(logs_root: Path, split_name: str) -> list[str]:
    """List a split's logs by name, without reading any of them.

    A log is any directory containing sync.arrow, at any depth under the
    split, which is how py123d finds them too.
    """
    log_names: list[str] = []
    for directory_path, directory_names, file_names in os.walk(
        logs_root / split_name,
        topdown=True,
    ):
        if "sync.arrow" not in file_names:
            continue
        log_names.append(Path(directory_path).name)
        directory_names[:] = []  # logs are never nested
    return sorted(log_names)


def _init_worker(config: Any) -> None:
    """Set up a worker process: one thread, its own policy and store writer.

    There is already one worker per core, so any library that opens a pool of
    its own oversubscribes the node. The writer has to be opened here rather
    than inherited, because an open write environment cannot survive a fork.
    """
    global _WORKER  # noqa: PLW0603
    torch.set_num_threads(1)
    cv2.setNumThreads(0)  # pyright: ignore[reportAttributeAccessIssue]
    pyarrow.set_cpu_count(1)
    pyarrow.set_io_thread_count(2)
    policy = build_policy(config)
    _WORKER = _Worker(
        config=config,
        policy=policy,
        writer=CacheStoreWriter(Path(policy.get_policy_config().cache_store_root)),
    )


def _build_log(log_name: str) -> _LogResult:
    """Build one log: enumerate its scenes, compute what is missing, store it.

    Each log is queued once and a worker takes one at a time, so this process
    is the only one writing that log's environments, which is what LMDB needs.
    """
    assert _WORKER is not None, "worker was not initialized"
    config = _WORKER.config
    config.apply_overrides({"training": {"data": {"py123d_log_names": [log_name]}}})
    dataset = typing.cast(AbstractPolicyDataset, _WORKER.policy.build_dataset())
    tensor_codecs = dataset.cached_tensor_codecs()
    cache_store_root = Path(_WORKER.policy.get_policy_config().cache_store_root)
    force_recompute = bool(config.training.data.force_cache_rebuild)

    stored_addresses: dict[str, set[str]] = {}
    sensor_views: set[str] = set()
    pairs = written_bytes = 0

    for scene_index in range(len(dataset)):
        address = dataset.sample_cache_address(scene_index)
        for sensor_view in dataset.sample_sensor_views(scene_index):
            sensor_views.add(sensor_view)
            if sensor_view not in stored_addresses:
                stored_addresses[sensor_view] = (
                    set()
                    if force_recompute
                    else read_cached_tensor_addresses(
                        cache_store_root / sensor_view,
                        log_name,
                    )
                )
            if all(
                f"{address.sample_key}/{name}" in stored_addresses[sensor_view]
                for name in tensor_codecs
            ):
                continue

            precomputed_tensors = dataset.precompute_cached_tensors(
                scene_index,
                sensor_view,
            )
            missing_tensor_names = tensor_codecs.keys() - precomputed_tensors.keys()
            if missing_tensor_names:
                raise ValueError(
                    f"sample {address} of view {sensor_view} computed no value "
                    f"for declared cached tensors {sorted(missing_tensor_names)}; "
                    f"a cacheable builder emitted less than its declared outputs.",
                )

            for name, value in precomputed_tensors.items():
                if name in tensor_codecs:
                    written_bytes += _WORKER.writer.write(
                        sensor_view,
                        address.log_name,
                        address.sample_key,
                        name,
                        encode_tensor(value, tensor_codecs[name]),
                    )
            pairs += 1

    _WORKER.writer.close()
    return _LogResult(pairs, written_bytes, sorted(sensor_views), tensor_codecs)


def _worker_count(config: Any) -> int:
    return max(
        1,
        config.training.num_assigned_cpu_cores
        * config.training.data.workers_per_cpu_core,
    )


def _run_logs(
    config: Any,
    log_names: list[str],
    num_workers: int,
) -> list[_LogResult]:
    """Build every log through a worker pool, one log at a time per worker."""
    with ProcessPoolExecutor(
        max_workers=num_workers,
        mp_context=multiprocessing.get_context("fork"),
        initializer=_init_worker,
        initargs=(config,),
    ) as executor:
        futures = [executor.submit(_build_log, name) for name in log_names]
        return [
            future.result()  # re-raises what a worker raised
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Caching",
                unit="log",
            )
        ]


def _shard_record_path(
    cache_store_root: Path,
    shard_index: int,
    shard_count: int,
) -> Path:
    """Path of one shard's record of what it built.

    Args:
        cache_store_root: The store's root directory.
        shard_index: The shard's index within the run.
        shard_count: Shards the run was split into.

    Returns:
        The record's path, unique to the shard.
    """
    return (
        cache_store_root
        / _SHARD_RECORD_DIR
        / f"{shard_index:04d}_of_{shard_count:04d}.yaml"
    )


def _record_finished_shard(
    cache_store_root: Path,
    shard_index: int,
    shard_count: int,
    results: list[_LogResult],
    cache_finger_print: dict[str, str],
) -> bool:
    """Record what this shard built and report whether every shard has finished.

    A shard writes its own record and no other, so concurrent shards never
    contend for a file; the store holds every log once every record exists.

    Args:
        cache_store_root: The store's root directory.
        shard_index: The shard's index within the run.
        shard_count: Shards the run was split into.
        results: What this shard's logs contributed.
        cache_finger_print: The dataset's current ``cache_finger_print``, uniform
            across every shard of one run.

    Returns:
        True when every shard of the run has left a record.
    """
    path = _shard_record_path(cache_store_root, shard_index, shard_count)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = yaml.safe_dump(
        {
            "tensors": results[0].tensor_codecs,
            "views": sorted(
                {view for result in results for view in result.sensor_views},
            ),
            "cache_finger_print": cache_finger_print,
        },
        sort_keys=False,
    )
    # A shard that sees this record must see all of it: the last shard reads
    # every record the moment the final one appears.
    temporary_path = path.with_suffix(".tmp")
    temporary_path.write_text(record)
    os.replace(temporary_path, path)
    return all(
        _shard_record_path(cache_store_root, index, shard_count).is_file()
        for index in range(shard_count)
    )


def _manifest_from_shard_records(
    cache_store_root: Path,
    shard_count: int,
) -> CacheManifest:
    """Merge the shards' records into the store's manifest.

    Args:
        cache_store_root: The store's root directory.
        shard_count: Shards the run was split into.

    Returns:
        The store's self-description, over every shard.
    """
    records = [
        yaml.safe_load(
            _shard_record_path(cache_store_root, index, shard_count).read_text(),
        )
        for index in range(shard_count)
    ]
    return {
        "tensors": records[0]["tensors"],
        "views": sorted({view for record in records for view in record["views"]}),
        "cache_finger_print": records[0]["cache_finger_print"],
    }


def main() -> None:
    config = run_config.initialize_config()
    config.apply_overrides({"training": {"data": {"read_from_cache_store": False}}})

    split_name = config.training.data.py123d_split
    policy = build_policy(config)
    cache_store_root = Path(policy.get_policy_config().cache_store_root)
    LOG.info(
        "Starting cache build: policy %s, split %s, cache dir %s",
        config.policy.target,
        split_name,
        cache_store_root,
    )

    # A throwaway instance: nothing here reads scenes, only the dataset's own
    # declared cache_finger_print, checked against the store before any work
    # is dispatched.
    dataset = typing.cast(AbstractPolicyDataset, policy.build_dataset())
    if not dataset.cached_tensor_codecs():
        LOG.info(
            "Policy %s declares nothing cacheable; there is no store to build.",
            config.policy.target,
        )
        return
    # No cached tensor reads the future, so a future margin (the planning
    # head's) only cuts frames out of the store that a later run would then
    # have to compute live.
    future_margin = policy.get_policy_config().future_window_num_iterations
    if future_margin > 0:
        LOG.warning(
            "The config's future margin (%d ticks) keeps the last frames of "
            "every log out of the store; build with the planning head off to "
            "cover them.",
            future_margin,
        )
    cache_finger_print = dataset.cache_finger_print
    if not config.training.data.force_cache_rebuild:
        check_cache_finger_print(read_manifest(cache_store_root), cache_finger_print)

    logs_root = Path(config.training.data.py123d_data_root) / "logs"
    log_names = _find_log_names(logs_root, split_name)
    if not log_names:
        raise RuntimeError(
            f"no logs under {logs_root / split_name}; check the data root and split",
        )
    shard_index = int(config.training.data.cache_build_shard_index)
    shard_count = int(config.training.data.cache_build_shard_count)
    # Each shard gets different logs, so no two write the same file.
    shard_log_names = log_names[shard_index::shard_count]
    if not shard_log_names:
        raise RuntimeError(
            f"shard {shard_index} of {shard_count} covers none of the "
            f"{len(log_names)} logs; use at most that many shards",
        )
    if shard_count > 1:
        LOG.info(
            "Building shard %d of %d: %d of %d logs",
            shard_index,
            shard_count,
            len(shard_log_names),
            len(log_names),
        )

    num_workers = _worker_count(config)
    LOG.info("Building %d logs with %d workers", len(shard_log_names), num_workers)
    results = _run_logs(config, shard_log_names, num_workers)

    total_pairs = sum(result.sample_view_pairs for result in results)
    LOG.info(
        "Cached %d (sample, view) pairs (%.2f GB) over %d logs",
        total_pairs,
        sum(result.written_bytes for result in results) / 1e9,
        len(results),
    )
    # Only a finished store gets a manifest, so readers refuse a broken one. A
    # shard cannot see the other shards' logs, but it can see their records.
    if not _record_finished_shard(
        cache_store_root,
        shard_index,
        shard_count,
        results,
        cache_finger_print,
    ):
        LOG.info(
            "Shard %d of %d built; the store is sealed once every shard is.",
            shard_index,
            shard_count,
        )
        return
    write_manifest(
        cache_store_root,
        _manifest_from_shard_records(cache_store_root, shard_count),
    )
    LOG.info("Every shard of %d is built; wrote the manifest.", shard_count)


if __name__ == "__main__":
    main()
