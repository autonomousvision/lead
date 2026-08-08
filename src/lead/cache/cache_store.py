"""The cache store: prebuilt sample tensors under ``<root>/<view>/<log>/``,
addressed as ``<sample_key>/<tensor_name>``."""

import os
import typing
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

from lead.cache.codec import (
    CacheableValue,
    CodecSpec,
    decode_tensor,
    encode_tensor,
)
from lead.cache.lmdb_backend import LmdbCacheWriter, open_log_read_env

if TYPE_CHECKING:
    import lmdb

__all__ = [
    "CacheManifest",
    "CacheStoreReader",
    "CacheStoreWriter",
    "SampleCacheAddress",
    "check_cache_finger_print",
    "encode_tensor",
    "read_manifest",
    "write_manifest",
]

_MANIFEST_NAME = "manifest.yaml"


class SampleCacheAddress(typing.NamedTuple):
    """Where one sample's tensors live inside a store, identical across views."""

    log_name: str
    sample_key: str


class CacheManifest(TypedDict):
    """A store's self-description: the tensors it holds and the views it was built with."""

    tensors: dict[str, CodecSpec]
    views: list[str]
    # The dataset's cache_finger_print at build time; see check_cache_finger_print.
    cache_finger_print: dict[str, str]


def read_manifest(cache_store_root: str | Path) -> CacheManifest | None:
    """Read a store's manifest.

    Args:
        cache_store_root: The store's root directory.

    Returns:
        The manifest, or None when the store was never built.
    """
    import yaml

    path = Path(cache_store_root) / _MANIFEST_NAME
    if not path.is_file():
        return None
    return yaml.safe_load(path.read_text())


def check_cache_finger_print(
    manifest: CacheManifest | None,
    current: dict[str, str],
) -> None:
    """Refuse a store whose ``cache_finger_print`` does not match the caller's.

    A store with no manifest yet (never built) is not a mismatch; the caller
    is about to build it.

    Args:
        manifest: The store's manifest, or None when it was never built.
        current: The caller's current ``cache_finger_print``.

    Raises:
        ValueError: If the store was built and its fingerprint differs from
            ``current``.
    """
    if manifest is None:
        return
    stored = manifest.get("cache_finger_print", {})
    if stored == current:
        return
    changed = sorted(stored.keys() | current.keys())
    changed = [key for key in changed if stored.get(key) != current.get(key)]
    raise ValueError(
        f"cache store does not match the current config, in: {changed}\n"
        f"stored:  { {key: stored.get(key) for key in changed} }\n"
        f"current: { {key: current.get(key) for key in changed} }\n"
        f"Rebuild the store with training.data.force_cache_rebuild=true.",
    )


def write_manifest(
    cache_store_root: str | Path,
    manifest: CacheManifest,
) -> None:
    """Write the manifest declaring what a store holds.

    Args:
        cache_store_root: The store's root directory.
        manifest: The store's self-description (cached tensors and views).
    """
    import yaml

    path = Path(cache_store_root) / _MANIFEST_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    # Shards that finish together both write this; a reader must see one whole
    # file either way.
    temporary_path = path.with_suffix(f".{os.getpid()}.tmp")
    temporary_path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    os.replace(temporary_path, path)


class CacheStoreReader:
    """Reads prebuilt sample tensors. Safe to hand to forked DataLoader workers.

    LMDB environments are opened lazily per process and stripped when pickled,
    since neither the memory map nor the file descriptor survives a fork.
    """

    def __init__(
        self,
        cache_store_root: str | Path,
        cache_finger_print: dict[str, str],
    ) -> None:
        """Open a built store.

        Args:
            cache_store_root: The store's root directory.
            cache_finger_print: The caller's current ``cache_finger_print``,
                checked against the store's own.

        Raises:
            FileNotFoundError: If the store was never built.
            ValueError: If the store's fingerprint does not match ``cache_finger_print``.
        """
        self._cache_store_root = Path(cache_store_root)
        manifest = read_manifest(self._cache_store_root)
        if manifest is None:
            raise FileNotFoundError(
                f"no cache store under {self._cache_store_root}; build it with "
                f"lead.training.build_cache",
            )
        check_cache_finger_print(manifest, cache_finger_print)
        # The names of the cached tensors, as built.
        self.tensor_names: frozenset[str] = frozenset(manifest["tensors"])
        self._sensor_views: list[str] = manifest["views"]
        self._envs: dict[tuple[str, str], lmdb.Environment] | None = None

    def has_sensor_view(self, sensor_view: str) -> bool:
        """Whether the store was built with a sensor view.

        Args:
            sensor_view: The view subdir name.

        Returns:
            True when the view is part of the store.
        """
        return sensor_view in self._sensor_views

    def read(
        self,
        sensor_view: str,
        log_name: str,
        sample_key: str,
        tensor_names: Iterable[str],
    ) -> dict[str, CacheableValue] | None:
        """Read stored tensors of one sample.

        A sample the store does not hold at all is not an error: the scene
        enumeration a store was built with need not be the one reading it, so
        the caller computes such a sample live. A sample holding some but not
        all of the requested tensors is a damaged store and raises.

        Args:
            sensor_view: The view subdir name.
            log_name: The log of the sample.
            sample_key: The per-log sample key (anchor timestamp).
            tensor_names: The tensors to read.

        Returns:
            The decoded tensors, one per requested name, or None when the store
            holds none of them for this sample.

        Raises:
            KeyError: If the store holds only some of the requested tensors.
        """
        if self._envs is None:
            self._envs = {}
        env_key = (sensor_view, log_name)
        env = self._envs.get(env_key)
        if env is None:
            env = open_log_read_env(self._cache_store_root / sensor_view / log_name)
            self._envs[env_key] = env
        tensors: dict[str, CacheableValue] = {}
        missing_names: list[str] = []
        with env.begin() as txn:
            for name in tensor_names:
                tensor_blob = txn.get(f"{sample_key}/{name}".encode())
                if tensor_blob is None:
                    missing_names.append(name)
                    continue
                # Free for the bytes LMDB returns here; copies out of the
                # transaction should the env ever be opened with buffers=True.
                tensors[name] = decode_tensor(bytes(tensor_blob))
        if not tensors:
            return None
        # Omitting one would surface much later as a missing batch key.
        if missing_names:
            raise KeyError(
                f"{sorted(missing_names)} of sample {sample_key} of log "
                f"{log_name} are not in the {sensor_view} store under "
                f"{self._cache_store_root}, but its other tensors are; the "
                f"store is damaged, rebuild it with lead.training.build_cache.",
            )
        return tensors

    def close(self) -> None:
        """Release open environments. Idempotent."""
        for env in (self._envs or {}).values():
            env.close()
        self._envs = None

    def __getstate__(self) -> dict[str, typing.Any]:
        state = self.__dict__.copy()
        state["_envs"] = None
        return state


class CacheStoreWriter:
    """Writes one store during the cache build; one LMDB writer per (view, log)."""

    def __init__(self, cache_store_root: str | Path) -> None:
        """Open a store root for writing.

        Args:
            cache_store_root: The store's root directory.
        """
        self._cache_store_root = Path(cache_store_root)
        self._writers: dict[tuple[str, str], LmdbCacheWriter] = {}

    def write(
        self,
        sensor_view: str,
        log_name: str,
        sample_key: str,
        tensor_name: str,
        tensor_blob: bytes,
    ) -> int:
        """Store one already-encoded sample tensor.

        Args:
            sensor_view: The view subdir name.
            log_name: The log of the sample.
            sample_key: The per-log sample key (anchor timestamp).
            tensor_name: The tensor's name in the sample.
            tensor_blob: The encoded record from :func:`encode_tensor`.

        Returns:
            Bytes written, for progress reporting.
        """
        writer_key = (sensor_view, log_name)
        writer = self._writers.get(writer_key)
        if writer is None:
            writer = LmdbCacheWriter(
                self._cache_store_root / sensor_view,
                log_name,
            )
            self._writers[writer_key] = writer
        return writer.write(f"{sample_key}/{tensor_name}", tensor_blob)

    def flush(self) -> None:
        """Commit every open per-log writer's pending records."""
        for writer in self._writers.values():
            writer.flush()

    def close(self) -> None:
        """Flush and close every open per-log writer. Idempotent."""
        for writer in self._writers.values():
            writer.close()
        self._writers = {}
