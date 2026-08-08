"""LMDB storage for the cache store: one environment per log, fork-safe reads."""

from __future__ import annotations

import logging
from pathlib import Path

LOG = logging.getLogger(__name__)

# Reserved record key of a legacy per-log metadata stamp; stores built before
# the manifest existed hold one, so reads must keep skipping it.
_LEGACY_META_RECORD_KEY: bytes = b"__meta__"

# Initial address-space reservation per log. map_size is fixed at open(), before
# anything is encoded, and the final size is not knowable then: a scene costs far
# less under jpeg/png than under the lossless raw codec. Start modest and grow on
# overflow rather than guess.
_DEFAULT_MAP_SIZE_BYTES: int = 1 << 30  # 1 GiB per log, doubled on demand
_MAX_MAP_SIZE_BYTES: int = 1 << 40  # 1 TiB; a runaway backstop, never reached

_DEFAULT_RECORDS_PER_TRANSACTION: int = 64


def open_log_read_env(log_dir: Path, max_readers: int = 2048):
    """Open one log's environment for concurrent, lock-free reading.

    lock=False keeps the environment out of LMDB's reader table, which is what
    makes an environment inherited across fork() safe to keep using;
    readahead=False because access is random and readahead would only evict
    useful pages.

    Args:
        log_dir: The log's environment directory.
        max_readers: LMDB reader-table size of the environment.

    Returns:
        The open read-only LMDB environment.
    """
    import lmdb

    return lmdb.open(
        str(log_dir),
        readonly=True,
        lock=False,
        readahead=False,
        max_readers=max_readers,
        subdir=True,
    )


def read_cached_tensor_addresses(view_dir: str | Path, log_name: str) -> set[str]:
    """Tensor addresses (``<sample_key>/<tensor_name>``) already stored for a log.

    Read-only and safe to call before forking DataLoader workers, unlike
    opening a writer.

    Args:
        view_dir: One view's store dir, laid out as ``<view_dir>/<log_name>/``.
        log_name: The log to scan.

    Returns:
        The stored tensor addresses, or an empty set if the log has no store yet.
    """
    import lmdb

    log_dir = Path(view_dir) / log_name
    if not (log_dir / "data.mdb").is_file():
        return set()
    try:
        env = open_log_read_env(log_dir)
        try:
            with env.begin() as txn:
                return {
                    key.decode()
                    for key, _ in txn.cursor()
                    if key != _LEGACY_META_RECORD_KEY
                }
        finally:
            env.close()
    except lmdb.Error as error:
        # A damaged env (e.g. a killed build) means recompute, never crash.
        LOG.warning("unreadable store env %s (%s); recomputing its log", log_dir, error)
        return set()


class LmdbCacheWriter:
    """Exclusive writer for one log's environment. One instance per (view_dir, log_name)."""

    def __init__(
        self,
        view_dir: str | Path,
        log_name: str,
        map_size: int = _DEFAULT_MAP_SIZE_BYTES,
        records_per_transaction: int = _DEFAULT_RECORDS_PER_TRANSACTION,
    ) -> None:
        """Open (creating if needed) one log's environment for writing.

        Args:
            view_dir: One view's store dir.
            log_name: The log this writer owns.
            map_size: Initial LMDB address-space reservation in bytes.
            records_per_transaction: Tensor records batched per write transaction.
        """
        import lmdb

        self._lmdb = lmdb
        self._log_dir = Path(view_dir) / log_name
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._records_per_transaction = records_per_transaction
        self._map_size = map_size

        # sync=False: the cache is a derived artifact, durability buys nothing.
        # writemap=False: a write map ftruncates data.mdb to the full map_size,
        # so the file would report the full reservation for a few MB of data;
        # writing through the page cache lets the file grow as it fills, at no
        # measurable cost next to featurization.
        self._env = lmdb.open(
            str(self._log_dir),
            map_size=map_size,
            subdir=True,
            sync=False,
            writemap=False,
        )
        self._pending: list[tuple[bytes, bytes]] = []

    def write(self, tensor_address: str, tensor_blob: bytes) -> int:
        """Store one already-encoded sample tensor.

        Args:
            tensor_address: The record key, ``<sample_key>/<tensor_name>``.
            tensor_blob: The encoded record.

        Returns:
            Bytes written, for progress reporting.
        """
        self._pending.append((tensor_address.encode(), tensor_blob))
        if len(self._pending) >= self._records_per_transaction:
            self.flush()
        return len(tensor_blob)

    def _commit(self, items: list[tuple[bytes, bytes]]) -> None:
        """Write items in one transaction, doubling the map whenever it overflows.

        map_size has to be chosen before anything is encoded, so it is always a
        guess. Growing on MapFullError turns a wrong guess from a lost build
        into a resize. The failed transaction is aborted by the context manager
        before set_mapsize, which requires no open transaction.
        """
        assert self._env is not None
        while True:
            try:
                with self._env.begin(write=True) as txn:
                    for key, tensor_blob in items:
                        txn.put(key, tensor_blob)
                return
            except self._lmdb.MapFullError:
                if self._map_size >= _MAX_MAP_SIZE_BYTES:
                    raise
                self._map_size *= 2
                LOG.info(
                    "%s: growing lmdb map to %.1f GiB",
                    self._log_dir.name,
                    self._map_size / (1 << 30),
                )
                self._env.set_mapsize(self._map_size)

    def flush(self) -> None:
        """Commit pending records now. Idempotent."""
        if not self._pending:
            return
        self._commit(self._pending)
        self._pending.clear()

    def close(self) -> None:
        """Flush pending writes and release the environment. Idempotent."""
        if self._env is None:
            return
        self.flush()
        self._env.sync(True)
        self._env.close()
        self._env = None
