"""Background thread that runs a write function on submitted items in order."""

import logging
import queue
import typing
from threading import Thread

LOG = logging.getLogger(__name__)


class AsyncWriter:
    """Runs a write function on a background thread, never dropping an item.

    Items are processed in submission order, :meth:`submit` blocks while the
    queue is full, and a write error is re-raised on the next :meth:`submit`.
    """

    def __init__(
        self,
        write_fn: typing.Callable[..., None],
        queue_size: int = 10,
    ) -> None:
        """Start the background worker.

        Args:
            write_fn: Function invoked on the worker thread with the arguments
                of each :meth:`submit` call.
            queue_size: Maximum number of pending items before submit blocks.
        """
        self._write_fn = write_fn
        self._queue: queue.Queue[tuple | None] = queue.Queue(maxsize=queue_size)
        self._error: Exception | None = None
        self._worker = Thread(target=self._loop, daemon=True)
        self._worker.start()

    def submit(self, *args: object) -> None:
        """Queue one item for writing; blocks while the queue is full.

        Args:
            *args: Arguments passed through to the write function.
        """
        if self._error is not None:
            raise self._error
        self._queue.put(args)

    def close(self) -> None:
        """Write all pending items and stop the worker."""
        self._queue.put(None)
        self._worker.join()
        if self._error is not None:
            LOG.error(f"Async writing failed: {self._error}")

    def _loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            if self._error is not None:
                continue  # Drop further items; submit raises the error
            try:
                self._write_fn(*item)
            except Exception as error:
                LOG.exception("Async write worker failed")
                self._error = error
