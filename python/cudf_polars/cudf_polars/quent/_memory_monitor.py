# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A shared, refcounted background sampler of memory resource occupancy."""

from __future__ import annotations

import threading
import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import uuid
    from collections.abc import Callable

__all__ = ["MemoryResourceSampler"]


class MemoryResourceSampler:
    """
    Sample the memory resource on one thread, on behalf of every active query.

    The device is shared by all queries running on a worker, so the reading is
    taken once per tick and delivered to each attached query. A background
    thread runs only while at least one query is attached.

    Taking a reading (``read_bytes``) must be cheap and must not acquire a lock
    that the monitored code also takes. The sampler's own lock is never held by
    the allocation path; it only serializes :meth:`attach`, :meth:`detach`,
    :meth:`close` and the sampling thread.

    Monitoring failures never fail the monitored code. If reading or recording
    raises, the sampling thread stops and a :class:`RuntimeWarning` is emitted
    from the next :meth:`detach`.

    Parameters
    ----------
    read_bytes
        Callable that returns the current number of allocated device bytes.
    record
        Called as ``record(query_id, nbytes)`` once per attached query for each
        reading. Calls are serialized, and ``record`` is not called for a query
        after ``stop`` has been called for it.
    stop
        Called as ``stop(query_id)`` when a query detaches or the sampler is
        closed.
    interval_s
        Seconds between readings. Must be positive.
    name
        Name of the sampling thread.

    Raises
    ------
    ValueError
        If ``interval_s`` is not positive.
    """

    def __init__(
        self,
        *,
        read_bytes: Callable[[], int],
        record: Callable[[uuid.UUID, int], object],
        stop: Callable[[uuid.UUID], object],
        interval_s: float,
        name: str = "cudf-polars-memory-sampler",
    ) -> None:
        if interval_s <= 0:
            raise ValueError(f"interval_s must be positive, got {interval_s}")
        self._read_bytes = read_bytes
        self._record = record
        self._stop_query = stop
        self._interval_s = interval_s
        self._name = name
        self._lock = threading.Lock()
        # Queries currently attached, with the number of attachments of each.
        self._attached: dict[uuid.UUID, int] = {}
        self._thread: threading.Thread | None = None
        # Each thread gets its own stop event, so a thread that is being
        # retired never observes the restart of its successor.
        self._thread_stop: threading.Event | None = None
        self._error: Exception | None = None
        self._closed = False

    def attach(self, query_id: uuid.UUID) -> None:
        """
        Start sampling on behalf of ``query_id``.

        Takes one reading immediately, so even a query shorter than the
        sampling interval is sampled. Starts the sampling thread if this is
        the first attached query.
        """
        with self._lock:
            if self._closed:
                return
            self._attached[query_id] = self._attached.get(query_id, 0) + 1
            if self._attached[query_id] == 1:
                self._guarded(self._sample_one, query_id)
            thread = self._thread
            if thread is None or not thread.is_alive():
                self._start_thread()

    def detach(self, query_id: uuid.UUID) -> None:
        """
        Stop sampling on behalf of ``query_id``.

        Stops the sampling thread if this was the last attached query. No
        reading is recorded for ``query_id`` after this returns.
        """
        retired: threading.Thread | None = None
        with self._lock:
            count = self._attached.get(query_id)
            if count is None:
                return
            if count > 1:
                self._attached[query_id] = count - 1
                return
            del self._attached[query_id]
            self._guarded(self._stop_query, query_id)
            if not self._attached:
                retired = self._retire_thread()
            error, self._error = self._error, None
        if retired is not None:
            retired.join()
        self._warn(error)

    def close(self) -> None:
        """Detach every query and stop the sampling thread for good."""
        with self._lock:
            self._closed = True
            for query_id in self._attached:
                self._guarded(self._stop_query, query_id)
            self._attached.clear()
            retired = self._retire_thread()
            error, self._error = self._error, None
        if retired is not None:
            retired.join()
        self._warn(error)

    def _sample_one(self, query_id: uuid.UUID) -> None:
        self._record(query_id, self._read_bytes())

    def _sample_all(self) -> None:
        nbytes = self._read_bytes()
        for query_id in self._attached:
            self._record(query_id, nbytes)

    def _guarded(self, fn: Callable[..., object], *args: object) -> bool:
        """Call ``fn`` while holding the lock; remember rather than raise errors."""
        try:
            fn(*args)
        except Exception as error:
            if self._error is None:
                self._error = error
            return False
        return True

    def _start_thread(self) -> None:
        stop = threading.Event()
        thread = threading.Thread(
            target=self._run, args=(stop,), name=self._name, daemon=True
        )
        self._thread_stop = stop
        self._thread = thread
        thread.start()

    def _retire_thread(self) -> threading.Thread | None:
        """Tell the sampling thread to exit; the caller joins it without the lock."""
        thread, self._thread = self._thread, None
        stop, self._thread_stop = self._thread_stop, None
        if stop is not None:
            stop.set()
        return thread

    def _run(self, stop: threading.Event) -> None:
        # ``Event.wait`` returns as soon as the thread is retired rather than
        # after a full interval.
        while not stop.wait(self._interval_s):
            with self._lock:
                if stop.is_set():
                    return
                if not self._guarded(self._sample_all):
                    return

    def _warn(self, error: Exception | None) -> None:
        if error is not None:
            warnings.warn(
                f"{self._name} stopped after an error: {error!r}",
                RuntimeWarning,
                stacklevel=3,
            )
