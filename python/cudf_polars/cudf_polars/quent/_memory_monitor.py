# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Background sampling of memory resource occupancy."""

from __future__ import annotations

import contextlib
import threading
import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

__all__ = ["MemoryResourceSampler"]


class MemoryResourceSampler[StateT]:
    """
    Sample device memory usage during query execution.

    Parameters
    ----------
    read_bytes
        Callable that returns the current number of allocated device bytes. It's
        expected that this is cheap to read and doesn't acquire any kind of lock
        to compute the allocation.
    start
        Record the initial reading and return the sampling state.
    record
        Record a subsequent reading and return the updated sampling state.
    stop
        Stop the final sampling state.
    interval_s
        Seconds between readings. Must be positive.

    Raises
    ------
    ValueError
        If ``interval_s`` is not positive.
    """

    def __init__(
        self,
        *,
        read_bytes: Callable[[], int],
        start: Callable[[int], StateT],
        record: Callable[[StateT, int], StateT],
        stop: Callable[[StateT], object],
        interval_s: float,
    ) -> None:
        if interval_s <= 0:
            raise ValueError(f"interval_s must be positive, got {interval_s}")
        self._read_bytes = read_bytes
        self._start = start
        self._record = record
        self._stop = stop
        self._interval_s = interval_s

    @contextlib.contextmanager
    def monitor(self) -> Iterator[None]:
        """Sample until the context exits."""
        stop_event = threading.Event()
        error: Exception | None = None
        state: StateT | None = None

        def start() -> bool:
            nonlocal error, state
            try:
                state = self._start(self._read_bytes())
            except Exception as current_error:
                error = current_error
                stop_event.set()
                return False
            return True

        def sample() -> bool:
            nonlocal error, state
            assert state is not None
            try:
                state = self._record(state, self._read_bytes())
            except Exception as current_error:
                error = current_error
                stop_event.set()
                return False
            return True

        def run() -> None:
            while stop_event.wait(self._interval_s) is False and sample():
                pass

        thread: threading.Thread | None = None
        if start():
            thread = threading.Thread(
                target=run, name="cudf-polars-memory-sampler", daemon=True
            )
            thread.start()
        try:
            yield
        finally:
            stop_event.set()
            if thread is not None:
                thread.join()
            if state is not None:
                try:
                    self._stop(state)
                except Exception as current_error:
                    if error is None:
                        error = current_error
            if error is not None:
                warnings.warn(
                    f"cudf-polars-memory-sampler stopped after an error: {error!r}",
                    RuntimeWarning,
                    stacklevel=3,
                )
