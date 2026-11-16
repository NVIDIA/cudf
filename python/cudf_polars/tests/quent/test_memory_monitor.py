# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Quent memory resource sampler."""

from __future__ import annotations

import threading
import time
import uuid
from unittest.mock import MagicMock

import pytest

from cudf_polars.quent._context import QuentConfig
from cudf_polars.quent._memory_monitor import MemoryResourceSampler
from cudf_polars.quent._runtime import QuentWorkerRuntime


def _wait_for(condition, timeout_s: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.001)


def _sampler_threads() -> list[threading.Thread]:
    return [
        thread
        for thread in threading.enumerate()
        if thread.name == "cudf-polars-memory-sampler"
    ]


@pytest.mark.parametrize(
    "callback", ["read-immediate", "read-periodic", "record", "stop"]
)
def test_sampler_callback_errors_warn_instead_of_raising(callback: str) -> None:
    calls = 0

    def read_bytes() -> int:
        nonlocal calls
        calls += 1
        if callback == "read-immediate" or (callback == "read-periodic" and calls > 1):
            raise OSError("callback failed")
        return calls

    def record(*_: object) -> None:
        if callback == "record" and calls > 1:
            raise OSError("callback failed")

    def stop() -> None:
        if callback == "stop":
            raise OSError("callback failed")

    state = object()

    def start(nbytes: int) -> object:
        record(nbytes)
        return state

    def transition(current: object, nbytes: int) -> object:
        record(nbytes)
        return current

    sampler = MemoryResourceSampler(
        read_bytes=read_bytes,
        start=start,
        record=transition,
        stop=lambda _: stop(),
        interval_s=0.001,
    )
    if callback in {"read-periodic", "record"}:
        with (
            pytest.warns(RuntimeWarning, match="callback failed"),
            sampler.monitor(),
        ):
            _wait_for(lambda: calls > 1)


@pytest.mark.parametrize("interval_s", [0, -1.0])
def test_sampler_rejects_non_positive_interval(interval_s: float) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        MemoryResourceSampler(
            read_bytes=lambda: 0,
            start=lambda _: object(),
            record=lambda state, _: state,
            stop=lambda _: None,
            interval_s=interval_s,
        )


def _runtime(
    *,
    allocated_bytes,
    interval_s: float | None,
    events: list[tuple],
) -> tuple[QuentWorkerRuntime, uuid.UUID]:
    """Build a runtime whose sampler observer records ordered events."""

    def transition(previous, query, memory):
        handle = MagicMock()
        handle.sampling.side_effect = lambda *, query, memory: transition(
            handle, query, memory
        )
        handle.stopped.side_effect = lambda: events.append(("stopped", handle))
        events.append(("sampling", previous, handle, query, memory))
        return handle

    session = MagicMock()
    observer = session.binding_context.memory_resource_sampler_observer.return_value
    observer.handle.return_value.sampling.side_effect = lambda *, query, memory: (
        transition(None, query, memory)
    )
    resources = MagicMock()
    resources.sampled_memory_resource_id = uuid.uuid4()
    runtime = QuentWorkerRuntime(
        config=QuentConfig(device_memory_poll_interval_s=interval_s),
        session=session,
        worker_resources=resources,
        _worker_handle=MagicMock(),
        allocated_bytes=allocated_bytes,
    )
    return runtime, resources.sampled_memory_resource_id


def test_monitor_device_memory_stops_fsm_when_query_fails() -> None:
    events: list[tuple] = []
    runtime, _ = _runtime(allocated_bytes=lambda: 1, interval_s=3600, events=events)

    def failing_query() -> None:
        with runtime.monitor_device_memory(uuid.uuid4()):
            _wait_for(lambda: bool(events))
            raise RuntimeError("query failed")

    with pytest.raises(RuntimeError, match="query failed"):
        failing_query()

    assert [event[0] for event in events] == ["sampling", "stopped"]


def test_close_after_query_releases_allocated_bytes_reader() -> None:
    events: list[tuple] = []
    runtime, _ = _runtime(allocated_bytes=lambda: 1, interval_s=3600, events=events)
    query_id = uuid.uuid4()
    with runtime.monitor_device_memory(query_id):
        assert _sampler_threads()

    assert not _sampler_threads()
    assert [event[0] for event in events] == ["sampling", "stopped"]
    runtime.close()

    assert runtime.allocated_bytes is None


@pytest.mark.parametrize("interval_s", [0, -1.0])
def test_quent_config_rejects_non_positive_poll_interval(interval_s: float) -> None:
    with pytest.raises(ValueError, match="device_memory_poll_interval_s"):
        QuentConfig(device_memory_poll_interval_s=interval_s)


@pytest.mark.parametrize("interval_s", [None, 0.25])
def test_quent_config_poll_interval_roundtrips(tmp_path, interval_s) -> None:
    config = QuentConfig(
        output_root=str(tmp_path), device_memory_poll_interval_s=interval_s
    )
    assert QuentConfig._deserialize(config._serialize()) == config


def test_quent_config_polls_once_per_second_by_default() -> None:
    assert QuentConfig().device_memory_poll_interval_s == 0.1
