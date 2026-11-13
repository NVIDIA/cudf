# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Quent memory resource sampler."""

from __future__ import annotations

import itertools
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


def _sampler(
    *,
    interval_s: float = 0.001,
    read_bytes=None,
    events: list[tuple] | None = None,
) -> tuple[MemoryResourceSampler, list[tuple]]:
    """Build a sampler that records ``("record", query, bytes)``/``("stop", query)``."""
    events = [] if events is None else events
    counter = itertools.count(1)
    sampler = MemoryResourceSampler(
        read_bytes=read_bytes or (lambda: next(counter) * 1000),
        record=lambda query, nbytes: events.append(("record", query, nbytes)),
        stop=lambda query: events.append(("stop", query)),
        interval_s=interval_s,
    )
    return sampler, events


def _records(events: list[tuple], query: uuid.UUID | None = None) -> list[tuple]:
    return [
        event
        for event in list(events)
        if event[0] == "record" and (query is None or event[1] == query)
    ]


def _sampler_threads() -> list[threading.Thread]:
    return [
        thread
        for thread in threading.enumerate()
        if thread.name == "cudf-polars-memory-sampler"
    ]


def test_sampler_lifecycle_and_fanout() -> None:
    sampler, events = _sampler()
    q1, q2 = uuid.uuid4(), uuid.uuid4()
    sampler.attach(q1)
    sampler.attach(q2)

    assert _records(events, q1)[0] == ("record", q1, 1000)
    assert _records(events, q2)[0] == ("record", q2, 2000)
    _wait_for(
        lambda: bool(
            {event[2] for event in _records(events, q1)}
            & {event[2] for event in _records(events, q2)}
        )
    )

    sampler.detach(q1)
    stopped_at = len(events)
    _wait_for(lambda: len(events) > stopped_at)
    assert not _records(events[stopped_at:], q1)

    sampler.detach(q2)
    finished_at = len(events)
    time.sleep(0.01)
    assert len(events) == finished_at
    assert events.count(("stop", q1)) == events.count(("stop", q2)) == 1
    assert not _sampler_threads()


def test_sampler_attach_is_refcounted_and_restartable() -> None:
    sampler, events = _sampler(interval_s=3600)
    query = uuid.uuid4()
    sampler.attach(query)
    sampler.attach(query)
    sampler.detach(query)
    assert ("stop", query) not in events
    sampler.detach(query)
    sampler.detach(query)  # Unknown queries are ignored.

    sampler.attach(query)
    sampler.detach(query)
    assert events == [
        ("record", query, 1000),
        ("stop", query),
        ("record", query, 2000),
        ("stop", query),
    ]


def test_sampler_close_waits_for_read_and_stops_all_queries() -> None:
    read_started = threading.Event()
    finish_read = threading.Event()
    reads = 0

    def read_bytes() -> int:
        nonlocal reads
        reads += 1
        if reads > 2:
            read_started.set()
            finish_read.wait()
        return reads

    sampler, events = _sampler(read_bytes=read_bytes)
    queries = (uuid.uuid4(), uuid.uuid4())
    for query in queries:
        sampler.attach(query)
    _wait_for(read_started.is_set)

    close_thread = threading.Thread(target=sampler.close)
    close_thread.start()
    try:
        assert close_thread.is_alive()
    finally:
        finish_read.set()
        close_thread.join(timeout=10)

    assert not close_thread.is_alive()
    assert {("stop", query) for query in queries} <= set(events)
    sampler.close()  # Idempotent.
    sampler.attach(queries[0])  # Closed samplers remain closed.
    sampler.detach(queries[0])


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

    def stop(_: uuid.UUID) -> None:
        if callback == "stop":
            raise OSError("callback failed")

    sampler = MemoryResourceSampler(
        read_bytes=read_bytes,
        record=record,
        stop=stop,
        interval_s=0.001,
    )
    query = uuid.uuid4()
    sampler.attach(query)
    if callback in {"read-periodic", "record"}:
        _wait_for(lambda: calls > 1)
    with pytest.warns(RuntimeWarning, match="callback failed"):
        sampler.detach(query)


@pytest.mark.parametrize("interval_s", [0, -1.0])
def test_sampler_rejects_non_positive_interval(interval_s: float) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        MemoryResourceSampler(
            read_bytes=lambda: 0,
            record=lambda *_: None,
            stop=lambda *_: None,
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


def test_close_stops_sampling_and_releases_allocated_bytes_reader() -> None:
    events: list[tuple] = []
    runtime, _ = _runtime(allocated_bytes=lambda: 1, interval_s=3600, events=events)
    query_id = uuid.uuid4()
    monitor = runtime.monitor_device_memory(query_id)
    monitor.__enter__()
    assert _sampler_threads()

    runtime.close()

    assert runtime.allocated_bytes is None
    assert not _sampler_threads()
    assert [event[0] for event in events] == ["sampling", "stopped"]
    # A query that outlives the worker detaches harmlessly.
    monitor.__exit__(None, None, None)
    assert [event[0] for event in events] == ["sampling", "stopped"]


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
