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


def test_sampler_samples_immediately_and_repeatedly() -> None:
    sampler, events = _sampler()
    query = uuid.uuid4()
    sampler.attach(query)
    try:
        # The first reading is taken by `attach`, before the thread has ticked.
        assert events[0] == ("record", query, 1000)
        _wait_for(lambda: len(_records(events)) >= 3)
    finally:
        sampler.detach(query)
    nbytes = [record[2] for record in _records(events)]
    assert nbytes == [1000 * i for i in range(1, len(nbytes) + 1)]
    assert events[-1] == ("stop", query)


def test_sampler_does_not_hold_lock_while_reading() -> None:
    sampler: MemoryResourceSampler | None = None
    reads = 0

    def read_bytes() -> int:
        nonlocal reads
        assert sampler is not None
        assert sampler._lock.acquire(blocking=False)
        sampler._lock.release()
        reads += 1
        return reads

    sampler, events = _sampler(read_bytes=read_bytes)
    query = uuid.uuid4()
    sampler.attach(query)
    try:
        # Covers both the immediate read in attach and the background thread.
        _wait_for(lambda: len(_records(events)) >= 2)
    finally:
        sampler.detach(query)


def test_sampler_close_waits_for_in_flight_read() -> None:
    read_started = threading.Event()
    finish_read = threading.Event()
    reads = 0

    def read_bytes() -> int:
        nonlocal reads
        reads += 1
        if reads > 1:
            read_started.set()
            finish_read.wait()
        return reads

    sampler, _ = _sampler(read_bytes=read_bytes)
    sampler.attach(uuid.uuid4())
    _wait_for(read_started.is_set)

    close_thread = threading.Thread(target=sampler.close)
    close_thread.start()
    try:
        assert close_thread.is_alive()
    finally:
        finish_read.set()
        close_thread.join(timeout=10)
    assert not close_thread.is_alive()


def test_sampler_takes_one_reading_even_if_query_is_shorter_than_interval() -> None:
    sampler, events = _sampler(interval_s=3600)
    query = uuid.uuid4()
    sampler.attach(query)
    sampler.detach(query)
    assert events == [("record", query, 1000), ("stop", query)]


def test_sampler_does_not_sample_after_detach() -> None:
    sampler, events = _sampler()
    query = uuid.uuid4()
    sampler.attach(query)
    _wait_for(lambda: len(_records(events)) >= 2)
    sampler.detach(query)
    stopped_at = len(events)
    time.sleep(0.02)
    assert len(events) == stopped_at, "sampled after the last query detached"
    assert not _sampler_threads()


def test_sampler_stops_without_waiting_out_the_interval() -> None:
    sampler, _ = _sampler(interval_s=3600)
    query = uuid.uuid4()
    start = time.monotonic()
    sampler.attach(query)
    sampler.detach(query)
    assert time.monotonic() - start < 10


def test_sampler_thread_runs_only_while_a_query_is_attached() -> None:
    sampler, _ = _sampler(interval_s=3600)
    q1, q2 = uuid.uuid4(), uuid.uuid4()
    assert not _sampler_threads()

    sampler.attach(q1)
    (thread,) = _sampler_threads()
    sampler.attach(q2)
    assert _sampler_threads() == [thread], "queries share one thread"

    sampler.detach(q1)
    assert thread.is_alive(), "still needed by the other query"
    sampler.detach(q2)
    assert not _sampler_threads()

    # And it can start again.
    sampler.attach(q1)
    assert len(_sampler_threads()) == 1
    sampler.detach(q1)
    assert not _sampler_threads()


def test_sampler_reads_once_per_tick_and_fans_out_to_every_query() -> None:
    reads: list[int] = []
    counter = itertools.count(1)

    def read_bytes() -> int:
        reads.append(next(counter))
        return reads[-1]

    sampler, events = _sampler(read_bytes=read_bytes)
    q1, q2 = uuid.uuid4(), uuid.uuid4()
    sampler.attach(q1)
    sampler.attach(q2)
    try:
        _wait_for(
            lambda: len(_records(events, q1)) >= 3 and len(_records(events, q2)) >= 3
        )
    finally:
        sampler.detach(q1)
        sampler.detach(q2)

    # Each shared tick delivers the same reading to both queries...
    shared = {r[2] for r in _records(events, q1)} & {r[2] for r in _records(events, q2)}
    assert shared
    # ...from one read of the device, not one read per query.
    assert len(reads) < len(_records(events))
    # Each query is stopped exactly once.
    assert events.count(("stop", q1)) == 1
    assert events.count(("stop", q2)) == 1


def test_sampler_records_nothing_for_a_query_after_it_is_stopped() -> None:
    sampler, events = _sampler()
    q1, q2 = uuid.uuid4(), uuid.uuid4()
    sampler.attach(q1)
    sampler.attach(q2)
    _wait_for(lambda: len(_records(events, q1)) >= 2)
    sampler.detach(q1)
    stop_index = events.index(("stop", q1))
    q2_at_stop = len(_records(events[:stop_index], q2))
    # The thread keeps sampling on behalf of q2.
    _wait_for(lambda: len(_records(events, q2)) >= q2_at_stop + 2)
    sampler.detach(q2)
    assert not _records(events[stop_index:], q1)


def test_sampler_attach_is_reentrant_per_query() -> None:
    sampler, events = _sampler(interval_s=3600)
    query = uuid.uuid4()
    sampler.attach(query)
    sampler.attach(query)
    sampler.detach(query)
    assert ("stop", query) not in events
    sampler.detach(query)
    assert events == [("record", query, 1000), ("stop", query)]
    # Detaching an unknown query is a no-op.
    sampler.detach(uuid.uuid4())
    assert len(events) == 2


def test_sampler_error_warns_and_does_not_fail_the_body() -> None:
    def read_bytes() -> int:
        raise OSError("device lost")

    sampler, events = _sampler(read_bytes=read_bytes)
    query = uuid.uuid4()
    sampler.attach(query)
    time.sleep(0.01)
    # The warning is emitted when the query ends, not while it is running.
    with pytest.warns(RuntimeWarning, match="device lost"):
        sampler.detach(query)
    assert events == [("stop", query)]
    assert not _sampler_threads()


def test_sampler_error_in_record_does_not_leak_to_caller() -> None:
    sampler = MemoryResourceSampler(
        read_bytes=lambda: 1,
        record=MagicMock(side_effect=RuntimeError("emit failed")),
        stop=MagicMock(),
        interval_s=3600,
    )
    query = uuid.uuid4()
    sampler.attach(query)
    with pytest.warns(RuntimeWarning, match="emit failed"):
        sampler.detach(query)


def test_sampler_close_stops_attached_queries_and_the_thread() -> None:
    sampler, events = _sampler(interval_s=3600)
    q1, q2 = uuid.uuid4(), uuid.uuid4()
    sampler.attach(q1)
    sampler.attach(q2)
    sampler.close()
    assert {("stop", q1), ("stop", q2)} <= set(events)
    assert not _sampler_threads()
    # A closed sampler ignores further attachments.
    sampler.attach(q1)
    sampler.detach(q1)
    assert len(events) == 4


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
