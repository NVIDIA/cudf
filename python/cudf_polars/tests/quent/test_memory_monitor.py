# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for sampled memory-resource monitoring."""

from __future__ import annotations

import threading
import uuid
from unittest.mock import MagicMock

import pytest

from cudf_polars.quent import _memory_monitor


def test_samples_multiple_resources_on_one_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_thread = threading.Thread
    created_threads: list[threading.Thread] = []

    def make_thread(*args: object, **kwargs: object) -> threading.Thread:
        thread = original_thread(*args, **kwargs)  # type: ignore[arg-type]
        created_threads.append(thread)
        return thread

    monkeypatch.setattr(_memory_monitor.threading, "Thread", make_thread)

    sampled_twice = [threading.Event(), threading.Event()]
    read_counts = [0, 0]

    def reader(index: int) -> int:
        read_counts[index] += 1
        if read_counts[index] >= 2:
            sampled_twice[index].set()
        return index + 1

    with _memory_monitor.memory_resource_sampler(
        {
            uuid.uuid4(): lambda: reader(0),
            uuid.uuid4(): lambda: reader(1),
        },
        interval_s=0.001,
        quent_session=MagicMock(),
        query_id=uuid.uuid4(),
    ):
        assert all(event.wait(timeout=1) for event in sampled_twice)

    assert len(created_threads) == 1
    assert all(count >= 2 for count in read_counts)


def test_stops_started_samplers_when_a_later_reader_fails() -> None:
    first_fsm = MagicMock()
    session = MagicMock()
    observer = session.binding_context.memory_resource_sampler_observer.return_value
    observer.handle.return_value.sampling.return_value = first_fsm

    def fail() -> int:
        raise RuntimeError("host reader failed")

    with (
        pytest.raises(RuntimeError, match="host reader failed"),
        _memory_monitor.memory_resource_sampler(
            {
                uuid.uuid4(): lambda: 1,
                uuid.uuid4(): fail,
            },
            interval_s=0.1,
            quent_session=session,
            query_id=uuid.uuid4(),
        ),
    ):
        pytest.fail("sampler startup should fail")

    first_fsm.stopped.assert_called_once_with()


def test_does_not_join_thread_when_start_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fsms = [MagicMock(), MagicMock()]
    session = MagicMock()
    observer = session.binding_context.memory_resource_sampler_observer.return_value
    observer.handle.return_value.sampling.side_effect = fsms
    thread = MagicMock()
    thread.start.side_effect = RuntimeError("thread start failed")
    monkeypatch.setattr(
        _memory_monitor.threading, "Thread", MagicMock(return_value=thread)
    )

    with (
        pytest.raises(RuntimeError, match="thread start failed"),
        _memory_monitor.memory_resource_sampler(
            {
                uuid.uuid4(): lambda: 1,
                uuid.uuid4(): lambda: 2,
            },
            interval_s=0.1,
            quent_session=session,
            query_id=uuid.uuid4(),
        ),
    ):
        pytest.fail("sampler startup should fail")

    thread.join.assert_not_called()
    for fsm in fsms:
        fsm.stopped.assert_called_once_with()
