# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from typing import Any
from unittest.mock import Mock

import pytest

import cudf_polars.engine.core
from cudf_polars.streaming.actor_graph.collectives import closing
from cudf_polars.streaming.actor_graph.collectives.closing import (
    aclosing_collective,
    discard_pending_closes,
    wait_for_pending_closes,
)


class BlockedCollective:
    """Collective whose ``aclose()`` waits until ``released`` is set."""

    def __init__(self) -> None:
        self.close_started = threading.Event()
        self.released = threading.Event()
        self.closed = threading.Event()

    async def aclose(self) -> None:
        self.close_started.set()
        while not self.released.is_set():
            await asyncio.sleep(0.001)
        self.closed.set()


@pytest.fixture
def collective():
    collective = BlockedCollective()
    yield collective
    collective.released.set()


def cancel_inside(
    collective: BlockedCollective, comm: object, op_id: int, timeout: float
) -> None:
    async def run() -> None:
        entered = asyncio.Event()

        async def body() -> None:
            async with aclosing_collective(collective, comm, op_id):
                entered.set()
                await asyncio.Event().wait()

        task = asyncio.create_task(body())
        await asyncio.wait_for(entered.wait(), timeout=timeout)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=timeout)

    asyncio.run(run())


def test_cancelled_close_blocks_communicator_until_complete(
    pending_closes, collective, timeout_seconds: int
) -> None:
    comm = Mock()
    cancel_inside(collective, comm, 7, timeout_seconds)
    assert collective.close_started.is_set()
    assert not collective.closed.is_set()
    assert [close.op_id for close in pending_closes[comm]] == [7]

    collective.released.set()
    wait_for_pending_closes(comm, timeout=timeout_seconds)
    assert collective.closed.is_set()
    assert pending_closes == {}


def test_gate_timeout_marks_communicator_unusable(
    pending_closes, fake_collective, cancel_when
) -> None:
    comm = Mock()

    async def body() -> None:
        async with aclosing_collective(fake_collective, comm, 7):
            await fake_collective.extract_all(ordered=True)

    cancel_when(body, fake_collective.waiting)
    close_calls = fake_collective.close_calls
    start = time.monotonic()
    with pytest.raises(RuntimeError, match=r"operation IDs \[7\].*every rank"):
        wait_for_pending_closes(comm, timeout=0.2)
    assert time.monotonic() - start >= 0.2
    assert fake_collective.close_calls == close_calls + 1

    fake_collective.release()
    with pytest.raises(RuntimeError, match=r"operation IDs \[7\]"):
        wait_for_pending_closes(comm, timeout=30)
    assert fake_collective.close_calls == close_calls + 1


def test_discard_pending_closes_forgets_communicator(
    pending_closes, fake_collective, cancel_when
) -> None:
    comm = Mock()

    async def body() -> None:
        async with aclosing_collective(fake_collective, comm, 7):
            await fake_collective.extract_all(ordered=True)

    cancel_when(body, fake_collective.waiting)
    with pytest.raises(RuntimeError, match=r"operation IDs \[7\]"):
        wait_for_pending_closes(comm, timeout=0)

    discard_pending_closes(comm)
    assert pending_closes == {}
    wait_for_pending_closes(comm, timeout=0)


def test_background_close_failure_is_logged_and_blocks_communicator(
    monkeypatch, caplog, pending_closes, fake_collective, cancel_when, timeout_seconds
) -> None:
    comm = Mock()

    async def fail_close() -> None:
        raise ValueError("close failed")

    async def body() -> None:
        async with aclosing_collective(fake_collective, comm, 7):
            await fake_collective.extract_all(ordered=True)

    monkeypatch.setattr(fake_collective, "aclose", fail_close)
    with caplog.at_level(logging.ERROR, logger=closing.__name__):
        cancel_when(body, fake_collective.waiting)
    assert "operation ID 7 failed" in caplog.text
    assert [close.op_id for close in pending_closes[comm]] == [7]

    with pytest.raises(
        RuntimeError, match=r"operation IDs \[7\] was cancelled and failed"
    ) as excinfo:
        wait_for_pending_closes(comm, timeout=timeout_seconds)
    assert isinstance(excinfo.value.__cause__, ExceptionGroup)
    assert excinfo.value.__cause__.subgroup(ValueError) is not None


def test_background_close_of_skipped_finish_is_released(
    monkeypatch, caplog, pending_closes, fake_collective, cancel_when
) -> None:
    async def skipped_finish_close() -> None:
        raise RuntimeError("aclose() called before insert_finished()")

    async def body() -> None:
        async with aclosing_collective(fake_collective, Mock(), 7):
            await fake_collective.extract_all(ordered=True)

    monkeypatch.setattr(fake_collective, "aclose", skipped_finish_close)
    with caplog.at_level(logging.ERROR, logger=closing.__name__):
        cancel_when(body, fake_collective.waiting)
    assert caplog.text == ""
    assert pending_closes == {}


def test_cancelled_close_only_blocks_its_communicator(
    pending_closes, collective, timeout_seconds: int
) -> None:
    comm = Mock()
    cancel_inside(collective, comm, 7, timeout_seconds)
    wait_for_pending_closes(Mock(), timeout=0)
    assert set(pending_closes) == {comm}


def test_cancellation_while_closing_does_not_wait(
    pending_closes, collective, timeout_seconds: int
) -> None:
    comm = Mock()

    async def run() -> None:
        async def body() -> None:
            async with aclosing_collective(collective, comm, 3):
                pass

        async def close_started() -> None:
            while not collective.close_started.is_set():
                await asyncio.sleep(0.001)

        task = asyncio.create_task(body())
        await asyncio.wait_for(close_started(), timeout=timeout_seconds)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=timeout_seconds)

    asyncio.run(run())
    assert [close.op_id for close in pending_closes[comm]] == [3]


def test_close_is_awaited_on_success(pending_closes, collective) -> None:
    collective.released.set()

    async def run() -> None:
        async with aclosing_collective(collective, Mock(), 0):
            pass

    asyncio.run(run())
    assert collective.closed.is_set()
    assert pending_closes == {}


class SkippedFinishCollective:
    async def aclose(self) -> None:
        raise RuntimeError("aclose() called before insert_finished()")


def test_close_error_raises_on_success(pending_closes) -> None:
    async def run() -> None:
        async with aclosing_collective(SkippedFinishCollective(), Mock(), 0):
            pass

    with pytest.raises(RuntimeError, match="insert_finished"):
        asyncio.run(run())


def test_close_error_does_not_mask_body_error(pending_closes) -> None:
    async def run() -> None:
        async with aclosing_collective(SkippedFinishCollective(), Mock(), 0):
            raise ValueError("body failed")

    with pytest.raises(ValueError, match="body failed"):
        asyncio.run(run())


def test_evaluate_on_rank_refuses_communicator_with_pending_close(
    monkeypatch, pending_closes, collective, timeout_seconds: int
) -> None:
    comm = Mock()
    cancel_inside(collective, comm, 5, timeout_seconds)

    def fail_evaluate_on_rank(*args, **kwargs):
        pytest.fail("the query must not start")

    monkeypatch.setattr(closing, "PENDING_CLOSE_TIMEOUT", 0)
    monkeypatch.setattr(
        cudf_polars.engine.core, "_evaluate_on_rank", fail_evaluate_on_rank
    )
    unused: Any = None
    with pytest.raises(RuntimeError, match=r"operation IDs \[5\]"):
        cudf_polars.engine.core.evaluate_on_rank(
            unused, comm, unused, unused, unused, query_id=uuid.uuid4()
        )


def test_all_gather_host_data_refuses_communicator_with_pending_close(
    monkeypatch, pending_closes, collective, timeout_seconds: int
) -> None:
    comm = Mock()
    cancel_inside(collective, comm, 5, timeout_seconds)

    def fail_all_gather(*args, **kwargs):
        pytest.fail("the collective must not start")

    monkeypatch.setattr(closing, "PENDING_CLOSE_TIMEOUT", 0)
    monkeypatch.setattr(cudf_polars.engine.core, "AllGather", fail_all_gather)
    with pytest.raises(RuntimeError, match=r"operation IDs \[5\]"):
        cudf_polars.engine.core.all_gather_host_data(comm, Mock(), 0, b"")
