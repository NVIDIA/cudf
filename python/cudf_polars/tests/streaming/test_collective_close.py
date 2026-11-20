# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, NoReturn
from unittest.mock import Mock

import pytest

import cudf_polars.engine.core
from cudf_polars.streaming.actor_graph.collectives import closing
from cudf_polars.streaming.actor_graph.collectives.closing import (
    aclosing_collective,
    discard_pending_closes,
    wait_for_pending_closes,
)

if TYPE_CHECKING:
    from collections.abc import Callable


def test_cancelled_close_blocks_communicator_until_complete(
    fake_collective: Any,
    leave_pending_close: Callable[[Any, int], None],
    assert_pending_then_released: Callable[[Any, Any, int], None],
) -> None:
    comm = Mock()
    leave_pending_close(comm, 7)
    assert fake_collective.close_calls == 1
    assert_pending_then_released(comm, fake_collective, 7)


def test_gate_timeout_marks_communicator_unusable(
    fake_collective: Any, leave_pending_close: Callable[[Any, int], None]
) -> None:
    comm = Mock()
    leave_pending_close(comm, 7)
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


def test_gate_on_one_communicator_does_not_delay_another(
    fake_collective: Any,
    leave_pending_close: Callable[[Any, int], None],
    cancel_when: Callable[..., None],
    timeout_seconds: int,
) -> None:
    blocked_comm, other_comm = Mock(), Mock()
    leave_pending_close(blocked_comm, 1)
    other = type(fake_collective)()

    async def body() -> None:
        async with aclosing_collective(other, other_comm, 2):
            await other.extract_all(ordered=True)

    try:
        cancel_when(body, other.waiting)
    finally:
        other.release()

    with ThreadPoolExecutor(max_workers=1) as executor:
        blocked = executor.submit(wait_for_pending_closes, blocked_comm, 2)
        deadline = time.monotonic() + timeout_seconds
        while fake_collective.close_calls < 2:
            assert time.monotonic() < deadline
            time.sleep(0.001)
        start = time.monotonic()
        wait_for_pending_closes(other_comm, timeout=timeout_seconds)
        assert time.monotonic() - start < 1
        with pytest.raises(RuntimeError, match=r"operation IDs \[1\]"):
            blocked.result()


def test_discard_pending_closes_forgets_communicator(
    pending_closes: dict[Any, Any], leave_pending_close: Callable[[Any, int], None]
) -> None:
    comm = Mock()
    leave_pending_close(comm, 7)
    with pytest.raises(RuntimeError, match=r"operation IDs \[7\]"):
        wait_for_pending_closes(comm, timeout=0)

    discard_pending_closes(comm)
    assert pending_closes == {}
    wait_for_pending_closes(comm, timeout=0)


def test_background_close_failure_is_logged_and_blocks_communicator(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    pending_closes: dict[Any, Any],
    fake_collective: Any,
    leave_pending_close: Callable[[Any, int], None],
    timeout_seconds: int,
) -> None:
    async def fail_close() -> NoReturn:
        raise ValueError("close failed")

    monkeypatch.setattr(fake_collective, "aclose", fail_close)
    comm = Mock()
    with caplog.at_level(logging.ERROR, logger=closing.__name__):
        leave_pending_close(comm, 7)
    assert "operation ID 7 failed" in caplog.text
    assert [close.op_id for close in pending_closes[comm]] == [7]

    with pytest.raises(
        RuntimeError, match=r"operation IDs \[7\] was cancelled and failed"
    ) as excinfo:
        wait_for_pending_closes(comm, timeout=timeout_seconds)
    assert isinstance(excinfo.value.__cause__, ExceptionGroup)
    assert excinfo.value.__cause__.subgroup(ValueError) is not None


def test_background_close_of_skipped_finish_is_released(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    pending_closes: dict[Any, Any],
    fake_collective: Any,
    leave_pending_close: Callable[[Any, int], None],
) -> None:
    async def skipped_finish_close() -> NoReturn:
        raise RuntimeError("aclose() called before insert_finished()")

    monkeypatch.setattr(fake_collective, "aclose", skipped_finish_close)
    with caplog.at_level(logging.ERROR, logger=closing.__name__):
        leave_pending_close(Mock(), 7)
    assert caplog.text == ""
    assert pending_closes == {}


def test_cancelled_close_only_blocks_its_communicator(
    pending_closes: dict[Any, Any], leave_pending_close: Callable[[Any, int], None]
) -> None:
    comm = Mock()
    leave_pending_close(comm, 7)
    wait_for_pending_closes(Mock(), timeout=0)
    assert set(pending_closes) == {comm}


def test_cancellation_while_closing_does_not_wait(
    pending_closes: dict[Any, Any],
    fake_collective: Any,
    cancel_when: Callable[..., None],
) -> None:
    comm = Mock()

    async def body() -> None:
        async with aclosing_collective(fake_collective, comm, 3):
            pass

    cancel_when(body, fake_collective.closing)
    assert [close.op_id for close in pending_closes[comm]] == [3]


def test_close_is_awaited_on_success(
    pending_closes: dict[Any, Any], fake_collective: Any
) -> None:
    fake_collective.release()

    async def body() -> None:
        async with aclosing_collective(fake_collective, Mock(), 0):
            pass

    asyncio.run(body())
    assert fake_collective.close_calls == 1
    assert pending_closes == {}


class SkippedFinishCollective:
    async def aclose(self) -> NoReturn:
        raise RuntimeError("aclose() called before insert_finished()")


def test_close_error_raises_on_success(pending_closes: dict[Any, Any]) -> None:
    async def run() -> None:
        async with aclosing_collective(SkippedFinishCollective(), Mock(), 0):
            pass

    with pytest.raises(RuntimeError, match="insert_finished"):
        asyncio.run(run())


def test_close_error_does_not_mask_body_error(pending_closes: dict[Any, Any]) -> None:
    async def run() -> None:
        async with aclosing_collective(SkippedFinishCollective(), Mock(), 0):
            raise ValueError("body failed")

    with pytest.raises(ValueError, match="body failed"):
        asyncio.run(run())


class FailingCloseCollective:
    async def aclose(self) -> NoReturn:
        raise RuntimeError("wait failed")


def test_close_failure_is_not_masked_by_body_error(
    pending_closes: dict[Any, Any],
) -> None:
    async def run() -> None:
        async with aclosing_collective(FailingCloseCollective(), Mock(), 0):
            raise ValueError("body failed")

    with pytest.raises(RuntimeError, match="wait failed") as excinfo:
        asyncio.run(run())
    assert isinstance(excinfo.value.__context__, ValueError)


def test_evaluate_on_rank_refuses_communicator_with_pending_close(
    monkeypatch: pytest.MonkeyPatch,
    pending_closes: dict[Any, Any],
    leave_pending_close: Callable[[Any, int], None],
) -> None:
    comm = Mock()
    leave_pending_close(comm, 5)

    def fail_evaluate_on_rank(*args: object, **kwargs: object) -> NoReturn:
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
    monkeypatch: pytest.MonkeyPatch,
    pending_closes: dict[Any, Any],
    leave_pending_close: Callable[[Any, int], None],
) -> None:
    comm = Mock()
    leave_pending_close(comm, 5)

    def fail_all_gather(*args: object, **kwargs: object) -> NoReturn:
        pytest.fail("the collective must not start")

    monkeypatch.setattr(closing, "PENDING_CLOSE_TIMEOUT", 0)
    monkeypatch.setattr(cudf_polars.engine.core, "AllGather", fail_all_gather)
    with pytest.raises(RuntimeError, match=r"operation IDs \[5\]"):
        cudf_polars.engine.core.all_gather_host_data(comm, Mock(), 0, b"")
