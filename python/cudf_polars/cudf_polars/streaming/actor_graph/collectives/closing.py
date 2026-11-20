# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cancellation-aware release of rapidsmpf collectives."""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.streaming.coll.collective import Collective


PENDING_CLOSE_TIMEOUT = 10.0
"""Seconds :func:`wait_for_pending_closes` waits by default."""


@dataclass(frozen=True, eq=False, slots=True)
class _PendingClose:
    collective: Collective
    op_id: int


_lock = threading.Lock()
_pending: dict[Communicator, set[_PendingClose]] = {}
_background_closes: set[asyncio.Task[None]] = set()


def _discard(comm: Communicator, pending: _PendingClose) -> None:
    with _lock:
        closes = _pending.get(comm)
        if closes is not None:
            closes.discard(pending)
            if not closes:
                del _pending[comm]


async def _finish(comm: Communicator, pending: _PendingClose) -> None:
    with suppress(RuntimeError):
        await pending.collective.aclose()
    _discard(comm, pending)


async def _close_in_background(
    collective: Collective, comm: Communicator, op_id: int
) -> None:
    pending = _PendingClose(collective, op_id)
    with _lock:
        _pending.setdefault(comm, set()).add(pending)
    task = asyncio.create_task(_finish(comm, pending))
    _background_closes.add(task)
    task.add_done_callback(_background_closes.discard)
    await asyncio.sleep(0)


async def _aclose(
    collective: Collective, comm: Communicator, op_id: int, *, body_failed: bool
) -> None:
    try:
        await collective.aclose()
    except asyncio.CancelledError:
        await _close_in_background(collective, comm, op_id)
        raise
    except RuntimeError:
        if not body_failed:
            raise


@asynccontextmanager
async def aclosing_collective[CollectiveT: Collective](
    collective: CollectiveT, comm: Communicator, op_id: int
) -> AsyncIterator[CollectiveT]:
    """
    Release a collective on exit without letting cancellation wait for peers.

    On normal exit, wait for ``collective.aclose()``. If the block raised, a
    ``RuntimeError`` from ``aclose()`` (for example, because
    ``insert_finished()`` was skipped) is suppressed so the original error
    propagates.

    If the block or the close is cancelled, the close continues in the
    background and the collective is recorded as pending on ``comm``. Until
    that close completes, :func:`wait_for_pending_closes` refuses new
    collectives on ``comm``, which keeps ``op_id`` from being reused.

    Parameters
    ----------
    collective
        The collective to release.
    comm
        The communicator the collective is over.
    op_id
        The operation ID of the collective.

    Yields
    ------
    The collective.
    """
    try:
        yield collective
    except asyncio.CancelledError:
        await _close_in_background(collective, comm, op_id)
        raise
    except BaseException:
        await _aclose(collective, comm, op_id, body_failed=True)
        raise
    await _aclose(collective, comm, op_id, body_failed=False)


async def _finish_all(
    comm: Communicator, pending: tuple[_PendingClose, ...], timeout: float
) -> None:
    with suppress(TimeoutError):
        async with asyncio.timeout(timeout), asyncio.TaskGroup() as tg:
            for close in pending:
                tg.create_task(_finish(comm, close))


def wait_for_pending_closes(comm: Communicator, timeout: float | None = None) -> None:
    """
    Wait for cancelled collective closes on ``comm`` to complete.

    Call this before starting new collectives on ``comm``. rapidsmpf forbids
    reusing an operation ID until its ``aclose()`` returns, and forbids new
    collectives on a communicator after a cancelled ``aclose()`` whose
    collective never completes.

    Parameters
    ----------
    comm
        The communicator to check.
    timeout
        Seconds to wait for the pending closes to complete. Defaults to
        :data:`PENDING_CLOSE_TIMEOUT`.

    Raises
    ------
    RuntimeError
        If a pending close on ``comm`` does not complete within ``timeout``.
    """
    with _lock:
        pending = tuple(_pending.get(comm, ()))
    if not pending:
        return
    if timeout is None:
        timeout = PENDING_CLOSE_TIMEOUT
    with ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="cudf-polars-pending-close"
    ) as executor:
        executor.submit(asyncio.run, _finish_all(comm, pending, timeout)).result()
    with _lock:
        op_ids = sorted(close.op_id for close in _pending.get(comm, ()))
    if op_ids:
        raise RuntimeError(
            f"The close of the collectives with operation IDs {op_ids} was "
            f"cancelled and did not complete within {timeout} seconds. New "
            "collectives cannot run on this communicator until those "
            "collectives complete on every rank. Create a new engine to run "
            "more queries."
        )
