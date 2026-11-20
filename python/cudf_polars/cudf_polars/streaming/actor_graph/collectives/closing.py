# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cancellation-aware release of rapidsmpf collectives."""

from __future__ import annotations

import asyncio
import logging
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
"""
Seconds :func:`wait_for_pending_closes` waits by default.

A cancelled close usually completes soon after the query ends, once the peer
ranks, which are often cancelled too, insert their finish markers. The wait
covers that delay. If a peer never created the collective, the close never
completes, and the wait bounds how long the next query blocks before it
fails.
"""


@dataclass(frozen=True, eq=False, slots=True)
class _PendingClose:
    collective: Collective
    op_id: int


_SKIPPED_FINISH = "aclose() called before insert_finished()"

_logger = logging.getLogger(__name__)
_lock = threading.Lock()
_gate_locks: dict[Communicator, threading.Lock] = {}
_pending: dict[Communicator, set[_PendingClose]] = {}
_unusable: dict[Communicator, str] = {}
_background_closes: set[asyncio.Task[None]] = set()


def _discard(comm: Communicator, pending: _PendingClose) -> None:
    with _lock:
        closes = _pending.get(comm)
        if closes is not None:
            closes.discard(pending)
            if not closes:
                del _pending[comm]


def _is_skipped_finish(exc: RuntimeError) -> bool:
    return str(exc) == _SKIPPED_FINISH


async def _finish(comm: Communicator, pending: _PendingClose) -> None:
    try:
        await pending.collective.aclose()
    except RuntimeError as exc:
        if not _is_skipped_finish(exc):
            raise
    _discard(comm, pending)


async def _finish_or_log(comm: Communicator, pending: _PendingClose) -> None:
    try:
        await _finish(comm, pending)
    except Exception:
        _logger.exception(
            "The cancelled close of the collective with operation ID %d failed",
            pending.op_id,
        )


async def _close_in_background(
    collective: Collective, comm: Communicator, op_id: int
) -> None:
    pending = _PendingClose(collective, op_id)
    with _lock:
        _pending.setdefault(comm, set()).add(pending)
    task = asyncio.create_task(_finish_or_log(comm, pending))
    _background_closes.add(task)
    task.add_done_callback(_background_closes.discard)
    await asyncio.sleep(0)


async def aclose_collective(
    collective: Collective,
    comm: Communicator,
    op_id: int,
    *,
    error: BaseException | None = None,
) -> None:
    """
    Release a collective without letting cancellation wait for peers.

    Without ``error``, wait for ``collective.aclose()``. With ``error``, the
    ``RuntimeError`` that ``aclose()`` raises because ``insert_finished()``
    was skipped is suppressed so that the caller can raise ``error`` instead.
    Other close failures propagate.

    If ``error`` is a cancellation, or the close is cancelled, the close
    continues in the background and the collective is recorded as pending on
    ``comm``. Until that close completes, :func:`wait_for_pending_closes`
    refuses new collectives on ``comm``, which keeps ``op_id`` from being
    reused.

    Parameters
    ----------
    collective
        The collective to release.
    comm
        The communicator the collective is over.
    op_id
        The operation ID of the collective.
    error
        The exception that ended the use of the collective, if any.
    """
    if isinstance(error, asyncio.CancelledError):
        await _close_in_background(collective, comm, op_id)
        return
    try:
        await collective.aclose()
    except asyncio.CancelledError:
        await _close_in_background(collective, comm, op_id)
        raise
    except RuntimeError as exc:
        if error is None or not _is_skipped_finish(exc):
            raise


@asynccontextmanager
async def aclosing_collective[CollectiveT: Collective](
    collective: CollectiveT, comm: Communicator, op_id: int
) -> AsyncIterator[CollectiveT]:
    """
    Release a collective on exit without letting cancellation wait for peers.

    See :func:`aclose_collective` for how the collective is released.

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
    except BaseException as exc:
        await aclose_collective(collective, comm, op_id, error=exc)
        raise
    await aclose_collective(collective, comm, op_id)


def discard_pending_closes(comm: Communicator) -> None:
    """
    Forget the pending closes and unusable mark of ``comm``.

    Call this when ``comm`` is no longer used, so that the registry does not
    keep ``comm`` and its collectives alive. Closes that are still running
    continue in rapidsmpf.

    Parameters
    ----------
    comm
        The communicator that is no longer used.
    """
    with _lock:
        _pending.pop(comm, None)
        _unusable.pop(comm, None)
        _gate_locks.pop(comm, None)


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

    If a pending close fails or does not complete within ``timeout``,
    ``comm`` is marked unusable, and every later call raises at once.

    Calls that wait are serialized per communicator, so that two threads
    never close the same collective at the same time.

    The check is local to this rank. When it raises on one rank, the peer
    ranks do not know, and they can hang in their next collective while they
    wait for this rank. Every rank must shut down its engine and create a new
    one with a new communicator. A communicator that the caller owns stays
    unusable.

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
        If a pending close on ``comm`` fails or does not complete within
        ``timeout``, or if ``comm`` was already marked unusable.
    """
    if not _pending_closes(comm):
        return
    if timeout is None:
        timeout = PENDING_CLOSE_TIMEOUT
    with _lock:
        gate_lock = _gate_locks.setdefault(comm, threading.Lock())
    with gate_lock:
        pending = _pending_closes(comm)
        if not pending:
            return
        failure: Exception | None = None
        try:
            with ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="cudf-polars-pending-close"
            ) as executor:
                executor.submit(
                    asyncio.run, _finish_all(comm, pending, timeout)
                ).result()
        except Exception as exc:
            failure = exc
        with _lock:
            op_ids = sorted(close.op_id for close in _pending.get(comm, ()))
            if not op_ids:
                return
            outcome = (
                f"did not complete within {timeout} seconds"
                if failure is None
                else "failed"
            )
            message = _unusable[comm] = (
                f"The close of the collectives with operation IDs {op_ids} was "
                f"cancelled and {outcome}, so this communicator cannot run new "
                "collectives. Peer ranks may hang in their next collective. "
                "Shut down the engine on every rank and create a new one with "
                "a new communicator."
            )
    raise RuntimeError(message) from failure


def _pending_closes(comm: Communicator) -> tuple[_PendingClose, ...]:
    with _lock:
        if (message := _unusable.get(comm)) is not None:
            raise RuntimeError(message)
        return tuple(_pending.get(comm, ()))
