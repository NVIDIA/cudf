# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import logging
from typing import NoReturn
from unittest.mock import Mock

import pytest

from cudf_polars.streaming.actor_graph.collectives import slice_exchange
from cudf_polars.streaming.actor_graph.collectives.slice_exchange import SparseExchange


@pytest.mark.parametrize("start_early", [False, True])
def test_sparse_exchange_cancelled_finish_in_close_is_pending(
    monkeypatch,
    fake_collective,
    cancel_when,
    assert_pending_then_released,
    *,
    start_early: bool,
) -> None:
    monkeypatch.setattr(
        slice_exchange, "SparseAlltoall", lambda *_args, **_kwargs: fake_collective
    )
    comm = Mock(rank=0, nranks=2)

    async def close() -> None:
        exchange = SparseExchange(Mock(), comm, 5, sources=(1,), destinations=())
        if start_early:
            exchange.start_finish()
        await exchange.aclose()

    cancel_when(close, fake_collective.waiting)
    assert fake_collective.close_calls == 1
    assert_pending_then_released(comm, fake_collective, 5)


def test_sparse_exchange_cancelled_finish_in_body_is_pending(
    monkeypatch, fake_collective, cancel_when, assert_pending_then_released
) -> None:
    monkeypatch.setattr(
        slice_exchange, "SparseAlltoall", lambda *_args, **_kwargs: fake_collective
    )
    comm = Mock(rank=0, nranks=2)

    async def body() -> None:
        async with SparseExchange(
            Mock(), comm, 5, sources=(1,), destinations=()
        ) as exchange:
            await exchange.finish()

    cancel_when(body, fake_collective.waiting)
    assert_pending_then_released(comm, fake_collective, 5)


def test_sparse_exchange_logs_unawaited_finish_failure(
    monkeypatch, caplog, fake_collective, cancel_when, assert_pending_then_released
) -> None:
    async def fail_finish() -> NoReturn:
        raise ValueError("finish failed")

    monkeypatch.setattr(fake_collective, "insert_finished", fail_finish)
    monkeypatch.setattr(
        slice_exchange, "SparseAlltoall", lambda *_args, **_kwargs: fake_collective
    )
    comm = Mock(rank=0, nranks=2)
    entered = asyncio.Event()

    async def body() -> None:
        async with SparseExchange(
            Mock(), comm, 5, sources=(1,), destinations=()
        ) as exchange:
            exchange.start_finish()
            await asyncio.sleep(0)
            entered.set()
            await asyncio.Event().wait()

    with caplog.at_level(logging.ERROR, logger=slice_exchange.__name__):
        cancel_when(body, entered)
    assert "Finishing the sparse exchange failed" in caplog.text
    assert "finish failed" in caplog.text
    assert_pending_then_released(comm, fake_collective, 5)


def test_sparse_exchange_cancellation_does_not_wait_for_peers(
    monkeypatch, fake_collective, cancel_when, assert_pending_then_released
) -> None:
    monkeypatch.setattr(
        slice_exchange, "SparseAlltoall", lambda *_args, **_kwargs: fake_collective
    )
    comm = Mock(rank=0, nranks=2)
    entered = asyncio.Event()

    async def body() -> None:
        async with SparseExchange(Mock(), comm, 5, sources=(1,), destinations=()):
            entered.set()
            await asyncio.Event().wait()

    cancel_when(body, entered)
    assert_pending_then_released(comm, fake_collective, 5)


def test_sparse_exchange_cancelled_release_is_pending(
    monkeypatch, fake_collective, cancel_when, assert_pending_then_released
) -> None:
    fake_collective.finish_waits = False
    monkeypatch.setattr(
        slice_exchange, "SparseAlltoall", lambda *_args, **_kwargs: fake_collective
    )
    comm = Mock(rank=0, nranks=2)

    async def close() -> None:
        await SparseExchange(Mock(), comm, 5, sources=(1,), destinations=()).aclose()

    cancel_when(close, fake_collective.closing)
    assert_pending_then_released(comm, fake_collective, 5)
