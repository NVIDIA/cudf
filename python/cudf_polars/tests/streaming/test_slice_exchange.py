# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from unittest.mock import Mock

import pytest

from cudf_polars.streaming.actor_graph.collectives import slice_exchange
from cudf_polars.streaming.actor_graph.collectives.slice_exchange import SparseExchange


@pytest.mark.parametrize("start_early", [False, True])
def test_sparse_exchange_close_survives_cancellation(
    monkeypatch, timeout_seconds: int, *, start_early: bool
) -> None:
    """Finish before releasing the exchange despite repeated cancellation."""

    async def run() -> None:
        started = asyncio.Event()
        release = asyncio.Event()
        finished = asyncio.Event()

        class DelayedExchange:
            async def insert_finished(self, _context) -> None:
                started.set()
                await release.wait()
                finished.set()

        monkeypatch.setattr(
            slice_exchange,
            "SparseAlltoall",
            lambda *_args, **_kwargs: DelayedExchange(),
        )
        exchange = SparseExchange(
            Mock(), Mock(rank=0, nranks=2), 0, sources=(1,), destinations=()
        )

        if start_early:
            exchange.start_finish()
        task = asyncio.create_task(exchange.aclose())
        try:
            await asyncio.wait_for(started.wait(), timeout=timeout_seconds)
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert finished.is_set()
            await exchange.aclose()
        finally:
            release.set()

    asyncio.run(run())
