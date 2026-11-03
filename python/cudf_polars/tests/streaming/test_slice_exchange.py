# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest

import polars as pl

from cudf_streaming.table_chunk import (
    TableChunk,
    make_table_chunks_available_or_wait,
)

from cudf_polars.containers import DataFrame, DataType
from cudf_polars.streaming.actor_graph.collectives import slice_exchange
from cudf_polars.streaming.actor_graph.collectives.common import reserve_op_id
from cudf_polars.streaming.actor_graph.collectives.slice_exchange import (
    SliceRoute,
    SparseExchange,
)

if TYPE_CHECKING:
    from rapidsmpf.streaming.core.context import Context
    from rmm.pylibrmm.stream import Stream


def _chunk(values: list[int], context: Context, stream: Stream) -> TableChunk:
    frame = DataFrame.from_polars(
        pl.DataFrame({"value": pl.Series(values, dtype=pl.Int64())}), stream
    )
    return TableChunk.from_pylibcudf_table(
        frame.table,
        stream,
        exclusive_view=False,
        br=context.br(),
    )


def _values(chunk: TableChunk) -> list[int]:
    return (
        DataFrame.from_table(
            chunk.table_view(),
            ["value"],
            [DataType(pl.Int64())],
            chunk.stream,
        )
        .to_polars()["value"]
        .to_list()
    )


@pytest.mark.spmd
def test_sparse_exchange_sends_slice_and_chunk(spmd_engine) -> None:
    """A borrowed slice leaves its source intact and a whole chunk arrives."""
    if spmd_engine.comm.nranks != 2:
        pytest.skip("This test expects exactly two ranks.")

    async def run() -> None:
        context = spmd_engine.context
        rank = spmd_engine.comm.rank
        stream = context.br().stream_pool.get_stream()
        with reserve_op_id() as op_id:
            async with SparseExchange(
                context,
                spmd_engine.comm,
                op_id,
                sources=(0,) if rank == 1 else (),
                destinations=(1,) if rank == 0 else (),
            ) as exchange:
                if rank == 0:
                    source = _chunk([0, 1, 2, 3], context, stream)
                    await exchange.insert_slices(source, (SliceRoute(1, 1, 3),))
                    assert _values(source) == [0, 1, 2, 3]
                    await exchange.insert_chunk(1, _chunk([4, 5], context, stream))
                await exchange.finish()
                if rank == 1:
                    remote = exchange.extract(0)
                    chunks, _ = await make_table_chunks_available_or_wait(
                        context,
                        [
                            TableChunk.from_message(message, br=context.br())
                            for message in remote
                        ],
                        reserve_extra=0,
                        net_memory_delta=0,
                    )
                    assert [_values(chunk) for chunk in chunks] == [[1, 2], [4, 5]]

    asyncio.run(run())


def test_sparse_exchange_finish_survives_cancellation(monkeypatch) -> None:
    """Wait for exchange completion even after repeated cancellation."""

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
            Mock(), Mock(rank=0, nranks=2), 0, sources=(), destinations=(1,)
        )

        task = asyncio.create_task(exchange.finish())
        try:
            await asyncio.wait_for(started.wait(), timeout=5)
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
