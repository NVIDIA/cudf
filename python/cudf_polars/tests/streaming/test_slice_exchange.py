# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

import polars as pl

import pylibcudf as plc
from cudf_streaming.table_chunk import (
    TableChunk,
    make_table_chunks_available_or_wait,
)

from cudf_polars.containers import DataFrame, DataType
from cudf_polars.streaming.actor_graph.collectives.common import reserve_op_id
from cudf_polars.streaming.actor_graph.collectives.slice_exchange import (
    SliceRoute,
    SparseExchange,
)

if TYPE_CHECKING:
    from rapidsmpf.streaming.core.context import Context
    from rmm.pylibrmm.stream import Stream


def _chunk(values: list[int], context: Context, stream: Stream) -> TableChunk:
    return TableChunk.from_pylibcudf_table(
        plc.Table(
            [
                plc.Column.from_arrow(
                    pl.Series("value", values, dtype=pl.Int64()),
                    stream=stream,
                )
            ]
        ),
        stream,
        exclusive_view=True,
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
def test_sparse_exchange_sends_slice_and_moves_chunk(spmd_engine) -> None:
    """A borrowed slice keeps its source, while a whole-chunk send consumes it."""
    if spmd_engine.comm.nranks != 2:
        pytest.skip("This test expects exactly two ranks.")

    async def run() -> None:
        context = spmd_engine.context
        rank = spmd_engine.comm.rank
        with reserve_op_id() as op_id:
            async with SparseExchange(
                context,
                spmd_engine.comm,
                op_id,
                sources=(0,) if rank == 1 else (),
                destinations=(1,) if rank == 0 else (),
            ) as exchange:
                if rank == 0:
                    source = _chunk(
                        [0, 1, 2, 3],
                        context,
                        context.br().stream_pool.get_stream(),
                    )
                    await exchange.insert_slices(source, (SliceRoute(1, 1, 3),))
                    assert _values(source) == [0, 1, 2, 3]
                    moved = _chunk(
                        [4, 5],
                        context,
                        context.br().stream_pool.get_stream(),
                    )
                    await exchange.insert_chunk(1, moved)
                    with pytest.raises(ValueError, match="released"):
                        moved.is_available()
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


def test_sparse_exchange_finish_survives_cancellation() -> None:
    """Keep the borrowed exchange alive until finish completes after cancellation."""

    async def run() -> None:
        release = asyncio.Event()

        class DelayedExchange:
            async def insert_finished(self, _context) -> None:
                await release.wait()

        exchange = SparseExchange.__new__(SparseExchange)
        exchange.context = None
        exchange._exchange = DelayedExchange()
        exchange._finish_task = None
        exchange._finished = False
        exchange._closed = False

        task = asyncio.create_task(exchange.finish())
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert exchange._finished

    asyncio.run(run())
