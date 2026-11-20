# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for RapidsMPF AllGather functionality."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import polars as pl

import pylibcudf as plc
from cudf_streaming.table_chunk import TableChunk
from rapidsmpf.streaming.coll.allgather import AllGather

from cudf_polars.dsl.ir import IRExecutionContext
from cudf_polars.streaming.actor_graph import utils as actor_graph_utils
from cudf_polars.streaming.actor_graph.collectives import allgather
from cudf_polars.streaming.actor_graph.collectives.allgather import AllGatherManager
from cudf_polars.streaming.actor_graph.collectives.closing import (
    aclosing_collective,
    wait_for_pending_closes,
)
from cudf_polars.streaming.actor_graph.collectives.common import reserve_op_id
from cudf_polars.streaming.actor_graph.utils import allgather_reduce


async def _test_allgather(engine) -> None:
    """Very simple test that AllGatherManager can concatenate tables."""
    context = engine.context
    comm = engine.comm
    stream = context.br().stream_pool.get_stream()

    # Create simple test tables with different sizes
    tables = [
        plc.Table(
            [
                plc.Column.from_arrow(
                    pl.Series([i] * num_elements, dtype=pl.Int32()), stream=stream
                )
            ]
        )
        for i, num_elements in enumerate([100, 200, 300])
    ]

    # Insert tables into AllGatherManager
    with ThreadPoolExecutor(max_workers=1) as executor:
        ir_context = IRExecutionContext(executor)
        async with AllGatherManager(context, comm, 0) as allgather:
            with allgather.inserting() as inserter:
                for i, table in enumerate(tables):
                    await inserter.insert(
                        i,
                        TableChunk.from_pylibcudf_table(
                            table, stream, exclusive_view=True, br=context.br()
                        ),
                    )
            # Extract concatenated result
            result = await allgather.extract_concatenated(
                stream, ordered=True, ir_context=ir_context
            )

    # Verify the concatenated table has the expected shape
    assert result.num_rows() == 600  # 100 + 200 + 300
    assert result.num_columns() == 1

    # Verify the column type is correct
    col = result.columns()[0]
    assert col.size() == 600
    assert col.type().id().value == plc.types.TypeId.INT32.value


def test_allgather(spmd_engine) -> None:
    asyncio.run(_test_allgather(spmd_engine))


def test_allgather_manager_cancellation_does_not_wait_for_peers(
    monkeypatch, fake_collective, cancel_when, assert_pending_then_released
) -> None:
    monkeypatch.setattr(
        allgather, "AllGather", lambda *_args, **_kwargs: fake_collective
    )
    comm = Mock()

    async def body() -> None:
        async with AllGatherManager(Mock(), comm, 4):
            await fake_collective.extract_all(ordered=True)

    cancel_when(body, fake_collective.waiting)
    assert fake_collective.close_calls == 1
    assert_pending_then_released(comm, fake_collective, 4)


async def _test_allgather_reduce(engine) -> None:
    """Test allgather_reduce with single and multiple values."""
    context = engine.context
    comm = engine.comm

    # Test with a single value
    (result,) = await allgather_reduce(context, comm, 0, 42)
    assert result == 42  # Single rank, so sum is just the local value

    # Test with multiple values
    results = await allgather_reduce(context, comm, 1, 10, 20, 30)
    assert results == (10, 20, 30)  # Single rank, so sums are just the local values


def test_allgather_reduce(spmd_engine) -> None:
    asyncio.run(_test_allgather_reduce(spmd_engine))


def test_allgather_reduce_cancellation_does_not_wait_for_peers(
    monkeypatch, fake_collective, cancel_when, assert_pending_then_released
) -> None:
    monkeypatch.setattr(actor_graph_utils, "PackedData", Mock())
    monkeypatch.setattr(
        actor_graph_utils, "AllGather", lambda *_args, **_kwargs: fake_collective
    )
    comm = Mock(nranks=2)

    cancel_when(lambda: allgather_reduce(Mock(), comm, 2, 42), fake_collective.waiting)
    assert_pending_then_released(comm, fake_collective, 2)


def test_real_allgather_cancelled_close_completes_on_new_loop(
    spmd_engine, pending_closes, cancel_when, timeout_seconds
) -> None:
    comm = spmd_engine.comm
    entered = asyncio.Event()

    with reserve_op_id() as op_id:

        async def body() -> None:
            async with aclosing_collective(
                AllGather(spmd_engine.context, comm, op_id), comm, op_id
            ):
                entered.set()
                await asyncio.Event().wait()

        cancel_when(body, entered)
        assert [close.op_id for close in pending_closes[comm]] == [op_id]
        wait_for_pending_closes(comm, timeout=timeout_seconds)
        assert pending_closes == {}

    result = (
        pl.LazyFrame({"a": [1, 2, 3]})
        .select(pl.col("a").sum())
        .collect(engine=spmd_engine)
    )
    assert result["a"].to_list() == [6]
