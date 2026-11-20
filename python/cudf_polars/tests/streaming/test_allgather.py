# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for RapidsMPF AllGather functionality."""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import pytest

import polars as pl

import pylibcudf as plc
from cudf_streaming.table_chunk import TableChunk

from cudf_polars.dsl.ir import IRExecutionContext
from cudf_polars.streaming.actor_graph import utils as actor_graph_utils
from cudf_polars.streaming.actor_graph.collectives import allgather
from cudf_polars.streaming.actor_graph.collectives.allgather import AllGatherManager
from cudf_polars.streaming.actor_graph.collectives.closing import (
    wait_for_pending_closes,
)
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
    monkeypatch, pending_closes, timeout_seconds: int
) -> None:
    close_started = threading.Event()
    peers_finished = threading.Event()

    class PendingAllGather:
        async def aclose(self) -> None:
            close_started.set()
            while not peers_finished.is_set():
                await asyncio.sleep(0.001)

    monkeypatch.setattr(
        allgather, "AllGather", lambda *_args, **_kwargs: PendingAllGather()
    )
    comm = Mock()

    async def run() -> None:
        entered = asyncio.Event()

        async def body() -> None:
            async with AllGatherManager(Mock(), comm, 4):
                entered.set()
                await asyncio.Event().wait()

        task = asyncio.create_task(body())
        await asyncio.wait_for(entered.wait(), timeout=timeout_seconds)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=timeout_seconds)

    asyncio.run(run())
    assert close_started.is_set()
    assert [close.op_id for close in pending_closes[comm]] == [4]
    peers_finished.set()
    wait_for_pending_closes(comm, timeout=timeout_seconds)
    assert pending_closes == {}


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
    monkeypatch, pending_closes, timeout_seconds: int
) -> None:
    extracting = threading.Event()
    peers_finished = threading.Event()

    class PendingAllGather:
        def __enter__(self) -> PendingAllGather:
            return self

        def __exit__(self, *args: object) -> None:
            pass

        def insert(self, *args: object) -> None:
            pass

        async def extract_all(self, *, ordered: bool) -> list:
            extracting.set()
            await asyncio.Event().wait()
            return []

        async def aclose(self) -> None:
            while not peers_finished.is_set():
                await asyncio.sleep(0.001)

    monkeypatch.setattr(actor_graph_utils, "PackedData", Mock())
    monkeypatch.setattr(
        actor_graph_utils, "AllGather", lambda *_args, **_kwargs: PendingAllGather()
    )
    comm = Mock(nranks=2)

    async def run() -> None:
        async def started() -> None:
            while not extracting.is_set():
                await asyncio.sleep(0.001)

        task = asyncio.create_task(allgather_reduce(Mock(), comm, 2, 42))
        await asyncio.wait_for(started(), timeout=timeout_seconds)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=timeout_seconds)

    asyncio.run(run())
    assert [close.op_id for close in pending_closes[comm]] == [2]
    peers_finished.set()
    wait_for_pending_closes(comm, timeout=timeout_seconds)
    assert pending_closes == {}
