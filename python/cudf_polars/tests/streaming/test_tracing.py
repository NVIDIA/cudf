# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integration tests for structlog tracing with rapidsmpf."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import textwrap
from typing import TYPE_CHECKING

import pytest

import polars as pl

from cudf_streaming.table_chunk import TableChunk

from cudf_polars.containers import DataFrame
from cudf_polars.streaming.actor_graph.tracing import ActorTracer, send_chunk

if TYPE_CHECKING:
    from cudf_polars.engine.spmd import SPMDEngine


@pytest.fixture
def chunk(spmd_engine: SPMDEngine) -> TableChunk:
    context = spmd_engine.context
    stream = context.br().stream_pool.get_stream()
    df = DataFrame.from_polars(pl.DataFrame({"x": [1, 2, 3]}), stream)
    return TableChunk.from_pylibcudf_table(
        df.table, stream, exclusive_view=True, br=context.br()
    )


@pytest.mark.spmd
def test_actor_tracer_counts_table_chunk_without_table_view(chunk: TableChunk) -> None:
    tracer = ActorTracer()
    tracer.add_chunk(chunk=chunk)
    assert tracer.chunk_count == 1
    assert tracer.row_count == 3


@pytest.mark.spmd
def test_send_chunk_traces_and_sends_message(
    spmd_engine: SPMDEngine, chunk: TableChunk
) -> None:
    context = spmd_engine.context
    ch_out = context.create_channel()
    tracer = ActorTracer()

    async def send_and_recv():
        async with asyncio.TaskGroup() as tg:
            recv_task = tg.create_task(ch_out.recv(context))
            tg.create_task(send_chunk(context, ch_out, chunk, 11, tracer=tracer))
        return recv_task.result()

    msg = asyncio.run(send_and_recv())

    assert msg is not None
    assert msg.sequence_number == 11
    assert TableChunk.from_message(msg, br=context.br()).shape[0] == 3
    assert tracer.chunk_count == 1
    assert tracer.row_count == 3


def test_structlog_streaming_node_events(timeout_seconds: int):
    """Test that structlog emits 'Streaming Actor' events when tracing is enabled."""
    pytest.importorskip("structlog")
    code = textwrap.dedent("""\
    import rmm
    import polars as pl

    rmm.mr.set_current_device_resource(rmm.mr.ManagedMemoryResource())
    from cudf_polars.engine.spmd import SPMDEngine

    df = pl.DataFrame({"x": range(100), "y": ["a", "b"] * 50})
    q = df.lazy().filter(pl.col("x") > 50).group_by("y").agg(pl.col("x").sum())
    with SPMDEngine(executor_options={"max_rows_per_partition": 10}) as engine:
        q.collect(engine=engine)
    """)

    env = os.environ.copy()
    env["CUDF_POLARS_LOG_TRACES"] = "1"

    with subprocess.Popen(
        [sys.executable, "-c", code],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ) as proc:
        result, _ = proc.communicate(timeout=timeout_seconds)

    assert b"Streaming Actor" in result
    assert b"scope=actor" in result or b"'scope': 'actor'" in result
    assert b"actor_ir_id=" in result
    assert b"actor_ir_type=" in result
    assert b"chunk_count=" in result


def test_structlog_contains_expected_ir_types(timeout_seconds: int):
    """Test that structlog output contains expected IR types for a query."""
    pytest.importorskip("structlog")
    code = textwrap.dedent("""\
    import rmm
    import polars as pl

    rmm.mr.set_current_device_resource(rmm.mr.ManagedMemoryResource())
    from cudf_polars.engine.spmd import SPMDEngine

    df = pl.DataFrame({"x": range(100), "y": ["a", "b"] * 50})
    q = df.lazy().filter(pl.col("x") > 50).group_by("y").agg(pl.col("x").sum())
    with SPMDEngine(executor_options={"max_rows_per_partition": 10}) as engine:
        q.collect(engine=engine)
    """)

    env = os.environ.copy()
    env["CUDF_POLARS_LOG_TRACES"] = "1"

    with subprocess.Popen(
        [sys.executable, "-c", code],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ) as proc:
        result, _ = proc.communicate(timeout=timeout_seconds)

    assert b"ir_type=DataFrameScan" in result
    assert b"ir_type=Filter" in result
    assert b"ir_type=GroupBy" in result


def test_structlog_disabled_by_default(timeout_seconds: int):
    """Test that structlog does NOT emit events when CUDF_POLARS_LOG_TRACES is not set."""
    pytest.importorskip("structlog")
    code = textwrap.dedent("""\
    import rmm
    import polars as pl

    rmm.mr.set_current_device_resource(rmm.mr.ManagedMemoryResource())
    from cudf_polars.engine.spmd import SPMDEngine

    df = pl.DataFrame({"x": range(10), "y": ["a", "b"] * 5})
    q = df.lazy().filter(pl.col("x") > 5)
    with SPMDEngine(executor_options={"max_rows_per_partition": 5}) as engine:
        q.collect(engine=engine)
    """)

    env = os.environ.copy()
    env.pop("CUDF_POLARS_LOG_TRACES", None)

    with subprocess.Popen(
        [sys.executable, "-c", code],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ) as proc:
        result, _ = proc.communicate(timeout=timeout_seconds)

    assert b"Streaming Actor" not in result
