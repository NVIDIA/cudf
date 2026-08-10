# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integration tests for structlog tracing with rapidsmpf."""

from __future__ import annotations

import asyncio
import json
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


@pytest.mark.parametrize(
    "broadcast_limit,bloom_filter_max_size,join_strategy,method,reason,output_rows",
    [
        (1, 32 * 1024 * 1024, "shuffle", "bloom", "bloom_fits", 10),
        (64, 0, "shuffle", "broadcast_semi_join", "exact_domain_fits", 10),
        (
            1_000_000,
            32 * 1024 * 1024,
            "broadcast_left",
            "skip",
            "target_not_redistributed",
            None,
        ),
    ],
    ids=["bloom", "exact", "skip"],
)
def test_local_join_prefilter_trace_records_decision_and_effect(
    timeout_seconds: int,
    broadcast_limit: int,
    bloom_filter_max_size: int,
    join_strategy: str,
    method: str,
    reason: str,
    output_rows: int | None,
) -> None:
    """Trace a direct-input join prefilter selected through the public engine."""
    pytest.importorskip("structlog")
    code = textwrap.dedent(f"""\
    import json
    import os

    import polars as pl
    import rmm
    import structlog

    rmm.mr.set_current_device_resource(rmm.mr.ManagedMemoryResource())

    from cudf_polars.engine.spmd import SPMDEngine

    domain = (
        pl.LazyFrame({{"key": [1, 99], "active": [True, False]}})
        .filter("active")
        .select("key")
    )
    target = pl.LazyFrame(
        {{"key": [i % 100 for i in range(1_000)], "value": range(1_000)}}
    )
    query = domain.join(target, on="key")
    options = {{
        "join_filter_pushdown": {{
            "threshold": 0.5,
            "bloom_filter_max_size": {bloom_filter_max_size},
        }},
        "broadcast_limit": {broadcast_limit},
        "target_partition_size": 64,
        "max_rows_per_partition": 100,
    }}
    with SPMDEngine(executor_options=options) as engine:
        with structlog.testing.capture_logs() as logs:
            result = query.collect(engine=engine)

    (event,) = (
        log
        for log in logs
        if log.get("scope") == "actor" and "join_prefilters" in log
    )
    record = {{
        "result_rows": result.height,
        "join_strategy": event["decision"],
        "prefilter": event["join_prefilters"][0],
    }}
    print("PREFILTER_TRACE=" + json.dumps(record))
    """)

    env = os.environ.copy()
    env["CUDF_POLARS_LOG_TRACES"] = "1"
    result = subprocess.check_output(
        [sys.executable, "-c", code],
        env=env,
        stderr=subprocess.STDOUT,
        timeout=timeout_seconds,
    )
    (payload,) = (
        line.removeprefix(b"PREFILTER_TRACE=")
        for line in result.splitlines()
        if line.startswith(b"PREFILTER_TRACE=")
    )
    record = json.loads(payload)

    assert record["result_rows"] == 10
    assert record["join_strategy"] == join_strategy
    assert (
        record["prefilter"].items()
        >= {
            "target_side": "right",
            "domain_side": "left",
            "method": method,
            "reason": reason,
            "domain_rows": 1,
        }.items()
    )
    if output_rows is None:
        assert "input_rows" not in record["prefilter"]
        assert "output_rows" not in record["prefilter"]
    else:
        assert record["prefilter"]["estimated_cardinality"] == 1
        assert record["prefilter"]["input_rows"] == 1_000
        assert record["prefilter"]["output_rows"] == output_rows


@pytest.mark.parametrize(
    "broadcast_limit,bloom_filter_max_size,method,reason,output_rows",
    [
        (1, 32 * 1024 * 1024, "bloom", "bloom_fits", 20),
        (64, 0, "broadcast_semi_join", "exact_domain_fits", 20),
        (1, 0, "skip", "no_viable_filter", None),
    ],
    ids=["bloom", "exact", "skip"],
)
def test_standalone_prefilter_trace_records_decision_and_effect(
    timeout_seconds: int,
    broadcast_limit: int,
    bloom_filter_max_size: int,
    method: str,
    reason: str,
    output_rows: int | None,
) -> None:
    """Trace a prefilter pushed below an intervening join."""
    pytest.importorskip("structlog")
    code = textwrap.dedent(f"""\
    import json

    import polars as pl
    import rmm
    import structlog

    rmm.mr.set_current_device_resource(rmm.mr.ManagedMemoryResource())

    from cudf_polars.engine.spmd import SPMDEngine

    domain = (
        pl.LazyFrame(
            {{"p_partkey": range(10), "active": [True] * 2 + [False] * 8}}
        )
        .filter("active")
        .select("p_partkey")
    )
    target = (
        pl.LazyFrame(
            {{
                "l_partkey": [i % 10 for i in range(100)],
                "bridge_key": range(100),
                "value": range(100),
            }}
        )
        .join(pl.LazyFrame({{"bridge_key": range(100)}}), on="bridge_key")
        .with_columns((pl.col("value") + 1).alias("derived"))
    )
    query = domain.join(target, left_on="p_partkey", right_on="l_partkey")
    options = {{
        "join_filter_pushdown": {{
            "threshold": 0.5,
            "bloom_filter_max_size": {bloom_filter_max_size},
        }},
        "broadcast_limit": {broadcast_limit},
        "target_partition_size": 64,
        "max_rows_per_partition": 10,
    }}
    with SPMDEngine(executor_options=options) as engine:
        with structlog.testing.capture_logs() as logs:
            result = query.collect(engine=engine)

    (event,) = (
        log
        for log in logs
        if log.get("scope") == "actor"
        and log.get("prefilter", {{}}).get("placement") == "standalone"
    )
    record = {{
        "result_rows": result.height,
        "decision": event["decision"],
        "prefilter": event["prefilter"],
    }}
    print("PREFILTER_TRACE=" + json.dumps(record))
    """)

    env = os.environ.copy()
    env["CUDF_POLARS_LOG_TRACES"] = "1"
    result = subprocess.check_output(
        [sys.executable, "-c", code],
        env=env,
        stderr=subprocess.STDOUT,
        timeout=timeout_seconds,
    )
    (payload,) = (
        line.removeprefix(b"PREFILTER_TRACE=")
        for line in result.splitlines()
        if line.startswith(b"PREFILTER_TRACE=")
    )
    record = json.loads(payload)

    assert record["result_rows"] == 20
    assert record["decision"] == method
    assert (
        record["prefilter"].items()
        >= {
            "placement": "standalone",
            "method": method,
            "reason": reason,
            "domain_rows": 2,
        }.items()
    )
    if output_rows is None:
        assert "input_rows" not in record["prefilter"]
        assert "output_rows" not in record["prefilter"]
    else:
        assert record["prefilter"]["estimated_cardinality"] == 2
        assert record["prefilter"]["input_rows"] == 100
        assert record["prefilter"]["output_rows"] == output_rows


@pytest.mark.parametrize(
    "broadcast_limit,bloom_filter_max_size,method,reason,domain_rows",
    [
        (1, 32 * 1024 * 1024, "bloom", "bloom_fits", 15),
        (512, 0, "broadcast_semi_join", "exact_domain_fits", 15),
        (
            1_000_000,
            32 * 1024 * 1024,
            "bloom",
            "bloom_fits",
            15,
        ),
    ],
    ids=["bloom", "exact", "bloom_despite_intervening_broadcast"],
)
def test_indirect_prefilter_trace_records_decision_and_effect(
    timeout_seconds: int,
    broadcast_limit: int,
    bloom_filter_max_size: int,
    method: str,
    reason: str,
    domain_rows: int | None,
) -> None:
    """Trace a composite prefilter pushed below an intervening join."""
    pytest.importorskip("structlog")
    code = textwrap.dedent(f"""\
    import json

    import polars as pl
    import rmm
    import structlog

    rmm.mr.set_current_device_resource(rmm.mr.ManagedMemoryResource())

    from cudf_polars.engine.spmd import SPMDEngine

    nation = (
        pl.LazyFrame(
            {{"n_nationkey": range(10), "active": [True] * 5 + [False] * 5}}
        )
        .filter("active")
        .select("n_nationkey")
    )
    orders = pl.LazyFrame(
        {{
            "o_orderkey": range(90),
            "n_nationkey": [i % 10 for i in range(90)],
        }}
    )
    lineitem = pl.LazyFrame(
        {{
            "l_orderkey": [i % 90 for i in range(180)],
            "l_suppkey": [i % 60 for i in range(180)],
        }}
    )
    supplier = pl.LazyFrame(
        {{
            "s_suppkey": range(30),
            "s_nationkey": [i % 10 for i in range(30)],
        }}
    )
    query = (
        nation.join(orders, on="n_nationkey")
        .join(
            lineitem,
            left_on="o_orderkey",
            right_on="l_orderkey",
            maintain_order="left",
        )
        .join(
            supplier,
            left_on=("l_suppkey", "n_nationkey"),
            right_on=("s_suppkey", "s_nationkey"),
        )
    )
    options = {{
        "join_filter_pushdown": {{
            "threshold": 0.5,
            "bloom_filter_max_size": {bloom_filter_max_size},
        }},
        "broadcast_limit": {broadcast_limit},
        "target_partition_size": 64,
        "max_rows_per_partition": 100,
    }}
    with SPMDEngine(executor_options=options) as engine:
        with structlog.testing.capture_logs() as logs:
            result = query.collect(engine=engine)

    (event,) = (
        log
        for log in logs
        if log.get("scope") == "actor"
        and log.get("prefilter", {{}}).get("placement") == "standalone"
        and log.get("prefilter", {{}}).get("target_on") == ["l_suppkey"]
    )
    record = {{
        "result_rows": result.height,
        "prefilter": event["prefilter"],
    }}
    print("PREFILTER_TRACE=" + json.dumps(record))
    """)

    env = os.environ.copy()
    env["CUDF_POLARS_LOG_TRACES"] = "1"
    result = subprocess.check_output(
        [sys.executable, "-c", code],
        env=env,
        stderr=subprocess.STDOUT,
        timeout=timeout_seconds,
    )
    (payload,) = (
        line.removeprefix(b"PREFILTER_TRACE=")
        for line in result.splitlines()
        if line.startswith(b"PREFILTER_TRACE=")
    )
    record = json.loads(payload)

    assert record["result_rows"] == 45
    assert record["prefilter"]["target_on"] == ["l_suppkey"]
    assert (
        record["prefilter"].items()
        >= {
            "placement": "standalone",
            "method": method,
            "reason": reason,
            "domain_rows": domain_rows,
        }.items()
    )
    if method == "skip":
        assert "input_rows" not in record["prefilter"]
        assert "output_rows" not in record["prefilter"]
    else:
        assert record["prefilter"]["estimated_cardinality"] == domain_rows
        assert record["prefilter"]["input_rows"] == 180
        if method == "broadcast_semi_join":
            assert record["prefilter"]["output_rows"] == 45
        else:
            assert 45 <= record["prefilter"]["output_rows"] < 180


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
