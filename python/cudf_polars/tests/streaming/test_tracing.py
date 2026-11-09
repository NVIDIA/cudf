# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for streaming telemetry with rapidsmpf."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock

import pytest

import polars as pl

from cudf_streaming.table_chunk import TableChunk
from rapidsmpf.memory.buffer import MemoryType
from rapidsmpf.streaming.chunks.arbitrary import ArbitraryChunk
from rapidsmpf.streaming.core.message import Message

from cudf_polars.containers import DataFrame
from cudf_polars.streaming.actor_graph.io import Lineariser
from cudf_polars.streaming.actor_graph.tracing import (
    ActorTracer,
    record_channel_metrics,
    send_chunk,
)

if TYPE_CHECKING:
    import pathlib

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
def test_actor_metrics_count_table_chunk_without_table_view(chunk: TableChunk) -> None:
    tracer = ActorTracer()
    tracer.add_chunk(chunk=chunk)
    assert tracer.chunk_count == 1
    assert tracer.row_count == 3


def test_record_channel_metrics_sums_all_memory_types() -> None:
    input_channel = Mock()
    input_channel.metrics.return_value.recv_bytes = {
        MemoryType.DEVICE: 10,
        MemoryType.HOST: 4,
    }
    output_channel = Mock()
    output_channel.metrics.return_value.send_bytes = {
        MemoryType.DEVICE: 7,
        MemoryType.HOST: 3,
    }
    tracer = ActorTracer()

    record_channel_metrics(tracer, chs_in=(input_channel,), chs_out=(output_channel,))

    assert sum(tracer.input_bytes.values()) == 14
    assert sum(tracer.output_bytes.values()) == 10


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


@pytest.mark.spmd
def test_lineariser_backpressures_each_producer(spmd_engine: SPMDEngine) -> None:
    context = spmd_engine.context
    ch_out = context.create_channel()
    lineariser = Lineariser(context, ch_out, num_producers=2)
    produced: list[list[int]] = [[], []]
    output: list[int] = []

    async def run() -> list[list[int]]:
        release_gap = asyncio.Event()
        out_of_order_sent = asyncio.Event()

        async def producer(producer_id: int, sequence_numbers: list[int]) -> None:
            if producer_id == 1:
                await release_gap.wait()
            for sequence_number in sequence_numbers:
                ch_in = await lineariser.acquire(producer_id)
                produced[producer_id].append(sequence_number)
                await ch_in.send(
                    context,
                    Message(sequence_number, ArbitraryChunk(sequence_number)),
                )
                if sequence_number == 2:
                    out_of_order_sent.set()
            await lineariser.input_channels[producer_id].drain(context)

        async def consumer() -> None:
            while (msg := await ch_out.recv(context)) is not None:
                output.append(ArbitraryChunk.from_message(msg).release())

        async with asyncio.TaskGroup() as tg:
            tg.create_task(lineariser.drain())
            tg.create_task(producer(0, [0, 2, 4]))
            tg.create_task(producer(1, [1, 3, 5]))
            tg.create_task(consumer())

            await out_of_order_sent.wait()
            await asyncio.sleep(0)
            produced_before_gap = [values.copy() for values in produced]
            release_gap.set()

        return produced_before_gap

    produced_before_gap = asyncio.run(run())

    assert produced_before_gap == [[0, 2], []]
    assert output == list(range(6))


def test_parquet_scan_ordering_trace_from_set_sorted(
    tmp_path: pathlib.Path,
) -> None:
    pytest.importorskip("cudf_polars_quent")
    from cudf_polars.engine.spmd import SPMDEngine
    from cudf_polars.quent import QuentConfig

    source = tmp_path / "data.parquet"
    pl.DataFrame({"x": range(100), "y": range(100)}).write_parquet(
        source,
        row_group_size=10,
    )

    output_root = tmp_path / "quent-ordering"
    with SPMDEngine(
        executor_options={
            "dynamic_planning": {"infer_ordering": True},
            "target_partition_size": 1024,
            "quent_context": QuentConfig(output_root=str(output_root)),
        },
    ) as engine:
        result = (
            pl.scan_parquet(source).set_sorted("x").select("x").collect(engine=engine)
        )
        event_root = engine._quent_output_root
        assert event_root is not None
    operator_events: dict[str, list[dict[str, Any]]] = {}
    for path in event_root.rglob("*.ndjson"):
        if path.parent.name != "Operator":
            continue
        for line in path.read_text().splitlines():
            event = json.loads(line)
            operator_events.setdefault(event["id"], []).append(event["data"])
    decisions: list[tuple[str, str]] = []
    for events in operator_events.values():
        type_name = next(
            event["Declared"]["type_name"] for event in events if "Declared" in event
        )
        decisions.extend(
            (type_name, event["Statistics"]["values"]["decision"])
            for event in events
            if "Statistics" in event
        )
    assert result.height == 100
    assert ("StreamingScan", "parquet_ordering") in decisions


def test_local_join_prefilter_trace_records_decision_and_effect(
    tmp_path: pathlib.Path,
) -> None:
    """Trace a direct-input join prefilter selected through the public engine."""
    pytest.importorskip("cudf_polars_quent")
    from cudf_polars.engine.spmd import SPMDEngine
    from cudf_polars.quent import QuentConfig

    cases: list[tuple[str, bool, int, int, str, str, str, int | None, int | None]] = [
        ("bloom", False, 1, 32 * 1024 * 1024, "shuffle", "bloom", "bloom_fits", 1, 10),
        (
            "exact",
            False,
            64,
            0,
            "shuffle",
            "broadcast_semi_join",
            "exact_domain_fits",
            1,
            10,
        ),
        (
            "broadcast-skip",
            False,
            1_000_000,
            32 * 1024 * 1024,
            "broadcast_left",
            "skip",
            "target_not_redistributed",
            1,
            None,
        ),
        (
            "ordered-skip",
            True,
            1,
            32 * 1024 * 1024,
            "ordered_aligned",
            "skip",
            "target_not_redistributed",
            None,
            None,
        ),
    ]

    domain_path = tmp_path / "domain.parquet"
    target_path = tmp_path / "target.parquet"
    pl.DataFrame(
        {
            "key": range(100),
            "active": [i % 10 == 0 for i in range(100)],
        }
    ).write_parquet(domain_path)
    pl.DataFrame(
        {
            "key": range(1_000),
            "value": range(1_000),
        }
    ).write_parquet(target_path)
    records = {}
    output_root = tmp_path / "quent-local-prefilters"
    for (
        case_id,
        ordered,
        broadcast_limit,
        bloom_filter_max_size,
        expected_join_strategy,
        *_,
    ) in cases:
        if ordered:
            domain = (
                pl.scan_parquet(domain_path)
                .filter("active")
                .select("key")
                .set_sorted("key")
            )
            target = pl.scan_parquet(target_path).set_sorted("key")
        else:
            domain = (
                pl.LazyFrame({"key": [1, 99], "active": [True, False]})
                .filter("active")
                .select("key")
            )
            target = pl.LazyFrame(
                {
                    "key": [i % 100 for i in range(1_000)],
                    "value": range(1_000),
                }
            )
        options = {
            "join_filter_pushdown": {
                "threshold": 0.5,
                "bloom_filter_max_size": bloom_filter_max_size,
            },
            "broadcast_limit": broadcast_limit,
            "target_partition_size": 1 << 30 if ordered else 64,
            "max_rows_per_partition": 1_000_000 if ordered else 100,
            "quent_context": QuentConfig(output_root=str(output_root / case_id)),
        }
        with SPMDEngine(executor_options=options) as engine:
            result = domain.join(target, on="key").collect(engine=engine)
            event_root = engine._quent_output_root
            assert event_root is not None
        statistics = []
        prefilters = []
        for path in event_root.rglob("*.ndjson"):
            if path.parent.name != "Operator":
                continue
            for line in path.read_text().splitlines():
                event = json.loads(line)["data"]
                if "Statistics" in event:
                    statistics.append(event["Statistics"]["values"])
                if "RuntimePrefilterStatistics" in event:
                    prefilters.append(event["RuntimePrefilterStatistics"]["values"])
        (prefilter,) = prefilters
        (join_statistics,) = (
            values
            for values in statistics
            if values["decision"] == expected_join_strategy
        )
        records[case_id] = {
            "result_rows": result.height,
            "join_strategy": join_statistics["decision"],
            "prefilter": prefilter,
        }
    for (
        case_id,
        _,
        _,
        _,
        join_strategy,
        method,
        reason,
        domain_rows,
        output_rows,
    ) in cases:
        record = records[case_id]
        assert record["result_rows"] == 10
        assert record["join_strategy"] == join_strategy
        expected_prefilter: dict[str, str | int] = {
            "target_side": "right",
            "domain_side": "left",
            "method": method,
            "reason": reason,
        }
        if domain_rows is not None:
            expected_prefilter["domain_rows"] = domain_rows
        assert record["prefilter"].items() >= expected_prefilter.items()
        if output_rows is None:
            assert record["prefilter"]["input_rows"] is None
            assert record["prefilter"]["output_rows"] is None
        else:
            assert record["prefilter"]["estimated_cardinality"] == 1
            assert record["prefilter"]["input_rows"] == 1_000
            assert record["prefilter"]["output_rows"] == output_rows


def test_standalone_prefilter_trace_records_decision_and_effect(
    tmp_path: pathlib.Path,
) -> None:
    """Trace a prefilter pushed below an intervening join."""
    pytest.importorskip("cudf_polars_quent")
    from cudf_polars.engine.spmd import SPMDEngine
    from cudf_polars.quent import QuentConfig

    cases = [
        ("bloom", 1, 32 * 1024 * 1024, "bloom", "bloom_fits", 20),
        ("exact", 64, 0, "broadcast_semi_join", "exact_domain_fits", 20),
        ("skip", 1, 0, "skip", "no_viable_filter", None),
    ]
    records = {}
    output_root = tmp_path / "quent-standalone-prefilters"
    for case_id, broadcast_limit, bloom_filter_max_size, *_ in cases:
        domain = (
            pl.LazyFrame(
                {
                    "p_partkey": range(10),
                    "active": [True] * 2 + [False] * 8,
                }
            )
            .filter("active")
            .select("p_partkey")
        )
        target = (
            pl.LazyFrame(
                {
                    "l_partkey": [i % 10 for i in range(100)],
                    "bridge_key": range(100),
                    "value": range(100),
                }
            )
            .join(pl.LazyFrame({"bridge_key": range(100)}), on="bridge_key")
            .with_columns((pl.col("value") + 1).alias("derived"))
        )
        options = {
            "join_filter_pushdown": {
                "threshold": 0.5,
                "bloom_filter_max_size": bloom_filter_max_size,
            },
            "broadcast_limit": broadcast_limit,
            "target_partition_size": 64,
            "max_rows_per_partition": 10,
            "quent_context": QuentConfig(output_root=str(output_root / case_id)),
        }
        with SPMDEngine(executor_options=options) as engine:
            result = domain.join(
                target, left_on="p_partkey", right_on="l_partkey"
            ).collect(engine=engine)
            event_root = engine._quent_output_root
            assert event_root is not None
        prefilters = []
        for path in event_root.rglob("*.ndjson"):
            if path.parent.name != "Operator":
                continue
            for line in path.read_text().splitlines():
                event = json.loads(line)["data"]
                if "RuntimePrefilterStatistics" in event:
                    value = event["RuntimePrefilterStatistics"]["values"]
                    if value["placement"] == "standalone":
                        prefilters.append(value)
        (prefilter,) = prefilters
        records[case_id] = {
            "result_rows": result.height,
            "decision": prefilter["method"],
            "prefilter": prefilter,
        }
    for case_id, _, _, method, reason, output_rows in cases:
        record = records[case_id]
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
            assert record["prefilter"]["input_rows"] is None
            assert record["prefilter"]["output_rows"] is None
        else:
            assert record["prefilter"]["estimated_cardinality"] == 2
            assert record["prefilter"]["input_rows"] == 100
            assert record["prefilter"]["output_rows"] == output_rows


def test_indirect_prefilter_trace_records_decision_and_effect(
    tmp_path: pathlib.Path,
) -> None:
    """Trace a composite prefilter pushed below an intervening join."""
    pytest.importorskip("cudf_polars_quent")
    from cudf_polars.engine.spmd import SPMDEngine
    from cudf_polars.quent import QuentConfig

    cases = [
        ("bloom", 1, 32 * 1024 * 1024, "bloom", "bloom_fits", 15),
        ("exact", 512, 0, "broadcast_semi_join", "exact_domain_fits", 15),
        (
            "bloom_despite_intervening_broadcast",
            1_000_000,
            32 * 1024 * 1024,
            "bloom",
            "bloom_fits",
            15,
        ),
    ]
    records = {}
    output_root = tmp_path / "quent-indirect-prefilters"
    for case_id, broadcast_limit, bloom_filter_max_size, *_ in cases:
        nation = (
            pl.LazyFrame(
                {
                    "n_nationkey": range(10),
                    "active": [True] * 5 + [False] * 5,
                }
            )
            .filter("active")
            .select("n_nationkey")
        )
        orders = pl.LazyFrame(
            {
                "o_orderkey": range(90),
                "n_nationkey": [i % 10 for i in range(90)],
            }
        )
        lineitem = pl.LazyFrame(
            {
                "l_orderkey": [i % 90 for i in range(180)],
                "l_suppkey": [i % 60 for i in range(180)],
            }
        )
        supplier = pl.LazyFrame(
            {
                "s_suppkey": range(30),
                "s_nationkey": [i % 10 for i in range(30)],
            }
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
        options = {
            "join_filter_pushdown": {
                "threshold": 0.5,
                "bloom_filter_max_size": bloom_filter_max_size,
            },
            "broadcast_limit": broadcast_limit,
            "target_partition_size": 64,
            "max_rows_per_partition": 100,
            "quent_context": QuentConfig(output_root=str(output_root / case_id)),
        }
        with SPMDEngine(executor_options=options) as engine:
            result = query.collect(engine=engine)
            event_root = engine._quent_output_root
            assert event_root is not None
        prefilters = []
        for path in event_root.rglob("*.ndjson"):
            if path.parent.name != "Operator":
                continue
            for line in path.read_text().splitlines():
                event = json.loads(line)["data"]
                if "RuntimePrefilterStatistics" in event:
                    value = event["RuntimePrefilterStatistics"]["values"]
                    if value["placement"] == "standalone" and value["target_on"] == [
                        "l_suppkey"
                    ]:
                        prefilters.append(value)
        (prefilter,) = prefilters
        records[case_id] = {"result_rows": result.height, "prefilter": prefilter}
    for case_id, _, _, method, reason, domain_rows in cases:
        record = records[case_id]
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
        assert record["prefilter"]["estimated_cardinality"] == domain_rows
        assert record["prefilter"]["input_rows"] == 180
        if method == "broadcast_semi_join":
            assert record["prefilter"]["output_rows"] == 45
        else:
            assert 45 <= record["prefilter"]["output_rows"] < 180
