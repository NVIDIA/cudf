# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import datetime as dt
from unittest.mock import Mock

import pytest

import polars as pl
from polars.testing import assert_frame_equal

from cudf_streaming.channel_metadata import OrderScheme

from cudf_polars import Translator
from cudf_polars.engine.core import all_gather_host_data
from cudf_polars.engine.options import StreamingOptions
from cudf_polars.engine.spmd import SPMDEngine
from cudf_polars.streaming.actor_graph.collectives.common import reserve_op_id
from cudf_polars.streaming.actor_graph.collectives.overlap import (
    OverlapInput,
    RowCountOverlapPolicy,
)
from cudf_polars.streaming.actor_graph.core import evaluate_logical_plan
from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)
from cudf_polars.testing.engine_utils import warns_on_spmd
from cudf_polars.testing.io import make_partitioned_source
from cudf_polars.utils.config import ConfigOptions


@pytest.fixture
def engine(streaming_engine_factory):
    return streaming_engine_factory(
        StreamingOptions(max_rows_per_partition=2, fallback_mode="warn"),
    )


def test_multirank_overlap_input_requires_preparation() -> None:
    async def run() -> None:
        rolling_input = OverlapInput(
            Mock(),
            Mock(nranks=2),
            Mock(),
            RowCountOverlapPolicy(1, 1),
            duplicated=False,
            collective_id=0,
        )
        with pytest.raises(RuntimeError, match="async context manager"):
            await anext(aiter(rolling_input))

    asyncio.run(run())


def test_rolling_datetime(engine):
    dates = [
        "2020-01-01 13:45:48",
        "2020-01-01 16:42:13",
        "2020-01-01 16:45:09",
        "2020-01-02 18:12:48",
        "2020-01-03 19:45:32",
        "2020-01-08 23:16:43",
    ]
    df = (
        pl.DataFrame({"dt": dates, "a": [3, 7, 5, 9, 2, 1]})
        .with_columns(pl.col("dt").str.strptime(pl.Datetime("ns")))
        .lazy()
    )
    q = df.with_columns(pl.sum("a").rolling(index_column="dt", period="2d"))
    # HStack may redirect to Select before fallback; message differs by Polars IR / version.
    with warns_on_spmd(
        engine,
        UserWarning,
        match=r"This (HStack|selection) is not supported for multiple partitions\.",
    ):
        assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("closed", ["left", "right", "both", "none"])
def test_rolling_integer_period(engine, closed) -> None:
    df = pl.LazyFrame(
        {
            "orderby": [1, 4, 8, 10, 12, 13, 14, 22],
            "values": [1, 2, 3, 4, 5, 6, 7, 8],
        }
    )
    q = df.rolling("orderby", period="4i", closed=closed).agg(
        sum_values=pl.col("values").sum(),
        min_values=pl.col("values").min(),
        max_values=pl.col("values").max(),
        count=pl.len(),
    )

    assert_gpu_result_equal(q, engine=engine)


def test_rolling_preserves_index_ordering_metadata(spmd_engine_factory) -> None:
    engine = spmd_engine_factory(
        StreamingOptions(
            max_rows_per_partition=2,
            dynamic_planning=None,
            fallback_mode="raise",
            raise_on_fail=True,
        )
    )
    df = pl.LazyFrame(
        {
            "orderby": [1, 2, 3, 4, 5, 6],
            "values": [10, 20, 30, 40, 50, 60],
        }
    )
    q = (
        df.sort("orderby")
        .rolling("orderby", period="3i")
        .agg(sum_values=pl.col("values").sum())
    )
    ir = Translator(q._ldf.visit(), engine).translate_ir()
    metadata_collector = evaluate_logical_plan(
        ir, ConfigOptions.from_polars_engine(engine), collect_metadata=True
    )[1]
    assert metadata_collector is not None
    assert len(metadata_collector) == 1

    metadata = metadata_collector[0]
    assert metadata.partitioning is not None
    assert isinstance(metadata.partitioning.inter_rank, OrderScheme)
    assert metadata.partitioning.local == "inherit"
    (ordering,) = metadata.partitioning.inter_rank.orderings
    assert tuple(key.column_index for key in ordering.keys) == (0,)
    assert ordering.strict_boundaries is True


def test_fixed_size_rolling_with_ordered_passthrough_metadata(
    spmd_engine_factory,
) -> None:
    engine = spmd_engine_factory(
        StreamingOptions(
            max_rows_per_partition=2,
            dynamic_planning=None,
            fallback_mode="raise",
            raise_on_fail=True,
        )
    )
    df = pl.LazyFrame(
        {
            "orderby": [1, 2, 3, 4, 5, 6],
            "values": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        }
    )
    q = df.sort("orderby").select(
        "orderby",
        pl.col("values").rolling_mean(window_size=3).alias("mean_values"),
    )
    ir = Translator(q._ldf.visit(), engine).translate_ir()
    metadata_collector = evaluate_logical_plan(
        ir, ConfigOptions.from_polars_engine(engine), collect_metadata=True
    )[1]
    assert metadata_collector is not None
    assert len(metadata_collector) == 1

    metadata = metadata_collector[0]
    assert metadata.partitioning is not None
    assert isinstance(metadata.partitioning.inter_rank, OrderScheme)
    assert metadata.partitioning.local == "inherit"
    (ordering,) = metadata.partitioning.inter_rank.orderings
    assert tuple(key.column_index for key in ordering.keys) == (0,)
    assert ordering.strict_boundaries is True


@pytest.mark.parametrize(
    "df",
    [
        pl.LazyFrame(
            {
                "orderby": pl.Series([1, 4, 8, 10], dtype=pl.Int32),
                "values": [1, 2, 3, 4],
            }
        ),
        pl.LazyFrame(
            {
                "orderby": [1, 2, 3, 4, 5, 6],
                "values": [1, 2, 3, 4, 5, 6],
            }
        ).filter(pl.col("orderby") > 4),
        pl.LazyFrame(
            {
                "orderby": [1, 2, 2, 3],
                "values": [1, 2, 3, 4],
            }
        ),
    ],
    ids=["int32-index", "empty-input-partitions", "duplicate-boundary-index"],
)
def test_rolling_integer_edge_cases(engine, df) -> None:
    q = df.rolling("orderby", period="2i").agg(
        sum_values=pl.col("values").sum(),
        count=pl.len(),
    )

    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "orderby, period",
    [
        ([1, 10, 2, 3], "1i"),
        (
            pl.Series(
                [
                    dt.datetime(2020, 1, 1, 0, 0, 1),
                    dt.datetime(2020, 1, 1, 0, 0, 10),
                    dt.datetime(2020, 1, 1, 0, 0, 2),
                    dt.datetime(2020, 1, 1, 0, 0, 3),
                ],
                dtype=pl.Datetime("us"),
            ),
            "1s",
        ),
    ],
    ids=["integer", "datetime"],
)
def test_rolling_unsorted_across_chunks_raises(
    spmd_engine_factory, orderby, period
) -> None:
    engine = spmd_engine_factory(
        StreamingOptions(max_rows_per_partition=2, fallback_mode="raise"),
    )
    df = pl.LazyFrame({"orderby": orderby, "values": [1, 2, 3, 4]})
    q = df.rolling("orderby", period=period).agg(
        sum_values=pl.col("values").sum(),
    )

    with pytest.RaisesGroup(
        pytest.RaisesExc(
            RuntimeError,
            match=r"Index column.*in rolling is not sorted, please sort first",
        )
    ):
        q.collect(engine=engine)


@pytest.mark.parametrize("center", [False, True])
def test_fixed_size_rolling_mean(engine, center) -> None:
    df = pl.LazyFrame({"x": [1.0, 2.0, 4.0, 8.0, 16.0]})
    q = df.select(pl.col("x").rolling_mean(window_size=3, center=center))

    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "kind,offset,closed",
    [
        *(("range", "-4i", closed) for closed in ("left", "right", "both", "none")),
        ("range", "0i", "right"),
        ("count", None, None),
    ],
    ids=[
        "range-left",
        "range-right",
        "range-both",
        "range-none",
        "range-leading",
        "count",
    ],
)
@pytest.mark.engine_params(["dask", "ray"])
def test_rolling_multirank(engine, tmp_path, kind, offset, closed) -> None:
    source = tmp_path / "rolling.parquet"
    _write_rolling_source(source)
    _assert_rolling_result(engine, source, kind, offset, closed)


def _write_rolling_source(source, *, orderby_dtype=None) -> None:
    orderby: list[int] | pl.Series = [0, 1, 2, 3, 3, 4, 5, 6]
    if orderby_dtype is not None:
        orderby = pl.Series(orderby, dtype=orderby_dtype)
    make_partitioned_source(
        pl.DataFrame(
            {
                "orderby": orderby,
                "values": [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0],
            }
        ),
        source,
        "parquet",
        row_group_size=2,
    )


def _assert_rolling_result(engine, source, kind, offset, closed="right") -> None:
    q = pl.scan_parquet(source)
    if kind == "range":
        q = (
            q.sort("orderby")
            .rolling("orderby", period="4i", offset=offset, closed=closed)
            .agg(pl.col("values").sum())
        )
    else:
        q = q.select(pl.col("values").rolling_mean(window_size=3, center=True))
    assert_gpu_result_equal(q, engine=engine)


def _share_source_path(spmd_engine, source) -> str:
    path = str(source).encode() if spmd_engine.rank == 0 else b""
    with reserve_op_id() as op_id:
        return all_gather_host_data(
            spmd_engine.comm, spmd_engine.context.br(), op_id, path
        )[0].decode()


@pytest.mark.spmd
@pytest.mark.parametrize(
    "kind",
    ["range", "count"],
)
def test_rolling_multirank_spmd(spmd_engine, tmp_path, kind) -> None:
    source = tmp_path / "rolling.parquet"
    if spmd_engine.rank == 0:
        _write_rolling_source(source)
    source = _share_source_path(spmd_engine, source)
    _assert_rolling_result(spmd_engine, source, kind, "-2i")


@pytest.mark.spmd
def test_rolling_multirank_uint64(spmd_engine, tmp_path) -> None:
    source = tmp_path / "rolling-uint64.parquet"
    if spmd_engine.rank == 0:
        _write_rolling_source(source, orderby_dtype=pl.UInt64)
    source = _share_source_path(spmd_engine, source)
    _assert_rolling_result(spmd_engine, source, "range", "-2i")


@pytest.mark.spmd
def test_rolling_uint64_out_of_bounds(spmd_engine, tmp_path) -> None:
    if spmd_engine.comm.nranks != 2:
        pytest.skip("This test expects exactly two ranks.")

    source = tmp_path / "rolling-uint64.parquet"
    if spmd_engine.rank == 0:
        make_partitioned_source(
            pl.DataFrame(
                {
                    "orderby": pl.Series([2**63, 2**63 + 1], dtype=pl.UInt64),
                    "values": [1, 2],
                }
            ),
            source,
            "parquet",
            row_group_size=1,
        )
    source = _share_source_path(spmd_engine, source)
    q = (
        pl.scan_parquet(source)
        .sort("orderby")
        .rolling("orderby", period="2i")
        .agg(sum=pl.sum("values"))
    )

    with pytest.RaisesGroup(
        pytest.RaisesExc(
            NotImplementedError,
            match="outside the signed 64-bit range",
        )
    ):
        q.collect(engine=spmd_engine)


def test_fixed_size_rolling_mean_after_drop_nulls(engine) -> None:
    df = pl.LazyFrame({"x": [None, 1.0, 2.0, None, 4.0, 8.0, 16.0]})
    q = df.select(pl.col("x").drop_nulls().rolling_mean(window_size=2))

    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "within_expression", [False, True], ids=["select", "expression"]
)
def test_fixed_size_rolling_mean_after_unaligned_siblings(
    spmd_engine_factory,
    within_expression,
) -> None:
    engine = spmd_engine_factory(
        StreamingOptions(max_rows_per_partition=2, fallback_mode="warn")
    )
    df = pl.LazyFrame(
        {
            "x": [None, 1.0, 2.0, 4.0],
            "y": [8.0, 16.0, None, 32.0],
        }
    )
    x = pl.col("x").drop_nulls().rolling_mean(window_size=2)
    y = pl.col("y").drop_nulls().rolling_mean(window_size=2)
    expressions = [x + y] if within_expression else [x, y]
    q = df.select(expressions)

    with pytest.warns(
        UserWarning, match="HConcat is not supported for multiple partitions"
    ):
        assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "closed, period, offset",
    [
        ("left", "10i", "-5i"),
        ("left", "10i", "20i"),
        ("left", "10i", "-30i"),
        ("right", "10i", "-5i"),
        ("right", "10i", "20i"),
        ("right", "10i", "-30i"),
        ("both", "10i", "-5i"),
        ("both", "10i", "20i"),
        ("both", "10i", "-30i"),
        ("none", "10i", "-5i"),
        ("none", "10i", "20i"),
        ("none", "10i", "-30i"),
    ],
    ids=[
        "left-nonzero-overlap",
        "left-fully-leading",
        "left-fully-trailing",
        "right-nonzero-overlap",
        "right-fully-leading",
        "right-fully-trailing",
        "both-nonzero-overlap",
        "both-fully-leading",
        "both-fully-trailing",
        "none-nonzero-overlap",
        "none-fully-leading",
        "none-fully-trailing",
    ],
)
def test_rolling_integer_offset(engine, period, offset, closed) -> None:
    df = pl.LazyFrame(
        {
            "orderby": [0, 5, 10, 15, 20, 25, 30, 35],
            "values": [1, 2, 3, 4, 5, 6, 7, 8],
        }
    )
    q = df.rolling("orderby", period=period, offset=offset, closed=closed).agg(
        sum_values=pl.col("values").sum(),
        count=pl.len(),
    )

    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("closed", ["left", "right", "both", "none"])
def test_rolling_datetime_period(engine, closed) -> None:
    df = pl.LazyFrame(
        {
            "dt": pl.Series(
                [
                    dt.datetime(2020, 1, 1, 13, 45, 48),
                    dt.datetime(2020, 1, 1, 16, 42, 13),
                    dt.datetime(2020, 1, 1, 16, 45, 9),
                    dt.datetime(2020, 1, 2, 18, 12, 48),
                    dt.datetime(2020, 1, 3, 19, 45, 32),
                    dt.datetime(2020, 1, 8, 23, 16, 43),
                ],
                dtype=pl.Datetime("us"),
            ),
            "values": [3, 7, 5, 9, 2, 1],
        }
    )
    q = df.rolling("dt", period="2d", closed=closed).agg(
        sum_values=pl.col("values").sum()
    )

    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("closed", ["left", "right", "both", "none"])
def test_grouped_rolling_warns(engine, closed) -> None:
    df = pl.LazyFrame(
        {
            "key": ["a", "a", "a", "a", "b", "b", "b", "b"],
            "orderby": [1, 4, 8, 10, 1, 3, 6, 10],
            "values": [1, 2, 3, 4, 10, 20, 30, 40],
        }
    )
    q = df.rolling("orderby", period="4i", closed=closed, group_by="key").agg(
        sum_values=pl.col("values").sum(),
        count=pl.len(),
    )

    with warns_on_spmd(
        engine,
        UserWarning,
        match="Grouped or sliced rolling does not support multiple partitions",
    ):
        assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "expr",
    [
        pl.col("x").sum().over("g"),
        pl.len().over("g"),
        pl.col("x").sum().over("g", "g2"),
        pl.col("x").sum().over("g_null"),
        pl.col("x").sum().over("g", order_by="s"),
        pl.col("x").rank(method="dense", descending=True).over("g"),
        pl.col("x").rank(method="min").over("g", "g2"),
        pl.col("x").cum_sum().over("g", order_by="s"),
        pl.col("x").diff().over("g", order_by="s"),
        pl.col("x").shift(1).over("g", order_by="s"),
        pl.col("x").shift(-1, fill_value=0).over("g", order_by="s"),
        pl.when((pl.col("x") % 2) == 0)
        .then(None)
        .otherwise(pl.col("x"))
        .fill_null(strategy="forward")
        .over("g", order_by="s"),
    ],
    ids=[
        "single_key_sum",
        "len_over",
        "multi_key",
        "null_keys",
        "order_by",
        "rank_dense",
        "rank_min_multi_key",
        "cum_sum_order_by",
        "diff_order_by",
        "shift_order_by",
        "shift_fill_order_by",
        "fill_null_forward",
    ],
)
def test_over_select(engine, expr):
    df = pl.LazyFrame(
        {
            "g": [1, 1, 2, 2, 2, 1],
            "x": [1, 2, 3, 4, 5, 6],
            "g2": ["a", "b", "a", "b", "a", "b"],
            "g_null": [1, None, 1, None, 2, 1],
            "s": [6, 5, 4, 3, 2, 1],
        }
    )
    assert_gpu_result_equal(df.select(expr), engine=engine, check_row_order=True)


def test_over_select_static_planning_falls_back(spmd_engine_factory) -> None:
    engine = spmd_engine_factory(
        StreamingOptions(
            max_rows_per_partition=2,
            dynamic_planning=None,
            fallback_mode="warn",
        )
    )
    df = pl.LazyFrame({"g": [1, 1, 2, 2, 2, 1], "x": [1, 2, 3, 4, 5, 6]})
    q = df.select(pl.col("x").sum().over("g"))

    with warns_on_spmd(
        engine,
        UserWarning,
        match="This selection is not supported for multiple partitions.",
    ):
        assert_gpu_result_equal(q, engine=engine, check_row_order=True)


def test_rolling_sum_over(engine):
    df = (
        pl.LazyFrame(
            {
                "ric": ["A", "A", "A", "B", "B", "B"],
                "ts": [
                    dt.datetime(2025, 1, 1, 9, 0),
                    dt.datetime(2025, 1, 1, 9, 1),
                    dt.datetime(2025, 1, 1, 9, 3),
                    dt.datetime(2025, 1, 1, 9, 0),
                    dt.datetime(2025, 1, 1, 9, 2),
                    dt.datetime(2025, 1, 1, 9, 3),
                ],
                "price": [10.0, 11.0, 12.0, 20.0, 21.0, 22.0],
                "volume": [100, 200, 300, 400, 500, 600],
            }
        )
        .with_columns(notional=pl.col("price") * pl.col("volume"))
        .sort("ric", "ts")
    )
    q = df.with_columns(
        volume_before=pl.col("volume")
        .sum()
        .rolling("ts", period="2m", offset="-2m", closed="left")
        .over("ric"),
        notional_before=pl.col("notional")
        .sum()
        .rolling("ts", period="2m", offset="-2m", closed="left")
        .over("ric"),
        volume_after=pl.col("volume")
        .sum()
        .rolling("ts", period="2m", closed="right")
        .over("ric"),
    ).select(
        "ric",
        "ts",
        "volume_before",
        "notional_before",
        "volume_after",
    )
    expected = pl.DataFrame(
        {
            "ric": ["A", "A", "A", "B", "B", "B"],
            "ts": [
                dt.datetime(2025, 1, 1, 9, 0),
                dt.datetime(2025, 1, 1, 9, 1),
                dt.datetime(2025, 1, 1, 9, 3),
                dt.datetime(2025, 1, 1, 9, 0),
                dt.datetime(2025, 1, 1, 9, 2),
                dt.datetime(2025, 1, 1, 9, 3),
            ],
            "volume_before": [0, 100, 200, 0, 400, 500],
            "notional_before": [0.0, 1000.0, 2200.0, 0.0, 8000.0, 10500.0],
            "volume_after": [100, 300, 300, 400, 500, 1100],
        }
    )
    assert_frame_equal(q.collect(engine=engine), expected)


def test_rolling_common_aggs_over(engine):
    df = pl.LazyFrame(
        {
            "g": ["A", "A", "A", "B", "B", "B"],
            "ts": [1, 2, 4, 1, 3, 4],
            "x": [100, 200, 300, 400, 500, 600],
        }
    ).sort("g", "ts")
    q = df.select(
        pl.col("x").sum().rolling("ts", period="2i").over("g").alias("sum"),
        pl.col("x").min().rolling("ts", period="2i").over("g").alias("min"),
        pl.col("x").max().rolling("ts", period="2i").over("g").alias("max"),
        pl.col("x").mean().rolling("ts", period="2i").over("g").alias("mean"),
        pl.col("x").count().rolling("ts", period="2i").over("g").alias("count"),
        pl.len().rolling("ts", period="2i").over("g").alias("len"),
    )
    expected = pl.DataFrame(
        {
            "sum": [100, 300, 300, 400, 500, 1100],
            "min": [100, 100, 300, 400, 500, 500],
            "max": [100, 200, 300, 400, 500, 600],
            "mean": [100.0, 150.0, 300.0, 400.0, 500.0, 550.0],
            "count": pl.Series([1, 2, 1, 1, 1, 2], dtype=pl.UInt32),
            "len": pl.Series([1, 2, 1, 1, 1, 2], dtype=pl.UInt32),
        }
    )
    assert_frame_equal(q.collect(engine=engine), expected)


@pytest.mark.parametrize("strategy", ["forward", "backward"])
def test_over_cum_sum_fill_null_per_partition(engine, strategy):
    df = pl.LazyFrame(
        {
            "g": [1, 1, 1, 2, 2],
            "s": [1, 2, 3, 1, 2],
            "x": [10.0, None, 5.0, 20.0, None],
        }
    )
    expr = pl.col("x").cum_sum().fill_null(strategy=strategy).over("g", order_by="s")
    assert_gpu_result_equal(df.select(expr), engine=engine, check_row_order=True)


def test_over_shift_without_order_by_single_rank(spmd_engine_factory) -> None:
    engine = spmd_engine_factory(
        StreamingOptions(max_rows_per_partition=2, fallback_mode="raise"),
    )
    df = pl.LazyFrame(
        {
            "g": [1, 1, 2, 2, 2, 1],
            "x": [1, 2, 3, 4, 5, 6],
        }
    )
    q = df.select(pl.col("x").shift(1).over("g"))
    assert_gpu_result_equal(q, engine=engine, check_row_order=True)


@pytest.mark.parametrize(
    "expr",
    [
        pl.col("x").rank().fill_null(strategy="forward").over("g", order_by="x"),
        pl.col("x")
        .rank()
        .fill_null(strategy="forward")
        .fill_null(strategy="forward")
        .over("g", order_by="x"),
        pl.col("x").rank().abs().fill_null(strategy="forward").over("g", order_by="x"),
        pl.col("x")
        .cum_sum()
        .abs()
        .fill_null(strategy="forward")
        .over("g", order_by="x"),
        pl.col("x")
        .cum_sum()
        .fill_null(strategy="forward")
        .fill_null(strategy="forward")
        .over("g", order_by="x"),
        pl.col("x").shift(1).fill_null(strategy="forward").over("g"),
    ],
    ids=[
        "rank_fill",
        "rank_fill_fill",
        "rank_abs_fill",
        "cum_sum_abs_fill",
        "cum_sum_fill_fill",
        "shift_fill",
    ],
)
def test_over_fill_null_over_window_fails_translation(engine, expr):
    df = pl.LazyFrame({"g": [1, 1, 2, 2, 2, 1], "x": [1.0, None, 3, 4, None, 6]})
    assert_ir_translation_raises(df.select(expr), engine, NotImplementedError)


def test_over_with_columns(engine):
    df = pl.LazyFrame(
        {
            "g": [1, 1, 2, 2, 2, 1],
            "x": [1, 2, 3, 4, 5, 6],
        }
    )
    assert_gpu_result_equal(
        df.with_columns(pl.col("x").sum().over("g")),
        engine=engine,
        check_row_order=True,
    )


def test_over_colliding_internal_agg_names(engine):
    df = pl.LazyFrame(
        {
            "category": ["A", "A", "B", "B", "C"],
            "value": [20, 30, 15, 40, 35],
        }
    )
    q = df.select(
        pl.col("category"),
        pl.col("value"),
        pl.col("value").sum().over("category").alias("cat_sum"),
        pl.col("value").mean().over("category").alias("cat_avg"),
    ).sort("category", "value")
    assert_gpu_result_equal(q, engine=engine, check_row_order=True)


@pytest.mark.parametrize(
    "expr",
    [
        pl.col("x").sum().over(pl.col("g") % 2),
        pl.col("x").sum().over("g", pl.col("x") % 2),
    ],
    ids=["noncol_key", "mixed_col_and_expr_key"],
)
def test_over_noncol_key_fallback(request, streaming_engine_factory, expr) -> None:
    # Non-Col and mixed Col/expr partition-by keys are not yet supported for
    # multi-partition streaming and should fall back to single-partition.
    engine = streaming_engine_factory(
        StreamingOptions(max_rows_per_partition=2, fallback_mode="warn"),
    )
    if not isinstance(engine, SPMDEngine):
        # On Dask/Ray the fallback warning fires on worker processes and is
        # invisible to ``pytest.warns``.
        request.applymarker(
            pytest.mark.xfail(
                reason="https://github.com/NVIDIA/cudf/issues/22405",
                strict=False,
            )
        )
    df = pl.LazyFrame(
        {
            "g": [1, 1, 2, 2, 2, 1],
            "x": [1, 2, 3, 4, 5, 6],
        }
    )
    with pytest.warns(UserWarning, match=r"not supported for multiple partitions"):
        assert_gpu_result_equal(df.select(expr), engine=engine)


def test_over_mixed_keys(streaming_engine_factory) -> None:
    # Multiple over expressions with different partition-by keys are decomposed
    # into separate Over nodes (one per key group) and combined with HConcat.
    engine = streaming_engine_factory(
        StreamingOptions(max_rows_per_partition=2, fallback_mode="warn"),
    )
    df = pl.LazyFrame(
        {
            "g": [1, 1, 2, 2, 2, 1],
            "g2": ["a", "b", "a", "b", "a", "b"],
            "x": [1, 2, 3, 4, 5, 6],
        }
    )
    q = df.select(
        pl.col("x").sum().over("g").alias("s_g"),
        pl.col("x").sum().over("g2").alias("s_g2"),
    )
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


@pytest.mark.parametrize(
    "expr",
    [
        pl.col("x").sum().over("g"),
        pl.len().over("g"),
        pl.col("x").rank(method="dense").over("g"),
        pl.col("x").cum_sum().over("g", order_by="s"),
        pl.col("x").shift(1).over("g", order_by="s"),
    ],
    ids=[
        "scalar_sum",
        "scalar_len",
        "nonscalar_rank",
        "nonscalar_cum_sum",
        "nonscalar_shift",
    ],
)
@pytest.mark.parametrize("max_rows_per_partition", [1, 2])
def test_over_many_partitions(
    streaming_engine_factory, max_rows_per_partition, expr
) -> None:
    # Small max_rows_per_partition forces many chunks, exercising the AllGather
    # (scalar broadcast) and sort-and-split (non-scalar) paths across many
    # partitions. Two values cover both single-row chunks and multi-row chunks
    # so the within-chunk position sort is also exercised.
    engine = streaming_engine_factory(
        StreamingOptions(
            max_rows_per_partition=max_rows_per_partition, fallback_mode="warn"
        ),
    )
    df = pl.LazyFrame(
        {
            "g": [1, 1, 2, 2, 2, 1],
            "x": [1, 2, 3, 4, 5, 6],
            "s": [6, 5, 4, 3, 2, 1],
        }
    )
    assert_gpu_result_equal(df.select(expr), engine=engine, check_row_order=True)


@pytest.mark.parametrize(
    "expr",
    [
        pl.col("x").sum().over("g"),
        pl.col("x").rank(method="dense").over("g"),
    ],
    ids=["scalar_sum", "nonscalar_rank"],
)
def test_over_empty_input(streaming_engine_factory, expr) -> None:
    engine = streaming_engine_factory(
        StreamingOptions(max_rows_per_partition=2, fallback_mode="warn"),
    )
    df = pl.LazyFrame(
        {
            "g": pl.Series([], dtype=pl.Int64),
            "x": pl.Series([], dtype=pl.Int64),
        }
    )
    assert_gpu_result_equal(df.select(expr), engine=engine, check_row_order=True)


@pytest.mark.parametrize(
    "expr",
    [
        pl.col("x").sum().over("g"),
        pl.col("x").rank(method="dense").over("g"),
        pl.col("x").shift(1).over("g", order_by="x"),
    ],
    ids=["scalar_sum", "nonscalar_rank", "nonscalar_shift"],
)
def test_over_already_partitioned(streaming_engine_factory, expr) -> None:
    # broadcast_limit=0 disables broadcast joins entirely. Therefore, we should
    # already be shuffled on "g" after the join. The over("g") actor should
    # detect this and evaluate chunkwise without a second shuffle.
    engine = streaming_engine_factory(
        StreamingOptions(
            max_rows_per_partition=3, broadcast_limit=0, fallback_mode="warn"
        ),
    )
    left = pl.LazyFrame({"g": [1, 1, 2, 2, 2, 1], "x": [1, 2, 3, 4, 5, 6]})
    right = pl.LazyFrame({"g": [1, 2], "y": [10, 20]})
    assert_gpu_result_equal(
        left.join(right, on="g").with_columns(expr),
        engine=engine,
        check_row_order=False,
    )


def test_over_in_filter_unsupported(request, streaming_engine_factory) -> None:
    engine = streaming_engine_factory(
        StreamingOptions(max_rows_per_partition=1, fallback_mode="warn"),
    )
    q = pl.concat(
        [
            pl.LazyFrame({"k": ["x", "y"], "v": [3, 2]}),
            pl.LazyFrame({"k": ["x", "y"], "v": [5, 7]}),
        ]
    ).filter(pl.len().over("k") == 2)
    if not isinstance(engine, SPMDEngine):
        # On Dask/Ray the fallback warning fires on worker processes and is
        # invisible to ``pytest.warns``.
        request.applymarker(
            pytest.mark.xfail(
                reason="https://github.com/NVIDIA/cudf/issues/22405",
                strict=False,
            )
        )

    with pytest.warns(
        UserWarning,
        match=r"over\(...\) inside filter is not supported for multiple partitions.*",
    ):
        assert_gpu_result_equal(q, engine=engine)


@pytest.mark.spmd
def test_over_in_filter_after_concat_preserves_order(streaming_engine_factory) -> None:
    # https://github.com/NVIDIA/cudf/issues/22405: row order from an ordered
    # concat must survive the over(...)-in-filter fallback on multi-rank engines.
    engine = streaming_engine_factory(
        StreamingOptions(max_rows_per_partition=1, fallback_mode="silent"),
    )
    q = pl.concat(
        [
            pl.LazyFrame({"k": ["x", "y"], "v": [3, 2]}),
            pl.LazyFrame({"k": ["x", "y"], "v": [5, 7]}),
        ]
    ).filter(pl.len().over("k") == 2)
    assert_gpu_result_equal(q, engine=engine)
