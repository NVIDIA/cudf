# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for dynamic GroupBy operations using the rapidsmpf runtime."""

from __future__ import annotations

import asyncio
import contextlib
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import polars as pl

from cudf_streaming.table_chunk import TableChunk

from cudf_polars.containers import DataFrame
from cudf_polars.engine.options import StreamingOptions
from cudf_polars.streaming.actor_graph import groupby as groupby_actor_graph
from cudf_polars.streaming.actor_graph.collectives.shuffle import ShuffleManager
from cudf_polars.testing.asserts import assert_gpu_result_equal


@pytest.fixture(scope="module")
def df() -> pl.LazyFrame:
    """Create a test DataFrame for groupby."""
    return pl.LazyFrame(
        {
            "key": list(range(50)) * 3,  # 50 unique keys
            "key2": list(range(10)) * 15,  # 10 unique keys
            "value": range(150),
            "value2": [1.0, 2.0, 3.0, 4.0, 5.0] * 30,
            "x": range(150),
            "xx": list(range(75)) * 2,
            "y": [1, 2, 3] * 50,
            "z": [1.0, 2.0, 3.0, 4.0, 5.0] * 30,
        }
    )


@pytest.fixture
def strategy_chunk(spmd_engine) -> TableChunk:
    context = spmd_engine.context
    stream = context.br().stream_pool.get_stream()
    df = DataFrame.from_polars(pl.DataFrame({"key": range(8)}), stream)
    return TableChunk.from_pylibcudf_table(
        df.table, stream, exclusive_view=True, br=context.br()
    )


def test_dynamic_groupby_strategy_avoids_row_limit_allgather(
    monkeypatch, strategy_chunk
):
    """Avoid tree allgather when partial aggregate rows exceed cuDF's row limit."""

    async def fake_allgather_reduce(_context, _comm, _op_id, *local_values):
        estimated_size = strategy_chunk.data_alloc_size() * 4
        assert local_values == (estimated_size, 8, 4, 0)
        return (estimated_size, 8, 4, 0)

    monkeypatch.setattr(groupby_actor_graph, "allgather_reduce", fake_allgather_reduce)
    monkeypatch.setattr(groupby_actor_graph, "MAX_ROWS_PER_PARTITION", 2)
    tracer = SimpleNamespace(decision=None)

    output_count = asyncio.run(
        groupby_actor_graph._choose_strategy(
            None,
            None,
            4,
            strategy_chunk,
            1,
            True,  # noqa: FBT003
            [0],
            1_000_000_000,
            False,  # noqa: FBT003
            False,  # noqa: FBT003
            tracer,
        )
    )

    assert output_count == 4
    assert tracer.decision == "shuffle"


@pytest.mark.parametrize(
    "nranks,npartitions,expected",
    [
        (2, 5, [3, 2]),
        (3, 5, [2, 2, 1]),
        (4, 10, [3, 2, 3, 2]),
    ],
)
def test_partition_count_for_rank_uses_contiguous_ownership(
    nranks, npartitions, expected
):
    """GroupBy metadata uses the same uneven partition ownership as adjust_ordering."""
    counts = [
        groupby_actor_graph._partition_count_for_rank(rank, nranks, npartitions)
        for rank in range(nranks)
    ]
    assert counts == expected


@pytest.mark.parametrize("keys", [("key",), ("key", "key2")])
@pytest.mark.parametrize("agg", ["sum", "mean", "len", "min", "max"])
def test_dynamic_groupby_basic(df, streaming_engine, keys, agg):
    """Test dynamic groupby with various key and agg combinations."""
    expr = getattr(pl.col("value"), agg)()
    q = df.group_by(*keys).agg(expr)
    assert_gpu_result_equal(q, engine=streaming_engine, check_row_order=False)


def test_dynamic_groupby_tree_strategy(df, streaming_engine_factory):
    """Test that small output uses tree reduction (high target_partition_size)."""
    streaming_engine = streaming_engine_factory(
        StreamingOptions(target_partition_size=100_000_000),
    )
    q = df.group_by("key2").agg(pl.col("value").sum())
    assert_gpu_result_equal(q, engine=streaming_engine, check_row_order=False)


def test_dynamic_groupby_shuffle_strategy(streaming_engine_factory):
    """Test that large output uses shuffle (low target_partition_size)."""
    streaming_engine = streaming_engine_factory(
        StreamingOptions(target_partition_size=1000),
    )
    df = pl.LazyFrame({"key": range(1000), "value": range(1000)})
    q = df.group_by("key").agg(pl.col("value").sum())
    assert_gpu_result_equal(q, engine=streaming_engine, check_row_order=False)


@pytest.mark.parametrize("group_keys", [("key", "subkey"), ("key",)])
def test_dynamic_groupby_after_sort_on_group_keys(spmd_engine_factory, group_keys):
    """Group sorted data by the full sort key set or a sorted-key prefix."""
    streaming_engine = spmd_engine_factory(
        StreamingOptions(target_partition_size=128),
    )
    df = pl.LazyFrame(
        {
            "key": [0] * 16 + [1] * 16 + [2] * 16 + [3] * 16,
            "subkey": ([0] * 8 + [1] * 8) * 4,
            "value": range(64),
        }
    )
    q = (
        df.sort("key", "subkey")
        .group_by(*group_keys, maintain_order=True)
        .agg(pl.col("value").sum())
    )
    assert_gpu_result_equal(q, engine=streaming_engine)


def test_dynamic_groupby_single_group(streaming_engine):
    """Test dynamic groupby where all rows have the same key."""
    df = pl.LazyFrame({"key": [1] * 100, "value": range(100)})
    q = df.group_by("key").agg(pl.col("value").sum())
    assert_gpu_result_equal(q, engine=streaming_engine, check_row_order=False)


def test_dynamic_groupby_multiple_aggs(df, streaming_engine):
    """Test dynamic groupby with multiple aggregations."""
    q = df.group_by("key").agg(
        pl.col("value").sum().alias("value_sum"),
        pl.col("value").mean().alias("value_mean"),
        pl.col("value2").min().alias("value2_min"),
    )
    assert_gpu_result_equal(q, engine=streaming_engine, check_row_order=False)


def test_dynamic_groupby_maintain_order(df, streaming_engine):
    """Test dynamic groupby with maintain_order=True."""
    q = df.group_by("key", maintain_order=True).agg(pl.col("value").sum())
    assert_gpu_result_equal(q, engine=streaming_engine, check_row_order=False)


def test_dynamic_groupby_single_row(streaming_engine):
    """Test dynamic groupby on single-row DataFrame."""
    df = pl.LazyFrame({"key": [1], "value": [42]})
    q = df.group_by("key").agg(pl.col("value").sum())
    assert_gpu_result_equal(q, engine=streaming_engine, check_row_order=False)


# ---------------------------------------------------------------------------
# Tests migrated from tests/streaming/test_groupby.py
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op", ["sum", "mean", "len"])
@pytest.mark.parametrize("keys", [("y",), ("y", "z")])
def test_groupby(df, streaming_engine, op, keys):
    q = getattr(df.group_by(*keys), op)()
    assert_gpu_result_equal(q, engine=streaming_engine, check_row_order=False)


@pytest.mark.parametrize("op", ["sum", "mean", "len"])
@pytest.mark.parametrize("keys", [("y",), ("y", "z")])
def test_groupby_single_partitions(df, streaming_engine_factory, op, keys):
    streaming_engine = streaming_engine_factory(
        StreamingOptions(max_rows_per_partition=int(1e9)),
    )
    q = getattr(df.group_by(*keys), op)()
    assert_gpu_result_equal(q, engine=streaming_engine, check_row_order=False)


@pytest.mark.parametrize(
    "op", ["sum", "mean", "len", "count", "min", "max", "n_unique", "std", "var"]
)
@pytest.mark.parametrize("keys", [("y",), ("y", "z")])
def test_groupby_agg(df, streaming_engine, op, keys):
    agg = getattr(pl.col("x"), op)()
    q = df.group_by(*keys).agg(agg)
    assert_gpu_result_equal(q, engine=streaming_engine, check_row_order=False)


def test_groupby_sort_by_first_last(streaming_engine_factory):
    streaming_engine = streaming_engine_factory(
        StreamingOptions(max_rows_per_partition=2, fallback_mode="raise"),
    )
    df = pl.LazyFrame(
        {
            "g": ["B", "A", "C", "A", "B", "C", "A", "B"],
            "idx": [2, 3, 2, 1, 1, 1, 2, 3],
            "val": [40, 30, 60, 10, 30, 50, 20, 50],
        }
    )

    q = df.group_by("g").agg(
        pl.col("val").sum().alias("volume"),
        pl.col("val").sort_by("idx").first().alias("open"),
        pl.col("val").sort_by("idx").last().alias("close"),
    )
    assert_gpu_result_equal(q, engine=streaming_engine, check_row_order=False)


def test_groupby_sort_by_first_last_stable_ties(streaming_engine_factory):
    streaming_engine = streaming_engine_factory(
        StreamingOptions(max_rows_per_partition=2, fallback_mode="raise"),
    )
    df = pl.LazyFrame(
        {
            "g": ["A", "B", "A", "B", "A", "B"],
            "idx": [1, 1, 1, 1, 1, 1],
            "val": [10, 40, 20, 50, 30, 60],
        }
    )

    q = df.group_by("g").agg(
        pl.col("val").sort_by("idx", maintain_order=True).first().alias("first_tie"),
        pl.col("val").sort_by("idx", maintain_order=True).last().alias("last_tie"),
    )
    assert_gpu_result_equal(q, engine=streaming_engine, check_row_order=False)


def test_groupby_sort_by_stable_ties_with_preshuffle_fallback(
    streaming_engine_factory,
):
    streaming_engine = streaming_engine_factory(
        StreamingOptions(max_rows_per_partition=2, fallback_mode="raise"),
    )
    df = pl.LazyFrame(
        {
            "g": ["A", "B", "A", "B", "A", "B", "A", "B"],
            "idx": [1, 1, 1, 1, 1, 1, 1, 1],
            "val": [10, 100, 20, 200, 30, 300, 40, 400],
            "u": [1, 1, 2, 2, 3, 3, 4, 4],
        }
    )

    q = df.group_by("g").agg(
        pl.col("u").n_unique().alias("nu"),
        pl.col("val").sort_by("idx", maintain_order=True).first().alias("first_tie"),
        pl.col("val").sort_by("idx", maintain_order=True).last().alias("last_tie"),
    )
    with pytest.raises(NotImplementedError, match="input-order ties"):
        q.collect(engine=streaming_engine)


def test_groupby_sort_by_preserves_sorted_key_order(streaming_engine_factory):
    streaming_engine = streaming_engine_factory(
        StreamingOptions(max_rows_per_partition=2, fallback_mode="raise"),
    )
    df = pl.LazyFrame(
        {
            "g": ["A", "A", "B", "B", "C", "C"],
            "idx": [1, 2, 1, 2, 1, 2],
            "val": [10, 20, 30, 40, 50, 60],
        }
    )

    q = (
        df.sort("g", descending=True)
        .group_by("g", maintain_order=True)
        .agg(
            pl.col("val").sort_by("idx").first().alias("open"),
            pl.col("val").sort_by("idx").last().alias("close"),
        )
    )
    assert_gpu_result_equal(q, engine=streaming_engine, check_row_order=True)


@pytest.mark.parametrize("ddof", [0, 2, 50])
@pytest.mark.parametrize("agg", ["std", "var"])
def test_groupby_std_var_ddof(df, engine, agg, ddof):
    q = df.group_by("y").agg(getattr(pl.col("x"), agg)(ddof=ddof))
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


@pytest.mark.parametrize("fallback_mode", ["silent", "raise", "warn", "foo"])
def test_groupby_fallback(df, fallback_mode, spmd_engine_factory):
    streaming_engine = spmd_engine_factory(
        StreamingOptions(fallback_mode=fallback_mode),
    )
    match = "Failed to decompose groupby aggs"

    q = df.group_by("y").median()

    if fallback_mode == "silent":
        ctx = contextlib.nullcontext()
    elif fallback_mode == "raise":
        ctx = pytest.raises(
            NotImplementedError,
            match=match,
        )
    elif fallback_mode == "foo":
        ctx = pytest.raises(
            pl.exceptions.ComputeError,
            match="'foo' is not a valid StreamingFallbackMode",
        )
    else:
        ctx = pytest.warns(UserWarning, match=match)
    with ctx:
        assert_gpu_result_equal(q, engine=streaming_engine, check_row_order=False)


def test_groupby_agg_literal(df, streaming_engine):
    q = df.group_by("y").agg(1)
    assert_gpu_result_equal(q, engine=streaming_engine, check_row_order=False)


@pytest.mark.parametrize(
    "op",
    [
        pl.max("x") - pl.min("x"),
        pl.mean("x") * pl.sum("x"),
        pl.max("x") + pl.max("z"),
        pl.max("x") + 1,
    ],
)
def test_groupby_agg_binop(df: pl.LazyFrame, streaming_engine, op: pl.Expr) -> None:
    q = df.group_by("y").agg(op)
    assert_gpu_result_equal(q, engine=streaming_engine, check_row_order=False)


@pytest.mark.parametrize(
    "op, column_name",
    [
        (pl.max("x") - pl.min("x"), "x__max_min"),
        (pl.mean("x"), "x__mean_sum"),
    ],
)
def test_groupby_agg_duplicate(streaming_engine, op: pl.Expr, column_name: str) -> None:
    # Ensure that the column names we create internally don't collide with
    # the user's column names.
    df = pl.LazyFrame(
        {
            "x": [0, 1, 2, 3] * 2,
            column_name: [4, 5, 6, 7] * 2,
            "y": [1, 2, 1, 2] * 2,
        }
    )
    q = df.group_by("y").agg(op, pl.min(column_name))
    assert_gpu_result_equal(q, engine=streaming_engine, check_row_order=False)


def test_groupby_agg_empty(df: pl.LazyFrame, streaming_engine) -> None:
    q = df.group_by("y").agg()
    assert_gpu_result_equal(q, engine=streaming_engine, check_row_order=False)


@pytest.mark.filterwarnings("ignore:This slice not supported for multiple partitions.")
@pytest.mark.parametrize("zlice", [(0, 2), (2, 2), (-2, None)])
def test_groupby_then_slice(streaming_engine, zlice: tuple[int, int]) -> None:
    df = pl.LazyFrame(
        {
            "x": [0, 1, 2, 3] * 2,
            "y": [1, 2, 1, 2] * 2,
        }
    )
    q = df.group_by("y", maintain_order=True).max().slice(*zlice)
    assert_gpu_result_equal(q, engine=streaming_engine)


def test_groupby_on_equality(streaming_engine) -> None:
    # See: https://github.com/NVIDIA/cudf/issues/19152
    df = pl.LazyFrame(
        {
            "key1": [1, 1, 1, 2, 3, 1, 4, 6, 7],
            "key2": [2, 2, 2, 2, 6, 1, 4, 6, 8],
            "int32": pl.Series([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=pl.Int32()),
        }
    )
    q = df.group_by(pl.col("key1") == pl.col("key2")).agg(pl.col("int32").sum())
    assert_gpu_result_equal(q, engine=streaming_engine, check_row_order=False)


@pytest.mark.parametrize(
    "values",
    [
        [1, None, 2, None],
        [1, None, None, None],
    ],
)
def test_mean_partitioned(values: list[int | None], streaming_engine_factory) -> None:
    streaming_engine = streaming_engine_factory(
        StreamingOptions(max_rows_per_partition=2),
    )
    df = pl.LazyFrame(
        {
            "key1": [1, 1, 2, 2],
            "uint16_with_null": pl.Series(values, dtype=pl.UInt16()),
        }
    )
    q = df.group_by("key1").agg(pl.col("uint16_with_null").mean())
    assert_gpu_result_equal(q, engine=streaming_engine, check_row_order=False)


def test_groupby_literal_key(df, streaming_engine):
    q = (
        df.group_by(
            pl.lit(True).alias("key"),  # noqa: FBT003
            maintain_order=False,
        )
        .agg(pl.col("x").sum())
        .drop("key")
    )
    assert_gpu_result_equal(q, engine=streaming_engine)


# ---------------------------------------------------------------------------
# Tests migrated from tests/streaming/test_groupby.py
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op", ["sum", "mean", "len", "count"])
@pytest.mark.parametrize("keys", [("y",), ("y", "z")])
def test_groupby_agg_config_options(df, op, keys, streaming_engine_factory):
    streaming_engine = streaming_engine_factory(
        StreamingOptions(max_rows_per_partition=4),
    )
    agg = getattr(pl.col("x"), op)()
    if op in ("sum", "mean"):
        agg = agg.round(2)  # Unary test coverage
    q = df.group_by(*keys).agg(agg)
    assert_gpu_result_equal(q, engine=streaming_engine, check_row_order=False)


def test_groupby_count_type_mismatch(df, streaming_engine_factory):
    streaming_engine = streaming_engine_factory(
        StreamingOptions(target_partition_size=1),
    )
    q = df.group_by("key", maintain_order=True).agg(pl.col("value").count())
    assert_gpu_result_equal(q, engine=streaming_engine, check_row_order=False)


@pytest.mark.skip_on_streaming_engine(
    "patch.object on ShuffleManager.Inserter doesn't reach worker processes",
    engine=("dask", "ray"),
)
def test_shuffle_reduce_insert_finished_called_on_oom(streaming_engine_factory):
    streaming_engine = streaming_engine_factory(
        StreamingOptions(target_partition_size=10, max_rows_per_partition=5),
    )
    # Tests that an exception raised inside insert_hash() must not leave the
    # C++ ShufflerAsync without insert_finished() being called.

    def foo(*args, **kwargs):
        raise MemoryError("OOM in insert_hash()")

    df = pl.LazyFrame({"a": range(10), "b": range(10)})
    with (
        patch.object(ShuffleManager.Inserter, "insert_hash", foo),
        pytest.raises(MemoryError) as exc_info,
    ):
        df.group_by("a").agg(pl.col("b").sum()).collect(engine=streaming_engine)
    assert "OOM in insert_hash" in str(exc_info.value)
