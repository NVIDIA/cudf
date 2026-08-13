# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for CSE HStack handling in the streaming executor."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import polars as pl

from cudf_polars import Translator
from cudf_polars.containers import DataType
from cudf_polars.dsl import expr
from cudf_polars.dsl.expr import Col
from cudf_polars.dsl.expressions.base import ExecutionContext
from cudf_polars.dsl.ir import Filter, HStack
from cudf_polars.dsl.traversal import traversal
from cudf_polars.engine.options import StreamingOptions
from cudf_polars.streaming.parallel import lower_ir_graph
from cudf_polars.streaming.repartition import Repartition
from cudf_polars.streaming.statistics import collect_statistics
from cudf_polars.testing.asserts import assert_gpu_result_equal
from cudf_polars.testing.engine_utils import warns_on_spmd
from cudf_polars.utils.config import ConfigOptions, StreamingFallbackMode

if TYPE_CHECKING:
    import concurrent.futures


@pytest.fixture
def engine(streaming_engine_factory):
    # Override ``fallback_mode`` because the ``spmd-small`` baseline sets
    # it to ``SILENT``, which would swallow the ``UserWarning`` that
    # ``test_hstack_non_scalar_cse_fallback`` asserts on.
    return streaming_engine_factory(
        StreamingOptions(
            max_rows_per_partition=3,
            raise_on_fail=True,
            fallback_mode=StreamingFallbackMode.WARN,
        ),
    )


@pytest.fixture(scope="module")
def df():
    return pl.LazyFrame({"a": [1, 2, 3, 4, 5, 6, 7], "b": [2, 3, 4, 5, 6, 7, 8]})


def test_cse_agg_with_columns(df, engine):
    # CSE: col("a").sum() appears twice; Polars emits HStack(broadcast=False)
    # with the shared aggregation, then HStack(broadcast=True) with the outer
    # expressions. Exercises the Projection handler's outer_bcast_hstacks path.
    q = df.with_columns(
        pl.col("a").sum().alias("s"),
        (pl.col("a").sum() * 2).alias("s2"),
    )
    assert_gpu_result_equal(q, engine=engine)


def test_cse_agg_select(df, engine):
    # CSE: col("a").sum() appears twice in a select.
    # Exercises the Select handler with no outer broadcast HStack.
    q = df.select(
        pl.col("a").sum().alias("s"),
        (pl.col("a").sum() * 2).alias("s2"),
    )
    assert_gpu_result_equal(q, engine=engine)


def test_hstack_non_scalar_cse_fallback(df, engine):
    # Non-scalar CSE (head(5)) skips the CSE transform, falling through to the
    # non-pointwise HStack fallback in lower_ir_node.register(HStack).
    q = df.with_columns(
        pl.col("a").head(5).min().alias("min_5"),
        pl.col("a").head(5).max().alias("max_5"),
    )
    with warns_on_spmd(
        engine, UserWarning, match="not supported for multiple partitions"
    ):
        assert_gpu_result_equal(q, engine=engine)


def test_hstack_non_pointwise_redirect_covers_parallel_hstack_handler(
    engine, parquet_stats_executor: concurrent.futures.ThreadPoolExecutor
):
    """Filter → rec(HStack) so standalone non-pointwise HStack hits redirect to Select."""
    base = Translator(
        pl.LazyFrame({"a": [1, 2, 3], "b": [4, 5, 6]})._ldf.visit(), engine
    ).translate_ir()
    dt_a = base.schema["a"]
    sum_dtype = DataType(pl.Int64())
    agg_sum = expr.Agg(
        sum_dtype,
        "sum",
        None,
        ExecutionContext.FRAME,
        Col(dt_a, "a"),
    )
    hstack_schema = {**base.schema, "s": sum_dtype}
    hstack = HStack(
        hstack_schema,
        (expr.NamedExpr("s", agg_sum),),
        should_broadcast=True,
        df=base,
    )
    mask = expr.NamedExpr("m", expr.Literal(DataType(pl.Boolean()), value=True))
    root = Filter(hstack_schema, mask, hstack)
    config_options = ConfigOptions.from_polars_engine(engine)
    lower_ir_graph(
        root,
        config_options,
        collect_statistics(
            root,
            config_options,
            parquet_stats_executor,
        ),
    )


def test_with_columns_scalar_upstream_20981(engine):
    # Based on upstream-Polars unit test.
    lf = pl.LazyFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
    q = lf.with_columns(pl.col.a.mean())
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("comm_subexpr_elim", [True, False])
def test_cse_agg_shared_decomposition(
    engine,
    comm_subexpr_elim,
    parquet_stats_executor: concurrent.futures.ThreadPoolExecutor,
):
    df = pl.LazyFrame({"a": [1, 2, 3, 4, 5, 6]})
    q = df.with_columns(
        pl.col("a").sum().alias("s"),
        (pl.col("a").sum() * 2).alias("s2"),
    )
    opts = pl.QueryOptFlags(comm_subexpr_elim=comm_subexpr_elim)
    ir = Translator(
        q._ldf.with_optimizations(opts._pyoptflags).visit(), engine
    ).translate_ir()

    # With CSE, Polars lifts sum(a) into a nested HStack(should_broadcast=False).
    # Without CSE, a single flat HStack contains both expressions.
    inner_hstacks = [
        n for n in traversal([ir]) if isinstance(n, HStack) and not n.should_broadcast
    ]
    assert len(inner_hstacks) == (1 if comm_subexpr_elim else 0)

    config_options = ConfigOptions.from_polars_engine(engine)
    lowering = lower_ir_graph(
        ir,
        config_options,
        collect_statistics(ir, config_options, parquet_stats_executor),
    )

    # Both paths must lower to a single Repartition computing one aggregation.
    repartitions = [
        n for n in traversal([lowering.lowered]) if isinstance(n, Repartition)
    ]
    assert len(repartitions) == 1
    assert len(repartitions[0].children[0].exprs) == 1  # type: ignore[attr-defined]
    assert_gpu_result_equal(q, engine=engine, collect_kwargs={"optimizations": opts})


def test_hstack_with_cse_and_column_override(engine):
    # When a with_columns overrides a column and a CSE placeholder is hoisted,
    # other expressions in the same with_columns must still see the original
    # column, not the overridden value.
    df = pl.LazyFrame({"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8.0, 9.0]})
    q = df.with_columns(
        a=(1 + 3 * pl.col("a")) * (1 / pl.col("a")),
        c=pl.col("a") + pl.col("b") / 2,
        e=((pl.col("a") > pl.col("b")) & (pl.col("a") >= pl.col("z"))).cast(pl.Int64),
        k=2 // pl.col("a"),
    )
    assert_gpu_result_equal(q, engine=engine)
