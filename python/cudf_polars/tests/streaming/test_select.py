# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
from decimal import Decimal

import pytest

import polars as pl

import pylibcudf as plc

from cudf_polars import Translator
from cudf_polars.containers import DataType
from cudf_polars.dsl import expr
from cudf_polars.dsl.ir import HStack, Projection, Select
from cudf_polars.engine.options import StreamingOptions
from cudf_polars.streaming.select import _inline_hstack_false, _sub_expr
from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
)
from cudf_polars.testing.engine_utils import warns_on_spmd


@pytest.fixture
def engine(streaming_engine_factory):
    return streaming_engine_factory(
        StreamingOptions(max_rows_per_partition=3, fallback_mode="warn"),
    )


@pytest.fixture(scope="module")
def df():
    return pl.LazyFrame(
        {
            "a": [1, 2, 3, 4, 5, 6, 7],
            "b": [1, 1, 1, 1, 1, 1, 1],
            "c": [2, 4, 6, 8, 10, 12, 14],
        }
    )


def test_select(df, engine):
    query = df.select(
        pl.col("a") + pl.col("b"), (pl.col("a") * 2 + pl.col("b")).alias("d")
    )
    assert_gpu_result_equal(query, engine=engine)


@pytest.mark.parametrize("fallback_mode", ["silent", "raise", "warn", "foo"])
def test_select_reduce_fallback(df, spmd_engine_factory, fallback_mode):
    engine = spmd_engine_factory(
        StreamingOptions(max_rows_per_partition=3, fallback_mode=fallback_mode),
    )
    match = "This selection is not supported for multiple partitions."

    query = df.select(
        (pl.col("a") + pl.col("b")).max(),
        # NOTE: We don't support `median` yet
        (pl.col("a") * 2 + pl.col("b")).alias("d").median(),
    )

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
        assert_gpu_result_equal(query, engine=engine)


def test_select_fill_null_with_strategy(df, streaming_engine_factory):
    engine = streaming_engine_factory(
        StreamingOptions(max_rows_per_partition=3, fallback_mode="warn"),
    )
    q = df.select(pl.col("a").forward_fill())

    with warns_on_spmd(
        engine,
        UserWarning,
        match="fill_null with strategy other than 'zero' or 'one' is not supported for multiple partitions",
    ):
        assert_gpu_result_equal(q, engine=engine)


def test_select_scan_fill_null_with_strategy_fallback(tmp_path, spmd_engine_factory):
    engine = spmd_engine_factory(
        StreamingOptions(target_partition_size=1, fallback_mode="warn"),
    )
    path = tmp_path / "data.parquet"
    pl.DataFrame({"a": [None, 1, None, 3]}).write_parquet(path)
    q = pl.scan_parquet(path).select(pl.col("a").fill_null(strategy="forward"))

    with warns_on_spmd(
        engine,
        UserWarning,
        match="fill_null with strategy other than 'zero' or 'one' is not supported for multiple partitions",
    ):
        assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "aggs",
    [
        (pl.col("a").sum(),),
        (pl.col("a").min() + pl.col("a"),),
        (
            (pl.col("a") + pl.col("b")).sum(),
            (pl.col("a") * 2 + pl.col("b")).alias("d").min(),
        ),
        (pl.col("a").min() + pl.col("b").max(),),
        (pl.col("a") - (pl.col("b") + pl.col("c").max()).sum(),),
        (pl.col("b").len(),),
        (pl.col("a") - (pl.col("b") + pl.col("c").max()).mean(),),
        (
            pl.col("b").sum(),
            (pl.col("c").sum() + 1),
        ),
        (
            pl.col("b").n_unique(),
            (pl.col("c").n_unique() + 1),
        ),
        (pl.col("a").min(), pl.col("b"), pl.col("c").max()),
    ],
)
def test_select_aggs(df, engine, aggs):
    # Test supported aggs (e.g. "min", "max", "mean", "n_unique")
    query = df.select(*aggs)
    assert_gpu_result_equal(query, engine=engine)


@pytest.mark.parametrize(
    "aggs",
    [
        (pl.col("a").drop_nulls().n_unique(),),
        (pl.col("a").drop_nulls().sum(),),
        (pl.col("a").drop_nulls().min(),),
        (pl.col("a").drop_nulls().max(),),
        (pl.col("a").drop_nulls().mean(),),
        (pl.col("a").null_count(),),
    ],
)
def test_select_drop_nulls_aggs(engine, aggs):
    df = pl.LazyFrame({"a": [1, 2, None, 2, 3, None, 1, 4]})
    query = df.select(*aggs)
    assert_gpu_result_equal(query, engine=engine)


def test_select_with_cse_no_agg(df, engine):
    expr = pl.col("a") + pl.col("a")
    query = df.select(expr, (expr * 2).alias("b"), ((expr * 2) + 10).alias("c"))
    assert_gpu_result_equal(query, engine=engine)


def test_select_parquet_fast_count(tmp_path, df, engine):
    file = tmp_path / "data.parquet"
    df.collect().write_parquet(file)
    q = pl.scan_parquet(file).select(pl.len())
    assert_gpu_result_equal(q, engine=engine)


def test_select_literal(engine):
    # See: https://github.com/NVIDIA/cudf/issues/19147
    ldf = pl.LazyFrame({"a": list(range(10))})
    q = ldf.select(pl.lit(2).pow(pl.lit(-3, dtype=pl.Float32)))
    assert_gpu_result_equal(q, engine=engine)


def test_select_with_empty_partitions(engine):
    df = pl.concat(
        [
            pl.LazyFrame({"b": pl.Series([], dtype=pl.Decimal(15, 2))}),
            pl.LazyFrame({"b": pl.Series([], dtype=pl.Decimal(15, 2))}),
        ]
    )
    q = df.select(pl.col("b").sum() / Decimal("7.00"))
    # Polars pre their decimal overhaul: https://github.com/pola-rs/polars/issues/19784
    # returned a different precision and scale, so we skip dtype check
    assert_gpu_result_equal(q, engine=engine)


def test_select_mean_with_decimals(engine):
    df = pl.LazyFrame({"d": [Decimal("1.23")] * 4})
    q = df.select(pl.mean("d"))
    assert_gpu_result_equal(q, engine=engine)


def test_select_with_len(streaming_engine_factory):
    engine = streaming_engine_factory(
        StreamingOptions(max_rows_per_partition=3, fallback_mode="warn"),
    )
    df1 = pl.LazyFrame({"c0": [1] * 4})
    df2 = pl.LazyFrame({"c0": [2] * 4})
    q = pl.concat([df1.join(df2, how="cross"), df1.with_columns(pl.lit(None))]).select(
        pl.len()
    )
    with warns_on_spmd(
        engine,
        UserWarning,
        match="Cross join not support for multiple partitions",
    ):
        assert_gpu_result_equal(q, engine=engine)


def test_sub_expr_replaces_col():
    dt = DataType(pl.Int64())
    b = expr.Col(dt, "b")
    assert _sub_expr(expr.Col(dt, "a"), {"a": b}) is b


def test_sub_expr_reconstructs_when_nested_col_substituted():
    dt = DataType(pl.Int8())
    a = expr.Col(dt, "a")
    b = expr.Col(dt, "b")
    c = expr.Col(dt, "c")
    add = expr.BinOp(dt, plc.binaryop.BinaryOperator.ADD, a, c)
    out = _sub_expr(add, {"a": b})
    assert out.children[0] is b
    assert out.children[1] is c


def test_inline_hstack_false_noop_scan():
    ir = Translator(
        pl.LazyFrame({"a": [1]})._ldf.visit(), pl.GPUEngine()
    ).translate_ir()
    assert _inline_hstack_false(ir) is ir


def test_inline_hstack_false_inlines_hstack_false_chain():
    base = Translator(
        pl.LazyFrame({"a": [1, 2, 3]})._ldf.visit(), pl.GPUEngine()
    ).translate_ir()
    dt = base.schema["a"]
    inner = HStack(
        {**base.schema, "s": dt},
        (expr.NamedExpr("s", expr.Col(dt, "a")),),
        should_broadcast=False,
        df=base,
    )
    ir = Select(
        {"t": dt},
        (expr.NamedExpr("t", expr.Col(dt, "s")),),
        should_broadcast=True,
        df=inner,
    )
    out = _inline_hstack_false(ir)
    assert out.children[0] is base
    assert isinstance(out.exprs[0].value, expr.Col)
    assert out.exprs[0].value.name == "a"


def test_inline_hstack_false_non_select_hstack_parent_unchanged():
    base = Translator(
        pl.LazyFrame({"a": [1, 2, 3]})._ldf.visit(), pl.GPUEngine()
    ).translate_ir()
    dt = base.schema["a"]
    inner = HStack(
        {**base.schema, "s": dt},
        (expr.NamedExpr("s", expr.Col(dt, "a")),),
        should_broadcast=False,
        df=base,
    )
    proj = Projection(inner.schema, inner)
    assert _inline_hstack_false(proj) is proj


def test_select_with_mixed_fusable_non_fusable_exprs(df, engine):
    q = df.select(
        foo=pl.col("a").n_unique(),
        bar=pl.col("b").sum(),
        baz=pl.col("c").sum(),
    )
    assert_gpu_result_equal(q, engine=engine)
