# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)

# One partial result exercises selection; full and overfull requests cover the
# separate "all rows" behavior without repeating intermediate result lengths.
K_CASES = [0, 1, 4, 5]


@pytest.fixture
def df():
    return pl.LazyFrame(
        {
            "test": [2, 4, 1, 3],
            "val": [2, 4, 9, 3],
            "bool_val": [False, True, True, False],
            "str_value": ["d", "b", "a", "c"],
            "col_with_nulls": [2, 4, None, 3],
        }
    )


@pytest.mark.parametrize("col", ["test", "bool_val", "str_value", "col_with_nulls"])
@pytest.mark.parametrize("k", K_CASES)
def test_top_k(engine: pl.GPUEngine, df, col, k):
    q = df.select(pl.col(col).top_k(k))
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


@pytest.mark.parametrize("col", ["test", "bool_val", "str_value", "col_with_nulls"])
@pytest.mark.parametrize("k", K_CASES)
def test_bottom_k(engine: pl.GPUEngine, df, col, k):
    q = df.select(pl.col(col).bottom_k(k))
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


@pytest.mark.parametrize("method", ["top_k", "bottom_k"])
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize(
    "layout", ["nulls_first", "nulls_last", "no_nulls", "all_nulls", "one_row", "empty"]
)
@pytest.mark.parametrize("k", [0, 2, 4, 5, 7])
def test_top_k_sorted_input(engine: pl.GPUEngine, method, descending, layout, k):
    values: list[int | None] = [4, 3, 2, 1] if descending else [1, 2, 3, 4]
    if layout == "nulls_first":
        values = [None, None, *values]
    elif layout == "nulls_last":
        values = [*values, None, None]
    elif layout == "all_nulls":
        values = [None, None, None]
    elif layout == "one_row":
        values = [7]
    elif layout == "empty":
        values = []
    col = pl.col("a").set_sorted(descending=descending)
    expr = col.top_k(k) if method == "top_k" else col.bottom_k(k)
    q = pl.LazyFrame({"a": pl.Series(values, dtype=pl.Int64)}).select(expr)
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


@pytest.mark.parametrize("method", ["top_k", "bottom_k"])
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("k", [1, 3])
def test_top_k_sorted_input_nan(engine: pl.GPUEngine, method, descending, k):
    floats = [1.0, 2.0, float("nan"), float("nan")]
    values = [None, *(floats[::-1] if descending else floats)]
    col = pl.col("a").set_sorted(descending=descending)
    expr = col.top_k(k) if method == "top_k" else col.bottom_k(k)
    q = pl.LazyFrame({"a": values}).select(expr)
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


@pytest.mark.parametrize("method", ["top_k", "bottom_k"])
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("nulls_last", [False, True])
@pytest.mark.parametrize("k", [2, 5])
def test_top_k_sorted_input_flags(in_memory_engine, method, descending, nulls_last, k):
    values = [4, 3, 2, 1] if descending else [1, 2, 3, 4]
    values = [*values, None] if nulls_last else [None, *values]
    col = pl.col("a").set_sorted(descending=descending)
    expr = col.top_k(k) if method == "top_k" else col.bottom_k(k)
    q = pl.LazyFrame({"a": values}).select(expr)
    got = q.collect(engine=in_memory_engine)["a"]
    if k == 2:
        expected = [4, 3] if method == "top_k" else [1, 2]
    else:
        expected = values if descending == (method == "top_k") else values[::-1]
    assert got.to_list() == expected
    assert got.flags["SORTED_DESC" if method == "top_k" else "SORTED_ASC"]


@pytest.mark.parametrize("method", ["top_k_by", "bottom_k_by"])
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("nulls", ["first", "last", "none", "one_row", "empty"])
@pytest.mark.parametrize("k", [0, 2, 4, 5, 7])
def test_top_k_by_sorted_input(engine: pl.GPUEngine, method, descending, nulls, k):
    by: list[int | None] = [4, 3, 2, 1] if descending else [1, 2, 3, 4]
    values = [10, 20, 30, 40]
    if nulls == "first":
        by, values = [None, None, *by], [0, 0, *values]
    elif nulls == "last":
        by, values = [*by, None, None], [*values, 0, 0]
    elif nulls == "one_row":
        by, values = [7], [70]
    elif nulls == "empty":
        by, values = [], []
    by_expr = pl.col("by").set_sorted(descending=descending)
    col = pl.col("a")
    expr = (
        col.top_k_by(by_expr, k)
        if method == "top_k_by"
        else col.bottom_k_by(by_expr, k)
    )
    q = pl.LazyFrame(
        {
            "a": pl.Series(values, dtype=pl.Int64),
            "by": pl.Series(by, dtype=pl.Int64),
        }
    ).select(expr)
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


@pytest.mark.parametrize("method", ["top_k", "bottom_k", "top_k_by", "bottom_k_by"])
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("nulls_last", [False, True])
@pytest.mark.parametrize("k", [2, 5])
@pytest.mark.parametrize("dtype", [pl.Int64(), pl.Float64(), pl.String()])
def test_top_k_sorted_input_null_mask(
    engine: pl.GPUEngine, method, descending, nulls_last, k, dtype
):
    values: list[int | None] = [4, 3, 2, 1] if descending else [1, 2, 3, 4]
    values = [*values, None, None] if nulls_last else [None, None, *values]
    sorted_col = pl.col("a").set_sorted(descending=descending)
    col = pl.col("a")
    if method == "top_k":
        expr = sorted_col.top_k(k)
    elif method == "bottom_k":
        expr = sorted_col.bottom_k(k)
    elif method == "top_k_by":
        expr = col.top_k_by(sorted_col, k)
    else:
        expr = col.bottom_k_by(sorted_col, k)
    expr = expr.str.contains("") if dtype == pl.String() else expr.is_finite()
    df = pl.LazyFrame({"a": pl.Series(values, dtype=pl.Int64).cast(dtype)})
    assert_gpu_result_equal(df.select(expr), engine=engine, check_row_order=False)


@pytest.mark.parametrize("by", ["val", "str_value", "col_with_nulls"])
@pytest.mark.parametrize("k", K_CASES)
@pytest.mark.parametrize("reverse", [False, True])
def test_top_k_by(engine: pl.GPUEngine, df, by, k, reverse):
    q = df.select(pl.col("test").top_k_by(by, k, reverse=reverse))
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


@pytest.mark.parametrize("by", ["val", "str_value", "col_with_nulls"])
@pytest.mark.parametrize("k", K_CASES)
@pytest.mark.parametrize("reverse", [False, True])
def test_bottom_k_by(engine: pl.GPUEngine, df, by, k, reverse):
    q = df.select(pl.col("test").bottom_k_by(by, k, reverse=reverse))
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


def test_top_k_by_multiple_by_unsupported(in_memory_engine, df):
    q = df.select(pl.col("test").top_k_by(["val", "str_value"], 2))
    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)


@pytest.mark.parametrize(
    "k_expr",
    [
        pl.col("test").max(),
        pl.col("test").min(),
        pl.lit(2) + pl.lit(1),
    ],
)
def test_top_k_expr_k(engine: pl.GPUEngine, df, k_expr):
    q = df.select(pl.col("val").top_k(k_expr))
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


@pytest.mark.parametrize(
    "k_expr",
    [
        pl.col("test").max(),
        pl.col("test").min(),
        pl.lit(2) + pl.lit(1),
    ],
)
def test_bottom_k_expr_k(engine: pl.GPUEngine, df, k_expr):
    q = df.select(pl.col("val").bottom_k(k_expr))
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


@pytest.mark.parametrize(
    "k_expr",
    [
        pl.col("test").max(),
        pl.col("test").min(),
        pl.lit(2) + pl.lit(1),
    ],
)
@pytest.mark.parametrize("reverse", [False, True])
def test_top_k_by_expr_k(engine: pl.GPUEngine, df, k_expr, reverse):
    q = df.select(pl.col("val").top_k_by("col_with_nulls", k_expr, reverse=reverse))
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


@pytest.mark.parametrize("k_expr", [pl.col("test").max(), pl.lit(2.0)])
def test_top_k_by_scalar_key(engine: pl.GPUEngine, df, k_expr):
    # With only scalar keys every row ties, so only the row count is defined.
    q = df.select(pl.col("val").top_k_by(pl.col("test").max(), k_expr))
    got = q.collect(engine=engine)["val"]
    expect = q.collect()["val"]
    assert got.len() == expect.len()
    assert set(got) <= set(df.collect()["val"])


@pytest.mark.parametrize(
    "k_expr",
    [
        pl.lit(None, dtype=pl.Int64),
        pl.lit(-1),
        pl.lit(float("nan")),
        pl.lit("a"),
        pl.lit(pl.Series([], dtype=pl.Int64)),
        pl.lit(pl.Series([1, 2])),
    ],
)
@pytest.mark.parametrize("method", ["top_k", "top_k_by"])
def test_top_k_invalid_k_raises(in_memory_engine, df, k_expr, method):
    col = pl.col("val")
    expr = col.top_k(k_expr) if method == "top_k" else col.top_k_by("test", k_expr)
    with pytest.raises(pl.exceptions.PolarsError):
        df.select(expr).collect(engine=in_memory_engine)
