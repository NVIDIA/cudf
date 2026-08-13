# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import assert_gpu_result_equal
from cudf_polars.testing.engine_utils import is_streaming_engine


@pytest.mark.parametrize(
    "series, value",
    [
        (pl.Series([1, 2, 3]), 0),
        (pl.Series([1, 2, None]), None),
        (pl.Series([1.5, float("nan"), -3.0]), 2.5),
        (pl.Series([1.5, 2.5, -3.0]), None),
        (pl.Series(["a", "bb", None]), "zz"),
        (pl.Series(["a", "bb", "c"]), None),
        (pl.Series([], dtype=pl.Int64), 0),
        (pl.Series([None, None, None], dtype=pl.Int64), 7),
        (pl.Series([None, None, None], dtype=pl.Int64), None),
        (pl.Series([42], dtype=pl.Int64), 7),
        (pl.Series([None], dtype=pl.Int64), 7),
    ],
)
@pytest.mark.parametrize("n", [0, 1, 3])
def test_extend_constant(
    engine: pl.GPUEngine, series: pl.Series, value: int | float | str | None, n: int
) -> None:
    lf = pl.LazyFrame({"a": series})
    q = lf.select(pl.col("a").extend_constant(value, n))
    assert_gpu_result_equal(q, engine=engine)


def test_extend_constant_non_literal_value_and_n(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame({"a": [1, 2, 3], "v": [9, 9, 9], "n": [2, 2, 2]})
    q = lf.select(pl.col("a").extend_constant(pl.col("v").first(), pl.col("n").first()))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "n", [-1, pl.col("a").min().abs() * -1], ids=["literal", "expression"]
)
def test_extend_constant_negative_n(engine: pl.GPUEngine, n: int | pl.Expr) -> None:
    lf = pl.LazyFrame({"a": [1, 2, 3]})
    q = lf.select(pl.col("a").extend_constant(0, n))
    with pytest.raises(pl.exceptions.InvalidOperationError):
        q.collect()
    if is_streaming_engine(engine):
        with pytest.RaisesGroup(pl.exceptions.InvalidOperationError):
            q.collect(engine=engine)
    else:
        with pytest.raises(pl.exceptions.InvalidOperationError):
            q.collect(engine=engine)
