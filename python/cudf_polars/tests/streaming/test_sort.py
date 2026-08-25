# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.engine.options import StreamingOptions
from cudf_polars.testing.asserts import assert_gpu_result_equal


@pytest.fixture
def engine(streaming_engine_factory):
    return streaming_engine_factory(
        StreamingOptions(
            max_rows_per_partition=3,
            fallback_mode="raise",
            raise_on_fail=True,
        ),
    )


@pytest.fixture
def engine_large(streaming_engine_factory):
    return streaming_engine_factory(
        StreamingOptions(
            max_rows_per_partition=2_100,
            fallback_mode="raise",
            raise_on_fail=True,
        ),
    )


@pytest.fixture(scope="module")
def df():
    return pl.LazyFrame(
        {
            "x": [1, 2, 3, 4, 5, 6, 7],
            "y": [1, 6, 7, 2, 5, 4, 3],
            "z": ["e", "c", "b", "g", "a", "f", "d"],
        }
    )


def large_frames():
    x = [1.0] * 10_000
    x[-1] = float("nan")
    y = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 1000

    yield pytest.param(
        pl.LazyFrame(
            {
                "x": x,
            }
        ),
        ["x"],
        False,
        id="all_equal_one_nan",
    )

    yield pytest.param(
        pl.LazyFrame(
            {
                "x": x,
                "y": y,
            }
        ),
        ["x", "y"],
        False,
        id="two_cols",
    )

    idx = list(range(10_000))
    yield pytest.param(
        pl.LazyFrame(
            {
                "x": x,
                "y": y,
                "idx": idx,
            }
        ),
        ["x", "y"],
        True,
        id="two_col_stable",
    )


def test_sort(df, engine):
    q = df.sort(by=["y", "z"])
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("large_df,by,stable", list(large_frames()))
@pytest.mark.parametrize(
    "nulls_last,descending", [(True, False), (True, True), (False, True)]
)
def test_large_sort(large_df, by, engine_large, stable, nulls_last, descending):
    q = large_df.sort(
        by=by, nulls_last=nulls_last, maintain_order=stable, descending=descending
    )
    assert_gpu_result_equal(q, engine=engine_large)


def test_sort_head(df, engine):
    q = df.sort(by=["y", "z"]).head(2)
    assert_gpu_result_equal(q, engine=engine)


def test_sort_tail(df, engine):
    q = df.sort(by=["y", "z"]).tail(2)
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("offset", [1, -4])
def test_sort_slice(df, engine, offset):
    # Slice in the middle, which distributed sorts need to be careful with
    q = df.sort(by=["y", "z"]).slice(offset, 2)
    with pytest.raises(
        NotImplementedError,
        match=r"This slice not supported for multiple partitions.",
    ):
        assert_gpu_result_equal(q, engine=engine)


def test_sort_after_sparse_join(streaming_engine_factory):
    engine = streaming_engine_factory(
        StreamingOptions(max_rows_per_partition=4, raise_on_fail=True),
    )
    left = pl.LazyFrame({"foo": list(range(5)), "bar": list(range(5))})
    right = pl.LazyFrame({"foo": list(range(1))})
    q = left.join(right, on="foo", how="inner").sort(by=["foo"])
    assert_gpu_result_equal(q, engine=engine)


def test_sort_by_renamed_join_column(streaming_engine_factory):
    engine = streaming_engine_factory(
        StreamingOptions(max_rows_per_partition=1, raise_on_fail=True),
    )
    df1 = pl.LazyFrame({"k1": [1, 2], "text": ["A", "B"]})
    df2 = pl.LazyFrame({"k2": [1, 2], "text": ["y", "x"]})
    ctx = pl.SQLContext()
    ctx.register("df1", df1)
    ctx.register("df2", df2)
    q = ctx.execute(
        "SELECT df1.text AS t1, df2.text AS t2 "
        "FROM df1 INNER JOIN df2 ON df1.k1 = df2.k2 "
        "ORDER BY df2.text"
    )
    assert_gpu_result_equal(q, engine=engine)
