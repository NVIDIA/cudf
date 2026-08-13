# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import assert_gpu_result_equal


@pytest.fixture(scope="module")
def df():
    return pl.DataFrame(
        {
            "c": ["a", "b", "c", "d", "e", "f"],
            "a": [1, 2, 3, None, 4, 5],
            "b": pl.Series([None, None, 3, float("inf"), 4, 0], dtype=pl.Float64),
            "d": [-1, 2, -3, None, 4, -5],
        }
    )


@pytest.fixture(scope="module")
def pq_file(tmp_path_factory, df):
    tmp_path = tmp_path_factory.mktemp("parquet_filter")
    df.write_parquet(tmp_path / "tmp.pq", row_group_size=3)
    return pl.scan_parquet(tmp_path / "tmp.pq")


@pytest.mark.parametrize(
    "expr",
    [
        pl.col("a").is_in([0, 1]),
        pl.col("c").is_in(["a", "b"]),
        pl.col("a").is_between(0, 2),
        (pl.col("a") < 2).not_(),
        pl.lit(2) > pl.col("a"),
        pl.lit(2) >= pl.col("a"),
        pl.lit(2) < pl.col("a"),
        pl.lit(2) <= pl.col("a"),
        pl.lit(0) == pl.col("a"),
        pl.lit(1) != pl.col("a"),
        pl.col("a") == pl.col("d"),
        (pl.col("b") < pl.lit(2, dtype=pl.Float64).sqrt()),
        (pl.col("a") >= pl.lit(2)) & (pl.col("b") > 0),
        pl.col("b").is_finite(),
        pl.col("a").is_null(),
        pl.col("a").is_not_null(),
        pl.col("a").abs().is_between(0, 2),
        pl.col("a").ne_missing(pl.lit(None, dtype=pl.Int64)),
    ],
)
@pytest.mark.parametrize("selection", [["c", "b"], ["a"], ["a", "c"], ["b"], "c"])
@pytest.mark.parametrize("chunked", [False, True], ids=["unchunked", "chunked"])
def test_scan_by_hand(expr, selection, pq_file, chunked):
    q = pq_file.filter(expr).select(*selection)
    assert_gpu_result_equal(
        q,
        engine=pl.GPUEngine(
            executor="in-memory",
            raise_on_fail=True,
            parquet_options={"chunked": chunked},
        ),
    )


def test_parquet_filter_ne_missing(tmp_path):
    df = pl.DataFrame({"a": [1, 2, None, 3, None]})
    df.write_parquet(tmp_path / "df.parquet", row_group_size=3)
    q = pl.scan_parquet(tmp_path / "df.parquet").filter(
        pl.col("a").ne_missing(pl.lit(2, dtype=pl.Int64))
    )
    assert_gpu_result_equal(
        q, engine=pl.GPUEngine(executor="in-memory", raise_on_fail=True)
    )


def test_parquet_filter_boolean_column(engine: pl.GPUEngine, tmp_path):
    df = pl.DataFrame({"x": [1, 2, 3], "y": [True, False, True]})
    df.write_parquet(tmp_path / "df.parquet")
    q = pl.scan_parquet(tmp_path / "df.parquet").filter(pl.col("y"))
    assert_gpu_result_equal(q, engine=engine)


def test_jit_filter(pq_file):
    q = pq_file.filter((pl.col("a") >= 2) & (pl.col("a") <= 4)).select("a", "c")
    assert_gpu_result_equal(
        q,
        engine=pl.GPUEngine(
            executor="in-memory",
            raise_on_fail=True,
            parquet_options={"use_jit_filter": True},
        ),
    )
