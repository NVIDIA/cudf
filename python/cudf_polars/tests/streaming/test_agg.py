# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from decimal import Decimal

import pytest

import polars as pl

from cudf_polars.testing.asserts import assert_gpu_result_equal


@pytest.fixture
def decimal_df() -> pl.LazyFrame:
    return pl.LazyFrame(
        {
            "a": pl.Series(
                "a",
                [Decimal("0.10"), Decimal("1.10"), Decimal("100.10")],
                dtype=pl.Decimal(precision=9, scale=2),
            ),
        }
    )


def test_decimal_aggs(
    decimal_df: pl.LazyFrame, streaming_engine, xfail_decimal_sum_precision_polars_140
) -> None:
    q = decimal_df.with_columns(
        sum=pl.col("a").sum(),
        min=pl.col("a").min(),
        max=pl.col("a").max(),
        mean=pl.col("a").mean(),
        median=pl.col("a").median(),
    )
    assert_gpu_result_equal(q, engine=streaming_engine)


def test_mean_all_null(streaming_engine):
    lf = pl.LazyFrame({"a": [None, None]}, schema={"a": pl.Float64})
    q = lf.select(pl.col("a").mean())
    assert_gpu_result_equal(q, engine=streaming_engine)
