# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from decimal import Decimal

import pytest

import polars as pl

from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)
from cudf_polars.utils.versions import POLARS_VERSION_LT_138


@pytest.mark.parametrize(
    "subset",
    [
        None,
        ["a", "c"],
        ["b", "c", "d"],
        ["b", "d"],
        ["b", "c"],
        ["c", "e"],
        ["d", "e"],
        pl.selectors.string(),
        pl.selectors.integer(),
    ],
)
@pytest.mark.parametrize("predicate_pushdown", [False, True])
def test_scan_drop_nulls(engine: pl.GPUEngine, subset, predicate_pushdown):
    df = pl.LazyFrame(
        {
            "a": [1, 2, 3, 4],
            "b": [None, 4, 5, None],
            "c": [6, 7, None, None],
            "d": [8, None, 9, 10],
            "e": [None, None, "A", None],
        }
    )
    # Drop nulls are pushed into filters
    q = df.drop_nulls(subset)

    assert_gpu_result_equal(
        q,
        engine=engine,
        collect_kwargs={
            "optimizations": pl.QueryOptFlags(predicate_pushdown=predicate_pushdown)
        },
    )


def test_can_convert_lists(engine: pl.GPUEngine):
    df = pl.LazyFrame(
        {
            "a": pl.Series([[1, 2], [3]], dtype=pl.List(pl.Int8())),
            "b": pl.Series([[1], [2]], dtype=pl.List(pl.UInt16())),
            "c": pl.Series(
                [
                    [["1", "2", "3"], ["4", "567"]],
                    [["8", "9"], []],
                ],
                dtype=pl.List(pl.List(pl.String())),
            ),
            "d": pl.Series([[[1, 2]], []], dtype=pl.List(pl.List(pl.UInt16()))),
        }
    )

    assert_gpu_result_equal(df, engine=engine)


def test_dataframescan_with_decimals(engine: pl.GPUEngine):
    q = pl.LazyFrame(
        {
            "foo": [1, 2],
            "bar": [Decimal("1.23"), Decimal("4.56")],
        },
        schema={"foo": pl.Int64, "bar": pl.Decimal(precision=15, scale=2)},
    )
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.skipif(
    POLARS_VERSION_LT_138,
    reason="height parameter added in Polars 1.38",
)
def test_dataframescan_zero_width_with_rows(engine: pl.GPUEngine):
    df = pl.LazyFrame(height=5)
    q = df.select(pl.len())
    assert_gpu_result_equal(q, engine=engine)


def test_struct_literal_not_supported(engine: pl.GPUEngine):
    dtype = pl.Struct([pl.Field("a", pl.Int64), pl.Field("b", pl.String)])
    q = pl.LazyFrame().select(pl.lit(None, dtype=pl.Null).cast(dtype, strict=True))
    assert_ir_translation_raises(q, engine, NotImplementedError)
