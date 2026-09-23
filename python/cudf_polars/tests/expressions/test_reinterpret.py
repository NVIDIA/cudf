# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)
from cudf_polars.utils.versions import POLARS_VERSION_LT_139


@pytest.mark.parametrize(
    "series",
    [
        pl.Series(
            [0, 1, -1, 9223372036854775807, -9223372036854775808], dtype=pl.Int64
        ),
        pl.Series([0, 1, None, -1], dtype=pl.Int64),
        pl.Series([0, 1, 18446744073709551615], dtype=pl.UInt64),
        pl.Series([0, 1, None, 18446744073709551615], dtype=pl.UInt64),
        pl.Series([], dtype=pl.Int64),
    ],
)
@pytest.mark.parametrize("signed", [True, False])
def test_reinterpret(engine: pl.GPUEngine, series: pl.Series, *, signed: bool) -> None:
    lf = pl.LazyFrame({"a": series})
    q = lf.select(pl.col("a").reinterpret(signed=signed))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.skipif(
    POLARS_VERSION_LT_139, reason="reinterpret dtype argument added in 1.39"
)
@pytest.mark.parametrize(
    "source_dtype, target_dtype",
    [
        (pl.UInt32, pl.Float32),
        (pl.Int32, pl.Float32),
        (pl.Float32, pl.Int32),
        (pl.Float64, pl.Int64),
    ],
)
def test_reinterpret_across_kinds_raises(
    engine: pl.GPUEngine,
    source_dtype: pl.DataType,
    target_dtype: pl.DataType,
) -> None:
    lf = pl.LazyFrame({"a": pl.Series([0, 1, 2]).cast(source_dtype)})
    q = lf.select(pl.col("a").reinterpret(dtype=target_dtype))
    assert_ir_translation_raises(q, engine, NotImplementedError)
