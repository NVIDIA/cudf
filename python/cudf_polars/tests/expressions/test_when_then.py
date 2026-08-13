# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import polars as pl
from polars.exceptions import ShapeError

from cudf_polars.testing.asserts import assert_gpu_result_equal


def test_when_then_mismatched_branch_lengths_raises(
    engine_raise_on_fail: pl.GPUEngine,
) -> None:
    df = pl.LazyFrame({"x": range(5)})
    q = df.select(pl.when(True).then(pl.col("x").head(2)).otherwise(pl.col("x")))  # noqa: FBT003
    with pytest.raises(ShapeError):
        q.collect(engine=engine_raise_on_fail)


@pytest.mark.parametrize(
    "when",
    [
        pl.lit(value=True, dtype=pl.Boolean()),
        pl.lit(value=False, dtype=pl.Boolean()),
        pl.lit(None, dtype=pl.Boolean()),
        pl.col("c"),
    ],
)
@pytest.mark.parametrize("then", [pl.lit(10), pl.col("a")])
@pytest.mark.parametrize("otherwise", [pl.lit(-2), pl.col("b")])
def test_when_then(engine: pl.GPUEngine, when, then, otherwise):
    df = pl.LazyFrame(
        {
            "a": [1, 2, 3, 4, 5, 6, 7],
            "b": [10, 13, 11, 15, 16, 11, 10],
            "c": [None, True, False, False, True, True, False],
        }
    )

    q = df.select(pl.when(when).then(then).otherwise(otherwise))
    assert_gpu_result_equal(q, engine=engine)
