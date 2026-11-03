# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import polars as pl

from cudf_polars.testing.asserts import assert_gpu_result_equal


def test_hstack(engine: pl.GPUEngine):
    ldf = pl.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, 6, 7],
            "b": [1, 1, 1, 1, 1, 1, 1],
        }
    ).lazy()

    query = ldf.with_columns(pl.col("a") + pl.col("b"))
    assert_gpu_result_equal(query, engine=engine)


def test_hstack_with_cse(engine: pl.GPUEngine):
    ldf = pl.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, 6, 7],
            "b": [1, 1, 1, 1, 1, 1, 1],
        }
    ).lazy()

    expr = pl.col("a") + pl.col("b")
    query = ldf.with_columns(expr.alias("c"), expr.alias("d") * 2)
    assert_gpu_result_equal(query, engine=engine)
