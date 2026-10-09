# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.containers import DataType
from cudf_polars.dsl.ir import DataFrameScan, Empty, HConcat, IRExecutionContext
from cudf_polars.testing.asserts import assert_gpu_result_equal
from cudf_polars.utils.versions import POLARS_VERSION_LT_139


def test_hconcat(engine: pl.GPUEngine):
    ldf = pl.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, 6, 7],
            "b": [1, 1, 1, 1, 1, 1, 1],
        }
    ).lazy()
    ldf2 = ldf.select((pl.col("a") + pl.col("b")).alias("c"))
    query = pl.concat([ldf, ldf2], how="horizontal")
    assert_gpu_result_equal(query, engine=engine)


def test_hconcat_different_heights(engine: pl.GPUEngine):
    left = pl.LazyFrame({"a": [1, 2, 3, 4]})

    right = pl.LazyFrame({"b": [[1], [2]], "c": ["a", "bcde"]})

    q = pl.concat([left, right], how="horizontal")
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.skipif(POLARS_VERSION_LT_139, reason="strict keyword added in polars 1.39")
def test_hconcat_strict_different_heights():
    left = pl.LazyFrame({"a": [1, 2, 3]})
    right = pl.LazyFrame({"b": [4, 5]})
    q = pl.concat([left, right], how="horizontal", strict=True)
    with pytest.raises(pl.exceptions.ShapeError):
        q.collect()
    with pytest.raises(pl.exceptions.ShapeError):
        q.collect(engine=pl.GPUEngine(executor="in-memory", raise_on_fail=True))


def test_hconcat_should_broadcast():
    # HConcat with should_broadcast=True is used by the streaming engine to
    # recombine decomposed expressions. Test it by constructing IR directly.
    context = IRExecutionContext()

    df1 = pl.DataFrame({"a": [1, 2, 3]})
    df2 = pl.DataFrame({"b": [4.0]})  # 1 row to be broadcast to 3

    child1 = DataFrameScan({"a": DataType(pl.Int64())}, df1._df, None)
    child2 = DataFrameScan({"b": DataType(pl.Float64())}, df2._df, None)

    schema = {"a": DataType(pl.Int64()), "b": DataType(pl.Float64())}
    node = HConcat(schema, True, False, False, child1, child2)  # noqa: FBT003
    result = node.evaluate(cache={}, timer=None, context=context)

    polars_result = result.to_polars()
    assert polars_result.shape == (3, 2)
    assert polars_result["b"].to_list() == [4.0, 4.0, 4.0]


def test_empty_init():
    # Empty is an IR node used by the streaming engine to represent an empty
    # DataFrame with a known schema. Test direct construction.
    schema = {"a": DataType(pl.Int64()), "b": DataType(pl.String())}
    node = Empty(schema)
    assert node.schema is schema
    assert node.children == ()
    assert node._non_child_args == (schema,)
