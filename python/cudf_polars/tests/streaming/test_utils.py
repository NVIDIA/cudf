# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import polars as pl

import pylibcudf as plc

from cudf_polars.containers import DataFrame, DataType
from cudf_polars.dsl import expr
from cudf_polars.dsl.ir import IRExecutionContext
from cudf_polars.streaming.utils import _concat, _leaf_column_names
from cudf_polars.utils.cuda_stream import get_cuda_stream


def test_leaf_column_names():
    dt = DataType(pl.datatypes.Int32())
    a = expr.Col(dt, "a")
    b = expr.Literal(dt, 1)
    c = expr.Col(dt, "c")
    d = expr.BinOp(dt, plc.binaryop.BinaryOperator.ADD, a, b)
    e = expr.BinOp(dt, plc.binaryop.BinaryOperator.ADD, d, c)
    assert set(_leaf_column_names(e)) == {"a", "c"}


def test_concat_single_preserves_sortedness():
    stream = get_cuda_stream()
    expected = pl.DataFrame({"a": [1, 2]}).with_columns(pl.col("a").set_sorted())
    df = DataFrame.from_polars(expected, stream=stream)

    result = _concat(df, context=IRExecutionContext())

    assert result is df
    assert result.to_polars().flags == expected.flags
