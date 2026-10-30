# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

import polars as pl
from polars.testing import assert_frame_equal

import pylibcudf as plc

from cudf_polars.containers import Column, DataFrame, DataType
from cudf_polars.utils.cuda_stream import get_cuda_stream
from cudf_polars.utils.dtypes import (
    can_cast,
    is_order_preserving_cast,
    make_all_null_column,
    make_empty_column,
)

INT8 = plc.DataType(plc.TypeId.INT8)
INT16 = plc.DataType(plc.TypeId.INT16)
INT32 = plc.DataType(plc.TypeId.INT32)
INT64 = plc.DataType(plc.TypeId.INT64)
UINT8 = plc.DataType(plc.TypeId.UINT8)
UINT16 = plc.DataType(plc.TypeId.UINT16)
UINT32 = plc.DataType(plc.TypeId.UINT32)
UINT64 = plc.DataType(plc.TypeId.UINT64)
FLOAT32 = plc.DataType(plc.TypeId.FLOAT32)
FLOAT64 = plc.DataType(plc.TypeId.FLOAT64)
LIST = plc.DataType(plc.TypeId.LIST)
STRUCT = plc.DataType(plc.TypeId.STRUCT)


@pytest.mark.parametrize(
    "pltype",
    [
        pl.Struct({"a": pl.Binary(), "b": pl.Float32}),
        pl.List(pl.Object()),
        pl.List(pl.Array(pl.Int8, 2)),
        pl.Array(pl.String, 2),
        pl.Array(pl.Array(pl.Int8, 2), 3),
        pl.Struct({"a": pl.Array(pl.Int8, 2)}),
        pl.Binary(),
        pl.List(pl.Categorical()),
        pl.Struct({"a": pl.Enum(["a"])}),
        pl.Field("a", pl.Int8),
        pl.Object(),
        pl.Unknown(),
    ],
    ids=repr,
)
def test_unhandled_dtype_conversion_raises(pltype):
    with pytest.raises(NotImplementedError):
        _ = DataType(pltype)


def test_is_order_preserving_cast():
    assert is_order_preserving_cast(INT8, INT8)  # Same type
    assert is_order_preserving_cast(INT8, INT16)  # Smaller type
    assert is_order_preserving_cast(INT8, FLOAT32)  # Int to large enough float
    assert is_order_preserving_cast(UINT8, UINT16)  # Unsigned to larger unsigned
    assert is_order_preserving_cast(UINT8, FLOAT32)  # Unsigned to large enough float
    assert is_order_preserving_cast(FLOAT32, FLOAT64)  # Float to larger float
    assert is_order_preserving_cast(INT64, FLOAT32)  # Int any float
    assert is_order_preserving_cast(FLOAT32, INT32)  # Float to undersized int
    assert is_order_preserving_cast(FLOAT32, INT64)  # float to large int

    assert not is_order_preserving_cast(INT16, INT8)  # Bigger type
    assert not is_order_preserving_cast(INT8, UINT8)  # Different signedness
    assert not is_order_preserving_cast(FLOAT64, FLOAT32)  # Smaller float


def test_can_cast_nested():
    assert not can_cast(LIST, LIST)
    assert not can_cast(LIST, STRUCT)
    assert can_cast(INT8, INT16)


@pytest.mark.parametrize(
    "dtype",
    [pl.Enum(["a", "b"]), pl.Categorical()],
    ids=["enum", "categorical"],
)
def test_make_empty_column_dictionary(dtype):
    result = make_empty_column(DataType(dtype), get_cuda_stream())

    assert result.type().id() == plc.TypeId.DICTIONARY32
    assert result.size() == 0
    assert [child.type().id() for child in result.children()] == [
        plc.TypeId.INT32,
        plc.TypeId.STRING,
    ]


@pytest.mark.parametrize(
    "dtype",
    [
        pl.Int8(),
        pl.String(),
        pl.List(pl.Int8()),
        pl.Struct({"a": pl.Int8()}),
        pl.List(pl.Struct({"a": pl.Int8()})),
        pl.Struct({"a": pl.List(pl.Int8()), "b": pl.Struct({"c": pl.String()})}),
    ],
    ids=repr,
)
def test_make_all_null_column(dtype):
    stream = get_cuda_stream()
    dt = DataType(dtype)

    result = make_all_null_column(dt, 3, stream)

    assert_frame_equal(
        DataFrame([Column(result, name="a", dtype=dt)], stream=stream).to_polars(),
        pl.DataFrame({"a": pl.Series([None] * 3, dtype=dtype)}),
        check_dtypes=True,
    )


@pytest.mark.parametrize(
    "dtype",
    [pl.Enum(["a", "b"]), pl.Categorical()],
    ids=["enum", "categorical"],
)
def test_make_all_null_column_dictionary(dtype):
    result = make_all_null_column(DataType(dtype), 3, get_cuda_stream())

    assert result.type().id() == plc.TypeId.DICTIONARY32
    assert result.size() == 3
    assert result.null_count() == 3
    assert [child.type().id() for child in result.children()] == [
        plc.TypeId.INT32,
        plc.TypeId.STRING,
    ]
    assert result.children()[1].size() == 0
