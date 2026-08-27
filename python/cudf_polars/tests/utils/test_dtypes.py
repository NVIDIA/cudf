# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

import polars as pl

import pylibcudf as plc

from cudf_polars.containers import DataType
from cudf_polars.utils.dtypes import is_order_preserving_cast

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


@pytest.mark.parametrize(
    "pltype",
    [
        pl.Time(),
        pl.Struct({"a": pl.Binary(), "b": pl.Float32}),
        pl.List(pl.Object()),
        pl.List(pl.Array(pl.Int8, 2)),
        pl.Array(pl.String, 2),
        pl.Array(pl.Array(pl.Int8, 2), 3),
        pl.Struct({"a": pl.Array(pl.Int8, 2)}),
        pl.Binary(),
        pl.Categorical(),
        pl.Enum(["a", "b"]),
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
