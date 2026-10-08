# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

import polars as pl

import pylibcudf as plc

from cudf_polars.containers import DataType


def test_hash():
    dtype = pl.Int8()
    assert hash(dtype) == hash(DataType(dtype))


def test_eq():
    dtype = pl.Int8()
    data_type = DataType(dtype)

    assert data_type != dtype
    assert data_type == DataType(dtype)


def test_repr():
    data_type = DataType(pl.Int8())

    assert repr(data_type) == "<DataType(polars=Int8, plc=<type_id.INT8: 1>)>"


@pytest.mark.parametrize(
    "dtype, expected",
    [
        (
            pl.Struct({"a": pl.Int8, "b": pl.Int16}),
            [DataType(pl.Int8()), DataType(pl.Int16())],
        ),
        (
            pl.Struct({"a": pl.Struct({"b": pl.Int8()})}),
            [DataType(pl.Struct({"b": pl.Int8()}))],
        ),
        (pl.List(pl.Int8), [DataType(pl.Int8())]),
        (pl.Array(pl.Int8, 2), [DataType(pl.Int8())]),
        (pl.Int8(), []),
    ],
)
def test_children(dtype, expected):
    assert DataType(dtype).children == expected


def test_array_dtype_uses_physical_list():
    dtype = pl.Array(pl.Float32(), 3)
    result = DataType(dtype)

    assert result.polars_type == dtype
    assert result.id() == plc.TypeId.LIST


@pytest.mark.parametrize(
    "dtype, expected",
    [
        (pl.Categorical(), plc.TypeId.UINT32),
        (pl.Categorical("fruit"), plc.TypeId.UINT32),
        (pl.Categorical(pl.Categories("x", physical=pl.UInt8)), plc.TypeId.UINT8),
        (pl.Categorical(pl.Categories("y", "ns", pl.UInt16)), plc.TypeId.UINT16),
        (pl.Enum([]), plc.TypeId.UINT8),
        (pl.Enum([str(i) for i in range(3)]), plc.TypeId.UINT8),
    ],
    ids=[
        "global",
        "named",
        "uint8",
        "uint16",
        "enum0",
        "enum3",
    ],
)
def test_categorical_dtype_uses_physical_codes(dtype, expected):
    result = DataType(dtype)

    assert result.polars_type == dtype
    assert result.id() == expected
    assert result.is_categorical
    assert result.children == []


@pytest.mark.parametrize(
    "dtype",
    [
        pl.List(pl.Categorical()),
        pl.List(pl.List(pl.Enum(["x"]))),
        pl.Struct({"a": pl.Enum(["x"])}),
        pl.Array(pl.Enum(["x"]), 2),
    ],
    ids=repr,
)
def test_nested_categorical_raises(dtype):
    with pytest.raises(NotImplementedError, match="Categorical nested"):
        DataType(dtype)


def test_categorical_dtype_keeps_mapping_alive():
    dtype = DataType(pl.Categorical(pl.Categories("keepalive")))

    assert dtype._categorical_reference.dtype == dtype.polars_type
