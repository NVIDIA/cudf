# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

import polars as pl

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
            pl.Struct({"a": pl.Int8(), "b": pl.Int16()}),
            [DataType(pl.Int8()), DataType(pl.Int16())],
        ),
        (
            pl.Struct({"a": pl.Struct({"b": pl.Int8()})}),
            [DataType(pl.Struct({"b": pl.Int8()}))],
        ),
        (pl.List(pl.Int8()), [DataType(pl.Int8())]),
        (pl.Int8(), []),
    ],
)
def test_children(dtype, expected):
    assert DataType(dtype).children == expected


def test_common_decimal_type_raises():
    with pytest.raises(ValueError, match=r"Both inputs required to be decimal types."):
        DataType.common_decimal_dtype(
            DataType(pl.Float64()),
            DataType(pl.Float64()),
        )
