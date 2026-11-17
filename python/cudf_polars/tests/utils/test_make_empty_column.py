# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Tests for make_empty_column, including nested types."""

from __future__ import annotations

import pytest

import polars as pl

import pylibcudf as plc
from rmm.pylibrmm.stream import DEFAULT_STREAM

from cudf_polars.containers import DataType
from cudf_polars.utils.dtypes import make_empty_column


@pytest.mark.parametrize(
    "polars_dtype",
    [
        pl.Int32(),
        pl.Int64(),
        pl.Float64(),
        pl.String(),
        pl.Boolean(),
    ],
)
def test_flat_types(polars_dtype):
    dtype = DataType(polars_dtype)
    col = make_empty_column(dtype, DEFAULT_STREAM)
    assert col.size() == 0
    assert col.null_count() == 0
    assert col.type() == dtype.plc_type


@pytest.mark.parametrize(
    "inner",
    [pl.Int32(), pl.Int64(), pl.Float64(), pl.String()],
)
def test_list_type(inner):
    dtype = DataType(pl.List(inner))
    col = make_empty_column(dtype, DEFAULT_STREAM)

    assert col.size() == 0
    assert col.null_count() == 0
    assert col.type().id() == plc.TypeId.LIST

    offsets = col.child(0)
    assert offsets.size() == 1
    assert offsets.type().id() == plc.TypeId.INT32

    child = col.child(1)
    assert child.size() == 0
    assert child.type() == DataType(inner).plc_type


def test_nested_list_type():
    dtype = DataType(pl.List(pl.List(pl.Int32())))
    col = make_empty_column(dtype, DEFAULT_STREAM)

    assert col.size() == 0
    assert col.type().id() == plc.TypeId.LIST

    inner_list = col.child(1)
    assert inner_list.size() == 0
    assert inner_list.type().id() == plc.TypeId.LIST

    leaf = inner_list.child(1)
    assert leaf.size() == 0
    assert leaf.type().id() == plc.TypeId.INT32


def test_struct_type():
    dtype = DataType(pl.Struct({"a": pl.Int64, "b": pl.String}))
    col = make_empty_column(dtype, DEFAULT_STREAM)

    assert col.size() == 0
    assert col.null_count() == 0
    assert col.type().id() == plc.TypeId.STRUCT
    assert col.num_children() == 2

    assert col.child(0).type().id() == plc.TypeId.INT64
    assert col.child(0).size() == 0
    assert col.child(1).type().id() == plc.TypeId.STRING
    assert col.child(1).size() == 0


def test_struct_with_list_field():
    dtype = DataType(pl.Struct({"x": pl.Int64, "y": pl.List(pl.String)}))
    col = make_empty_column(dtype, DEFAULT_STREAM)

    assert col.size() == 0
    assert col.type().id() == plc.TypeId.STRUCT
    assert col.num_children() == 2

    assert col.child(0).type().id() == plc.TypeId.INT64

    list_child = col.child(1)
    assert list_child.type().id() == plc.TypeId.LIST
    assert list_child.size() == 0
    assert list_child.child(1).type().id() == plc.TypeId.STRING


def test_list_of_struct():
    dtype = DataType(pl.List(pl.Struct({"a": pl.Int32, "b": pl.Float64})))
    col = make_empty_column(dtype, DEFAULT_STREAM)

    assert col.size() == 0
    assert col.type().id() == plc.TypeId.LIST

    struct_child = col.child(1)
    assert struct_child.type().id() == plc.TypeId.STRUCT
    assert struct_child.num_children() == 2
    assert struct_child.child(0).type().id() == plc.TypeId.INT32
    assert struct_child.child(1).type().id() == plc.TypeId.FLOAT64
