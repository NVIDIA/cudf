# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING

import pytest

import polars as pl

import pylibcudf as plc

import cudf_polars.containers.column
import cudf_polars.containers.datatype
from cudf_polars.containers import Column, DataType
from cudf_polars.utils.cuda_stream import get_cuda_stream

if TYPE_CHECKING:
    from cudf_polars.typing import PolarsDataType


def _as_instance(dtype: PolarsDataType) -> pl.DataType:
    if isinstance(dtype, type):
        return dtype()
    if isinstance(dtype, pl.List):
        inner = _as_instance(dtype.inner)
        return pl.List(inner)
    if isinstance(dtype, pl.Struct):
        return pl.Struct(
            [pl.Field(f.name, _as_instance(f.dtype)) for f in dtype.fields]
        )
    return dtype


def test_non_scalar_access_raises():
    dtype = DataType(pl.Int8())
    column = Column(
        plc.column_factories.make_numeric_column(
            dtype.plc_type, 2, plc.MaskState.ALL_VALID
        ),
        dtype=dtype,
    )
    with pytest.raises(ValueError):
        _ = column.obj_scalar(stream=get_cuda_stream())


def test_obj_scalar_caching():
    stream = get_cuda_stream()
    dtype = DataType(pl.Int8())
    column = Column(
        plc.Column.from_iterable_of_py([1], dtype.plc_type),
        dtype=dtype,
    )
    assert column.obj_scalar(stream=stream).to_py(stream=stream) == 1
    # test caching behavior
    assert column.obj_scalar(stream=stream).to_py(stream=stream) == 1


def test_check_sorted():
    stream = get_cuda_stream()
    dtype = DataType(pl.Int8())
    column = Column(
        plc.Column.from_iterable_of_py([0, 1, 2], dtype.plc_type),
        dtype=dtype,
    )
    assert column.check_sorted(
        order=plc.types.Order.ASCENDING,
        null_order=plc.types.NullOrder.AFTER,
        stream=stream,
    )
    column.set_sorted(
        is_sorted=plc.types.Sorted.YES,
        order=plc.types.Order.ASCENDING,
        null_order=plc.types.NullOrder.AFTER,
    )
    assert column.check_sorted(
        order=plc.types.Order.ASCENDING,
        null_order=plc.types.NullOrder.AFTER,
        stream=stream,
    )


@pytest.mark.parametrize("length", [0, 1])
def test_length_leq_one_always_sorted(length):
    stream = get_cuda_stream()
    dtype = DataType(pl.Int8())
    column = Column(
        plc.column_factories.make_numeric_column(
            dtype.plc_type, length, plc.MaskState.ALL_VALID
        ),
        dtype=dtype,
    )
    assert column.check_sorted(
        order=plc.types.Order.ASCENDING,
        null_order=plc.types.NullOrder.AFTER,
        stream=stream,
    )
    assert column.check_sorted(
        order=plc.types.Order.DESCENDING,
        null_order=plc.types.NullOrder.AFTER,
        stream=stream,
    )

    column.set_sorted(
        is_sorted=plc.types.Sorted.NO,
        order=plc.types.Order.ASCENDING,
        null_order=plc.types.NullOrder.AFTER,
    )
    assert column.check_sorted(
        order=plc.types.Order.ASCENDING,
        null_order=plc.types.NullOrder.AFTER,
        stream=stream,
    )
    assert column.check_sorted(
        order=plc.types.Order.DESCENDING,
        null_order=plc.types.NullOrder.AFTER,
        stream=stream,
    )


def test_shallow_copy():
    dtype = DataType(pl.Int8())
    column = Column(
        plc.column_factories.make_numeric_column(
            dtype.plc_type, 2, plc.MaskState.ALL_VALID
        ),
        dtype=dtype,
    )
    copy = column.copy()
    copy = copy.set_sorted(
        is_sorted=plc.types.Sorted.YES,
        order=plc.types.Order.ASCENDING,
        null_order=plc.types.NullOrder.AFTER,
    )
    assert column.is_sorted == plc.types.Sorted.NO
    assert copy.is_sorted == plc.types.Sorted.YES


@pytest.mark.parametrize("typeid", [pl.Int8(), pl.Float32()])
def test_mask_nans(typeid):
    stream = get_cuda_stream()
    dtype = DataType(typeid)
    column = Column(
        plc.Column.from_iterable_of_py([0, 0, 0], dtype=dtype.plc_type), dtype=dtype
    )
    masked = column.mask_nans(stream=stream)
    assert column.null_count == masked.null_count


def test_mask_nans_float():
    stream = get_cuda_stream()
    dtype = DataType(pl.Float32())
    column = Column(
        plc.Column.from_iterable_of_py([0, 0, float("nan")], dtype=dtype.plc_type),
        dtype=dtype,
    )
    masked = column.mask_nans(stream=stream)
    assert masked.nan_count(stream=stream) == 0
    # test caching behavior
    assert masked.nan_count(stream=stream) == 0
    assert masked.slice((0, 2), stream=stream).null_count == 0
    assert masked.slice((2, 1), stream=stream).null_count == 1


def test_slice_none_returns_self():
    stream = get_cuda_stream()
    dtype = DataType(pl.Int8())
    column = Column(
        plc.column_factories.make_numeric_column(
            dtype.plc_type, 2, plc.MaskState.ALL_VALID
        ),
        dtype=dtype,
    )
    assert column.slice(None, stream=stream) is column


def test_deserialize_ctor_kwargs_invalid_dtype_and_kind():
    column_kwargs = {
        "is_sorted": plc.types.Sorted.NO,
        "order": plc.types.Order.ASCENDING,
        "null_order": plc.types.NullOrder.AFTER,
        "name": "test",
        "dtype": {"kind": "scalar", "name": "foo"},
    }
    with pytest.raises(NotImplementedError, match="Unknown scalar dtype name"):
        Column.deserialize_ctor_kwargs(column_kwargs)

    column_kwargs["dtype"]["kind"] = "bar"

    with pytest.raises(NotImplementedError, match="Unsupported kind"):
        Column.deserialize_ctor_kwargs(column_kwargs)


def test_deserialize_ctor_kwargs_list_dtype():
    pl_type = pl.List(pl.Int64())
    column_kwargs = {
        "is_sorted": plc.types.Sorted.NO,
        "order": plc.types.Order.ASCENDING,
        "null_order": plc.types.NullOrder.AFTER,
        "name": "test",
        "dtype": cudf_polars.containers.datatype._dtype_to_header(pl_type),
    }
    result = Column.deserialize_ctor_kwargs(column_kwargs)
    expected = {
        "is_sorted": plc.types.Sorted.NO,
        "order": plc.types.Order.ASCENDING,
        "null_order": plc.types.NullOrder.AFTER,
        "name": "test",
        "dtype": DataType(pl_type),
    }
    assert result == expected


def test_serialize_cache_miss():
    stream = get_cuda_stream()
    dtype = DataType(pl.Int8())
    column = Column(
        plc.column_factories.make_numeric_column(
            dtype.plc_type, 2, plc.MaskState.ALL_VALID
        ),
        dtype=dtype,
    )
    header, frames = column.serialize(stream=stream)
    assert header == {"column_kwargs": column.serialize_ctor_kwargs(), "frame_count": 2}
    assert len(frames) == 2
    assert frames[0].nbytes > 0
    assert frames[1].nbytes > 0

    # https://github.com/rapidsai/cudf/pull/18953
    # In a multi-GPU setup, we might attempt to deserialize a column
    # whose type we haven't seen before. polars lets you use either the
    # class (`pl.Int8`) or an instance (`pl.Int8()`) in most places
    # for types that are not parameterized. These are equal and have
    # the same hash, so they cache the same, but have some difference
    # in behavior (e.g. isinstance).
    cudf_polars.containers.datatype._from_polars.cache_clear()
    result = Column.deserialize(header, frames, stream=stream)
    assert result.dtype == dtype


# datetimes return instances of DataType, rather than DataTypeClass


@pytest.mark.parametrize(
    "dtype",
    [
        pl.Boolean,
        pl.Boolean(),
        pl.Date,
        pl.Date(),
        pl.Datetime,
        pl.Datetime(),
        pl.Duration,
        pl.Duration(),
        pl.Float32,
        pl.Float32(),
        pl.Int8,
        pl.Int8(),
        pl.List(pl.Int8()),
        pl.List(pl.Int8),
        pl.List(pl.Decimal(10)),
        pl.Object,
        pl.Object(),
        pl.String,
        pl.String(),
        pl.Time,
        pl.Time(),
        pl.UInt8,
        pl.UInt8(),
        pl.Struct([pl.Field("a", pl.Int8), pl.Field("b", pl.Int8)]),
        # These fail.
        pytest.param(
            pl.Binary,
            marks=pytest.mark.xfail(reason="Binary is not supported", strict=True),
        ),
        pytest.param(
            pl.Binary(),
            marks=pytest.mark.xfail(reason="Binary is not supported", strict=True),
        ),
        # These Error
        pytest.param(
            pl.Enum(["a", "b"]),
            marks=pytest.mark.xfail(reason="Enum is not supported", strict=True),
        ),
        pytest.param(
            pl.Array(pl.Int8, shape=(1,)),
            marks=pytest.mark.xfail(reason="Array[Int8] is not supported", strict=True),
        ),
    ],
)
def test_dtype_header_roundtrip(dtype: pl.DataType):
    dt = _as_instance(dtype)
    header = cudf_polars.containers.datatype._dtype_to_header(dt)
    result = cudf_polars.containers.datatype._dtype_from_header(header)
    assert result == dt


@pytest.mark.parametrize(
    "val, plc_tid, pl_type",
    [
        (1, plc.TypeId.INT64, pl.Int64()),
        (True, plc.TypeId.BOOL8, pl.Boolean()),
    ],
)
def test_astype_to_string(val, plc_tid, pl_type):
    stream = get_cuda_stream()
    col = Column(
        plc.Column.from_iterable_of_py([val], plc.DataType(plc_tid), stream=stream),
        dtype=DataType(pl_type),
    )
    target_dtype = DataType(pl.String())
    result = col.astype(target_dtype, stream=stream)
    assert result.dtype == target_dtype


def test_astype_duration_to_narrower_integer():
    stream = get_cuda_stream()
    col = Column(
        plc.unary.cast(
            plc.Column.from_iterable_of_py(
                [1, 2, -3], plc.DataType(plc.TypeId.INT64), stream=stream
            ),
            plc.DataType(plc.TypeId.DURATION_MICROSECONDS),
            stream=stream,
        ),
        dtype=DataType(pl.Duration(time_unit="us")),
    )
    target_dtype = DataType(pl.Int32())
    result = col.astype(target_dtype, stream=stream)
    assert result.dtype == target_dtype
    assert result.obj.type().id() == plc.TypeId.INT32


def test_astype_from_string_unsupported():
    stream = get_cuda_stream()
    col = Column(
        plc.Column.from_iterable_of_py(
            ["True"], plc.DataType(plc.TypeId.STRING), stream=stream
        ),
        dtype=DataType(pl.String()),
    )
    with pytest.raises(pl.exceptions.InvalidOperationError):
        col.astype(DataType(pl.Boolean()), stream=stream)


def test_astype_to_string_unsupported():
    stream = get_cuda_stream()
    col = Column(
        plc.Column.from_scalar(
            plc.Scalar.from_py(datetime.datetime(2020, 1, 1), stream=stream),
            1,
            stream=stream,
        ),
        dtype=DataType(pl.Datetime(time_unit="ns")),
    )
    with pytest.raises(pl.exceptions.InvalidOperationError):
        col.astype(DataType(pl.String()), stream=stream)
