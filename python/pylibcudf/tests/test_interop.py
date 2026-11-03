# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import decimal

import cupy as cp
import nanoarrow
import nanoarrow.device
import numpy as np
import pyarrow as pa
import pytest
from utils import assert_column_eq, assert_table_eq

import rmm

import pylibcudf as plc


def test_list_dtype_roundtrip():
    list_type = pa.list_(pa.int32())
    plc_type = plc.DataType.from_arrow(list_type)

    assert plc_type == plc.types.DataType(plc.types.TypeId.LIST)

    with pytest.raises(ValueError):
        plc_type.to_arrow()

    arrow_type = plc_type.to_arrow(value_type=list_type.value_type)
    assert arrow_type == list_type


def test_struct_dtype_roundtrip():
    struct_type = pa.struct([("a", pa.int32()), ("b", pa.string())])
    plc_type = plc.DataType.from_arrow(struct_type)

    assert plc_type == plc.types.DataType(plc.types.TypeId.STRUCT)

    with pytest.raises(ValueError):
        plc_type.to_arrow()

    arrow_type = plc_type.to_arrow(
        fields=[struct_type.field(i) for i in range(struct_type.num_fields)],
    )
    assert arrow_type == struct_type


def test_table_with_nested_dtype_to_arrow():
    result = plc.Table(
        [plc.Column.from_arrow(pa.array([[{"": 1}]]))]
    ).to_arrow()
    expected_schema = pa.schema(
        [
            pa.field(
                "",
                pa.list_(
                    pa.field(
                        "",
                        pa.struct([pa.field("", pa.int64(), nullable=False)]),
                        nullable=False,
                    )
                ),
                nullable=False,
            )
        ]
    )
    assert result.schema == expected_schema


def test_decimal128_roundtrip():
    decimal_type = pa.decimal128(10, 2)
    plc_type = plc.DataType.from_arrow(decimal_type)

    assert plc_type.id() == plc.types.TypeId.DECIMAL128

    with pytest.raises(ValueError):
        plc_type.to_arrow()

    arrow_type = plc_type.to_arrow(precision=decimal_type.precision)
    assert arrow_type == decimal_type


@pytest.mark.parametrize(
    "data_type",
    [
        plc.types.DataType(plc.types.TypeId.DECIMAL32),
        plc.types.DataType(plc.types.TypeId.DECIMAL64),
    ],
)
def test_decimal_other(data_type):
    precision = 3

    with pytest.raises(ValueError):
        data_type.to_arrow()

    arrow_type = data_type.to_arrow(precision=precision)
    assert arrow_type == pa.decimal128(precision, 0)


@pytest.mark.parametrize(
    "plc_type",
    [plc.TypeId.DECIMAL128, plc.TypeId.DECIMAL64, plc.TypeId.DECIMAL32],
)
def test_decimal_respect_metadata_precision(plc_type):
    precision, scale = 3, 2
    pa_type = {
        plc.TypeId.DECIMAL128: pa.decimal128,
        plc.TypeId.DECIMAL64: pa.decimal64,
        plc.TypeId.DECIMAL32: pa.decimal32,
    }[plc_type]
    expected = pa.array(
        [decimal.Decimal("1.23"), None], type=pa_type(precision, scale)
    )
    plc_column = plc.unary.cast(
        plc.Column.from_arrow(expected), plc.DataType(plc_type, scale=-scale)
    )
    result = plc_column.to_arrow(
        metadata=plc.interop.ColumnMetadata(precision=precision)
    )
    assert result.equals(expected)


@pytest.mark.parametrize("precision", [0, 39])
def test_decimal_precision_metadata_out_of_range(precision):
    scale = 2
    expected = pa.array(
        [decimal.Decimal("1.23"), None], type=pa.decimal128(3, scale)
    )
    plc_column = plc.unary.cast(
        plc.Column.from_arrow(expected),
        plc.DataType(plc.TypeId.DECIMAL128, scale=-scale),
    )
    with pytest.raises(TypeError):
        plc_column.to_arrow(
            metadata=plc.interop.ColumnMetadata(precision=precision),
        )


def test_round_trip_dlpack_plc_table():
    expected = pa.table({"a": [1, 2, 3], "b": [5, 6, 7]})
    plc_table = plc.Table.from_arrow(expected)
    result = plc.interop.from_dlpack(plc.interop.to_dlpack(plc_table))
    assert_table_eq(expected, result)


@pytest.mark.parametrize("array", [np.array, cp.array])
def test_round_trip_dlpack_array(array, patch_cupy_stream):
    with patch_cupy_stream:
        arr = array([1, 2, 3])
    result = plc.interop.from_dlpack(arr.__dlpack__())
    expected = pa.table({"a": [1, 2, 3]})
    assert_table_eq(expected, result)


def test_to_dlpack_error():
    plc_table = plc.Table.from_arrow(
        pa.table({"a": [1, None, 3], "b": [5, 6, 7]})
    )
    with pytest.raises(ValueError, match="Cannot create a DLPack tensor"):
        plc.interop.from_dlpack(plc.interop.to_dlpack(plc_table))


def test_from_dlpack_error():
    with pytest.raises(ValueError, match="Invalid PyCapsule object"):
        plc.interop.from_dlpack(1)


# We can't control the stream nanoarrow uses internally so we must disable the stream
# testing for these tests.
@pytest.mark.uses_custom_stream
def test_device_interop_column():
    pa_arr = pa.array([{"a": [1, None]}, None, {"b": [None, 4]}])
    plc_col = plc.Column.from_arrow(pa_arr)

    na_arr = nanoarrow.device.c_device_array(plc_col)
    new_col = plc.Column.from_arrow(na_arr)
    assert_column_eq(pa_arr, new_col)


@pytest.mark.uses_custom_stream
def test_device_interop_table():
    # Have to manually construct the schema to ensure that names match. pyarrow will
    # assign names to nested types automatically otherwise.
    schema = pa.schema(
        [
            pa.field("", pa.int64()),
            pa.field("", pa.float64()),
            pa.field("", pa.string()),
            pa.field("", pa.list_(pa.field("", pa.int64()))),
            pa.field("", pa.struct([pa.field("", pa.float64())])),
        ]
    )
    pa_tbl = pa.table(
        [
            [1, None, 3],
            [1.0, 2.0, None],
            ["a", "b", None],
            [[1, None], None, [2]],
            [{"a": 1.0}, None, {"b": 2.0}],
        ],
        schema=schema,
    )
    plc_table = plc.Table.from_arrow(pa_tbl)

    na_arr = nanoarrow.device.c_device_array(plc_table)
    actual_schema = pa.schema(na_arr.schema)
    assert actual_schema.equals(pa_tbl.schema)

    new_tbl = plc.Table.from_arrow(na_arr)
    assert_table_eq(pa_tbl, new_tbl)


@pytest.mark.parametrize(
    "data",
    [
        [[1, 2, 3], [4, 5, 6]],
        [[1, 2, 3], [None, 5, 6]],
        [[[1]], [[2]]],
        [[{"a": 1}], [{"b": 2}]],
    ],
)
def test_column_from_arrow_stream(data):
    pa_arr = pa.chunked_array(data)
    col = plc.Column.from_arrow(pa_arr)
    assert_column_eq(pa_arr, col)


def test_arrow_object_lifetime():
    def f():
        # Store a temporary so it is cached in the frame when the exception is raised
        t = plc.Table.from_arrow(pa.Table.from_pydict({"a": [1]}))  # noqa: F841
        raise ValueError("test exception")

    # Nested try-excepts are necessary for Python to extend the lifetime of the stack
    # frame of f enough to be problematic.
    try:
        try:
            previous = rmm.mr.get_current_device_resource()
            rmm.mr.set_current_device_resource(
                rmm.mr.CudaAsyncMemoryResource()
            )
            f()
        finally:
            rmm.mr.set_current_device_resource(previous)
    except ValueError:
        # Ignore the exception. A failure in this test is a seg fault
        pass
