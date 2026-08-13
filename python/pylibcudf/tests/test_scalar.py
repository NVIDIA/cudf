# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import datetime
import decimal

import pyarrow as pa
import pytest

import pylibcudf as plc
from pylibcudf.types import DataType, TypeId


@pytest.fixture(scope="module")
def np():
    return pytest.importorskip("numpy")


@pytest.fixture(
    params=[
        True,
        False,
        -1,
        0,
        1 - 1.0,
        0.0,
        1.52,
        "",
        "a1!",
        datetime.datetime(2020, 1, 1),
        datetime.datetime(2020, 1, 1, microsecond=1),
        datetime.timedelta(1),
        datetime.timedelta(days=1, microseconds=1),
        decimal.Decimal("3.14"),
    ],
    ids=repr,
)
def py_scalar(request):
    return request.param


def test_from_py(py_scalar):
    result = plc.Scalar.from_py(py_scalar)
    expected = pa.scalar(py_scalar)
    if isinstance(py_scalar, decimal.Decimal):
        # libcudf decimals don't have precision so we must use to_py
        # instead of to_arrow
        assert result.to_py() == expected.as_py()
    else:
        assert result.to_arrow().equals(expected)
    if isinstance(py_scalar, decimal.Decimal):
        assert (
            result.type()
            .to_arrow(precision=len(py_scalar.as_tuple().digits))
            .equals(expected.type)
        )
    else:
        assert result.type().to_arrow().equals(expected.type)


def test_to_py_none():
    assert plc.Scalar.from_py(None, DataType(TypeId.INT8)).to_py() is None


def test_to_py(py_scalar):
    if isinstance(py_scalar, (datetime.datetime, datetime.timedelta)):
        with pytest.raises(NotImplementedError):
            plc.Scalar.from_py(py_scalar).to_py()
        assert py_scalar == plc.Scalar.from_py(py_scalar).to_arrow().as_py()
    else:
        assert py_scalar == plc.Scalar.from_py(py_scalar).to_py()


@pytest.mark.parametrize(
    "val,tid",
    [
        (1, TypeId.INT8),
        (1, TypeId.INT16),
        (1, TypeId.INT32),
        (1, TypeId.INT64),
        (1, TypeId.UINT8),
        (1, TypeId.UINT16),
        (1, TypeId.UINT32),
        (1, TypeId.UINT64),
        (1, TypeId.FLOAT32),
        (1, TypeId.DURATION_NANOSECONDS),
        (1, TypeId.DURATION_MICROSECONDS),
        (1, TypeId.DURATION_MILLISECONDS),
        (1, TypeId.DURATION_SECONDS),
        (1.0, TypeId.FLOAT32),
        (1.5, TypeId.FLOAT64),
        (float("inf"), TypeId.FLOAT32),
        (float("-inf"), TypeId.FLOAT32),
        (float("inf"), TypeId.FLOAT64),
        (float("-inf"), TypeId.FLOAT64),
        ("str", TypeId.STRING),
        (True, TypeId.BOOL8),
        (0, TypeId.BOOL8),
        (1, TypeId.BOOL8),
        (datetime.timedelta(1), TypeId.DURATION_SECONDS),
        (datetime.timedelta(1), TypeId.DURATION_MILLISECONDS),
        (datetime.timedelta(1), TypeId.DURATION_NANOSECONDS),
        (datetime.datetime(2020, 1, 1), TypeId.TIMESTAMP_SECONDS),
        (datetime.datetime(2020, 1, 1), TypeId.TIMESTAMP_MILLISECONDS),
        (datetime.datetime(2020, 1, 1), TypeId.TIMESTAMP_NANOSECONDS),
    ],
)
def test_from_py_with_dtype(val, tid):
    dtype = DataType(tid)
    result = plc.Scalar.from_py(val, dtype)
    expected = pa.scalar(val).cast(dtype.to_arrow())
    assert result.to_arrow().equals(expected)


@pytest.mark.parametrize(
    "val,tid,error,msg",
    [
        (
            -1,
            TypeId.UINT8,
            ValueError,
            "Cannot assign negative value to UINT8 scalar",
        ),
        (
            -1,
            TypeId.UINT16,
            ValueError,
            "Cannot assign negative value to UINT16 scalar",
        ),
        (
            -1,
            TypeId.UINT32,
            ValueError,
            "Cannot assign negative value to UINT32 scalar",
        ),
        (
            -1,
            TypeId.UINT64,
            ValueError,
            "Cannot assign negative value to UINT64 scalar",
        ),
        (
            2,
            TypeId.BOOL8,
            ValueError,
            "Cannot convert 2 to BOOL8 scalar",
        ),
        (
            "str",
            TypeId.INT32,
            TypeError,
            "Cannot convert str to Scalar with dtype INT32",
        ),
        (
            True,
            TypeId.INT32,
            TypeError,
            "Cannot convert bool to Scalar with dtype INT32",
        ),
        (
            1.5,
            TypeId.INT32,
            TypeError,
            "Cannot convert float to Scalar with dtype INT32",
        ),
        (
            datetime.datetime(2020, 1, 1),
            TypeId.INT32,
            TypeError,
            "Cannot convert datetime to Scalar with dtype INT32",
        ),
        (
            datetime.timedelta(days=1, microseconds=1),
            TypeId.INT32,
            TypeError,
            "Cannot convert timedelta to Scalar with dtype INT32",
        ),
    ],
)
def test_from_py_with_dtype_errors(val, tid, error, msg):
    dtype = DataType(tid)
    with pytest.raises(error, match=msg):
        plc.Scalar.from_py(val, dtype)


@pytest.mark.parametrize(
    "val, tid",
    [
        (-(2**7) - 1, TypeId.INT8),
        (2**7, TypeId.INT8),
        (2**15, TypeId.INT16),
        (2**31, TypeId.INT32),
        (2**63, TypeId.INT64),
        (2**8, TypeId.UINT8),
        (2**16, TypeId.UINT16),
        (2**32, TypeId.UINT32),
        (2**64, TypeId.UINT64),
        (float(2**150), TypeId.FLOAT32),
        (float(-(2**150)), TypeId.FLOAT32),
        (datetime.timedelta.max, TypeId.DURATION_NANOSECONDS),
        (datetime.timedelta.max, TypeId.DURATION_MICROSECONDS),
        (datetime.datetime.max, TypeId.TIMESTAMP_NANOSECONDS),
    ],
)
def test_from_py_overflow_errors(val, tid):
    dtype = DataType(tid)
    with pytest.raises(OverflowError, match="out of range"):
        plc.Scalar.from_py(val, dtype)


@pytest.mark.parametrize("val", [[1], {1: 1}])
def test_from_py_notimplemented(val):
    with pytest.raises(NotImplementedError):
        plc.Scalar.from_py(val)


def test_from_py_typeerror():
    with pytest.raises(TypeError):
        plc.Scalar.from_py(object)


def test_from_py_none_no_type_raises():
    with pytest.raises(ValueError):
        plc.Scalar.from_py(None)


def test_from_py_none():
    result = plc.Scalar.from_py(None, plc.DataType(plc.TypeId.STRING))
    expected = pa.scalar(None, type=pa.string())
    assert result.to_arrow().equals(expected)


@pytest.mark.parametrize(
    "np_type",
    [
        "bool_",
        "str_",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float32",
        "float64",
    ],
)
def test_from_numpy(np, np_type):
    np_klass = getattr(np, np_type)
    np_val = np_klass("1" if np_type == "str_" else 1)
    result = plc.Scalar.from_numpy(np_val)
    expected = pa.scalar(np_val)
    assert result.to_arrow().equals(expected)


@pytest.mark.parametrize("np_type", ["datetime64", "timedelta64"])
def test_from_numpy_notimplemented(np, np_type):
    np_val = getattr(np, np_type)(1, "ns")
    with pytest.raises(NotImplementedError):
        plc.Scalar.from_numpy(np_val)


def test_from_numpy_typeerror(np):
    with pytest.raises(TypeError):
        plc.Scalar.from_numpy(np.void(5))


def test_round_trip_scalar_through_column(py_scalar):
    result = plc.Column.from_scalar(
        plc.Scalar.from_py(py_scalar), 1
    ).to_scalar()
    expected = pa.scalar(py_scalar)
    if isinstance(py_scalar, decimal.Decimal):
        # libcudf decimals don't have precision so we must use to_py
        # instead of to_arrow
        assert result.to_py() == expected.as_py()
    else:
        assert result.to_arrow().equals(expected)


def test_non_constant_column_to_scalar_raises():
    with pytest.raises(
        ValueError, match="to_scalar only works for columns of size 1"
    ):
        plc.Column.from_arrow(pa.array([0, 1])).to_scalar()


def test_roundtrip_python_decimal():
    d = decimal.Decimal("3.14")
    assert d == plc.Scalar.from_py(d).to_py()
