# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from decimal import Decimal

import cupy as cp
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import cudf
from cudf.core.column.column import _can_values_be_equal, as_column
from cudf.core.column.decimal import DecimalColumn
from cudf.testing import assert_eq


@pytest.fixture
def pandas_input(all_supported_types_as_str):
    dtype = all_supported_types_as_str
    rng = np.random.default_rng(seed=0)
    size = 100

    def random_ints(dtype, size):
        dtype_min = np.iinfo(dtype).min
        dtype_max = np.iinfo(dtype).max
        rng = np.random.default_rng(seed=0)
        return rng.integers(dtype_min, dtype_max, size=size, dtype=dtype)

    try:
        dtype = np.dtype(dtype)
    except TypeError:
        if dtype == "category":
            data = random_ints(np.int64, size)
        else:
            raise
    else:
        if dtype.kind == "b":
            data = rng.choice([False, True], size=size)
        elif dtype.kind in ("m", "M"):
            # datetime or timedelta
            data = random_ints(np.int64, size)
        elif dtype.kind == "U":
            # Unicode strings of integers like "12345"
            # As of pandas 3.0, pandas interprets numpy string type as object
            data = random_ints(np.int64, size).astype(dtype.str)
            data = pd.array(data, dtype=pd.StringDtype(na_value=np.nan))
            dtype = None
        elif dtype.kind == "f":
            # floats in [0.0, 1.0)
            data = rng.random(size=size, dtype=dtype)
        else:
            data = random_ints(dtype, size)
    return pd.Series(data, dtype=dtype)


def str_host_view(list_of_str, to_dtype):
    return np.concatenate(
        [np.frombuffer(s.encode("utf-8"), dtype=to_dtype) for s in list_of_str]
    )


def test_column_set_equal_length_object_by_mask():
    # Series.__setitem__ might bypass some of the cases
    # handled in column.__setitem__ so this test is needed

    data = cudf.Series([0, 0, 1, 1, 1])._column
    replace_data = cudf.Series([100, 200, 300, 400, 500])._column
    bool_col = cudf.Series([True, True, True, True, True])._column

    data[bool_col] = replace_data
    assert_eq(
        cudf.Series._from_column(data),
        cudf.Series._from_column(replace_data),
    )

    data = cudf.Series([0, 0, 1, 1, 1])._column
    bool_col = cudf.Series([True, False, True, False, True])._column
    data[bool_col] = replace_data

    assert_eq(
        cudf.Series._from_column(data),
        cudf.Series([100, 0, 300, 1, 500]),
    )


@pytest.mark.parametrize("offset", [0, 1, 15])
@pytest.mark.parametrize("size", [50, 10, 0])
def test_column_offset_and_size(pandas_input, offset, size):
    col = as_column(pandas_input)

    col = col.slice(offset, offset + size)

    if isinstance(col.dtype, cudf.CategoricalDtype):
        assert col.size == col.codes.size

    got = cudf.Series._from_column(col)

    if offset is None:
        offset = 0
    if size is None:
        size = 100
    else:
        size = size + offset

    slicer = slice(offset, size)
    expect = pandas_input.iloc[slicer].reset_index(drop=True)

    assert_eq(expect, got)


def column_slicing_test(col, offset, size, cast_to_float=False):
    col_slice = col.slice(offset, offset + size)
    series = cudf.Series._from_column(col)
    sliced_series = cudf.Series._from_column(col_slice)

    if cast_to_float:
        pd_series = series.astype(float).to_pandas()
        sliced_series = sliced_series.astype(float)
    else:
        pd_series = series.to_pandas()

    if isinstance(col.dtype, cudf.CategoricalDtype):
        # The cudf.Series is constructed from an already sliced column, whereas
        # the pandas.Series is constructed from the unsliced series and then
        # sliced, so the indexes should be different and we must ignore it.
        # However, we must compare these as frames, not raw arrays,  because
        # numpy comparison of categorical values won't work.
        assert_eq(
            pd_series[offset : offset + size].reset_index(drop=True),
            sliced_series.reset_index(drop=True),
        )
    else:
        assert_eq(
            np.asarray(pd_series[offset : offset + size]),
            sliced_series.to_numpy(),
        )


@pytest.mark.parametrize("offset", [0, 1, 15])
@pytest.mark.parametrize("size", [50, 10, 0])
def test_column_slicing(pandas_input, offset, size):
    col = as_column(pandas_input)
    column_slicing_test(col, offset, size)


@pytest.mark.parametrize("offset", [0, 1, 15])
@pytest.mark.parametrize("size", [50, 10, 0])
@pytest.mark.parametrize("precision", [2, 3, 5])
@pytest.mark.parametrize("scale", [0, 1, 2])
@pytest.mark.parametrize(
    "decimal_type",
    [cudf.Decimal128Dtype, cudf.Decimal64Dtype, cudf.Decimal32Dtype],
)
def test_decimal_column_slicing(offset, size, precision, scale, decimal_type):
    col = as_column(pd.Series(np.random.default_rng(seed=0).random(1000)))
    col = col.astype(decimal_type(precision, scale))
    column_slicing_test(col, offset, size, True)


@pytest.mark.parametrize(
    "data",
    [
        np.array([[23, 68, 2, 38, 9, 83, 72, 6, 98, 30]]),
        np.array([[1, 2], [7, 6]]),
    ],
)
def test_column_series_multi_dim(data):
    with pytest.raises(ValueError):
        cudf.Series(data)

    with pytest.raises(ValueError):
        as_column(data)


@pytest.mark.parametrize(
    ("data", "error"),
    [
        ([1, "1.0", "2", -3], cudf.errors.MixedTypeError),
        ([np.nan, 0, "null", cp.nan], cudf.errors.MixedTypeError),
        (
            [np.int32(4), np.float64(1.5), np.float32(1.290994), np.int8(0)],
            None,
        ),
    ],
)
def test_column_mixed_dtype(data, error):
    if error is None:
        cudf.Series(data)
    else:
        with pytest.raises(error):
            cudf.Series(data)


@pytest.mark.parametrize(
    "scalar",
    [np.nan, pd.Timedelta(days=1), pd.Timestamp(2020, 1, 1)],
    ids=repr,
)
def test_as_column_scalar_with_nan(nan_as_null, scalar):
    size = 5
    expected = (
        cudf.Series([scalar] * size, nan_as_null=nan_as_null)
        .dropna()
        .to_numpy()
    )

    got = (
        cudf.Series._from_column(
            as_column(scalar, length=size, nan_as_null=nan_as_null)
        )
        .dropna()
        .to_numpy()
    )

    np.testing.assert_equal(expected, got)


@pytest.mark.parametrize("data", [[1.1, 2.2, 3.3, 4.4], [1, 2, 3, 4]])
def test_column_series_cuda_array_dtype(data, float_types_as_str):
    psr = pd.Series(np.asarray(data, dtype=float_types_as_str))
    sr = cudf.Series(cp.asarray(data, dtype=float_types_as_str))

    assert_eq(psr, sr)

    psr = pd.Series(data, dtype=float_types_as_str)
    sr = cudf.Series(data, dtype=float_types_as_str)

    assert_eq(psr, sr)


def test_column_zero_length_slice():
    # see https://github.com/rapidsai/cudf/pull/4777
    x = cudf.DataFrame({"a": [1]})
    the_column = x[1:]["a"]._column

    expect = np.array([], dtype="int8")
    got = cp.asarray(the_column.data).get()

    np.testing.assert_array_equal(expect, got)


def test_column_chunked_array_creation():
    pyarrow_array = pa.array([1, 2, 3] * 1000)
    chunked_array = pa.chunked_array(pyarrow_array)

    actual_column = as_column(chunked_array, dtype=np.dtype(np.float64))
    expected_column = as_column(pyarrow_array, dtype=np.dtype(np.float64))

    assert_eq(
        cudf.Series._from_column(actual_column),
        cudf.Series._from_column(expected_column),
    )

    actual_column = as_column(chunked_array)
    expected_column = as_column(pyarrow_array)

    assert_eq(
        cudf.Series._from_column(actual_column),
        cudf.Series._from_column(expected_column),
    )


@pytest.mark.parametrize("box", [cp.asarray, np.asarray])
@pytest.mark.parametrize(
    "data",
    [
        np.array([1, 2, 3, 4, 5], dtype="uint8"),
        np.array([], dtype="uint8"),
        np.array([255], dtype="uint8"),
    ],
)
def test_as_column_buffer(box, data):
    expected = as_column(data)
    boxed = box(data)
    actual_column = as_column(
        cudf.core.buffer.as_buffer(
            boxed if isinstance(boxed, cp.ndarray) else boxed.data
        ),
        dtype=data.dtype,
    )
    assert_eq(
        cudf.Series._from_column(actual_column),
        cudf.Series._from_column(expected),
    )


def test_can_cast_safely_same_kind():
    # 'i' -> 'i'
    data = cudf.Series([1, 2, 3], dtype="int32")._column
    to_dtype = np.dtype("int64")

    assert data.can_cast_safely(to_dtype)

    data = cudf.Series([1, 2, 3], dtype="int64")._column
    to_dtype = np.dtype("int32")

    assert data.can_cast_safely(to_dtype)

    data = cudf.Series([1, 2, 2**31], dtype="int64")._column
    assert not data.can_cast_safely(to_dtype)

    # 'u' -> 'u'
    data = cudf.Series([1, 2, 3], dtype="uint32")._column
    to_dtype = np.dtype("uint64")

    assert data.can_cast_safely(to_dtype)

    data = cudf.Series([1, 2, 3], dtype="uint64")._column
    to_dtype = np.dtype("uint32")

    assert data.can_cast_safely(to_dtype)

    data = cudf.Series([1, 2, 2**33], dtype="uint64")._column
    assert not data.can_cast_safely(to_dtype)

    # 'f' -> 'f'
    data = cudf.Series([np.inf, 1.0], dtype="float64")._column
    to_dtype = np.dtype("float32")
    assert data.can_cast_safely(to_dtype)

    data = cudf.Series(
        [float(np.finfo("float32").max) * 2, 1.0], dtype="float64"
    )._column
    to_dtype = np.dtype("float32")
    assert not data.can_cast_safely(to_dtype)


def test_can_cast_safely_empty():
    # An empty column can be cast to any numeric dtype losslessly.
    data = cudf.Series([], dtype="float64")._column
    for to_dtype in ["int8", "int64", "uint32", "float32", "bool"]:
        assert data.can_cast_safely(np.dtype(to_dtype))


def test_can_cast_safely_mixed_kind():
    data = cudf.Series([1, 2, 3], dtype="int32")._column
    to_dtype = np.dtype("float32")
    assert data.can_cast_safely(to_dtype)

    # too big to fit into f32 exactly
    data = cudf.Series([1, 2, 2**24 + 1], dtype="int32")._column
    assert not data.can_cast_safely(to_dtype)

    data = cudf.Series([1, 2, 3], dtype="uint32")._column
    to_dtype = np.dtype("float32")
    assert data.can_cast_safely(to_dtype)

    # too big to fit into f32 exactly
    data = cudf.Series([1, 2, 2**24 + 1], dtype="uint32")._column
    assert not data.can_cast_safely(to_dtype)

    to_dtype = np.dtype("float64")
    assert data.can_cast_safely(to_dtype)

    data = cudf.Series([1.0, 2.0, 3.0], dtype="float32")._column
    to_dtype = np.dtype("int32")
    assert data.can_cast_safely(to_dtype)

    # not integer float
    data = cudf.Series([1.0, 2.0, 3.5], dtype="float32")._column
    assert not data.can_cast_safely(to_dtype)

    data = cudf.Series([10.0, 11.0, 2000.0], dtype="float64")._column
    assert data.can_cast_safely(to_dtype)

    # float out of int range
    data = cudf.Series([1.0, 2.0, 1.0 * (2**31)], dtype="float32")._column
    assert not data.can_cast_safely(to_dtype)

    # negative signed integers casting to unsigned integers
    data = cudf.Series([-1, 0, 1], dtype="int32")._column
    to_dtype = np.dtype("uint32")
    assert not data.can_cast_safely(to_dtype)


def test_can_cast_safely_has_nulls():
    data = cudf.Series([1, 2, 3, None], dtype="float32")._column
    to_dtype = np.dtype("int64")

    assert data.can_cast_safely(to_dtype)

    data = cudf.Series([1, 2, 3.1, None], dtype="float32")._column
    assert not data.can_cast_safely(to_dtype)


@pytest.mark.parametrize(
    "vals, src, to, expected",
    [
        # same kind ('i' -> 'i', 'u' -> 'u'): exact dtype bounds fit
        # inclusively
        (
            [np.iinfo(np.int8).min, np.iinfo(np.int8).max],
            "int64",
            "int8",
            True,
        ),
        (
            [np.iinfo(np.int16).min, np.iinfo(np.int16).max],
            "int64",
            "int16",
            True,
        ),
        (
            [np.iinfo(np.int8).min, np.iinfo(np.int8).max + 1],
            "int64",
            "int8",
            False,
        ),
        ([0, np.iinfo(np.uint16).max], "uint64", "uint16", True),
        ([0, np.iinfo(np.uint16).max + 1], "uint64", "uint16", False),
        # int -> uint: the exact unsigned max is representable
        ([0, np.iinfo(np.uint8).max], "int64", "uint8", True),
        ([0, np.iinfo(np.uint32).max], "int64", "uint32", True),
        ([0, np.iinfo(np.uint8).max + 1], "int64", "uint8", False),
        # uint -> int: the exact signed max is representable
        ([0, np.iinfo(np.int32).max], "uint64", "int32", True),
        ([0, np.iinfo(np.int64).max], "uint64", "int64", True),
        ([0, np.iinfo(np.int32).max + 1], "uint64", "int32", False),
    ],
)
def test_can_cast_safely_dtype_bounds(vals, src, to, expected):
    # A value equal to the target dtype's max must be considered castable
    # (the bound check is inclusive).
    data = cudf.Series(vals, dtype=src)._column
    assert bool(data.can_cast_safely(np.dtype(to))) is expected


@pytest.mark.parametrize(
    "data,pyarrow_kwargs,cudf_kwargs",
    [
        (
            [100, 200, 300],
            {"type": pa.decimal128(3)},
            {"dtype": cudf.core.dtypes.Decimal128Dtype(3, 0)},
        ),
        (
            [{"a": 1, "b": 3}, {"c": 2, "d": 4}],
            {},
            {},
        ),
        (
            [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],
            {},
            {},
        ),
    ],
)
def test_as_column_arrow_array(data, pyarrow_kwargs, cudf_kwargs):
    pyarrow_data = pa.array(data, **pyarrow_kwargs)
    cudf_from_pyarrow = as_column(pyarrow_data)
    expected = as_column(data, **cudf_kwargs)
    assert_eq(
        cudf.Series._from_column(cudf_from_pyarrow),
        cudf.Series._from_column(expected),
    )


@pytest.mark.parametrize(
    "left, right, expected",
    [
        (np.dtype(np.int64), np.dtype(np.int64), True),
        (np.dtype(np.int64), np.dtype(np.float32), True),
        (np.dtype(np.int64), cudf.Decimal64Dtype(10, 5), True),
        (np.dtype(np.int64), np.dtype(object), False),
        (np.dtype("datetime64[ns]"), np.dtype("datetime64[ms]"), True),
        (np.dtype("timedelta64[ns]"), np.dtype("timedelta64[ms]"), True),
        (np.dtype("timedelta64[ns]"), np.dtype("datetime64[ms]"), False),
        (cudf.CategoricalDtype([1]), np.dtype(np.int64), True),
        (cudf.CategoricalDtype([1]), np.dtype("datetime64[ms]"), False),
    ],
)
def test__can_values_be_equal(left, right, expected):
    assert _can_values_be_equal(left, right) is expected
    assert _can_values_be_equal(right, left) is expected


def test_string_int_to_ipv4():
    gsr = cudf.Series([0, None, 0, 698875905, 2130706433, 700776449]).astype(
        "uint32"
    )
    expected = cudf.Series(
        ["0.0.0.0", None, "0.0.0.0", "41.168.0.1", "127.0.0.1", "41.197.0.1"]
    )

    got = cudf.Series._from_column(gsr._column.int2ip())

    assert_eq(expected, got)


def test_string_int_to_ipv4_dtype_fail(numeric_types_as_str):
    if numeric_types_as_str == "uint32":
        pytest.skip(f"int2ip passes with {numeric_types_as_str}")
    gsr = cudf.Series([1, 2, 3, 4, 5]).astype(numeric_types_as_str)
    with pytest.raises(TypeError):
        gsr._column.int2ip()


def test_datetime_can_cast_safely():
    sr = cudf.Series(
        ["1679-01-01", "2000-01-31", "2261-01-01"], dtype="datetime64[ms]"
    )
    assert sr._column.can_cast_safely(np.dtype("datetime64[ns]"))

    sr = cudf.Series(
        ["1677-01-01", "2000-01-31", "2263-01-01"], dtype="datetime64[ms]"
    )

    assert sr._column.can_cast_safely(np.dtype("datetime64[ns]")) is False

    # Pre-epoch values below the target range are also unsafe.
    sr = cudf.Series(["1600-01-01", "2000-01-31"], dtype="datetime64[ms]")
    assert sr._column.can_cast_safely(np.dtype("datetime64[ns]")) is False

    # Casting to a coarser resolution truncates sub-resolution
    # components, so it is not considered safe; equal resolution is.
    sr = cudf.Series(["2000-01-31"], dtype="datetime64[ns]")
    assert sr._column.can_cast_safely(np.dtype("datetime64[s]")) is False
    assert sr._column.can_cast_safely(np.dtype("datetime64[ns]"))


@pytest.mark.parametrize(
    "data_",
    [
        [Decimal("1.1"), Decimal("2.2"), Decimal("3.3"), Decimal("4.4")],
        [Decimal("-1.1"), Decimal("2.2"), Decimal("3.3"), Decimal("4.4")],
        [1],
        [-1],
        [1, 2, 3, 4],
        [42, 17, 41],
        [1, 2, None, 4],
        [None, None, None],
        [],
    ],
)
@pytest.mark.parametrize(
    "typ_",
    [
        pa.decimal32(precision=4, scale=2),
        pa.decimal64(precision=5, scale=3),
        pa.decimal128(precision=6, scale=4),
    ],
)
def test_round_trip_decimal_column(data_, typ_):
    pa_arr = pa.array(data_, type=typ_)
    decimal_col = DecimalColumn.from_arrow(pa_arr)
    result = decimal_col.to_arrow()

    # Round-trip should preserve the exact PyArrow decimal type
    assert result.equals(pa_arr)


def test_from_arrow_max_precision_decimal64():
    # Decimal64 max precision is 18, so 19 should raise ValueError
    with pytest.raises(ValueError):
        DecimalColumn.from_arrow(
            pa.array([1, 2, 3], type=pa.decimal64(scale=0, precision=19))
        )


def test_from_arrow_max_precision_decimal32():
    # Decimal32 max precision is 9, so 10 should raise ValueError
    with pytest.raises(ValueError):
        DecimalColumn.from_arrow(
            pa.array([1, 2, 3], type=pa.decimal32(scale=0, precision=10))
        )
