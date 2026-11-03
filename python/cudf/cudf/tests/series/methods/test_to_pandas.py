# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import datetime
import decimal

import cupy as cp
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import cudf
from cudf.testing import _object_array_equal_nan, assert_eq


def test_to_pandas_index_true_timezone():
    data = [
        "2008-05-12",
        "2008-12-12",
        "2009-05-12",
    ]
    dti = cudf.DatetimeIndex(data).tz_localize("UTC")
    ser = cudf.Series(dti, index=list("abc"))
    result = ser.to_pandas(index=True)
    expected = pd.Series(pd.to_datetime(data, utc=True), index=list("abc"))
    assert_eq(result, expected)


@pytest.mark.parametrize(
    "sr_data,expected_psr",
    [
        (
            pa.array([1, 2, None, 3], type=pa.uint8()),
            pd.Series([1, 2, None, 3], dtype=pd.UInt8Dtype()),
        ),
        (
            pa.array([23, None, None, 32], type=pa.uint16()),
            pd.Series([23, None, None, 32], dtype=pd.UInt16Dtype()),
        ),
        (
            pa.array([None, 123, None, 1], type=pa.uint32()),
            pd.Series([None, 123, None, 1], dtype=pd.UInt32Dtype()),
        ),
        (
            pa.array([234, 2323, 23432, None, None, 224], type=pa.uint64()),
            pd.Series(
                [234, 2323, 23432, None, None, 224], dtype=pd.UInt64Dtype()
            ),
        ),
        (
            pa.array([-10, 1, None, -1, None, 3], type=pa.int8()),
            pd.Series([-10, 1, None, -1, None, 3], dtype=pd.Int8Dtype()),
        ),
        (
            pa.array([111, None, 222, None, 13], type=pa.int16()),
            pd.Series([111, None, 222, None, 13], dtype=pd.Int16Dtype()),
        ),
        (
            pa.array([11, None, 22, 33, None, 2, None, 3], type=pa.int32()),
            pd.Series(
                [11, None, 22, 33, None, 2, None, 3], dtype=pd.Int32Dtype()
            ),
        ),
        (
            pa.array(
                [32431, None, None, 32322, 0, 10, -32324, None],
                type=pa.int64(),
            ),
            pd.Series(
                [32431, None, None, 32322, 0, 10, -32324, None],
                dtype=pd.Int64Dtype(),
            ),
        ),
        (
            pa.array(
                [True, None, False, None, False, True, True, False],
                type=pa.bool_(),
            ),
            pd.Series(
                [True, None, False, None, False, True, True, False],
                dtype=pd.BooleanDtype(),
            ),
        ),
        (
            pa.array(
                [
                    "abc",
                    "a",
                    None,
                    "hello world",
                    "foo buzz",
                    "",
                    None,
                    "rapids ai",
                ],
                type=pa.string(),
            ),
            pd.Series(
                [
                    "abc",
                    "a",
                    None,
                    "hello world",
                    "foo buzz",
                    "",
                    None,
                    "rapids ai",
                ],
                dtype=pd.StringDtype(na_value=np.nan),
            ),
        ),
        (
            pa.array(
                [1, 2, None, 10.2, None],
                type=pa.float32(),
            ),
            pd.Series(
                [1, 2, None, 10.2, None],
                dtype=pd.Float32Dtype(),
            ),
        ),
    ],
)
def test_series_to_pandas_nullable_dtypes(sr_data, expected_psr):
    sr = cudf.Series(sr_data)
    actual_psr = sr.to_pandas(nullable=True)

    assert_eq(actual_psr, expected_psr)


@pytest.mark.parametrize(
    "scalar",
    [
        1,
        1.0,
        "a",
        datetime.datetime(2020, 1, 1),
        datetime.timedelta(1),
        {"1": 2},
        [1],
        decimal.Decimal("1.0"),
    ],
)
def test_series_to_pandas_arrow_type_nullable_raises(scalar):
    pa_array = pa.array([scalar, None])
    ser = cudf.Series(pa_array)
    with pytest.raises(ValueError, match=r".* cannot both be set"):
        ser.to_pandas(nullable=True, arrow_type=True)


def test_to_pandas_nullable_integer():
    gsr_not_null = cudf.Series([1, 2, 3])
    gsr_has_null = cudf.Series([1, 2, None])

    psr_not_null = pd.Series([1, 2, 3], dtype="int64")
    psr_has_null = pd.Series([1, 2, None], dtype="Int64")

    assert_eq(gsr_not_null.to_pandas(), psr_not_null)
    assert_eq(gsr_has_null.to_pandas(nullable=True), psr_has_null)


def test_to_pandas_nullable_bool():
    gsr_not_null = cudf.Series([True, False, True])
    gsr_has_null = cudf.Series([True, False, None])

    psr_not_null = pd.Series([True, False, True], dtype="bool")
    psr_has_null = pd.Series([True, False, None], dtype="boolean")

    assert_eq(gsr_not_null.to_pandas(), psr_not_null)
    assert_eq(gsr_has_null.to_pandas(nullable=True), psr_has_null)


@pytest.mark.parametrize(
    "scalar",
    [
        1,
        1.0,
        "a",
        datetime.datetime(2020, 1, 1),
        datetime.timedelta(1),
        {"1": 2},
        [1],
        decimal.Decimal("1.0"),
    ],
)
def test_series_to_pandas_arrow_type(scalar):
    pa_array = pa.array([scalar, None])
    ser = cudf.Series(pa_array)
    result = ser.to_pandas(arrow_type=True)
    expected = pd.Series(pd.arrays.ArrowExtensionArray(pa_array))
    pd.testing.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "data",
    [
        [1000000, 200000, 3000000],
        [1000000, 200000, None],
        [],
        [None],
        [None, None, None, None, None],
        [12, 12, 22, 343, 4353534, 435342],
        np.array([10, 20, 30, None, 100]),
        cp.asarray([10, 20, 30, 100]),
    ],
)
def test_timedelta_series_to_pandas(data, timedelta_types_as_str):
    gsr = cudf.Series(data, dtype=timedelta_types_as_str)

    expected = np.array(
        cp.asnumpy(data) if isinstance(data, cp.ndarray) else data,
        dtype=timedelta_types_as_str,
    )

    expected = pd.Series(expected)
    actual = gsr.to_pandas()

    assert_eq(expected, actual)


def test_writable_numpy_array_timedelta():
    gi = cudf.Index([1, 2, 3], dtype="timedelta64[ns]")
    expected_flags = pd.Index(
        [1, 2, 3], dtype="timedelta64[ns]"
    )._data._ndarray.flags

    actual_flags = gi.to_pandas()._data._ndarray.flags
    assert expected_flags.c_contiguous == actual_flags.c_contiguous
    assert expected_flags.f_contiguous == actual_flags.f_contiguous
    assert expected_flags.writeable == actual_flags.writeable
    assert expected_flags.aligned == actual_flags.aligned
    assert expected_flags.writebackifcopy == actual_flags.writebackifcopy


@pytest.mark.parametrize("nulls", ["some", "all"])
def test_to_from_pandas_nulls(nulls):
    data = np.arange(1, 10)
    pd_data = pd.Series(data.astype("datetime64[ns]"))
    if nulls == "some":
        # Fill half the values with NaT
        pd_data[list(range(0, len(pd_data), 2))] = np.datetime64("nat", "ns")
    elif nulls == "all":
        # Fill all the values with NaT
        pd_data[:] = np.datetime64("nat", "ns")
    gdf_data = cudf.Series(pd_data)

    expect = pd_data
    got = gdf_data.to_pandas()

    assert_eq(expect, got)


@pytest.mark.parametrize("data", [["a"], ["a", None], [None]])
def test_string_export(data):
    ps = pd.Series(data, dtype="str", name="nice name")
    gs = cudf.Series(data, dtype="str", name="nice name")

    expect = ps
    got = gs.to_pandas()
    assert_eq(expect, got)

    expect = np.array(ps)
    got = gs.to_numpy()

    try:
        assert_eq(expect, got)
    except AssertionError:
        _object_array_equal_nan(expect, got)

    expect = pa.Array.from_pandas(ps)
    got = gs.to_arrow()
    assert pa.Array.equals(expect, got)
