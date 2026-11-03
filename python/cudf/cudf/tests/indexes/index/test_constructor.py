# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import re

import cupy as cp
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import cudf
from cudf.testing import assert_eq


def test_create_interval_index_from_list():
    interval_list = [
        np.nan,
        pd.Interval(2.0, 3.0, closed="right"),
        pd.Interval(3.0, 4.0, closed="right"),
    ]

    expected = pd.Index(interval_list)
    actual = cudf.Index(interval_list)

    assert_eq(expected, actual)


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
        [1000000, 200000, 3000000],
        [1000000, 200000, None],
        [1],
        [12, 11, 232, 223432411, 2343241, 234324, 23234],
        [12, 11, 2.32, 2234.32411, 2343.241, 23432.4, 23234],
        [1.321, 1132.324, 23223231.11, 233.41, 0.2434, 332, 323],
        [12, 11, 2.32, 2234.32411, 2343.241, 23432.4, 23234],
    ],
)
def test_infer_timedelta_index(data, timedelta_types_as_str):
    gdi = cudf.Index(data, dtype=timedelta_types_as_str)
    pdi = gdi.to_pandas()

    assert_eq(pdi, gdi)


def test_pandas_as_index():
    # Define Pandas Indexes
    pdf_int_index = pd.Index([1, 2, 3, 4, 5])
    pdf_uint_index = pd.Index([1, 2, 3, 4, 5])
    pdf_float_index = pd.Index([1.0, 2.0, 3.0, 4.0, 5.0])
    pdf_datetime_index = pd.DatetimeIndex(
        [1000000, 2000000, 3000000, 4000000, 5000000]
    )
    pdf_category_index = pd.CategoricalIndex(["a", "b", "c", "b", "a"])

    # Define cudf Indexes
    gdf_int_index = cudf.Index(pdf_int_index)
    gdf_uint_index = cudf.Index(pdf_uint_index)
    gdf_float_index = cudf.Index(pdf_float_index)
    gdf_datetime_index = cudf.Index(pdf_datetime_index)
    gdf_category_index = cudf.Index(pdf_category_index)

    # Check instance types
    assert isinstance(gdf_int_index, cudf.Index)
    assert isinstance(gdf_uint_index, cudf.Index)
    assert isinstance(gdf_float_index, cudf.Index)
    assert isinstance(gdf_datetime_index, cudf.DatetimeIndex)
    assert isinstance(gdf_category_index, cudf.CategoricalIndex)

    # Check equality
    assert_eq(pdf_int_index, gdf_int_index)
    assert_eq(pdf_uint_index, gdf_uint_index)
    assert_eq(pdf_float_index, gdf_float_index)
    assert_eq(pdf_datetime_index, gdf_datetime_index)
    assert_eq(pdf_category_index, gdf_category_index)

    assert_eq(
        pdf_category_index.codes,
        gdf_category_index.codes.astype(
            pdf_category_index.codes.dtype
        ).to_numpy(),
    )


def test_from_pandas_str():
    idx = ["a", "b", "c"]
    pidx = pd.Index(idx, name="idx")
    gidx_1 = cudf.Index(idx, name="idx")
    gidx_2 = cudf.from_pandas(pidx)

    assert_eq(gidx_1, gidx_2)


def test_from_pandas_gen():
    idx = [2, 4, 6]
    pidx = pd.Index(idx, name="idx")
    gidx_1 = cudf.Index(idx, name="idx")
    gidx_2 = cudf.from_pandas(pidx)

    assert_eq(gidx_1, gidx_2)


@pytest.mark.parametrize(
    "data",
    [
        range(0),
        range(1),
        range(0, 1),
        range(0, 5),
        range(1, 10),
        range(1, 10, 1),
        range(1, 10, 3),
        range(10, 1, -3),
        range(-5, 10),
    ],
)
def test_range_index_from_range(data):
    assert_eq(pd.Index(data), cudf.Index(data))


@pytest.mark.parametrize("data", [[1, 2, 3, 4], []])
@pytest.mark.parametrize("name", [1, "a", None])
def test_index_basic(data, all_supported_types_as_str, name, request):
    pdi = pd.Index(data, dtype=all_supported_types_as_str, name=name)
    gdi = cudf.Index(data, dtype=all_supported_types_as_str, name=name)
    if all_supported_types_as_str == "category" and len(data) == 0:
        # As of pandas 3.0, empty default type of object isn't
        # necessarily equivalent to cuDF's empty default type of
        # pandas.StringDtype
        pdi = pdi.set_categories(pd.Index([], dtype=gdi.categories.dtype))

    assert_eq(pdi, gdi)


@pytest.mark.parametrize(
    "data,nan_idx,NA_idx",
    [([1, 2, 3, None], None, 3), ([2, 3, np.nan, None], 2, 3)],
)
def test_index_nan_as_null(data, nan_idx, NA_idx, nan_as_null):
    idx = cudf.Index(data, nan_as_null=nan_as_null)

    if nan_as_null is not False:
        if nan_idx is not None:
            assert idx[nan_idx] is cudf.NA
    else:
        if nan_idx is not None:
            assert np.isnan(idx[nan_idx])

    if NA_idx is not None:
        if idx.dtype.kind == "f":
            assert np.isnan(idx[NA_idx])
        else:
            assert idx[NA_idx] is cudf.NA


def test_index_constructor_integer(default_integer_bitwidth):
    got = cudf.Index([1, 2, 3])
    expect = cudf.Index([1, 2, 3], dtype=f"int{default_integer_bitwidth}")

    assert_eq(expect, got)


def test_index_constructor_float(default_float_bitwidth):
    got = cudf.Index([1.0, 2.0, 3.0])
    expect = cudf.Index(
        [1.0, 2.0, 3.0], dtype=f"float{default_float_bitwidth}"
    )

    assert_eq(expect, got)


def test_index_error_list_index():
    s = cudf.Series([[1, 2], [2], [4]])
    with pytest.raises(
        NotImplementedError,
        match=re.escape(
            "Unsupported column type passed to create an "
            "Index: <class 'cudf.core.column.lists.ListColumn'>"
        ),
    ):
        cudf.Index(s)


@pytest.mark.parametrize(
    "data",
    [
        [
            pd.Timestamp("1970-01-01 00:00:00.000000001"),
            pd.Timestamp("1970-01-01 00:00:00.000000002"),
            12,
            20,
        ],
        [
            pd.Timedelta(10),
            pd.Timedelta(20),
            12,
            20,
        ],
        [1, 2, 3, 4],
    ],
)
def test_index_mixed_dtype_error(data):
    pi = pd.Index(data, dtype="object")
    with pytest.raises(TypeError):
        cudf.Index(pi)


@pytest.mark.parametrize("cls", [pd.DatetimeIndex, pd.TimedeltaIndex])
def test_index_date_duration_freq_error(cls):
    s = cls([1, 2, 3], freq="infer")
    with cudf.option_context("mode.pandas_compatible", True):
        with pytest.raises(NotImplementedError):
            cudf.Index(s)


def test_index_empty_from_pandas(all_supported_types_as_str):
    pidx = pd.Index([], dtype=all_supported_types_as_str)
    gidx = cudf.from_pandas(pidx)
    if all_supported_types_as_str == "category":
        # As of pandas 3.0, empty default type of object isn't
        # necessarily equivalent to cuDF's empty default type of
        # pandas.StringDtype
        pidx = pidx.set_categories(pd.Index([], dtype=gidx.categories.dtype))
    assert_eq(pidx, gidx)


def test_empty_index_init():
    pidx = pd.Index([])
    gidx = cudf.Index([])
    assert_eq(pidx, gidx)


@pytest.mark.parametrize("data", [[1, 2, 3], range(0, 10)])
def test_index_with_index_dtype(request, data, all_supported_types_as_str):
    pidx = pd.Index(data)
    gidx = cudf.Index(data)

    expected = pd.Index(pidx, dtype=all_supported_types_as_str)
    actual = cudf.Index(gidx, dtype=all_supported_types_as_str)

    assert_eq(expected, actual)


def test_period_index_error():
    pidx = pd.PeriodIndex(data=[pd.Period("2020-01")])
    with pytest.raises(NotImplementedError):
        cudf.from_pandas(pidx)
    with pytest.raises(NotImplementedError):
        cudf.Index(pidx)
    with pytest.raises(NotImplementedError):
        cudf.Series(pidx)
    with pytest.raises(NotImplementedError):
        cudf.Series(pd.Series(pidx))
    with pytest.raises(NotImplementedError):
        cudf.Series(pd.array(pidx))


@pytest.mark.parametrize("value", [cudf.DataFrame(range(1)), 11])
def test_index_from_dataframe_scalar_raises(value):
    with pytest.raises(TypeError):
        cudf.Index(value)


@pytest.mark.parametrize(
    "data",
    [
        cp.ones(5, dtype=cp.float16),
        np.ones(5, dtype="float16"),
        pd.Series([0.1, 1.2, 3.3], dtype="float16"),
        pytest.param(
            pa.array(np.ones(5, dtype="float16")),
            marks=pytest.mark.xfail(
                reason="https://issues.apache.org/jira/browse/ARROW-13762"
            ),
        ),
    ],
)
def test_index_raises_float16(data):
    with pytest.raises(TypeError):
        cudf.Index(data)


def test_from_pandas_rangeindex_return_rangeindex():
    pidx = pd.RangeIndex(start=3, stop=9, step=3, name="a")
    result = cudf.Index(pidx)
    expected = cudf.RangeIndex(start=3, stop=9, step=3, name="a")
    assert_eq(result, expected, exact=True)


def test_Index_init_with_nans():
    with cudf.option_context("mode.pandas_compatible", True):
        gi = cudf.Index([1, 2, 3, np.nan])
    assert gi.dtype == np.dtype("float64")
    pi = pd.Index([1, 2, 3, np.nan])
    assert_eq(pi, gi)


def test_roundtrip_index_plc_column():
    index = cudf.Index([1])
    expect = cudf.Index(index)
    actual = cudf.Index.from_pylibcudf(*expect.to_pylibcudf())
    assert_eq(expect, actual)


def test_categorical_index_with_dtype():
    dtype = cudf.CategoricalDtype(categories=["a", "z", "c"])
    gi = cudf.Index(["z", "c", "a"], dtype=dtype)
    pi = pd.Index(["z", "c", "a"], dtype=dtype.to_pandas())

    assert_eq(gi, pi)
    assert_eq(gi.dtype, pi.dtype)
    assert_eq(gi.dtype.categories, pi.dtype.categories)
