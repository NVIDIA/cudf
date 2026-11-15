# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cupy as cp
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import cudf
from cudf.core._compat import PANDAS_GE_31
from cudf.testing import assert_eq


@pytest.mark.skipif(not PANDAS_GE_31, reason="pandas 3.1 dtype selectors")
@pytest.mark.parametrize("kind", ["category", "interval"])
@pytest.mark.parametrize("selector", ["class", "family", "exact", "mixed"])
@pytest.mark.parametrize("keyword", ["include", "exclude"])
def test_select_dtypes_native_dtype_specs(kind, selector, keyword):
    pdf = pd.DataFrame(
        {
            "category": pd.Categorical(["a", "b", "a"]),
            "ordered": pd.Categorical(["a", "b", "a"], ordered=True),
            "right": pd.arrays.IntervalArray.from_breaks([0, 1, 2, 3]),
            "left": pd.arrays.IntervalArray.from_breaks(
                [0, 1, 2, 3], closed="left"
            ),
            "integer": [1, 2, 3],
        }
    )
    dtype_name = "CategoricalDtype" if kind == "category" else "IntervalDtype"
    pd_type = getattr(pd, dtype_name)
    gd_type = getattr(cudf, dtype_name)
    if selector == "exact":
        args = (["a", "b"],) if kind == "category" else ("int64", "right")
        pd_spec, gd_spec = pd_type(*args), gd_type(*args)
    elif selector == "family":
        pd_spec, gd_spec = pd_type(), gd_type()
    else:
        pd_spec, gd_spec = pd_type, gd_type
        if selector == "mixed":
            pd_spec, gd_spec = [pd_spec, "int64"], [gd_spec, "int64"]
    assert_eq(
        pdf.select_dtypes(**{keyword: pd_spec}),
        cudf.from_pandas(pdf).select_dtypes(**{keyword: gd_spec}),
    )


@pytest.mark.skipif(not PANDAS_GE_31, reason="pandas 3.1 dtype selectors")
def test_select_dtypes_nested_object_metadata():
    pdf = pd.DataFrame(
        {
            "string": pd.array(["x", "y"], dtype="str"),
            "object": np.array([{"k": 1}, None], dtype=object),
            "arrow": pd.array(
                [{"k": 1}, None],
                dtype=pd.ArrowDtype(pa.struct([("k", pa.int64())])),
            ),
        }
    )
    with cudf.option_context("mode.pandas_compatible", True):
        got = cudf.from_pandas(pdf).select_dtypes(
            include=object, exclude=pd.api.extensions.ExtensionDtype
        )
    assert list(got.columns) == ["object"]
    assert_eq(pdf[["object"]], got)


@pytest.mark.skipif(
    not PANDAS_GE_31, reason="pandas 3.1 safe integer inference"
)
@pytest.mark.parametrize("kind", ["Series", "DataFrame"])
@pytest.mark.parametrize(
    "value",
    [float(2**63), float(-(2**63)), np.nextafter(float(2**63), 0), 1.00000001],
)
def test_convert_dtypes_exact_int64_bounds(kind, value):
    data = [value, None]
    if kind == "DataFrame":
        data = {"a": data}
    expected = getattr(pd, kind)(data)
    got = cudf.from_pandas(expected).convert_dtypes()
    assert_eq(expected.convert_dtypes(), got)


@pytest.mark.skipif(not PANDAS_GE_31, reason="pandas 3.1 0D input validation")
@pytest.mark.parametrize("array_module", [np, cp])
def test_datetimeindex_rejects_zero_dimensional_input(array_module):
    with pytest.raises(ValueError, match="0-dim input"):
        cudf.DatetimeIndex(array_module.array(0, dtype="int64"))


@pytest.mark.parametrize(
    "range_args", [(0, 10, 2), (-13, 17, 4), (7, -14, -3), (0, 0, 1)]
)
@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("value", [-20, -1, 0, 7, 20])
def test_rangeindex_searchsorted_integer_controls(range_args, side, value):
    gidx = cudf.RangeIndex(*range_args)
    ascending = gidx.step > 0
    ordered = np.asarray(list(range(*range_args)))
    if not ascending:
        ordered = ordered[::-1]
    expected = np.searchsorted(
        ordered,
        value,
        side=side if ascending else {"left": "right", "right": "left"}[side],
    )
    if not ascending:
        expected = len(gidx) - expected
    assert gidx.searchsorted(value, side=side, ascending=ascending) == expected


@pytest.mark.skipif(
    not PANDAS_GE_31, reason="pandas 3.1 integer search positions"
)
@pytest.mark.parametrize(
    "range_args", [(0, 10, 2), (-13, 17, 4), (7, -14, -3), (0, 0, 1)]
)
@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize(
    "value", [-np.inf, -12.5, -0.1, 0.0, 6.5, np.inf, np.nan]
)
@pytest.mark.parametrize("na_position", ["first", "last"])
def test_rangeindex_searchsorted_float(range_args, side, value, na_position):
    gidx = cudf.RangeIndex(*range_args)
    ascending = gidx.step > 0
    ordered = np.asarray(list(range(*range_args)))
    if not ascending:
        ordered = ordered[::-1]
    expected = np.searchsorted(
        ordered,
        value,
        side=side if ascending else {"left": "right", "right": "left"}[side],
    )
    if not ascending:
        expected = len(gidx) - expected
    got = gidx.searchsorted(
        value, side=side, ascending=ascending, na_position=na_position
    )
    assert got == expected
    assert got == cudf.Index(
        list(range(*range_args)), dtype="int64"
    ).searchsorted(
        value, side=side, ascending=ascending, na_position=na_position
    )


@pytest.mark.parametrize("storage", ["python", "pyarrow"])
def test_get_dummies_all_string_storages(storage):
    dtype = pd.StringDtype(storage=storage, na_value=np.nan)
    pdf = pd.DataFrame(
        {"a": pd.array(["x", "y", "x"], dtype=dtype), "b": [1, 2, 3]}
    )
    assert_eq(pd.get_dummies(pdf), cudf.get_dummies(cudf.from_pandas(pdf)))


@pytest.mark.parametrize("name", [None, "name"])
@pytest.mark.parametrize("timezone", [None, "US/Eastern"])
@pytest.mark.parametrize("sliced", [False, True])
def test_datetimeindex_quarter_preserves_name(name, timezone, sliced):
    expected = pd.date_range(
        "2020-01-01", periods=5, freq="90D", tz=timezone, name=name
    )
    if sliced:
        expected = expected[1:]
    actual = cudf.from_pandas(expected).quarter
    assert actual.name == expected.name
    # Native quarter fields use int8 rather than pandas' int32.
    assert_eq(expected.quarter, actual, exact=False)
