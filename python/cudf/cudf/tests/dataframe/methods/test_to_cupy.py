# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import cupy as cp
import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


def test_empty_dataframe_to_cupy():
    df = cudf.DataFrame()

    # Check fully empty dataframe.
    mat = df.to_cupy()
    assert mat.shape == (0, 0)
    mat = df.to_numpy()
    assert mat.shape == (0, 0)

    rng = np.random.default_rng(seed=0)
    df = cudf.DataFrame(
        {
            "a": rng.random(10),
            "b": rng.random(10),
            "c": rng.random(10),
        }
    )

    # Check all columns in empty dataframe.
    mat = df.head(0).to_cupy()
    assert mat.shape == (0, 3)


def test_dataframe_to_cupy():
    nelem = 123
    rng = np.random.default_rng(seed=0)
    df = cudf.DataFrame(
        {
            "a": rng.random(nelem),
            "b": rng.random(nelem),
            "c": rng.random(nelem),
            "d": rng.random(nelem),
        }
    )

    # Check all columns
    mat = df.to_cupy()
    assert mat.shape == (nelem, 4)
    assert mat.strides == (8, 984)

    mat = df.to_numpy()
    assert mat.shape == (nelem, 4)
    assert mat.strides == (8, 984)
    for i, k in enumerate(df.columns):
        np.testing.assert_array_equal(df[k].to_numpy(), mat[:, i])

    # Check column subset
    mat = df[["a", "c"]].to_cupy().get()
    assert mat.shape == (nelem, 2)

    for i, k in enumerate("ac"):
        np.testing.assert_array_equal(df[k].to_numpy(), mat[:, i])


@pytest.mark.parametrize("in_dtype", ["int32", "int64", "float32", "float64"])
@pytest.mark.parametrize("out_dtype", ["int32", "int64", "float32", "float64"])
def test_dataframe_to_cupy_dtype(in_dtype, out_dtype):
    data = np.arange(12, dtype=in_dtype).reshape(3, 4)
    df = cudf.DataFrame(data)

    result = df.to_cupy(dtype=out_dtype)

    assert result.dtype == np.dtype(out_dtype)
    np.testing.assert_allclose(result.get(), data.astype(out_dtype))


@pytest.mark.parametrize("has_nulls", [False, True])
@pytest.mark.parametrize("use_na_value", [False, True])
def test_dataframe_to_cupy_single_column(has_nulls, use_na_value):
    nelem = 10
    data = np.arange(nelem, dtype=np.float64)

    if has_nulls:
        data = data.astype("object")
        data[::2] = None

    df = cudf.DataFrame({"a": data})

    if has_nulls and not use_na_value:
        result = df.to_cupy()
        expected = df.fillna(np.nan).to_cupy()
        assert_eq(result, expected)
        return

    na_value = 0.0 if use_na_value else None
    expected = (
        cp.asarray(df["a"].fillna(na_value))
        if has_nulls
        else cp.asarray(df["a"])
    )
    result = df.to_cupy(na_value=na_value)
    assert result.shape == (nelem, 1)
    assert_eq(result.ravel(), expected)


def test_dataframe_to_cupy_null_values():
    df = cudf.DataFrame()

    nelem = 123
    na = -10000

    refvalues = {}
    rng = np.random.default_rng(seed=0)
    for k in "abcd":
        data = rng.random(nelem)
        boolmask = rng.choice([True, False], size=nelem)
        data[~boolmask] = na
        df[k] = data
        df.loc[~boolmask, k] = None
        refvalues[k] = data

    result = df.to_cupy()
    expected = df.fillna(np.nan).to_cupy()
    assert_eq(result, expected)

    result = df.to_numpy()
    expected = df.fillna(np.nan).to_numpy()
    assert_eq(result, expected)

    for k in df.columns:
        df[k] = df[k].fillna(na)

    mat = df.to_numpy()
    for i, k in enumerate(df.columns):
        np.testing.assert_array_equal(refvalues[k], mat[:, i])


@pytest.mark.parametrize("method", ["to_cupy", "to_numpy"])
@pytest.mark.parametrize("value", [1, True, 1.5])
@pytest.mark.parametrize("constructor", ["DataFrame", "Series"])
def test_to_array_categorical(method, value, constructor):
    data = [value]
    expected = getattr(pd, constructor)(data, dtype="category").to_numpy()
    result = getattr(
        getattr(cudf, constructor)(data, dtype="category"), method
    )()
    assert_eq(result, expected)


@pytest.mark.parametrize("method", ["to_cupy", "to_numpy"])
@pytest.mark.parametrize(
    "source,out_dtype",
    [
        (cudf.Series(["1", "2", "3"]), "float32"),
        (cudf.Series(["1", "2", "3"]), "int64"),
        (
            cudf.Series(["2020-01-01", "2020-01-02"], dtype="datetime64[ns]"),
            "int64",
        ),
        (cudf.Series([True, False, True]), "int32"),
        (cudf.Series([1, 2, 3], dtype="int32"), "float64"),
    ],
)
def test_to_array_dtype_matches_astype(method, source, out_dtype):
    expected = getattr(source.astype(out_dtype), method)()
    result = getattr(source, method)(dtype=out_dtype)
    assert str(result.dtype) == out_dtype
    assert_eq(result, expected)


@pytest.mark.parametrize("constructor", ["DataFrame", "Series"])
@pytest.mark.parametrize(
    "data,dtype",
    [
        ([1, 2, 3], "int64"),
        ([1.5, 2.5, 3.5], "float64"),
        ([True, False, True], "bool"),
    ],
)
def test_to_numpy_object_dtype_boxes_values(constructor, data, dtype):
    # requesting dtype=object from to_numpy should box the
    # native Python values, matching pandas — not stringify them (which
    # would happen if we routed through cudf.dtype(object), the
    # string dtype).
    pd_obj = getattr(pd, constructor)(data, dtype=dtype)
    cudf_obj = getattr(cudf, constructor)(data, dtype=dtype)

    expected = pd_obj.to_numpy(dtype=object)
    result = cudf_obj.to_numpy(dtype=object)

    assert result.dtype == np.dtype("O")
    assert_eq(result, expected)
    # Ensure boxed values are real Python scalars, not strings.
    for value in result.flat:
        assert not isinstance(value, str)


@pytest.mark.parametrize("constructor", ["DataFrame", "Series"])
@pytest.mark.parametrize(
    "data",
    [
        [],
        [None],
        [None, None],
        ["a", "b", None],
    ],
)
def test_to_numpy_object_dtype_preserves_none_string_nulls(constructor, data):
    values = pd.Series(data, dtype=object, name="x")
    if constructor == "DataFrame":
        pd_obj = pd.DataFrame({"x": values})
    else:
        pd_obj = values
    cudf_obj = getattr(cudf, constructor)(pd_obj)

    expected = pd_obj.to_numpy(dtype=object)
    result = cudf_obj.to_numpy(dtype=object)
    null_mask = pd.isna(expected)

    assert result.dtype == np.dtype("O")
    np.testing.assert_array_equal(result, expected)
    assert all(value is None for value in result[null_mask].flat)

    expected = pd_obj.to_numpy(dtype=object, na_value="missing")
    result = cudf_obj.to_numpy(dtype=object, na_value="missing")
    np.testing.assert_array_equal(result, expected)
    assert all(value == "missing" for value in result[null_mask].flat)
