# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from cudf.pandas import LOADED

if not LOADED:
    raise ImportError("These tests must be run with cudf.pandas loaded")

import numpy as np
import pandas as pd


@pytest.fixture(autouse=True)
def fail_on_fallback(monkeypatch):
    monkeypatch.setenv("CUDF_PANDAS_FAIL_ON_FALLBACK", "True")


@pytest.fixture
def dataframe():
    df = pd.DataFrame(
        {
            "a": [1, 1, 1, 2, 3],
            "b": [1, 2, 3, 4, 5],
            "c": [1.2, 1.3, 1.5, 1.7, 1.11],
        }
    )
    return df


@pytest.fixture
def series(dataframe):
    return dataframe["a"]


@pytest.fixture
def array(series):
    return series.values


@pytest.mark.parametrize(
    "op",
    [
        "sum",
        "min",
        "max",
        "mean",
        "std",
        "var",
        "prod",
        "median",
    ],
)
def test_no_fallback_in_reduction_ops(series, op):
    s = series
    getattr(s, op)()


def test_groupby(dataframe):
    df = dataframe
    df.groupby("a", sort=True).max()


def test_no_fallback_in_binops(dataframe):
    df = dataframe
    df + df
    df - df
    df * df
    df**df
    df[["a", "b"]] & df[["a", "b"]]
    df <= df


@pytest.mark.parametrize(
    "agg_name",
    [
        "sum",
        "mean",
        "min",
        "max",
        "count",
        "std",
        "var",
        "median",
    ],
)
def test_no_fallback_in_groupby_rolling_aggs(request, dataframe, agg_name):
    if agg_name == "median":
        request.applymarker(
            pytest.mark.xfail(
                reason="groupby().rolling().median() not yet implemented"
            )
        )
    df = dataframe
    getattr(df.groupby("a").rolling(2), agg_name)()


def test_no_fallback_in_rolling_apply(series, dataframe):
    def sum_minus_min(a):
        m = a[0]
        s = 0.0
        for v in a:
            s += v
            if v < m:
                m = v
        return s - m

    series.rolling(2, min_periods=1).apply(sum_minus_min, raw=True)
    dataframe.rolling(2, min_periods=1).apply(sum_minus_min, raw=True)


def test_no_fallback_in_concat(dataframe):
    df = dataframe
    pd.concat([df, df])


def test_no_fallback_in_get_shape(dataframe):
    df = dataframe
    df.shape


@pytest.mark.parametrize("mixed_dtypes", [False, True])
@pytest.mark.parametrize(
    "key", [[], [False, False], slice(0, -1), np.array([])]
)
def test_loc_setitem_empty_selection(key, mixed_dtypes):
    data = {"a": [1, 2]}
    if mixed_dtypes:
        data["b"] = ["x", "y"]
    df = pd.DataFrame(data)
    expected = df.copy()

    df.loc[key] = 1.5

    pd.testing.assert_frame_equal(df, expected, check_exact=True)


def test_no_fallback_in_array_ufunc_op(array):
    np.add(array, array)


def test_no_fallback_in_merge(dataframe):
    df = dataframe
    pd.merge(df * df, df + df, how="inner")
    pd.merge(df * df, df + df, how="outer")
    pd.merge(df * df, df + df, how="left")
    pd.merge(df * df, df + df, how="right")


def test_no_fallback_in_memory_usage_and_sizeof(dataframe, series):
    df = dataframe
    s = series
    i = df.index

    df.memory_usage()
    df.memory_usage(index=False)
    df.memory_usage(index=True, deep=True)
    df.__sizeof__()

    s.memory_usage()
    s.memory_usage(index=False)
    s.memory_usage(index=True, deep=True)
    s.__sizeof__()

    i.memory_usage()
    with pytest.warns(UserWarning, match="The deep parameter is ignored"):
        i.memory_usage(deep=True)
        i.__sizeof__()


def test_no_fallback_in_named_agg(dataframe):
    df = dataframe
    df.groupby("a").agg(
        min_b=pd.NamedAgg(column="b", aggfunc="min"),
        max_c=pd.NamedAgg(column="c", aggfunc="max"),
    )
