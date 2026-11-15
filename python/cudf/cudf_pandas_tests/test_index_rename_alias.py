# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as xpd
import pytest

import cudf
from cudf.pandas import LOADED
from cudf.testing import assert_eq

if not LOADED:
    raise ImportError("These tests must be run with cudf.pandas loaded")

pd = xpd._fsproxy_slow


def _index(kind):
    if kind == "range":
        return pd.RangeIndex(2, name="before")
    if kind == "datetime":
        return pd.DatetimeIndex(["2020-01-01", "2020-01-02"], name="before")
    if kind == "timedelta":
        return pd.TimedeltaIndex(["0D", "1D"], name="before")
    if kind == "categorical":
        return pd.CategoricalIndex(["a", "b"], name="before")
    if kind == "interval":
        return pd.IntervalIndex.from_breaks([0, 1, 2], name="before")
    return pd.Index(["a", "b"], name="before")


@pytest.mark.parametrize(
    "kind", ["base", "range", "datetime", "timedelta", "interval"]
)
@pytest.mark.parametrize("inplace", [False, True])
def test_categorical_categories_rename_updates_cpu_owner(kind, inplace):
    expected = pd.Categorical.from_codes([0, 1], categories=_index(kind))
    actual = xpd.Categorical.from_codes([0, 1], categories=_index(kind))
    expected_result = expected.categories.rename(None, inplace=inplace)
    result = actual.categories.rename(None, inplace=inplace)
    assert actual._fsproxy_wrapped.categories.name == expected.categories.name
    pd.testing.assert_series_equal(
        pd.Series(expected), pd.Series(actual._fsproxy_slow)
    )
    if inplace:
        assert result is None
    else:
        assert_eq(expected_result, result._fsproxy_slow)
        assert actual._fsproxy_wrapped.categories.name == "before"


@pytest.mark.parametrize(
    "kind",
    ["base", "range", "datetime", "timedelta", "categorical", "interval"],
)
@pytest.mark.parametrize("inplace", [False, True])
def test_index_rename_preserves_gpu_owner(kind, inplace):
    expected = pd.Series([1, 2], index=_index(kind))
    actual = xpd.Series([1, 2], index=xpd.Index(_index(kind)))
    index = actual.index
    assert isinstance(index._fsproxy_wrapped, cudf.Index)
    expected_result = expected.index.rename(None, inplace=inplace)
    result = index.rename(None, inplace=inplace)
    assert isinstance(actual._fsproxy_wrapped, cudf.Series)
    assert isinstance(index._fsproxy_wrapped, cudf.Index)
    assert_eq(expected, actual._fsproxy_slow)
    if inplace:
        assert result is None
    else:
        assert_eq(expected_result, result._fsproxy_slow)
        assert index.name == "before"


def test_categories_from_named_series_can_rename_inplace():
    expected = pd.Series(["1", "2", "3"], name="unlabeled")
    actual = xpd.Series(["1", "2", "3"], name="unlabeled")
    expected_cat = (
        expected.astype("category")
        .array.as_ordered()
        .set_categories(expected, ordered=True)
    )
    actual_cat = (
        actual.astype("category")
        .array.as_ordered()
        .set_categories(actual, ordered=True)
    )
    expected_cat.categories.rename(None, inplace=True)
    actual_cat.categories.rename(None, inplace=True)
    pd.testing.assert_series_equal(
        pd.Series(expected_cat), pd.Series(actual_cat._fsproxy_slow)
    )
