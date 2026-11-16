# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as xpd
import pyarrow as pa
import pytest

from cudf.core._compat import PANDAS_GE_31
from cudf.pandas import LOADED
from cudf.pandas.module_accelerator import disable_module_accelerator
from cudf.testing import assert_eq

if not LOADED:
    raise ImportError("These tests must be run with cudf.pandas loaded")

pd = xpd._fsproxy_slow


@pytest.mark.skipif(
    not PANDAS_GE_31, reason="pandas 3.1 no-op index sort shares storage"
)
@pytest.mark.parametrize("ascending", [False, True])
def test_monotonic_index_sort_shares_storage(ascending):
    index = [0, 1, 2] if ascending else [2, 1, 0]
    source = xpd.DataFrame(
        {"a": [0.0, 1.0, 2.0]}, index=xpd.Index(index, name="rows")
    )
    result = source.sort_index(level="rows", ascending=ascending)
    assert np.shares_memory(source["a"].values, result["a"].values)
    result.iloc[0, 0] = 99
    assert source.iloc[0, 0] == 0


@pytest.mark.parametrize("dtype", [">i4", ">i8", ">f8"])
@pytest.mark.parametrize("box", ["series", "frame", "dict"])
def test_non_native_byteorder_constructor(dtype, box):
    def construct(module):
        values = np.array([0, 256], dtype=dtype)
        series = module.Series(values, dtype=dtype)
        if box == "series":
            return series
        if box == "dict":
            return module.DataFrame({"a": series, "b": series.copy()})
        return module.DataFrame(pd.DataFrame({"a": pd.Series(values)}))

    expected = construct(pd)
    actual = construct(xpd)
    assert_eq(actual._fsproxy_slow, expected)
    selected = actual if box == "series" else actual["a"]
    expected_selected = expected if box == "series" else expected["a"]
    assert selected.dtype == expected_selected.dtype
    assert selected.tolist() == [0, 256]


@pytest.mark.parametrize("box", [list, tuple])
@pytest.mark.parametrize("dtype", [None, "datetime64[ns]"])
def test_high_precision_datetime_constructor(box, dtype):
    values = box(
        [
            "2017-01-01 12:00:00.000000001",
            None,
            "2017-01-02 12:00:00.000000009",
        ]
    )
    expected = pd.DatetimeIndex(values, dtype=dtype)
    actual = xpd.DatetimeIndex(values, dtype=dtype)
    assert_eq(actual._fsproxy_slow, expected)
    assert actual.dtype == expected.dtype


@pytest.mark.skipif(
    not PANDAS_GE_31, reason="pandas 3.1 explicit tick frequency"
)
@pytest.mark.parametrize("frequency", ["24h", "48h", "120min"])
def test_datetime_constructor_preserves_requested_frequency(frequency):
    expected = pd.DatetimeIndex(
        pd.date_range("2000-01-01", periods=5, freq=frequency), freq=frequency
    )
    actual = xpd.DatetimeIndex(
        xpd.date_range("2000-01-01", periods=5, freq=frequency), freq=frequency
    )
    assert actual.freqstr == expected.freqstr
    assert_eq(actual._fsproxy_slow, expected)


@pytest.mark.skipif(
    not PANDAS_GE_31, reason="pandas 3.1 constructor axis isolation"
)
def test_shared_constructor_axes_have_independent_names():
    labels = xpd.Index(["a", "b"])
    result = xpd.DataFrame([[1, 2], [3, 4]], index=labels, columns=labels)
    result.index.name = "rows"
    assert result.columns.name is None
    assert labels.name is None


@pytest.mark.skipif(
    not PANDAS_GE_31, reason="pandas 3.1 categorical repr layout"
)
@pytest.mark.parametrize(
    "values", [["a", "bb", "ccc"], ["あ", "いい", "ううう"]]
)
@pytest.mark.parametrize("name", [None, "letters"])
def test_truncated_categorical_index_repr(values, name):
    expected = pd.CategoricalIndex(values * 100, name=name)
    actual = xpd.CategoricalIndex(values * 100, name=name)
    assert repr(actual) == repr(expected)


@pytest.mark.parametrize("name", [0.55555555, np.float32(0.55555555)])
def test_float_series_name_display_precision(name):
    expected = pd.Series([1.0], name=name)
    actual = xpd.Series([1.0], name=name)
    with pd.option_context("display.precision", 3):
        assert repr(actual) == repr(expected)


@pytest.mark.skipif(
    not PANDAS_GE_31, reason="pandas 3.1 Arrow conversion policy"
)
def test_from_arrow_uses_pandas_conversion_policy():
    table = pa.table({"a": pa.array(["a", "b", None])})
    with pytest.warns(pd.errors.Pandas4Warning, match="future.infer_string"):
        with pd.option_context("future.infer_string", False):
            with disable_module_accelerator():
                expected = pd.DataFrame.from_arrow(table)
            actual = xpd.DataFrame.from_arrow(table)
    with disable_module_accelerator():
        pd.testing.assert_frame_equal(actual._fsproxy_slow, expected)
    assert actual["a"].dtype == expected["a"].dtype
