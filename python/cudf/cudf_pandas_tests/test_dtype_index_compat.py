# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as xpd
import pytest

import cudf
from cudf.core._compat import PANDAS_GE_31
from cudf.pandas import LOADED
from cudf.testing import assert_eq

if not LOADED:
    raise ImportError("These tests must be run with cudf.pandas loaded")

pd = xpd._fsproxy_slow


@pytest.mark.parametrize(
    "range_args", [(0, 12, 2), (-13, 17, 4), (7, -14, -3)]
)
@pytest.mark.parametrize(
    "bounds", [(-12.5, 6.5), (-0.1, np.inf), (np.inf, -np.inf)]
)
@pytest.mark.parametrize("kind", ["Series", "DataFrame"])
def test_rangeindex_float_label_slice(range_args, bounds, kind):
    if not PANDAS_GE_31 and range_args[2] < 0 and bounds == (np.inf, -np.inf):
        pytest.skip("Existing descending infinite-bound slice limitation")
    index = pd.RangeIndex(*range_args)
    data = list(range(len(index)))
    if kind == "DataFrame":
        data = {"a": data}
    expected = getattr(pd, kind)(data, index=index).loc[slice(*bounds)]
    actual = getattr(xpd, kind)(data, index=xpd.RangeIndex(*range_args)).loc[
        slice(*bounds)
    ]
    if PANDAS_GE_31:
        assert isinstance(actual._fsproxy_wrapped, getattr(cudf, kind))
    assert_eq(expected, actual._fsproxy_slow)


@pytest.mark.parametrize("storage", ["python", "pyarrow"])
def test_get_dummies_string_storage(storage):
    dtype = pd.StringDtype(storage=storage, na_value=np.nan)
    expected = pd.DataFrame({"a": pd.array(["x", "y", "x"], dtype=dtype)})
    actual = xpd.DataFrame({"a": xpd.array(["x", "y", "x"], dtype=dtype)})
    result = xpd.get_dummies(actual)
    assert isinstance(result._fsproxy_wrapped, cudf.DataFrame)
    assert_eq(pd.get_dummies(expected), result._fsproxy_slow)


@pytest.mark.skipif(not PANDAS_GE_31, reason="pandas 3.1 integer inference")
def test_convert_dtypes_float_int64_overflow():
    values = [float(2**63), None]
    expected = pd.Series(values).convert_dtypes()
    actual = xpd.Series(values).convert_dtypes()
    assert isinstance(actual._fsproxy_wrapped, cudf.Series)
    assert_eq(expected, actual._fsproxy_slow)


@pytest.mark.parametrize("name", [None, "name"])
@pytest.mark.parametrize("timezone", [None, "US/Eastern"])
def test_datetimeindex_quarter_preserves_name(name, timezone):
    expected = pd.date_range(
        "2020-01-01", periods=5, freq="90D", tz=timezone, name=name
    ).quarter
    actual = xpd.date_range(
        "2020-01-01", periods=5, freq="90D", tz=timezone, name=name
    ).quarter
    assert isinstance(actual._fsproxy_wrapped, cudf.Index)
    assert actual.name == expected.name
    assert_eq(expected, actual._fsproxy_slow, exact=False)
