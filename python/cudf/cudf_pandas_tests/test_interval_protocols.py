# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re

import pandas as pd
import pyarrow as pa
import pytest

from cudf.core._compat import PANDAS_GE_31
from cudf.pandas.fast_slow_proxy import _slow_arg
from cudf.pandas.module_accelerator import disable_module_accelerator


@pytest.mark.parametrize("timezone", [None, "Europe/Brussels"])
@pytest.mark.parametrize("closed", ["left", "right", "both", "neither"])
@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("sliced", [False, True])
def test_interval_arrow_protocol(timezone, closed, missing, sliced):
    def make_array():
        result = pd.arrays.IntervalArray.from_breaks(
            pd.date_range("2017", periods=5, freq="D", tz=timezone),
            closed=closed,
        )
        if missing:
            result[2] = None
        return result[1:] if sliced else result

    try:
        with disable_module_accelerator():
            expected = pa.array(make_array())
    except TypeError as error:
        # pandas 3.0 does not support timezone-aware interval conversion.
        with pytest.raises(TypeError, match=re.escape(str(error))):
            pa.array(make_array())
        return

    actual = pa.array(make_array())
    assert actual.equals(expected)


@pytest.mark.skipif(
    not PANDAS_GE_31, reason="Index arrays become read-only in pandas 3.1"
)
@pytest.mark.parametrize("kind", ["interval", "datetime", "timedelta"])
@pytest.mark.parametrize("operation", ["setitem", "mask"])
def test_series_sharing_readonly_index_array(kind, operation):
    if kind == "interval":
        index = pd.interval_range(0, 3)
        value = pd.Interval(8, 9)
    elif kind == "datetime":
        index = pd.date_range("2017", periods=3)
        value = pd.Timestamp("2020")
    else:
        index = pd.timedelta_range("0 days", periods=3)
        value = pd.Timedelta("8 days")
    expected_index = _slow_arg(index).copy(deep=True)
    result = pd.Series(index.array, copy=False)

    with pytest.raises(ValueError, match="read-only"):
        if operation == "setitem":
            result.iloc[0] = value
        else:
            result.mask(result == index[0], value, inplace=True)

    with disable_module_accelerator():
        pd.testing.assert_index_equal(_slow_arg(index), expected_index)
