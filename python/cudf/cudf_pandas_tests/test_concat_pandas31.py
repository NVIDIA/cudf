# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import zoneinfo

import pandas as xpd
import pytest

import cudf
from cudf.core._compat import PANDAS_GE_31
from cudf.pandas import LOADED
from cudf.testing import assert_eq

if not LOADED:
    raise ImportError("These tests must be run with cudf.pandas loaded")

pd = xpd._fsproxy_slow


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("ignore_index", [False, True])
@pytest.mark.skipif(
    not PANDAS_GE_31, reason="pandas 3.1 object timestamp reductions"
)
def test_concat_aware_and_naive_nat(reverse, ignore_index):
    timestamp = pd.Timestamp(
        "2024-05-13 12:00:00", tz=zoneinfo.ZoneInfo("America/New_York")
    )
    expected_inputs = [pd.Series([timestamp]), pd.Series([pd.NaT])]
    actual_inputs = [xpd.Series([timestamp]), xpd.Series([pd.NaT])]
    if reverse:
        expected_inputs.reverse()
        actual_inputs.reverse()
    expected = pd.concat(expected_inputs, ignore_index=ignore_index)
    actual = xpd.concat(actual_inputs, ignore_index=ignore_index)
    assert_eq(expected, actual._fsproxy_slow)
    assert actual.min() == expected.min() == timestamp
    assert actual.max() == expected.max() == timestamp


def test_concat_same_timezone_retains_gpu():
    first = xpd.Series(
        xpd.date_range("2024-05-13", periods=1, tz=zoneinfo.ZoneInfo("UTC"))
    )
    second = xpd.Series(
        xpd.date_range("2024-05-14", periods=1, tz=zoneinfo.ZoneInfo("UTC"))
    )
    result = xpd.concat([first, second], ignore_index=True)
    assert isinstance(result._fsproxy_wrapped, cudf.Series)
    expected = pd.concat(
        [first._fsproxy_slow, second._fsproxy_slow], ignore_index=True
    )
    assert_eq(expected, result._fsproxy_slow)
