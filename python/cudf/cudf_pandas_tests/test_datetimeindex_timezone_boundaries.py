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


@pytest.mark.skipif(
    not PANDAS_GE_31, reason="pandas 3.1 timezone bounds checks"
)
@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
def test_index_localization_rejects_nat_sentinel(unit):
    offset = np.timedelta64(9 * 3600 + 18 * 60 + 59, "s")
    ticks = np.iinfo(np.int64).min + int(offset // np.timedelta64(1, unit))
    values = np.array([ticks], dtype="int64").view(f"M8[{unit}]")
    expected = pd.DatetimeIndex(values)
    actual = xpd.DatetimeIndex(values)
    with pytest.raises(pd.errors.OutOfBoundsDatetime, match="underflows past"):
        expected.tz_localize("Asia/Tokyo")
    with pytest.raises(pd.errors.OutOfBoundsDatetime, match="underflows past"):
        actual.tz_localize("Asia/Tokyo")


@pytest.mark.skipif(
    not PANDAS_GE_31, reason="pandas 3.1 timezone bounds checks"
)
@pytest.mark.parametrize(
    "ticks,timezone",
    [
        (np.iinfo(np.int64).max, "Etc/GMT-9"),
        (np.iinfo(np.int64).min + 1, "US/Pacific"),
    ],
)
def test_index_removing_timezone_rejects_wrap(ticks, timezone):
    values = np.array([ticks], dtype="int64").view("M8[ns]")
    expected = pd.DatetimeIndex(values).tz_localize("UTC").tz_convert(timezone)
    actual = xpd.DatetimeIndex(values).tz_localize("UTC").tz_convert(timezone)
    with pytest.raises(pd.errors.OutOfBoundsDatetime):
        expected.tz_localize(None)
    with pytest.raises(pd.errors.OutOfBoundsDatetime):
        actual.tz_localize(None)


def test_index_normal_timezone_localization_keeps_gpu():
    values = ["2024-01-01", "2024-01-02"]
    expected = pd.DatetimeIndex(values).tz_localize("Europe/Berlin")
    actual = xpd.DatetimeIndex(values).tz_localize("Europe/Berlin")
    assert isinstance(actual._fsproxy_wrapped, cudf.DatetimeIndex)
    assert_eq(expected, actual._fsproxy_slow)


@pytest.mark.skipif(
    not PANDAS_GE_31, reason="pandas 3.1 anchored frequency preservation"
)
def test_index_localization_preserves_weekly_frequency():
    expected = pd.date_range(
        "2020-03-01", periods=8, freq="W-SUN"
    ).tz_localize("Europe/London")
    actual = xpd.date_range("2020-03-01", periods=8, freq="W-SUN").tz_localize(
        "Europe/London"
    )
    assert isinstance(actual._fsproxy_wrapped, cudf.DatetimeIndex)
    assert_eq(expected, actual._fsproxy_slow)
    assert actual.freq == expected.freq
