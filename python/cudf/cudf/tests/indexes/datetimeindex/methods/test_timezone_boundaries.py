# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.core._compat import PANDAS_GE_31
from cudf.testing import assert_eq


@pytest.mark.skipif(
    not PANDAS_GE_31, reason="pandas 3.1 timezone bounds checks"
)
@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
def test_timezone_bounds_fallback_before_localization(unit):
    offset = np.timedelta64(9 * 3600 + 18 * 60 + 59, "s")
    ticks = np.iinfo(np.int64).min + int(offset // np.timedelta64(1, unit))
    index = cudf.DatetimeIndex(
        np.array([ticks], dtype="int64").view(f"M8[{unit}]")
    )
    with cudf.option_context("mode.pandas_compatible", True):
        with pytest.raises(NotImplementedError, match="near datetime bounds"):
            index.tz_localize("Asia/Tokyo")
    np.testing.assert_array_equal(
        index._column.astype(np.dtype("int64")).to_numpy(),
        np.array([ticks], dtype="int64"),
    )


@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
def test_timezone_localization_normal_range_stays_supported(unit):
    expected = pd.DatetimeIndex(["2024-01-01", "2024-01-02"]).as_unit(unit)
    actual = cudf.from_pandas(expected)
    with cudf.option_context("mode.pandas_compatible", True):
        result = actual.tz_localize("Europe/Berlin")
    assert_eq(expected.tz_localize("Europe/Berlin"), result)


@pytest.mark.skipif(
    not PANDAS_GE_31, reason="pandas 3.1 anchored frequency preservation"
)
@pytest.mark.parametrize(
    "freq,start", [("W-SUN", "2020-03-01"), ("2W-MON", "2020-03-02")]
)
def test_weekly_frequency_retains_anchor(freq, start):
    index = cudf.date_range(start, periods=3, freq=freq)
    expected = pd.date_range(start, periods=3, freq=freq)
    expected_freq = pd.tseries.frequencies.to_offset(freq)
    assert_eq(expected, index)
    assert index.freq == expected_freq
    result = index.tz_localize("Europe/London")
    assert_eq(expected.tz_localize("Europe/London"), result)
    assert result.freq == expected_freq
