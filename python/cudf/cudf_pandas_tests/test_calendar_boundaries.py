# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as xpd
import pytest

from cudf.pandas import LOADED
from cudf.testing import assert_eq

if not LOADED:
    raise ImportError("These tests must be run with cudf.pandas loaded")

pd = xpd._fsproxy_slow


@pytest.mark.parametrize(
    "value,timezone",
    [
        ("32767-12-31T23:30:00", "Etc/GMT-1"),
        ("-32768-01-01T00:30:00", "Etc/GMT+1"),
    ],
)
@pytest.mark.parametrize("component", ["year", "isocalendar"])
def test_calendar_local_year_overflow_matches_pandas(
    value, timezone, component
):
    values = np.array([None, value, None], dtype="datetime64[us]")
    expected = (
        pd.Series(values)
        .dt.tz_localize("UTC")
        .dt.tz_convert(timezone)
        .iloc[1:]
    )
    actual = (
        xpd.Series(values)
        .dt.tz_localize("UTC")
        .dt.tz_convert(timezone)
        .iloc[1:]
    )
    expected = getattr(expected.dt, component)
    actual = getattr(actual.dt, component)
    if component == "isocalendar":
        expected, actual = expected(), actual()
    assert_eq(expected, actual._fsproxy_slow)
