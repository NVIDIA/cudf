# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


def _series(value, timezone, unit, sliced):
    values = np.array([None, value, None], dtype=f"datetime64[{unit}]")
    expected = pd.Series(
        pd.DatetimeIndex(values).tz_localize("UTC").tz_convert(timezone)
    )
    actual = cudf.from_pandas(expected)
    return (
        (expected.iloc[1:], actual.iloc[1:]) if sliced else (expected, actual)
    )


@pytest.mark.parametrize("unit", ["s", "us"])
@pytest.mark.parametrize("sliced", [False, True])
@pytest.mark.parametrize(
    "value,timezone",
    [
        ("32767-12-31T23:30:00", "Etc/GMT-1"),
        ("-32768-01-01T00:30:00", "Etc/GMT+1"),
    ],
)
@pytest.mark.parametrize(
    "component",
    ["year", "day_of_week", "day_of_year", "is_leap_year", "isocalendar"],
)
def test_calendar_rejects_local_year_overflow(
    value, timezone, unit, sliced, component
):
    _, actual = _series(value, timezone, unit, sliced)
    with pytest.raises(
        NotImplementedError, match="outside the int16 year range"
    ):
        result = getattr(actual.dt, component)
        if component == "isocalendar":
            result()


@pytest.mark.parametrize("unit", ["s", "us"])
@pytest.mark.parametrize("sliced", [False, True])
@pytest.mark.parametrize(
    "value,timezone",
    [
        ("32768-01-01T00:30:00", "Etc/GMT+1"),
        ("-32769-12-31T23:30:00", "Etc/GMT-1"),
    ],
)
def test_calendar_accepts_local_year_within_bounds(
    value, timezone, unit, sliced
):
    expected, actual = _series(value, timezone, unit, sliced)
    assert_eq(expected.dt.year, actual.dt.year)
