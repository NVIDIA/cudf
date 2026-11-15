# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as xpd
import pytest

pd = xpd._fsproxy_slow


@pytest.mark.parametrize("unit", ["us", "ns"])
@pytest.mark.parametrize("right_ticks", [1, 2])
@pytest.mark.parametrize("numpy_left", [False, True])
def test_timestamp_subtract_overflow(unit, right_ticks, numpy_left):
    left_value = np.datetime64(np.iinfo(np.int64).min + 1, unit)
    right_value = np.datetime64(right_ticks, unit)
    expected_left = left_value if numpy_left else pd.Timestamp(left_value)
    actual_left = left_value if numpy_left else xpd.Timestamp(left_value)

    # pandas 3.0 used a bare assertion for the reserved NaT value.
    with pytest.raises(
        (pd.errors.OutOfBoundsDatetime, AssertionError)
    ) as error:
        expected_left - pd.Timestamp(right_value)

    with pytest.raises(type(error.value)):
        actual_left - xpd.Timestamp(right_value)


def test_wide_timestamp_reflected_subtract():
    left = np.datetime64("10000-01-01", "us")
    right = np.datetime64("1970-01-01", "us")
    expected = pd.Timestamp(left) - pd.Timestamp(right)
    result = xpd.Timestamp(left) - xpd.Timestamp(right)

    assert result == expected
    assert result.unit == expected.unit
