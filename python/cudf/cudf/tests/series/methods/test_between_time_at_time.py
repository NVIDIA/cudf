# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

import cudf
from cudf.testing import assert_eq


@pytest.fixture
def gsr_psr():
    dates = cudf.date_range("2018-04-09", periods=4, freq="1D20min")
    gsr = cudf.Series([1, 2, 3, 4], index=dates)
    return gsr, gsr.to_pandas()


def test_series_between_time_basic(gsr_psr):
    gsr, psr = gsr_psr
    expected = psr.between_time("0:20", "0:40")
    actual = gsr.between_time("0:20", "0:40")
    assert_eq(actual, expected)


def test_series_at_time_basic():
    dates = cudf.date_range("2018-04-09", periods=4, freq="12h")
    gsr = cudf.Series([1, 2, 3, 4], index=dates)
    psr = gsr.to_pandas()

    expected = psr.at_time("12:00")
    actual = gsr.at_time("12:00")
    assert_eq(actual, expected)