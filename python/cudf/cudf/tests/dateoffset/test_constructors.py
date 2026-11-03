# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import re

import numpy as np
import pandas as pd
import pytest

import cudf


@pytest.mark.parametrize("period", [1.5, 0.5, "string", "1", "1.0"])
@pytest.mark.parametrize("freq", ["years", "months"])
def test_construction_invalid(period, freq):
    kwargs = {freq: period}
    with pytest.raises(ValueError):
        cudf.DateOffset(**kwargs)


@pytest.mark.parametrize(
    "unit", ["nanoseconds", "microseconds", "milliseconds", "seconds"]
)
def test_construct_max_offset(unit):
    cudf.DateOffset(**{unit: np.iinfo("int64").max})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seconds": np.iinfo("int64").max + 1},
        {"seconds": np.iinfo("int64").max, "minutes": 1},
        {"minutes": np.iinfo("int64").max},
    ],
)
def test_offset_construction_overflow(kwargs):
    with pytest.raises(NotImplementedError):
        cudf.DateOffset(**kwargs)


@pytest.mark.parametrize(
    "unit",
    [
        "years",
        "months",
        "weeks",
        "days",
        "hours",
        "minutes",
        "seconds",
        "milliseconds",
        "microseconds",
        "nanoseconds",
    ],
)
@pytest.mark.parametrize("period", [0.5, -0.5, 0.71])
def test_offset_no_fractional_periods(unit, period):
    with pytest.raises(
        ValueError, match=re.escape("Non-integer periods not supported")
    ):
        cudf.DateOffset(**{unit: period})


def test_dateoffset_instance_subclass_check():
    assert not issubclass(pd.DateOffset, cudf.DateOffset)
    assert not isinstance(pd.DateOffset(), cudf.DateOffset)


@pytest.mark.parametrize("freqstr", ["ME", "YE-DEC"])
def test_dateoffset_freq_edgecases(freqstr):
    expect = pd.tseries.frequencies.to_offset(freqstr)
    got = cudf.DateOffset._from_freqstr(freqstr)
    assert got == expect
