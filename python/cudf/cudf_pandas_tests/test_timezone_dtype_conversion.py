# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from zoneinfo import ZoneInfo

import pandas as xpd
import pytest

import cudf
from cudf.pandas import LOADED
from cudf.testing import assert_eq

if not LOADED:
    raise ImportError("These tests must be run with cudf.pandas loaded")

pd = xpd._fsproxy_slow


@pytest.mark.parametrize("timezone", ["Europe/Paris", "US/Eastern"])
@pytest.mark.parametrize("kind", ["Series", "DataFrame"])
def test_timezone_extension_dtype_can_return_to_gpu(timezone, kind):
    values = pd.Series(
        pd.DatetimeIndex(
            ["2020-01-01", None, "2020-02-01"], tz=ZoneInfo(timezone)
        ),
        name="date",
    )
    expected = values if kind == "Series" else values.to_frame()
    actual = getattr(xpd, kind)(expected)
    assert isinstance(actual._fsproxy_wrapped, getattr(cudf, kind))
    # A timezone dtype's isnative flag is false even though it has no NumPy
    # byte-order mismatch. It must remain eligible for GPU conversion.
    assert_eq(expected, actual._fsproxy_slow)
    result = actual.copy()
    assert isinstance(actual._fsproxy_wrapped, getattr(cudf, kind))
    assert isinstance(result._fsproxy_wrapped, getattr(cudf, kind))
    assert_eq(expected, result._fsproxy_slow)
