# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as xpd
import pytest

from cudf.core._compat import PANDAS_GE_31
from cudf.pandas import LOADED
from cudf.testing import assert_eq

if not LOADED:
    raise ImportError("These tests must be run with cudf.pandas loaded")

pd = xpd._fsproxy_slow


@pytest.mark.skipif(not PANDAS_GE_31, reason="New pandas 3.1 block assignment")
@pytest.mark.parametrize("dtype", ["period[D]", "datetime64[ns, UTC]"])
@pytest.mark.parametrize("shape", [(2,), (2, 2), (2, 1)])
def test_datetimelike_iloc_ranges_select_rectangle(dtype, shape):
    def wrap(values):
        if dtype == "period[D]":
            return pd.arrays.PeriodArray(values.ravel(), dtype=dtype).reshape(
                values.shape
            )
        return (
            pd.array(values.ravel().view("datetime64[ns]"))
            .tz_localize("UTC")
            .reshape(values.shape)
        )

    values = np.arange(12, dtype="int64").reshape(4, 3)
    replacement = np.arange(100, 100 + np.prod(shape), dtype="int64").reshape(
        shape
    )
    actual = xpd.DataFrame(wrap(values))
    actual.iloc[range(2), range(2)] = wrap(replacement)
    values[np.ix_([0, 1], [0, 1])] = replacement
    assert_eq(pd.DataFrame(wrap(values)), actual._fsproxy_slow)


def test_numeric_iloc_ranges_keep_gpu_execution(monkeypatch):
    actual = xpd.DataFrame(np.arange(12).reshape(4, 3))
    with monkeypatch.context() as context:
        context.setenv("CUDF_PANDAS_FAIL_ON_FALLBACK", "1")
        actual.iloc[range(2), range(2)] = np.array([100, 101])
    expected = pd.DataFrame(
        [[100, 101, 2], [100, 101, 5], [6, 7, 8], [9, 10, 11]]
    )
    assert_eq(expected, actual._fsproxy_slow)
