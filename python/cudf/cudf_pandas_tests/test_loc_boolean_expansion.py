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
    not PANDAS_GE_31,
    reason="pandas 3.1 supports partial boolean column expansion",
)
def test_loc_partial_boolean_expansion_keeps_object_nan():
    expected = pd.DataFrame(index=range(4))
    actual = xpd.DataFrame(index=range(4))
    expected.loc[[0, 1, 2], "data"] = [True, False, True]
    actual.loc[[0, 1, 2], "data"] = [True, False, True]
    result = actual._fsproxy_slow
    assert_eq(expected, result)
    assert result["data"].iloc[3] is np.nan


@pytest.mark.parametrize(
    "values,nrows,key",
    [
        ([True, False, True], 3, slice(None)),
        ([1, 2, 3], 4, [0, 1, 2]),
        (["a", "b", "c"], 3, slice(None)),
    ],
)
def test_loc_column_expansion_keeps_supported_gpu_cases(values, nrows, key):
    expected = pd.DataFrame(index=range(nrows))
    actual = xpd.DataFrame(index=range(nrows))
    expected.loc[key, "data"] = values
    actual.loc[key, "data"] = values
    assert isinstance(actual._fsproxy_wrapped, cudf.DataFrame)
    assert_eq(expected, actual._fsproxy_slow)
