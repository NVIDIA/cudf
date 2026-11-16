# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect

import numpy as np
import pandas as pd
import pytest

from cudf.pandas import LOADED
from cudf.pandas.fast_slow_proxy import _State

if not LOADED:
    raise ImportError("These tests must be run with cudf.pandas loaded")


@pytest.mark.parametrize("deep", [None, False, True])
@pytest.mark.parametrize("first_cpu_only", [False, True])
def test_series_copy_does_not_inherit_another_receivers_transfer_restriction(
    monkeypatch, deep, first_cpu_only
):
    # Exercise first access to the class's shared method descriptor in either
    # order, independently of methods accessed by preceding tests.
    descriptor = inspect.getattr_static(pd.Series, "copy")
    monkeypatch.setattr(descriptor, "_attr", None)
    seed = (
        pd.Series([1, "a"], dtype=object).astype(object)
        if first_cpu_only
        else pd.Series([1.0, 2.0])
    )
    seed.copy()

    source = pd.Series(np.arange(10), dtype="float64")
    result = source.copy(deep=deep)
    assert np.may_share_memory(source.values, result.values) == (
        deep is not True
    )
    assert np.shares_memory(source.values, result.values) == (deep is not True)
    result.iloc[0] = np.nan
    assert np.isnan(result.iloc[0])
    assert source.iloc[0] == 0

    # Copying the original receiver must continue to preserve its own data.
    pd.testing.assert_series_equal(seed.copy(), seed)


def test_saved_copy_method_uses_current_receiver_transfer_restriction():
    source = pd.Series([1.0, 2.0])
    saved_copy = source.copy
    source.force_state(_State.SLOW)
    result = saved_copy()
    assert source._fsproxy_state is _State.SLOW
    assert result._fsproxy_state is _State.SLOW
    pd.testing.assert_series_equal(result, source)
