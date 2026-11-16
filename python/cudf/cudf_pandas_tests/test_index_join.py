# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as xpd
import pytest

from cudf.pandas import LOADED
from cudf.pandas.fast_slow_proxy import _State
from cudf.testing import assert_eq

if not LOADED:
    raise ImportError("These tests must be run with cudf.pandas loaded")

pd = xpd._fsproxy_slow


@pytest.mark.parametrize(
    "other_kind,how",
    [("self", how) for how in ("left", "right", "inner", "outer")]
    + [("subset", how) for how in ("left", "right")],
)
@pytest.mark.parametrize("return_indexers", [False, True])
def test_rangeindex_join_preserves_input_identity(
    monkeypatch, other_kind, how, return_indexers
):
    expected_index = pd.RangeIndex(10, name="positions")
    index = xpd.RangeIndex(10, name="positions")
    expected_other = (
        expected_index
        if other_kind == "self"
        else pd.RangeIndex(2, 5, name="positions")
    )
    other = (
        index
        if other_kind == "self"
        else xpd.RangeIndex(2, 5, name="positions")
    )
    expected = expected_index.join(
        expected_other, how=how, return_indexers=return_indexers
    )
    with monkeypatch.context() as context:
        context.setenv("CUDF_PANDAS_FAIL_ON_FALLBACK", "1")
        actual = index.join(other, how=how, return_indexers=return_indexers)

    if return_indexers:
        result = actual[0]
        expected_result = expected[0]
        assert_eq(expected[1], actual[1])
        assert_eq(expected[2], actual[2])
    else:
        result, expected_result = actual, expected
    assert result._fsproxy_state is _State.FAST
    if expected_result is expected_index:
        assert result is index
    elif expected_result is expected_other:
        assert result is other
    else:
        assert result is not index and result is not other
    assert_eq(expected_result, result._fsproxy_slow)
