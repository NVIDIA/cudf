# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as xpd
import pytest

import cudf
from cudf.pandas import LOADED
from cudf.testing import assert_eq

if not LOADED:
    raise ImportError("These tests must be run with cudf.pandas loaded")

pd = xpd._fsproxy_slow


@pytest.mark.parametrize("values", [[0, 1, 2], [5, 3, 1], [], [5]])
@pytest.mark.parametrize("dtype", ["int32", "int64"])
@pytest.mark.parametrize("explicit_index", [False, True])
def test_set_index_signed_progression_matches_pandas(
    values, dtype, explicit_index
):
    expected = pd.DataFrame(
        {"key": pd.Series(values, dtype=dtype), "value": range(len(values))}
    )
    actual = xpd.DataFrame(expected)
    expected_key = (
        pd.Index(expected["key"], name="key") if explicit_index else "key"
    )
    actual_key = xpd.Index(expected_key) if explicit_index else "key"
    result = actual.set_index(actual_key)
    expected = expected.set_index(expected_key)
    assert isinstance(result._fsproxy_wrapped, cudf.DataFrame)
    assert_eq(expected, result._fsproxy_slow)
    pd.testing.assert_index_equal(
        expected.index, result._fsproxy_slow.index, exact=True
    )
