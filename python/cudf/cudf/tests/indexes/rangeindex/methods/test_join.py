# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


def test_rangeindex_join_user_option(default_integer_bitwidth):
    # Test that RangeIndex is materialized into 32 bit index under user
    # configuration for join.
    idx1 = cudf.RangeIndex(0, 10, name="a")
    idx2 = cudf.RangeIndex(5, 15, name="b")

    actual = idx1.join(idx2, how="inner", sort=True)
    expected = idx1.to_pandas().join(idx2.to_pandas(), how="inner", sort=True)
    assert actual.dtype == cudf.dtype(f"int{default_integer_bitwidth}")
    # exact=False to ignore dtype comparison,
    # because `default_integer_bitwidth` is cudf only option
    assert_eq(expected, actual, exact=False)


@pytest.mark.parametrize("how", ["left", "right", "inner", "outer"])
@pytest.mark.parametrize("return_indexers", [False, True])
def test_rangeindex_join_self_identity(how, return_indexers):
    index = cudf.RangeIndex(10, name="positions")
    reference = pd.RangeIndex(10, name="positions")
    expected = reference.join(
        reference, how=how, return_indexers=return_indexers
    )
    actual = index.join(index, how=how, return_indexers=return_indexers)
    if return_indexers:
        result, left_indexer, right_indexer = actual
        assert_eq(expected[0], result)
        assert left_indexer is right_indexer is None
    else:
        result = actual
        assert_eq(expected, result)
    assert result is index
