# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cupy as cp
import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.core._compat import PANDAS_GE_31
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "method", ["pad", "ffill", "backfill", "bfill", "nearest"]
)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("nan_as_null", [True, False])
@pytest.mark.skipif(
    not PANDAS_GE_31, reason="pandas 3.1 missing-target fill semantics"
)
def test_get_indexer_inexact_nan_target(method, dtype, nan_as_null):
    source = pd.Index([1, 2, 3, 4, 5], dtype=dtype)
    target = [0.0, np.nan, 3.0, np.nan]
    actual_target = cudf.Index(target, nan_as_null=nan_as_null)
    expected = source.get_indexer(target, method=method)
    actual = cudf.from_pandas(source).get_indexer(actual_target, method=method)
    np.testing.assert_array_equal(expected, cp.asnumpy(actual))


@pytest.mark.skipif(
    not PANDAS_GE_31, reason="pandas 3.1 MultiIndex list-key precision"
)
@pytest.mark.parametrize("label", [1.1, float(np.float32(1.1))])
def test_float32_loc_list_preserves_key_precision(label):
    source = pd.Series([0, 1], index=np.array([1.1, 2.2], dtype="float32"))
    actual = cudf.from_pandas(source)
    if PANDAS_GE_31 and label == 1.1:
        with pytest.raises(KeyError):
            source.loc[[label]]
        with pytest.raises(KeyError):
            actual.loc[[label]]
    else:
        assert_eq(source.loc[[label]], actual.loc[[label]])
    assert actual.loc[1.1] == source.loc[1.1]


@pytest.mark.skipif(
    not PANDAS_GE_31,
    reason="pandas 3.1 validates every MultiIndex tuple width",
)
@pytest.mark.parametrize(
    "values",
    [
        [("foo", "one", "B"), ("bar",)],
        [("foo", "one")],
        [("foo", "one", "B"), ("bar", "two", "A", "x")],
    ],
)
def test_multiindex_isin_rejects_each_malformed_tuple(values):
    source = pd.MultiIndex.from_product(
        [["foo", "bar"], ["one", "two"], ["A", "B"]]
    )
    with pytest.raises(
        ValueError,
        match=r"MultiIndex\.isin expects an iterable of tuples of length 3",
    ):
        source.isin(values)
    with pytest.raises(
        ValueError,
        match=r"MultiIndex\.isin expects an iterable of tuples of length 3",
    ):
        cudf.from_pandas(source).isin(values)


@pytest.mark.parametrize("method", ["pad", "backfill", "nearest"])
def test_get_indexer_nonmissing_targets_still_match_pandas(method):
    source = pd.Index([1.0, 2.0, 3.0])
    target = [0.0, 2.5, 3.0]
    np.testing.assert_array_equal(
        source.get_indexer(target, method=method),
        cp.asnumpy(
            cudf.from_pandas(source).get_indexer(target, method=method)
        ),
    )
