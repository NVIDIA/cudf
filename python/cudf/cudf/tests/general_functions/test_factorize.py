# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cupy as cp
import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


def test_cudf_factorize_array():
    data = [1, 2, 3, 4, 5]

    parr = np.array(data)
    garr = cp.array(data)

    expect = pd.factorize(parr)
    got = cudf.factorize(garr)

    assert len(expect) == len(got)

    np.testing.assert_array_equal(expect[0], got[0].get())
    np.testing.assert_array_equal(expect[1], got[1].get())


def test_factorize_code_pandas_compatibility():
    psr = pd.Series([1, 2, 3, 4, 5])
    gsr = cudf.from_pandas(psr)

    expect = pd.factorize(psr)
    got = cudf.factorize(gsr)
    assert_eq(got[0], expect[0])
    assert_eq(got[1], expect[1])


def test_factorize_result_classes():
    data = [1, 2, 3]

    labels, cats = cudf.factorize(cudf.Series(data))

    assert isinstance(labels, cp.ndarray)
    assert isinstance(cats, cudf.Index)

    labels, cats = cudf.factorize(cudf.Index(data))

    assert isinstance(labels, cp.ndarray)
    assert isinstance(cats, cudf.Index)

    labels, cats = cudf.factorize(cp.array(data))

    assert isinstance(labels, cp.ndarray)
    assert isinstance(cats, cp.ndarray)

    # ``np.ndarray`` input must round-trip to ``np.ndarray`` outputs to
    # mirror ``pd.factorize(np.ndarray)``; previously cudf returned an
    # ``Index`` for ``cats`` here.
    np_arr = np.array(data)
    labels, cats = cudf.factorize(np_arr)
    assert isinstance(labels, np.ndarray)
    assert isinstance(cats, np.ndarray)
    p_labels, p_cats = pd.factorize(np_arr)
    np.testing.assert_array_equal(labels, p_labels)
    np.testing.assert_array_equal(cats, p_cats)


def test_factorize_rangeindex_preserves_class():
    # ``cudf.factorize(RangeIndex)`` must dispatch to ``RangeIndex.factorize``
    # so ``cats`` stays a ``RangeIndex`` (matching ``pd.factorize``).
    ri = cudf.RangeIndex(10)
    p_ri = pd.RangeIndex(10)

    for sort in [False, True]:
        labels, cats = cudf.factorize(ri, sort=sort)
        p_labels, p_cats = pd.factorize(p_ri, sort=sort)
        assert isinstance(cats, cudf.RangeIndex)
        assert_eq(cats, p_cats, exact=True)
        np.testing.assert_array_equal(labels.get(), p_labels)

    # decreasing range -> sort matters
    ri2 = ri[::-1]
    p_ri2 = p_ri[::-1]
    for sort in [False, True]:
        labels, cats = cudf.factorize(ri2, sort=sort)
        p_labels, p_cats = pd.factorize(p_ri2, sort=sort)
        assert isinstance(cats, cudf.RangeIndex)
        assert_eq(cats, p_cats, exact=True)
        np.testing.assert_array_equal(labels.get(), p_labels)


@pytest.mark.parametrize("sort", [False, True])
def test_factorize_multiindex(sort):
    # The uniques are the distinct rows (without level names), matching
    # pandas.
    pmi = pd.MultiIndex.from_arrays(
        [["bar", "baz", "foo", "bar"], [2, 3, 1, 2]], names=["a", "b"]
    )
    gmi = cudf.Index(pmi)

    expected_codes, expected_uniques = pmi.factorize(sort=sort)
    codes, uniques = gmi.factorize(sort=sort)

    np.testing.assert_array_equal(codes.get(), expected_codes)
    assert_eq(uniques, expected_uniques)


def test_factorize_multiindex_empty():
    pmi = pd.MultiIndex.from_arrays(
        [pd.Index([], dtype=object), pd.Index([], dtype="f4")]
    )
    gmi = cudf.Index(pmi)

    codes, uniques = gmi.factorize()
    assert len(codes) == 0
    assert_eq(uniques, gmi[:0])
