# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import cupy as cp
import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "data",
    [
        [1, 2, 3, 2, 1],
        [1, 2, None, 3, 1, 1],
        [],
        ["a", "b", "c", None, "z", "a"],
    ],
)
@pytest.mark.parametrize("use_na_sentinel", [True, False])
@pytest.mark.parametrize("sort", [True, False])
def test_series_factorize_use_na_sentinel(data, use_na_sentinel, sort):
    gsr = cudf.Series(data)
    psr = gsr.to_pandas(nullable=True)

    expected_labels, expected_cats = psr.factorize(
        use_na_sentinel=use_na_sentinel, sort=sort
    )
    actual_labels, actual_cats = gsr.factorize(
        use_na_sentinel=use_na_sentinel, sort=sort
    )
    assert_eq(expected_labels, actual_labels.get())
    assert_eq(expected_cats, actual_cats.to_pandas(nullable=True))


def test_factorize_series_obj():
    rng = np.random.default_rng(seed=0)

    arr = rng.integers(2, size=10, dtype=np.int32)
    ser = cudf.Series(arr)

    uvals, labels = ser.factorize()
    unique_values, indices = np.unique(arr, return_index=True)
    expected_values = unique_values[np.argsort(indices)]

    np.testing.assert_array_equal(labels.to_numpy(), expected_values)
    assert isinstance(uvals, cp.ndarray)
    assert isinstance(labels, cudf.Index)

    encoder = {labels[idx]: idx for idx in range(len(labels))}
    handcoded = [encoder[v] for v in arr]
    np.testing.assert_array_equal(uvals.get(), handcoded)


@pytest.mark.parametrize(
    "index",
    [
        None,
        [
            2992443.0,
            2992447.0,
            2992466.0,
            2992440.0,
            2992441.0,
            2992442.0,
            2992444.0,
            2992445.0,
            2992446.0,
            2992448.0,
        ],
    ],
)
def test_factorize_series_index(index):
    data = ["C", "H", "C", "W", "W", "W", "W", "W", "C", "W"]
    ser = cudf.Series(data, index=index)
    pser = pd.Series(data, index=index)
    result_unique, result_labels = ser.factorize()
    expected_unique, expected_labels = pser.factorize()
    assert_eq(result_unique.get(), expected_unique)
    assert_eq(result_labels, expected_labels)


def test_cudf_factorize_series():
    data = [1, 2, 3, 4, 5]

    psr = pd.Series(data)
    gsr = cudf.Series(data)

    expect = pd.factorize(psr)
    got = cudf.factorize(gsr)

    assert len(expect) == len(got)

    np.testing.assert_array_equal(expect[0], got[0].get())
    np.testing.assert_array_equal(expect[1], got[1].values.get())


@pytest.mark.parametrize(
    "data",
    [
        ["abc", "def", "abc", "a", "def", None],
        [10, 20, 100, -10, 0, 1, None, 10, 100],
    ],
)
def test_category_dtype_factorize(data):
    gs = cudf.Series(data, dtype="category")
    ps = gs.to_pandas()

    actual_codes, actual_uniques = gs.factorize()
    expected_codes, expected_uniques = ps.factorize()

    assert_eq(actual_codes, expected_codes)
    assert_eq(actual_uniques, expected_uniques)
