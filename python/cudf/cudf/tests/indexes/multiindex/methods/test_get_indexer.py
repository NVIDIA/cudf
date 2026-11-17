# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq
from cudf.testing._utils import assert_exceptions_equal


@pytest.mark.parametrize(
    "data",
    [
        [(1, 1, 1), (1, 1, 2), (1, 2, 1), (1, 2, 3), (2, 1, 1), (2, 2, 1)],
        [(2, 1, 1), (1, 2, 3), (1, 2, 1), (1, 1, 2), (2, 2, 1), (1, 1, 1)],
        [(1, 1, 1), (1, 1, 2), (1, 1, 24), (1, 2, 3), (2, 1, 1), (2, 2, 1)],
    ],
)
@pytest.mark.parametrize("key", [[(1, 2, 3)], [(9, 9, 9)]])
@pytest.mark.parametrize("method", [None, "ffill", "bfill"])
def test_get_indexer_multi_numeric(data, key, method):
    idx = pd.MultiIndex.from_tuples(data)
    pi = idx.sort_values()
    gi = cudf.from_pandas(pi)

    expected = pi.get_indexer(key, method=method)
    got = gi.get_indexer(key, method=method)

    assert_eq(expected, got)

    with cudf.option_context("mode.pandas_compatible", True):
        got = gi.get_indexer(key, method=method)

    assert_eq(expected, got, check_dtype=True)


@pytest.mark.parametrize(
    "key",
    [
        ((1, 2, 3),),
        ((2, 1, 1),),
        ((9, 9, 9),),
    ],
)
@pytest.mark.parametrize("method", [None, "ffill", "bfill"])
def test_get_indexer_multi_numeric_deviate(key, method):
    pi = pd.MultiIndex.from_tuples(
        [(2, 1, 1), (1, 2, 3), (1, 2, 1), (1, 1, 10), (1, 1, 1), (2, 2, 1)]
    ).sort_values()
    gi = cudf.from_pandas(pi)

    expected = pi.get_indexer(key, method=method)
    got = gi.get_indexer(key, method=method)

    assert_eq(expected, got)


@pytest.mark.parametrize("method", ["ffill", "bfill"])
def test_get_indexer_multi_error(method):
    pi = pd.MultiIndex.from_tuples(
        [(2, 1, 1), (1, 2, 3), (1, 2, 1), (1, 1, 10), (1, 1, 1), (2, 2, 1)]
    )
    gi = cudf.from_pandas(pi)

    assert_exceptions_equal(
        pi.get_indexer,
        gi.get_indexer,
        lfunc_args_and_kwargs=(
            [],
            {"target": ((1, 2, 3),), "method": method},
        ),
        rfunc_args_and_kwargs=(
            [],
            {"target": ((1, 2, 3),), "method": method},
        ),
    )


@pytest.mark.parametrize(
    "data",
    [
        [
            ("a", "a", "a"),
            ("a", "a", "b"),
            ("a", "b", "a"),
            ("a", "b", "c"),
            ("b", "a", "a"),
            ("b", "c", "a"),
        ],
        [
            ("a", "a", "b"),
            ("a", "b", "c"),
            ("b", "a", "a"),
            ("a", "a", "a"),
            ("a", "b", "a"),
            ("b", "c", "a"),
        ],
        [
            ("a", "a", "a"),
            ("a", "b", "c"),
            ("b", "a", "a"),
            ("a", "a", "b"),
            ("a", "b", "a"),
            ("b", "c", "a"),
        ],
    ],
)
@pytest.mark.parametrize(
    "key", [[("a", "b", "c"), ("b", "c", "a")], [("z", "z", "z")]]
)
@pytest.mark.parametrize("method", [None, "ffill", "bfill"])
def test_get_indexer_multi_string(data, key, method):
    idx = pd.MultiIndex.from_tuples(data)
    pi = idx.sort_values()
    gi = cudf.from_pandas(pi)

    expected = pi.get_indexer(key, method=method)
    got = gi.get_indexer(key, method=method)

    assert_eq(expected, got)
