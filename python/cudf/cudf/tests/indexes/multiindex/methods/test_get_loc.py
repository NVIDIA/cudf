# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

import cudf
from cudf.core._compat import PANDAS_GE_31
from cudf.testing import assert_eq
from cudf.testing._utils import assert_exceptions_equal, expect_warning_if


@pytest.mark.parametrize(
    "data",
    [
        [(1, 1, 1), (1, 1, 2), (1, 2, 1), (1, 2, 3), (2, 1, 1), (2, 2, 1)],
        [(2, 1, 1), (1, 2, 3), (1, 2, 1), (1, 1, 2), (2, 2, 1), (1, 1, 1)],
        [(1, 1, 1), (1, 1, 2), (1, 1, 2), (1, 2, 3), (2, 1, 1), (2, 2, 1)],
    ],
)
@pytest.mark.parametrize("key", [1, (1, 2), (1, 2, 3), (2, 1, 1), (9, 9, 9)])
def test_get_loc_multi_numeric(data, key):
    idx = pd.MultiIndex.from_tuples(data)
    pi = idx.sort_values()
    gi = cudf.from_pandas(pi)

    if key not in pi:
        assert_exceptions_equal(
            lfunc=pi.get_loc,
            rfunc=gi.get_loc,
            lfunc_args_and_kwargs=([], {"key": key}),
            rfunc_args_and_kwargs=([], {"key": key}),
        )
    else:
        expected = pi.get_loc(key)
        got = gi.get_loc(key)

        assert_eq(expected, got)


@pytest.mark.parametrize(
    "key, result",
    [
        (1, slice(1, 5, 1)),  # deviates
        ((1, 2), slice(1, 3, 1)),
        ((1, 2, 3), slice(1, 2, None)),
        ((2, 1, 1), slice(0, 1, None)),
        ((9, 9, 9), None),
    ],
)
def test_get_loc_multi_numeric_deviate(key, result):
    pi = pd.MultiIndex.from_tuples(
        [(2, 1, 1), (1, 2, 3), (1, 2, 1), (1, 1, 1), (1, 1, 1), (2, 2, 1)]
    )
    gi = cudf.from_pandas(pi)

    with expect_warning_if(
        isinstance(key, tuple) and (not PANDAS_GE_31 or len(key) < pi.nlevels),
        pd.errors.PerformanceWarning,
    ):
        key_flag = key not in pi

    if key_flag:
        with expect_warning_if(
            isinstance(key, tuple)
            and (not PANDAS_GE_31 or len(key) < pi.nlevels),
            pd.errors.PerformanceWarning,
        ):
            assert_exceptions_equal(
                lfunc=pi.get_loc,
                rfunc=gi.get_loc,
                lfunc_args_and_kwargs=([], {"key": key}),
                rfunc_args_and_kwargs=([], {"key": key}),
            )
    else:
        expected = (
            result.start
            if PANDAS_GE_31
            and isinstance(key, tuple)
            and len(key) == pi.nlevels
            else result
        )
        got = gi.get_loc(key)

        assert_eq(expected, got)


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
        [
            ("a", "a", "a"),
            ("a", "a", "b"),
            ("a", "a", "b"),
            ("a", "b", "c"),
            ("b", "a", "a"),
            ("b", "c", "a"),
        ],
        [
            ("a", "a", "b"),
            ("b", "a", "a"),
            ("b", "a", "a"),
            ("a", "a", "a"),
            ("a", "b", "a"),
            ("b", "c", "a"),
        ],
    ],
)
@pytest.mark.parametrize(
    "key", ["a", ("a", "a"), ("a", "b", "c"), ("b", "c", "a"), ("z", "z", "z")]
)
def test_get_loc_multi_string(data, key):
    idx = pd.MultiIndex.from_tuples(data)
    pi = idx.sort_values()
    gi = cudf.from_pandas(pi)

    if key not in pi:
        assert_exceptions_equal(
            lfunc=pi.get_loc,
            rfunc=gi.get_loc,
            lfunc_args_and_kwargs=([], {"key": key}),
            rfunc_args_and_kwargs=([], {"key": key}),
        )
    else:
        expected = pi.get_loc(key)
        got = gi.get_loc(key)

        assert_eq(expected, got)
