# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import collections
import itertools
import string

import pytest

import cudf
from cudf.testing import assert_eq


@pytest.fixture(
    scope="module",
    params=["default", "rangeindex", "intindex", "strindex"],
)
def index(request):
    n = 12
    if request.param == "rangeindex":
        return cudf.RangeIndex(2, n + 2)
    elif request.param == "intindex":
        return cudf.Index(
            [2, 3, 4, 1, 0, 5, 6, 8, 7, 9, 10, 13], dtype="int32"
        )
    elif request.param == "strindex":
        return cudf.Index(list(string.ascii_lowercase[:n]))
    elif request.param == "default":
        return None


@pytest.fixture(
    scope="module",
    params=[
        ["a", "a", "b", "b", "c", "c", "c", "d", "d", "d", "d", "d"],
        [1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4],
    ],
    ids=["str-group", "int-group"],
)
def df(index, request):
    # Sampling tests only read this shared source dataframe.
    return cudf.DataFrame(
        {"a": request.param, "b": request.param, "v": request.param},
        index=index,
    )


@pytest.fixture(params=["a", ["a", "b"]], ids=["single-col", "two-col"])
def by(request):
    return request.param


def expected(df, *, n=None, frac=None):
    value_counts = collections.Counter(df.a.to_numpy())
    if n is not None:
        values = list(
            itertools.chain.from_iterable(
                itertools.repeat(v, n) for v in value_counts.keys()
            )
        )
    elif frac is not None:
        values = list(
            itertools.chain.from_iterable(
                itertools.repeat(v, round(count * frac))
                for v, count in value_counts.items()
            )
        )
    else:
        raise ValueError("Must provide either n or frac")
    values = cudf.Series(sorted(values), dtype=df.a.dtype)
    return cudf.DataFrame({"a": values, "b": values, "v": values})


@pytest.mark.parametrize("n", [None, 0, 1, 2])
def test_constant_n_no_replace(df, by, n):
    result = df.groupby(by).sample(n=n).sort_values("a")
    n = 1 if n is None else n
    assert_eq(expected(df, n=n), result.reset_index(drop=True))


def test_constant_n_no_replace_too_large_raises(df):
    with pytest.raises(ValueError):
        df.groupby("a").sample(n=3)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_constant_n_replace(df, by, n):
    result = df.groupby(by).sample(n=n, replace=True).sort_values("a")
    assert_eq(expected(df, n=n), result.reset_index(drop=True))


def test_invalid_arguments(df):
    with pytest.raises(ValueError):
        df.groupby("a").sample(n=1, frac=0.1)


def test_not_implemented_arguments(df):
    with pytest.raises(NotImplementedError):
        # These are valid weights, but we don't implement this yet.
        df.groupby("a").sample(n=1, weights=[1 / len(df)] * len(df))


@pytest.mark.parametrize("frac", [0, 1 / 3, 1 / 2, 2 / 3, 1])
@pytest.mark.parametrize("replace", [False, True])
def test_fraction_rounding(df, by, frac, replace):
    result = df.groupby(by).sample(frac=frac, replace=replace).sort_values("a")
    assert_eq(expected(df, frac=frac), result.reset_index(drop=True))
