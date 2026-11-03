# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import re

import cupy as cp
import numpy as np
import pandas as pd
import pytest

import cudf
from cudf import MultiIndex
from cudf.testing import assert_eq
from cudf.testing._utils import assert_exceptions_equal


def test_multiindex_is_unique_monotonic():
    pidx = pd.MultiIndex(
        [
            ["a", "b", "c"],
            ["house", "store", "forest"],
            ["clouds", "clear", "storm"],
            ["fire", "smoke", "clear"],
        ],
        [
            [0, 0, 0, 0, 1, 1, 2],
            [1, 1, 1, 1, 0, 0, 2],
            [0, 0, 2, 2, 2, 0, 1],
            [0, 0, 0, 1, 2, 0, 1],
        ],
    )
    pidx.names = ["alpha", "location", "weather", "sign"]
    gidx = cudf.from_pandas(pidx)

    assert pidx.is_unique == gidx.is_unique
    assert pidx.is_monotonic_increasing == gidx.is_monotonic_increasing
    assert pidx.is_monotonic_decreasing == gidx.is_monotonic_decreasing


@pytest.mark.parametrize(
    "testarr",
    [
        (
            [
                ["bar", "bar", "foo", "foo", "qux", "qux", "qux"],
                ["one", "two", "one", "two", "one", "two", "two"],
            ],
            ["first", "second"],
        ),
        (
            [
                ["bar", "bar", "foo", "foo", "qux", "qux"],
                ["one", "two", "one", "two", "one", "two"],
            ],
            ["first", "second"],
        ),
    ],
)
def test_multiindex_tuples_is_unique_monotonic(testarr):
    tuples = list(zip(*testarr[0], strict=True))

    index = MultiIndex.from_tuples(tuples, names=testarr[1])
    index_pd = pd.MultiIndex.from_tuples(tuples, names=testarr[1])

    assert index.is_unique == index_pd.is_unique
    assert index.is_monotonic_increasing == index_pd.is_monotonic_increasing
    assert index.is_monotonic_decreasing == index_pd.is_monotonic_decreasing


@pytest.fixture(
    params=[
        "from_product",
        "from_tuples",
        "from_arrays",
        "init",
    ]
)
def midx(request):
    if request.param == "from_product":
        return cudf.MultiIndex.from_product([[0, 1], [1, 0]])
    elif request.param == "from_tuples":
        return cudf.MultiIndex.from_tuples([(0, 1), (0, 0), (1, 1), (1, 0)])
    elif request.param == "from_arrays":
        return cudf.MultiIndex.from_arrays([[0, 0, 1, 1], [1, 0, 1, 0]])
    elif request.param == "init":
        return cudf.MultiIndex(
            levels=[[0, 1], [0, 1]], codes=[[0, 0, 1, 1], [1, 0, 1, 0]]
        )
    else:
        raise NotImplementedError(f"{request.param} not implemented")


def test_multindex_constructor_levels_always_indexes(midx):
    assert_eq(midx.levels[0], cudf.Index([0, 1]))
    assert_eq(midx.levels[1], cudf.Index([0, 1]))


def test_bool_raises():
    assert_exceptions_equal(
        lfunc=bool,
        rfunc=bool,
        lfunc_args_and_kwargs=[[cudf.MultiIndex.from_arrays([range(1)])]],
        rfunc_args_and_kwargs=[[pd.MultiIndex.from_arrays([range(1)])]],
    )


def test_multi_index_contains_hashable():
    gidx = cudf.MultiIndex.from_tuples(
        zip(["foo", "bar", "baz"], [1, 2, 3], strict=True)
    )
    pidx = gidx.to_pandas()

    assert_exceptions_equal(
        lambda: [] in gidx,
        lambda: [] in pidx,
        lfunc_args_and_kwargs=((),),
        rfunc_args_and_kwargs=((),),
    )


def test_multiindex_codes():
    midx = cudf.MultiIndex.from_tuples(
        [("a", "b"), ("a", "c"), ("b", "c")], names=["A", "Z"]
    )

    for p_array, g_array in zip(
        midx.to_pandas().codes, midx.codes, strict=True
    ):
        assert_eq(p_array, g_array)


def test_multiindex_values_pandas_compatible():
    midx = cudf.MultiIndex.from_tuples([(10, 12), (8, 9), (3, 4)])
    with cudf.option_context("mode.pandas_compatible", True):
        with pytest.raises(NotImplementedError):
            midx.values


@pytest.mark.parametrize("bad", ["foo", ["foo"]])
def test_multiindex_set_names_validation(bad):
    mi = cudf.MultiIndex.from_tuples([(0, 0), (0, 1), (1, 0), (1, 1)])
    with pytest.raises(ValueError):
        mi.names = bad


def test_multiindex_levels():
    gidx = cudf.MultiIndex.from_product(
        [range(3), ["one", "two"]], names=["first", "second"]
    )
    pidx = gidx.to_pandas()

    assert_eq(gidx.levels[0], pidx.levels[0])
    assert_eq(gidx.levels[1], pidx.levels[1])


def test_multiindex_iter_error():
    midx = cudf.MultiIndex(
        levels=[[1, 3, 4, 5], [1, 2, 5]],
        codes=[[0, 0, 1, 2, 3], [0, 2, 1, 1, 0]],
        names=["x", "y"],
    )

    with pytest.raises(
        TypeError,
        match=re.escape(
            f"{midx.__class__.__name__} object is not iterable. "
            f"Consider using `.to_arrow()`, `.to_pandas()` or `.to_numpy()` "
            f"if you wish to iterate over the values."
        ),
    ):
        iter(midx)


def test_multiindex_values():
    midx = cudf.MultiIndex(
        levels=[[1, 3, 4, 5], [1, 2, 5]],
        codes=[[0, 0, 1, 2, 3], [0, 2, 1, 1, 0]],
        names=["x", "y"],
    )

    result = midx.values

    assert isinstance(result, cp.ndarray)
    np.testing.assert_array_equal(
        result.get(), np.array([[1, 1], [1, 5], [3, 2], [4, 2], [5, 1]])
    )


@pytest.mark.parametrize(
    "pdi",
    [
        pd.MultiIndex(
            levels=[[], [], []],
            codes=[[], [], []],
            names=["one", "two", "three"],
        ),
        pd.MultiIndex.from_tuples([(1, 2), (3, 4)]),
    ],
)
def test_multiindex_empty(pdi):
    gdi = cudf.from_pandas(pdi)

    assert_eq(pdi.empty, gdi.empty)


@pytest.mark.parametrize(
    "pdi",
    [
        pd.MultiIndex(
            levels=[[], [], []],
            codes=[[], [], []],
            names=["one", "two", "three"],
        ),
        pd.MultiIndex.from_tuples([(1, 2), (3, 4)]),
    ],
)
def test_multiindex_size(pdi):
    gdi = cudf.from_pandas(pdi)

    assert_eq(pdi.size, gdi.size)
