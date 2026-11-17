# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import datetime
import re

import cupy as cp
import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "values, item, expected",
    [
        [[1, 2, 3], 2, True],
        [[1, 2, 3], 4, False],
        [[1, 2, 3], "a", False],
        [["a", "b", "c"], "a", True],
        [["a", "b", "c"], "ab", False],
        [["a", "b", "c"], 6, False],
        [pd.Categorical(["a", "b", "c"]), "a", True],
        [pd.Categorical(["a", "b", "c"]), "ab", False],
        [pd.Categorical(["a", "b", "c"]), 6, False],
        [pd.date_range("20010101", periods=5, freq="D"), 20000101, False],
        [
            pd.date_range("20010101", periods=5, freq="D"),
            datetime.datetime(2000, 1, 1),
            False,
        ],
        [
            pd.date_range("20010101", periods=5, freq="D"),
            datetime.datetime(2001, 1, 1),
            True,
        ],
    ],
)
@pytest.mark.parametrize(
    "box",
    [cudf.Index, lambda x: cudf.Series(index=x)],
    ids=["index", "series"],
)
def test_contains(values, item, expected, box):
    assert (item in box(values)) is expected


@pytest.mark.parametrize(
    "testlist",
    [
        [1, 2, 3, 4],
        [1, 2, 3, 4, None],
        [1, 2, 3, 3, 4],
        [10, 9, 8, 7],
        [10, 9, 8, 8, 7],
        [1, 2, 3, 4, np.nan],
        [10, 9, 8, np.nan, 7],
        [10, 9, 8, 8, 7, np.nan],
        ["c", "d", "e", "f"],
        ["c", "d", "e", "e", "f"],
        ["c", "d", "e", "f", None],
        ["z", "y", "x", "r"],
        ["z", "y", "x", "x", "r"],
    ],
)
def test_index_is_unique_monotonic(testlist):
    index = cudf.Index(testlist)
    index_pd = pd.Index(testlist)

    assert index.is_unique == index_pd.is_unique
    assert index.is_monotonic_increasing == index_pd.is_monotonic_increasing
    assert index.is_monotonic_decreasing == index_pd.is_monotonic_decreasing


def test_name():
    idx = cudf.Index(np.asarray([4, 5, 6, 10]), name="foo")
    assert idx.name == "foo"


def test_index_names():
    idx = cudf.Index([1, 2, 3], name="idx")
    assert idx.names == ("idx",)


@pytest.mark.parametrize("data", [[1, 2, 3, 4], []])
def test_index_empty(data, all_supported_types_as_str):
    pdi = pd.Index(data, dtype=all_supported_types_as_str)
    gdi = cudf.Index(data, dtype=all_supported_types_as_str)

    assert pdi.empty == gdi.empty


@pytest.mark.parametrize("data", [[1, 2, 3, 4], []])
def test_index_size(data, all_supported_types_as_str):
    pdi = pd.Index(data, dtype=all_supported_types_as_str)
    gdi = cudf.Index(data, dtype=all_supported_types_as_str)

    assert pdi.size == gdi.size


@pytest.mark.parametrize("data", [[], [1]])
def test_index_iter_error(data, all_supported_types_as_str):
    gdi = cudf.Index(data, dtype=all_supported_types_as_str)

    with pytest.raises(
        TypeError,
        match=re.escape(
            f"{gdi.__class__.__name__} object is not iterable. "
            f"Consider using `.to_arrow()`, `.to_pandas()` or `.to_numpy()` "
            f"if you wish to iterate over the values."
        ),
    ):
        iter(gdi)


def test_index_values():
    gidx = cudf.Index([1, 2, 3])
    pidx = gidx.to_pandas()

    assert_eq(pidx.values, gidx.values)


def test_index_null_values():
    gidx = cudf.Index([1.0, None, 3, 0, None])
    result = gidx.values
    expected = cp.array([1.0, np.nan, 3, 0, np.nan])
    assert_eq(result, expected)


@pytest.mark.parametrize(
    "data",
    [
        [1, 2, 3],
        [np.nan, 10, 15, 16],
        range(0, 10),
        [np.nan, None, 10, 20],
        ["ab", "zx", "pq"],
        ["ab", "zx", None, "pq"],
    ],
)
def test_index_hasnans(data):
    gs = cudf.Index(data, nan_as_null=False)
    if isinstance(gs, cudf.RangeIndex):
        with pytest.raises(NotImplementedError):
            gs.to_pandas(nullable=True)
    else:
        ps = gs.to_pandas(nullable=True)
        # Check type to avoid mixing Python bool and NumPy bool
        assert isinstance(gs.hasnans, bool)
        assert gs.hasnans == ps.hasnans


@pytest.mark.parametrize("data", [[0, 1, 2], [1.1, 2.3, 4.5]])
@pytest.mark.parametrize("needle", [0, 1, 2.3])
def test_index_contains_float_int(data, numeric_types_as_str, needle):
    gidx = cudf.Index(data=data, dtype=numeric_types_as_str)
    pidx = gidx.to_pandas()

    actual = needle in gidx
    expected = needle in pidx

    assert_eq(actual, expected)


def test_index_constructor():
    gidx = cudf.Index([1, 2, 3])

    assert gidx._constructor is cudf.Index


@pytest.mark.parametrize(
    "data,expected_type",
    [
        ([], "empty"),
        ([1, 2, 3], "int64"),
        ([1.0, 2.0, 3.0], "float64"),
        (["a", "b", "c"], "string"),
        ([True, False, True], "boolean"),
    ],
)
def test_index_inferred_type(data, expected_type):
    gidx = cudf.Index(data)
    pidx = pd.Index(data)

    assert_eq(gidx.inferred_type, pidx.inferred_type)
