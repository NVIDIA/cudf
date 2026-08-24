# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import itertools

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


def test_dataframe_empty_sort_index():
    pdf = pd.DataFrame({"x": []})
    gdf = cudf.DataFrame(pdf)

    expect = pdf.sort_index()
    got = gdf.sort_index()

    assert_eq(expect, got, check_index_type=True)


@pytest.mark.parametrize(
    "index",
    [
        pd.RangeIndex(0, 3, 1),
        [3.0, 1.0, np.nan],
        # Test for single column MultiIndex
        pd.MultiIndex.from_arrays(
            [
                [2, 0, 1],
            ]
        ),
        pd.RangeIndex(2, -1, -1),
    ],
)
@pytest.mark.parametrize("axis", [0, 1, "index", "columns"])
@pytest.mark.parametrize("na_position", ["first", "last"])
def test_dataframe_sort_index(
    request, index, axis, ascending, inplace, ignore_index, na_position
):
    pdf = pd.DataFrame(
        {"b": [1, 3, 2], "a": [1, 4, 3], "c": [4, 1, 5]},
        index=index,
    )
    gdf = cudf.DataFrame(pdf)

    expected = pdf.sort_index(
        axis=axis,
        ascending=ascending,
        ignore_index=ignore_index,
        inplace=inplace,
        na_position=na_position,
    )
    got = gdf.sort_index(
        axis=axis,
        ascending=ascending,
        ignore_index=ignore_index,
        inplace=inplace,
        na_position=na_position,
    )
    # As of pandas 3.0, pandas sometimes returns a RangeIndex
    # instead of an Index[int64]
    if inplace is True:
        assert_eq(pdf, gdf)
    else:
        assert_eq(expected, got)


@pytest.mark.parametrize("axis", [0, 1, "index", "columns"])
@pytest.mark.parametrize(
    "level",
    [
        0,
        "b",
        1,
        ["b"],
        "a",
        ["a", "b"],
        ["b", "a"],
        [0, 1],
        [1, 0],
        [0, 2],
        None,
    ],
)
@pytest.mark.parametrize("na_position", ["first", "last"])
def test_dataframe_mulitindex_sort_index(
    request, axis, level, ascending, inplace, ignore_index, na_position
):
    pdf = pd.DataFrame(
        {
            "b": [1.0, 3.0, np.nan],
            "a": [1, 4, 3],
            1: ["a", "b", "c"],
            "e": [3, 1, 4],
            "d": [1, 2, 8],
        }
    ).set_index(["b", "a", 1])
    gdf = cudf.DataFrame(pdf)

    expected = pdf.sort_index(
        axis=axis,
        level=level,
        ascending=ascending,
        inplace=inplace,
        na_position=na_position,
        ignore_index=ignore_index,
    )
    got = gdf.sort_index(
        axis=axis,
        level=level,
        ascending=ascending,
        ignore_index=ignore_index,
        inplace=inplace,
        na_position=na_position,
    )

    if inplace is True:
        assert_eq(pdf, gdf)
    else:
        assert_eq(expected, got)


def test_sort_index_axis_1_ignore_index_true_columnaccessor_state_names():
    gdf = cudf.DataFrame([[1, 2, 3]], columns=["b", "a", "c"])
    result = gdf.sort_index(axis=1, ignore_index=True)
    assert result._data.names == tuple(result._data.keys())


@pytest.mark.parametrize(
    "levels",
    itertools.chain.from_iterable(
        itertools.permutations(range(3), n) for n in range(1, 4)
    ),
    ids=str,
)
def test_multiindex_sort_index_partial(levels):
    df = pd.DataFrame(
        {
            "a": [3, 3, 3, 1, 1, 1, 2, 2],
            "b": [4, 2, 7, -1, 11, -2, 7, 7],
            "c": [4, 4, 2, 3, 3, 3, 1, 1],
            "val": [1, 2, 3, 4, 5, 6, 7, 8],
        }
    ).set_index(["a", "b", "c"])
    cdf = cudf.from_pandas(df)

    expect = df.sort_index(level=levels, sort_remaining=True)
    got = cdf.sort_index(level=levels, sort_remaining=True)
    assert_eq(expect, got)


def test_df_cat_sort_index():
    df = cudf.DataFrame(
        {
            "a": pd.Categorical(list("aababcabbc"), categories=list("abc")),
            "b": np.arange(10),
        }
    )

    got = df.set_index("a").sort_index()
    expect = df.to_pandas().set_index("a").sort_index()

    assert_eq(got, expect)


@pytest.mark.parametrize("level", [0, 1, "c1", ["c2"], ["c2", "c1"], [1, 0]])
@pytest.mark.parametrize("sort_remaining", [True, False])
@pytest.mark.parametrize("ascending", [True, False])
@pytest.mark.parametrize("na_position", ["first", "last"])
def test_sort_index_axis_1_multiindex_level(
    level, sort_remaining, ascending, na_position
):
    # axis=1 previously ignored level= and sort_remaining= silently
    pdf = pd.DataFrame(
        [[1, 2, 3, 4]],
        columns=pd.MultiIndex.from_tuples(
            [("b", 2), ("a", 1), (np.nan, 3), ("a", 4)], names=["c1", "c2"]
        ),
    )
    gdf = cudf.DataFrame(pdf)

    expect = pdf.sort_index(
        axis=1,
        level=level,
        sort_remaining=sort_remaining,
        ascending=ascending,
        na_position=na_position,
    )
    got = gdf.sort_index(
        axis=1,
        level=level,
        sort_remaining=sort_remaining,
        ascending=ascending,
        na_position=na_position,
    )
    assert_eq(expect, got)


def test_sort_index_axis_1_level_out_of_bounds():
    pdf = pd.DataFrame(
        [[1, 2]],
        columns=pd.MultiIndex.from_tuples([("a", 1), ("b", 2)]),
    )
    gdf = cudf.DataFrame(pdf)
    with pytest.raises(IndexError):
        gdf.sort_index(axis=1, level=5)


@pytest.mark.parametrize(
    "level, ascending",
    [
        (["c1", "c2"], [True, False]),
        (["c2", "c1"], [False, True]),
        ([0, 1], [False, False]),
    ],
)
def test_sort_index_axis_1_per_level_ascending(level, ascending):
    pdf = pd.DataFrame(
        [[1, 2, 3, 4]],
        columns=pd.MultiIndex.from_tuples(
            [("b", 2), ("a", 1), ("b", 3), ("a", 4)], names=["c1", "c2"]
        ),
    )
    gdf = cudf.DataFrame(pdf)

    expect = pdf.sort_index(axis=1, level=level, ascending=ascending)
    got = gdf.sort_index(axis=1, level=level, ascending=ascending)
    assert_eq(expect, got)


def test_sort_index_axis_1_ascending_length_mismatch_raises():
    pdf = pd.DataFrame(
        [[1, 2]],
        columns=pd.MultiIndex.from_tuples([("a", 1), ("b", 2)]),
    )
    gdf = cudf.DataFrame(pdf)
    with pytest.raises(ValueError):
        pdf.sort_index(axis=1, level=[0, 1], ascending=[True])
    with pytest.raises(ValueError):
        gdf.sort_index(axis=1, level=[0, 1], ascending=[True])


def test_sort_index_axis_1_unknown_level_name_raises():
    pdf = pd.DataFrame(
        [[1, 2]],
        columns=pd.MultiIndex.from_tuples(
            [("a", 1), ("b", 2)], names=["c1", "c2"]
        ),
    )
    gdf = cudf.DataFrame(pdf)
    with pytest.raises(KeyError):
        pdf.sort_index(axis=1, level="nope")
    with pytest.raises(KeyError):
        gdf.sort_index(axis=1, level="nope")


@pytest.mark.parametrize("level", [0, "cols"])
@pytest.mark.parametrize("ascending", [True, False])
def test_sort_index_axis_1_flat_columns_level(level, ascending):
    # a flat columns axis accepts level 0 / its own name like pandas
    pdf = pd.DataFrame(
        [[1, 2, 3]], columns=pd.Index(["b", "c", "a"], name="cols")
    )
    gdf = cudf.DataFrame(pdf)

    expect = pdf.sort_index(axis=1, level=level, ascending=ascending)
    got = gdf.sort_index(axis=1, level=level, ascending=ascending)
    assert_eq(expect, got)


@pytest.mark.parametrize("na_position", ["first", "last"])
@pytest.mark.parametrize("ascending", [True, False])
def test_sort_index_axis_1_flat_nan_labels_na_position(na_position, ascending):
    # the plain (no level) axis=1 path previously used python sorted(),
    # which cannot honor na_position for NaN labels
    pdf = pd.DataFrame(
        [[1, 2, 3]], columns=pd.Index(["b", np.nan, "a"], dtype="object")
    )
    gdf = cudf.DataFrame(pdf)

    expect = pdf.sort_index(
        axis=1, ascending=ascending, na_position=na_position
    )
    got = gdf.sort_index(axis=1, ascending=ascending, na_position=na_position)
    assert_eq(expect, got)
