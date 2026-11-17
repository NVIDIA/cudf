# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize("axis", [0, "index"])
def test_dataframe_index_rename(axis):
    pdf = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    gdf = cudf.DataFrame(pdf)

    expect = pdf.rename(mapper={1: 5, 2: 6}, axis=axis)
    got = gdf.rename(mapper={1: 5, 2: 6}, axis=axis)

    assert_eq(expect, got)

    expect = pdf.rename(index={1: 5, 2: 6})
    got = gdf.rename(index={1: 5, 2: 6})

    assert_eq(expect, got)

    expect = pdf.rename({1: 5, 2: 6})
    got = gdf.rename({1: 5, 2: 6})

    assert_eq(expect, got)

    # `pandas` can support indexes with mixed values. We throw a
    # `NotImplementedError`.
    with pytest.raises(NotImplementedError):
        gdf.rename(mapper={1: "x", 2: "y"}, axis=axis)


def test_dataframe_MI_rename():
    gdf = cudf.DataFrame(
        {"a": np.arange(10), "b": np.arange(10), "c": np.arange(10)}
    )
    gdg = gdf.groupby(["a", "b"]).count()
    pdg = gdg.to_pandas()

    expect = pdg.rename(mapper={1: 5, 2: 6}, axis=0)
    got = gdg.rename(mapper={1: 5, 2: 6}, axis=0)

    assert_eq(expect, got)


@pytest.mark.parametrize("axis", [1, "columns"])
def test_dataframe_column_rename(axis):
    pdf = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    gdf = cudf.DataFrame(pdf)

    expect = pdf.rename(mapper=lambda name: 2 * name, axis=axis)
    got = gdf.rename(mapper=lambda name: 2 * name, axis=axis)

    assert_eq(expect, got)

    expect = pdf.rename(columns=lambda name: 2 * name)
    got = gdf.rename(columns=lambda name: 2 * name)

    assert_eq(expect, got)

    rename_mapper = {"a": "z", "b": "y", "c": "x"}
    expect = pdf.rename(columns=rename_mapper)
    got = gdf.rename(columns=rename_mapper)

    assert_eq(expect, got)


def test_rename_reset_label_dtype():
    data = {1: [2]}
    col_mapping = {1: "a"}
    result = cudf.DataFrame(data).rename(columns=col_mapping)
    expected = pd.DataFrame(data).rename(columns=col_mapping)
    assert_eq(result, expected)


def test_dataframe_rename_columns_reinfers_label_dtype():
    # pandas rebuilds the columns Index from the transformed labels on
    # rename, re-inferring its dtype (int8 -> int64) rather than
    # preserving it.
    gdf = cudf.DataFrame([[1, 2, 3]])
    gdf.columns = cudf.Index([4, 5, 6], dtype=np.int8)
    result = gdf.rename({4: 50}, axis="columns").columns
    expected = pd.Index([50, 5, 6], dtype=np.int64)
    assert_eq(result, expected)


def test_dataframe_rename_duplicate_column():
    gdf = cudf.DataFrame({"a": [1, 2, 3], "b": [3, 4, 5]})
    with pytest.raises(
        ValueError, match="Duplicate column names are not allowed"
    ):
        gdf.rename(columns={"a": "b"}, inplace=True)


@pytest.mark.parametrize("level", ["x", 0])
def test_rename_for_level_MultiIndex_dataframe(level):
    data = {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}
    index = {0: 123, 1: 4, 2: 6}
    pdf = pd.DataFrame(
        data,
        index=pd.MultiIndex.from_tuples([(0, 1, 2), (1, 2, 3), (2, 3, 4)]),
    )
    pdf.index.names = ["x", "y", "z"]
    gdf = cudf.from_pandas(pdf)

    expect = pdf.rename(index=index, level=level)
    got = gdf.rename(index=index, level=level)

    assert_eq(expect, got)


@pytest.mark.parametrize(
    "columns",
    [{"a": "f", "b": "g"}, {1: 3, 2: 4}, lambda s: 2 * s],
)
@pytest.mark.parametrize("level", [0, 1])
def test_rename_for_level_MultiColumn_dataframe(columns, level):
    gdf = cudf.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    gdf.columns = pd.MultiIndex.from_tuples([("a", 1), ("a", 2), ("b", 1)])

    pdf = gdf.to_pandas()

    expect = pdf.rename(columns=columns, level=level)
    got = gdf.rename(columns=columns, level=level)

    assert_eq(expect, got)


def test_rename_for_level_RangeIndex_dataframe():
    gdf = cudf.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    pdf = gdf.to_pandas()

    expect = pdf.rename(columns={"a": "f"}, index={0: 3, 1: 4}, level=0)
    got = gdf.rename(columns={"a": "f"}, index={0: 3, 1: 4}, level=0)

    assert_eq(expect, got)


def test_rename_for_level_is_None_MC():
    gdf = cudf.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    gdf.columns = pd.MultiIndex.from_tuples([("a", 1), ("a", 2), ("b", 1)])
    pdf = gdf.to_pandas()

    expect = pdf.rename(columns={"a": "f"}, level=None)
    got = gdf.rename(columns={"a": "f"}, level=None)

    assert_eq(expect, got)


@pytest.mark.parametrize("level", [0, 1])
def test_rename_for_level_unnamed_MultiIndex_dataframe(level):
    # GH: renaming a level of an *unnamed* MultiIndex must overwrite that
    # level rather than insert a spurious extra level (the level name is
    # ``None``, so it cannot be used as the ColumnAccessor key).
    pdf = pd.DataFrame(
        {"d": [0, 1, 2]},
        index=pd.MultiIndex.from_arrays(
            [["A", "B", "B"], ["cat", "cat", "cat"]]
        ),
    )
    gdf = cudf.from_pandas(pdf)
    mapper = {"A": "Apple", "B": "B"}

    expect = pdf.rename(index=mapper, level=level)
    got = gdf.rename(index=mapper, level=level)

    assert_eq(expect, got)
    assert got.index.nlevels == 2
