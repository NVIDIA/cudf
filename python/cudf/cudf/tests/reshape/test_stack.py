# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize("nulls", ["none", "some"])
def test_df_stack(nulls, all_supported_types_as_str):
    if (
        all_supported_types_as_str not in ["float32", "float64"]
        and nulls == "some"
    ):
        pytest.skip(
            reason=f"nulls not supported in {all_supported_types_as_str}"
        )
    elif all_supported_types_as_str == "category":
        pytest.skip(reason="category not applicable for test")

    num_cols = 2
    num_rows = 10
    pdf = pd.DataFrame()
    rng = np.random.default_rng(seed=0)
    for i in range(num_cols):
        colname = str(i)
        data = rng.integers(0, 26, num_rows).astype(all_supported_types_as_str)
        if nulls == "some":
            idx = rng.choice(num_rows, size=int(num_rows / 2), replace=False)
            data[idx] = np.nan
        pdf[colname] = data

    gdf = cudf.from_pandas(pdf)

    got = gdf.stack()
    expect = pdf.stack()

    assert_eq(expect, got)


def test_df_stack_reset_index():
    df = cudf.DataFrame(
        {
            "a": [1, 2, 3, 4],
            "b": [10, 11, 12, 13],
            "c": ["ab", "cd", None, "gh"],
        }
    )
    df = df.set_index(["a", "b"])
    pdf = df.to_pandas()

    expected = pdf.stack()
    actual = df.stack()

    assert_eq(expected, actual)

    expected = expected.reset_index()
    actual = actual.reset_index()

    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "tuples",
    [
        [("A", "cat"), ("A", "dog"), ("B", "cat"), ("B", "dog")],
        [("A", "cat"), ("B", "bird"), ("A", "dog"), ("B", "dog")],
    ],
)
@pytest.mark.parametrize(
    "level",
    [
        -1,
        0,
        1,
        "letter",
        "animal",
        [0, 1],
        [1, 0],
        ["letter", "animal"],
        ["animal", "letter"],
    ],
)
@pytest.mark.parametrize(
    "index",
    [
        pd.RangeIndex(2, name="range"),
        pd.Index([9, 8], name="myindex"),
        pd.MultiIndex.from_arrays(
            [
                ["A", "B"],
                [101, 102],
            ],
            names=["first", "second"],
        ),
    ],
)
def test_df_stack_multiindex_column_axis(tuples, index, level, dropna):
    if isinstance(level, list) and len(level) > 1 and not dropna:
        pytest.skip(
            "Stacking multiple levels with dropna==False is unsupported."
        )
    columns = pd.MultiIndex.from_tuples(tuples, names=["letter", "animal"])

    pdf = pd.DataFrame(
        data=[[1, 2, 3, 4], [2, 4, 6, 8]], columns=columns, index=index
    )
    gdf = cudf.from_pandas(pdf)

    with pytest.warns(FutureWarning):
        got = gdf.stack(level=level, dropna=dropna, future_stack=False)
    with pytest.warns(pd.errors.Pandas4Warning):
        expect = pdf.stack(level=level, dropna=dropna, future_stack=False)

    assert_eq(expect, got, check_dtype=False)

    got = gdf.stack(level=level, future_stack=True)
    expect = pdf.stack(level=level, future_stack=True)

    assert_eq(expect, got, check_dtype=False)


def test_df_stack_mixed_dtypes():
    pdf = pd.DataFrame(
        {
            "A": pd.Series([1, 2, 3], dtype="f4"),
            "B": pd.Series([4, 5, 6], dtype="f8"),
        }
    )

    gdf = cudf.from_pandas(pdf)

    got = gdf.stack()
    expect = pdf.stack()

    assert_eq(expect, got, check_dtype=False)


@pytest.mark.parametrize("level", [["animal", "hair_length"], [1, 2]])
def test_df_stack_multiindex_column_axis_pd_example(level):
    columns = pd.MultiIndex.from_tuples(
        [
            ("A", "cat", "long"),
            ("B", "cat", "long"),
            ("A", "dog", "short"),
            ("B", "dog", "short"),
        ],
        names=["exp", "animal", "hair_length"],
    )
    rng = np.random.default_rng(seed=0)
    df = pd.DataFrame(rng.standard_normal(size=(4, 4)), columns=columns)

    with pytest.warns(pd.errors.Pandas4Warning):
        expect = df.stack(level=level, future_stack=False)
    gdf = cudf.from_pandas(df)
    with pytest.warns(FutureWarning):
        got = gdf.stack(level=level, future_stack=False)

    assert_eq(expect, got)

    expect = df.stack(level=level, future_stack=True)
    got = gdf.stack(level=level, future_stack=True)

    assert_eq(expect, got)


def test_df_stack_int_level_names_resolved_positionally():
    # integer level *names* must not hijack positional level lookup
    # (pandas' get_level_values resolves integer arguments by name first)
    columns = pd.MultiIndex.from_tuples([("a", "x"), ("b", "y")], names=[1, 0])
    pdf = pd.DataFrame([[1, 2], [3, 4]], columns=columns)
    gdf = cudf.from_pandas(pdf)
    for level in (0, 1):
        assert_eq(
            pdf.stack(level=level, future_stack=True),
            gdf.stack(level=level, future_stack=True),
        )


@pytest.mark.parametrize("level", [2, -3])
def test_df_stack_out_of_bounds_level_raises(level):
    columns = pd.MultiIndex.from_tuples([("a", "x"), ("b", "y")])
    gdf = cudf.DataFrame([[1, 2]], columns=columns)
    with pytest.raises(IndexError, match="Too many levels"):
        gdf.stack(level=level)


def test_df_stack_duplicate_level_name_raises():
    columns = pd.MultiIndex.from_tuples(
        [("a", "x"), ("b", "y")], names=["c", "c"]
    )
    gdf = cudf.DataFrame([[1, 2]], columns=columns)
    with pytest.raises(ValueError, match="occurs multiple times"):
        gdf.stack(level="c")


def test_df_stack_unsorted_column_permutation_appearance_order():
    # a 3-cycle column permutation: the previous argsort-based reordering
    # misaligned column data for non-involution permutations; stacked keys
    # are emitted in appearance order like pandas
    columns = pd.MultiIndex.from_tuples(
        [("b", 1), ("c", 2), ("a", 3)], names=["l0", "l1"]
    )
    pdf = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=columns)
    gdf = cudf.from_pandas(pdf)
    assert_eq(
        pdf.stack("l0", future_stack=True),
        gdf.stack("l0", future_stack=True),
    )
