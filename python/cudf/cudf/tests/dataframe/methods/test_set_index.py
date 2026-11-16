# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.filterwarnings(
    "ignore:The inplace keyword in DataFrame\\.set_index is deprecated:pandas.errors.Pandas4Warning"
)
@pytest.mark.parametrize(
    "index",
    [
        "a",
        ["a", "b"],
        pd.CategoricalIndex(["I", "II", "III", "IV", "V"]),
        pd.Series(["h", "i", "k", "l", "m"]),
        ["b", pd.Index(["I", "II", "III", "IV", "V"])],
        ["c", [11, 12, 13, 14, 15]],
        pd.MultiIndex(
            levels=[
                ["I", "II", "III", "IV", "V"],
                ["one", "two", "three", "four", "five"],
            ],
            codes=[[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]],
            names=["col1", "col2"],
        ),
        pd.RangeIndex(0, 5),  # corner case
        [pd.Series(["h", "i", "k", "l", "m"]), pd.RangeIndex(0, 5)],
        [
            pd.MultiIndex(
                levels=[
                    ["I", "II", "III", "IV", "V"],
                    ["one", "two", "three", "four", "five"],
                ],
                codes=[[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]],
                names=["col1", "col2"],
            ),
            pd.RangeIndex(0, 5),
        ],
    ],
)
@pytest.mark.parametrize("append", [True, False])
def test_set_index(index, drop, append, inplace):
    gdf = cudf.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": ["a", "b", "c", "d", "e"],
            "c": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    pdf = gdf.to_pandas()

    expected = pdf.set_index(index, drop=drop, append=append)
    actual = gdf.set_index(index, inplace=inplace, drop=drop, append=append)

    if inplace:
        assert actual is None
        actual = gdf
    assert_eq(expected, actual)


@pytest.mark.parametrize("index", ["a", pd.Index([1, 1, 2, 2, 3])])
def test_set_index_verify_integrity(index):
    gdf = cudf.DataFrame(
        {
            "a": [1, 1, 2, 2, 5],
            "b": ["a", "b", "c", "d", "e"],
            "c": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    with pytest.raises(ValueError):
        gdf.set_index(index, verify_integrity=True)


def test_set_index_multi(drop):
    nelem = 10
    rng = np.random.default_rng(seed=0)
    a = np.arange(nelem)
    rng.shuffle(a)
    df = pd.DataFrame(
        {
            "a": a,
            "b": rng.integers(0, 4, size=nelem),
            "c": rng.uniform(low=0, high=4, size=nelem),
            "d": rng.choice(["green", "black", "white"], nelem),
        }
    )
    df["e"] = df["d"].astype("category")
    gdf = cudf.DataFrame(df)

    assert_eq(gdf.set_index("a", drop=drop), gdf.set_index(["a"], drop=drop))
    assert_eq(
        df.set_index(["b", "c"], drop=drop),
        gdf.set_index(["b", "c"], drop=drop),
    )
    assert_eq(
        df.set_index(["d", "b"], drop=drop),
        gdf.set_index(["d", "b"], drop=drop),
    )
    assert_eq(
        df.set_index(["b", "d", "e"], drop=drop),
        gdf.set_index(["b", "d", "e"], drop=drop),
    )


def test_df_cat_set_index():
    df = cudf.DataFrame(
        {
            "a": pd.Categorical(list("aababcabbc"), categories=list("abc")),
            "b": np.arange(10),
        }
    )
    got = df.set_index("a")

    pddf = df.to_pandas()
    expect = pddf.set_index("a")

    assert_eq(got, expect)


def test_df_set_index_from_series():
    df = cudf.DataFrame(
        {
            "a": list(range(10)),
            "b": list(range(0, 20, 2)),
        }
    )

    # Check set_index(Series)
    df2 = df.set_index(df["b"])
    assert list(df2.columns) == ["a", "b"]
    sliced_strided = df2.loc[2:6]
    assert len(sliced_strided) == 3
    assert list(sliced_strided.index.values) == [2, 4, 6]


def test_df_set_index_from_name():
    df = cudf.DataFrame(
        {
            "a": list(range(10)),
            "b": list(range(0, 20, 2)),
        }
    )

    # Check set_index(column_name)
    df2 = df.set_index("b")
    # 1 less column because 'b' is used as index
    assert list(df2.columns) == ["a"]
    sliced_strided = df2.loc[2:6]
    assert len(sliced_strided) == 3
    assert list(sliced_strided.index.values) == [2, 4, 6]


def test_categorical_index():
    pdf = pd.DataFrame(
        {"a": [1, 2, 3], "index": pd.Categorical(["a", "b", "c"])}
    )
    initial_df = cudf.from_pandas(pdf)
    pdf = pdf.set_index("index")
    gdf1 = cudf.from_pandas(pdf)
    gdf2 = cudf.DataFrame(
        {"a": [1, 2, 3], "index": pd.Categorical(["a", "b", "c"])}
    )
    assert_eq(initial_df.index, gdf2.index)
    gdf2 = gdf2.set_index("index")

    assert isinstance(gdf1.index, cudf.CategoricalIndex)
    assert_eq(pdf, gdf1)
    assert_eq(pdf.index, gdf1.index)
    assert_eq(
        pdf.index.codes,
        gdf1.index.codes.astype(pdf.index.codes.dtype).to_numpy(),
    )

    assert isinstance(gdf2.index, cudf.CategoricalIndex)
    assert_eq(pdf, gdf2)
    assert_eq(pdf.index, gdf2.index)
    assert_eq(
        pdf.index.codes,
        gdf2.index.codes.astype(pdf.index.codes.dtype).to_numpy(),
    )
