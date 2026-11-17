# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "data, gkey",
    [
        (
            {
                "id": ["a", "a", "a", "b", "b", "b", "c", "c", "c"],
                "val1": [5, 4, 6, 4, 8, 7, 4, 5, 2],
                "val2": [4, 5, 6, 1, 2, 9, 8, 5, 1],
                "val3": [4, 5, 6, 1, 2, 9, 8, 5, 1],
            },
            ["id"],
        ),
        (
            {
                "id": [0, 0, 0, 0, 1, 1, 1],
                "a": [1, 3, 4, 2.0, -3.0, 9.0, 10.0],
                "b": [10.0, 23, -4.0, 2, -3.0, None, 19.0],
            },
            ["id", "a"],
        ),
        pytest.param(
            {
                "id": ["a", "a", "b", "b", "c", "c"],
                "val1": [None, None, None, None, None, None],
            },
            ["id"],
            marks=pytest.mark.xfail(
                reason="cuDF null values are None instead of NaN"
            ),
        ),
    ],
)
@pytest.mark.parametrize("periods", [-2, 0, 5])
def test_groupby_pct_change(data, gkey, periods):
    gdf = cudf.DataFrame(data)
    pdf = gdf.to_pandas()

    actual = gdf.groupby(gkey).pct_change(
        periods=periods,
    )
    expected = pdf.groupby(gkey).pct_change(
        periods=periods,
    )

    assert_eq(expected, actual)


@pytest.mark.parametrize("periods", [-5, 5])
def test_groupby_pct_change_multiindex_dataframe(periods):
    gdf = cudf.DataFrame(
        {
            "a": [1, 1, 2, 2],
            "b": [1, 1, 2, 3],
            "c": [2, 3, 4, 5],
            "d": [6, 8, 9, 1],
        }
    ).set_index(["a", "b"])

    actual = gdf.groupby(level=["a", "b"]).pct_change(periods)
    expected = gdf.to_pandas().groupby(level=["a", "b"]).pct_change(periods)

    assert_eq(expected, actual)


def test_groupby_pct_change_empty_columns():
    gdf = cudf.DataFrame(
        {
            "id": cudf.Series([], dtype="object"),
            "val1": cudf.Series([], dtype="float64"),
            "val2": cudf.Series([], dtype="float64"),
        }
    )
    pdf = gdf.to_pandas()

    actual = gdf.groupby("id").pct_change()
    expected = pdf.groupby("id").pct_change()

    assert_eq(expected, actual)


@pytest.mark.parametrize("periods", [-1, 1])
def test_groupby_pct_change_no_fill(periods):
    # pandas 3.0 removed fill_method: values are not forward-filled, so
    # NaN propagates into the percent change
    data = {
        "key": ["a"] * 5 + ["b"] * 5,
        "vals": [3.0, None, None, 1.0, 2.0] * 2,
    }
    pdf = pd.DataFrame(data)
    gdf = cudf.DataFrame(data)

    expected = pdf.groupby("key")["vals"].pct_change(periods=periods)
    actual = gdf.groupby("key")["vals"].pct_change(periods=periods)

    assert_eq(expected, actual)
