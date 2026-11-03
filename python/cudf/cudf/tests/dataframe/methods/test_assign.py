# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


def test_assign():
    gdf = cudf.DataFrame({"x": [1, 2, 3]})
    gdf2 = gdf.assign(y=gdf.x + 1)
    assert list(gdf.columns) == ["x"]
    assert list(gdf2.columns) == ["x", "y"]

    np.testing.assert_equal(gdf2.y.to_numpy(), [2, 3, 4])


@pytest.mark.parametrize(
    "mapping",
    [
        {"y": 1, "z": lambda df: df["x"] + df["y"]},
        {
            "x": lambda df: df["x"] * 2,
            "y": lambda df: 2,
            "z": lambda df: df["x"] / df["y"],
        },
    ],
)
def test_assign_callable(mapping):
    df = pd.DataFrame({"x": [1, 2, 3]})
    cdf = cudf.from_pandas(df)
    expect = df.assign(**mapping)
    actual = cdf.assign(**mapping)
    assert_eq(expect, actual)


@pytest.mark.parametrize(
    "codes",
    [
        [0],
        [0, 1, 2],
        [0, 1, -1],
        [0, 1, 2, -1],
    ],
)
@pytest.mark.parametrize(
    "categories",
    [
        ["aa", "bb", "cc"],
        [2, 4, 10],
    ],
)
def test_categorical_assignment(codes, categories):
    data = pd.Categorical.from_codes(codes, categories)
    pd_df = pd.DataFrame({"a": np.ones(len(data))})
    cd_df = cudf.from_pandas(pd_df)

    pd_cat_series = pd.Series(data)
    # assign categorical series
    pd_df.assign(cat_col=pd_cat_series)
    cd_df.assign(cat_col=pd_cat_series)
    assert_eq(pd_df, cd_df)

    # assign categorical array
    # needed for dask_cudf support for including file name
    # as a categorical column
    # see issue: https://github.com/NVIDIA/cudf/issues/2269
    pd_df = pd.DataFrame({"a": np.ones(len(data))})
    cd_df = cudf.from_pandas(pd_df)

    pd_df.assign(cat_col=data)
    cd_df.assign(cat_col=data)
    assert_eq(pd_df, cd_df)
