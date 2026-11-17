# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "by",
    [
        lambda: "a",
        lambda: "b",
        lambda: ["a", "b"],
        lambda: "c",
        lambda: pd.Series([1, 2, 1, 2, 1, 2]),
        lambda: pd.Series(["x", "y", "y", "x", "z", "x"]),
    ],
)
def test_groupby_ngroup(by, ascending, sort):
    df_ngroup = cudf.DataFrame(
        {
            "a": [2, 2, 1, 1, 2, 3],
            "b": [1, 2, 1, 2, 1, 2],
            "c": ["a", "a", "b", "c", "d", "c"],
        },
        index=[1, 3, 5, 7, 4, 2],
    )
    df_ngroup.index.name = "foo"
    by = by()
    expected = (
        df_ngroup.to_pandas()
        .groupby(by, sort=sort)
        .ngroup(ascending=ascending)
    )
    actual = df_ngroup.groupby(by, sort=sort).ngroup(ascending=ascending)
    assert_eq(expected, actual, check_dtype=False)


def test_ngroup_singleton_groups_index():
    # every group a singleton with sort=True: ngroup must be relabeled
    # with the original index, not the group-key values
    pdf = pd.DataFrame({"id": [3, 1, 2], "val": [10, 20, 30]})
    gdf = cudf.DataFrame(pdf)

    expect = pdf.groupby("id", sort=True).ngroup()
    got = gdf.groupby("id", sort=True).ngroup()

    assert_eq(expect, got)
