# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_groupby_results_equal


@pytest.mark.parametrize("by", ["a", ["a", "b"], ["a", "c"]])
def test_groupby_nunique(by):
    pdf = pd.DataFrame(
        {"a": [1, 1, 1, 2, 3], "b": [1, 2, 2, 2, 1], "c": [1, 2, None, 4, 5]}
    )
    gdf = cudf.from_pandas(pdf)

    expect = pdf.groupby(by).nunique()
    got = gdf.groupby(by).nunique()

    assert_groupby_results_equal(expect, got, check_dtype=False)


def test_nunique_dropna(dropna):
    gdf = cudf.DataFrame(
        {
            "a": [1, 1, 2],
            "b": [4, None, 5],
            "c": [None, None, 7],
            "d": [1, 1, 3],
        }
    )
    pdf = gdf.to_pandas()

    result = gdf.groupby("a")["b"].nunique(dropna=dropna)
    expected = pdf.groupby("a")["b"].nunique(dropna=dropna)
    assert_groupby_results_equal(result, expected, check_dtype=False)


def test_groupby_nunique_series():
    pdf = pd.DataFrame({"a": [1, 1, 1, 2, 2, 2], "b": [1, 2, 3, 1, 1, 2]})
    gdf = cudf.from_pandas(pdf)

    assert_groupby_results_equal(
        pdf.groupby("a")["b"].nunique(),
        gdf.groupby("a")["b"].nunique(),
        check_dtype=False,
    )
