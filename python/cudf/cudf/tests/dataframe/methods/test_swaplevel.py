# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


def test_dataframe_swaplevel_axis_0():
    midx = cudf.MultiIndex(
        levels=[
            ["Work"],
            ["Final exam", "Coursework"],
            ["History", "Geography"],
            ["January", "February", "March", "April"],
        ],
        codes=[[0, 0, 0, 0], [0, 0, 1, 1], [0, 1, 0, 1], [0, 1, 2, 3]],
        names=["a", "b", "c", "d"],
    )
    cdf = cudf.DataFrame(
        {
            "Grade": ["A", "B", "A", "C"],
            "Percentage": ["95", "85", "95", "75"],
        },
        index=midx,
    )
    pdf = cdf.to_pandas()

    assert_eq(pdf.swaplevel(), cdf.swaplevel())
    assert_eq(pdf.swaplevel(), cdf.swaplevel(-2, -1, 0))
    assert_eq(pdf.swaplevel(1, 2), cdf.swaplevel(1, 2))
    assert_eq(cdf.swaplevel(2, 1), cdf.swaplevel(1, 2))
    assert_eq(pdf.swaplevel(-1, -3), cdf.swaplevel(-1, -3))
    assert_eq(pdf.swaplevel("a", "b", 0), cdf.swaplevel("a", "b", 0))
    assert_eq(cdf.swaplevel("a", "b"), cdf.swaplevel("b", "a"))


def test_dataframe_swaplevel_stacked_names():
    # the index produced by an unstack/stack round trip keys its levels
    # positionally while carrying real level names; swaplevel must
    # preserve the names, not re-derive them from the internal keys
    pidx = pd.MultiIndex.from_product(
        [[2000, 2001], [1, 2], [1, 15]], names=["year", "month", "day"]
    )
    pdf = pd.DataFrame(
        {"A": range(8), "B": range(8, 16)}, index=pidx, dtype="float64"
    )
    cdf = cudf.DataFrame(pdf)

    expected = pdf.unstack(1).stack(future_stack=True).swaplevel(1, 2)
    result = cdf.unstack(1).stack(future_stack=True).swaplevel(1, 2)
    assert list(result.index.names) == ["year", "month", "day"]
    assert_eq(expected.sort_index(), result.sort_index())


def test_dataframe_swaplevel_TypeError():
    cdf = cudf.DataFrame(
        {"a": [1, 2, 3], "c": [10, 20, 30]}, index=["x", "y", "z"]
    )

    with pytest.raises(TypeError):
        cdf.swaplevel()


def test_dataframe_swaplevel_axis_1():
    midx = cudf.MultiIndex(
        levels=[
            ["b", "a"],
            ["bb", "aa"],
            ["bbb", "aaa"],
        ],
        codes=[[0, 0, 1, 1], [0, 1, 0, 1], [0, 1, 0, 1]],
        names=[None, "a", "b"],
    )
    cdf = cudf.DataFrame(
        data=[[45, 30, 100, 90], [200, 100, 50, 80]],
        columns=midx,
    )
    pdf = cdf.to_pandas()

    assert_eq(pdf.swaplevel(1, 2, 1), cdf.swaplevel(1, 2, 1))
    assert_eq(pdf.swaplevel("a", "b", 1), cdf.swaplevel("a", "b", 1))
    assert_eq(cdf.swaplevel(2, 1, 1), cdf.swaplevel(1, 2, 1))
    assert_eq(pdf.swaplevel(0, 2, 1), cdf.swaplevel(0, 2, 1))
    assert_eq(pdf.swaplevel(2, 0, 1), cdf.swaplevel(2, 0, 1))
    assert_eq(cdf.swaplevel("a", "a", 1), cdf.swaplevel("b", "b", 1))
