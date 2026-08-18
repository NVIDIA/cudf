# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "aggfunc", ["mean", "count", {"D": "sum", "E": "count"}]
)
def test_pivot_table_simple(aggfunc):
    rng = np.random.default_rng(seed=0)
    fill_value = 0
    pdf = pd.DataFrame(
        {
            "A": ["one", "one", "two", "three"] * 6,
            "B": ["A", "B", "C"] * 8,
            "C": ["foo", "foo", "foo", "bar", "bar", "bar"] * 4,
            "D": rng.standard_normal(size=24),
            "E": rng.standard_normal(size=24),
        }
    )
    expected = pd.pivot_table(
        pdf,
        values=["D", "E"],
        index=["A", "B"],
        columns=["C"],
        aggfunc=aggfunc,
        fill_value=fill_value,
    )
    cdf = cudf.DataFrame(pdf)
    actual = cudf.pivot_table(
        cdf,
        values=["D", "E"],
        index=["A", "B"],
        columns=["C"],
        aggfunc=aggfunc,
        fill_value=fill_value,
    )
    assert_eq(expected, actual, check_dtype=False)


@pytest.mark.parametrize(
    "aggfunc", ["mean", "count", {"D": "sum", "E": "count"}]
)
def test_dataframe_pivot_table_simple(aggfunc):
    rng = np.random.default_rng(seed=0)
    fill_value = 0
    pdf = pd.DataFrame(
        {
            "A": ["one", "one", "two", "three"] * 6,
            "B": ["A", "B", "C"] * 8,
            "C": ["foo", "foo", "foo", "bar", "bar", "bar"] * 4,
            "D": rng.standard_normal(size=24),
            "E": rng.standard_normal(size=24),
        }
    )
    expected = pdf.pivot_table(
        values=["D", "E"],
        index=["A", "B"],
        columns=["C"],
        aggfunc=aggfunc,
        fill_value=fill_value,
    )
    cdf = cudf.DataFrame(pdf)
    actual = cdf.pivot_table(
        values=["D", "E"],
        index=["A", "B"],
        columns=["C"],
        aggfunc=aggfunc,
        fill_value=fill_value,
    )
    assert_eq(expected, actual, check_dtype=False)


@pytest.mark.parametrize("index", ["A", ["A"]])
@pytest.mark.parametrize("columns", ["C", ["C"]])
def test_pivot_table_scalar_index_columns(index, columns):
    data = {
        "A": ["one", "one", "two", "three"] * 6,
        "B": ["A", "B", "C"] * 8,
        "C": ["foo", "foo", "foo", "bar", "bar", "bar"] * 4,
        "D": range(24),
        "E": range(24),
    }
    result = cudf.DataFrame(data).pivot_table(
        values="D", index=index, columns=columns, aggfunc="sum"
    )
    expected = pd.DataFrame(data).pivot_table(
        values="D", index=index, columns=columns, aggfunc="sum"
    )
    assert_eq(result, expected)


@pytest.mark.parametrize("aggfunc", ["sum", "min", "max"])
@pytest.mark.parametrize("fill_value", [None, 0])
def test_pivot_table_sparse_int_fill_value(aggfunc, fill_value):
    # pandas promotes integer values to float64 when the reshape leaves
    # missing cells unfilled (fill_value=None), and keeps the integer
    # dtype when they are filled
    data = {
        "i": ["r1", "r1", "r2"],
        "c": ["a", "b", "a"],
        "v": [1, 2, 3],
    }
    with cudf.option_context("mode.pandas_compatible", True):
        result = cudf.DataFrame(data).pivot_table(
            index="i",
            columns="c",
            values="v",
            aggfunc=aggfunc,
            fill_value=fill_value,
        )
    expected = pd.DataFrame(data).pivot_table(
        index="i",
        columns="c",
        values="v",
        aggfunc=aggfunc,
        fill_value=fill_value,
    )
    assert_eq(expected, result)
