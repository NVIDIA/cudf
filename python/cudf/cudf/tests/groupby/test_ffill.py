# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

import cudf
from cudf.testing import assert_groupby_results_equal
from cudf.testing.dataset_generator import rand_dataframe


def test_groupby_select_then_ffill():
    pdf = pd.DataFrame(
        {
            "a": [1, 1, 1, 2, 2],
            "b": [1, None, None, 2, None],
            "c": [3, None, None, 4, None],
        }
    )
    gdf = cudf.from_pandas(pdf)

    expected = pdf.groupby("a")["c"].ffill()
    actual = gdf.groupby("a")["c"].ffill()

    assert_groupby_results_equal(expected, actual)


@pytest.mark.xfail(
    reason="decimal64 .to_pandas() fillna null with None instead of NaN"
)
def test_groupby_ffill_multi_value():
    nelem = 20
    t = rand_dataframe(
        dtypes_meta=[
            {"dtype": "int64", "null_frequency": 0, "cardinality": 10},
            {"dtype": "int64", "null_frequency": 0.4, "cardinality": 10},
            {"dtype": "float32", "null_frequency": 0.4, "cardinality": 10},
            {
                "dtype": "datetime64[ms]",
                "null_frequency": 0.4,
                "cardinality": 10,
            },
            {
                "dtype": "timedelta64[ns]",
                "null_frequency": 0.4,
                "cardinality": 10,
            },
            {"dtype": "decimal64", "null_frequency": 0.4, "cardinality": 10},
            {"dtype": "str", "null_frequency": 0.4, "cardinality": 10},
        ],
        rows=nelem,
        use_threads=False,
        seed=0,
    )
    key_col = "0"
    value_cols = ["1", "2", "3", "4", "5", "6"]
    pdf = t.to_pandas()
    gdf = cudf.from_pandas(pdf)

    expect = pdf.groupby(key_col).ffill()
    got = gdf.groupby(key_col).ffill()

    assert_groupby_results_equal(expect[value_cols], got[value_cols])


@pytest.mark.parametrize("method", ["ffill", "bfill"])
@pytest.mark.parametrize("limit", [0, 1, 2, -1, None])
@pytest.mark.parametrize(
    "values",
    [
        [1.0, None, None, None, 2.0, None, None],
        ["x", None, None, None, "y", None, None],
    ],
)
def test_groupby_fill_limit(method, limit, values):
    # interleaved keys: limit counts group-relative positions
    keys = ["a", "b"] * 7
    data = {"key": keys, "val": [v for v in values for _ in range(2)]}
    pdf = pd.DataFrame(data)
    gdf = cudf.DataFrame(data)

    expect = getattr(pdf.groupby("key"), method)(limit=limit)
    got = getattr(gdf.groupby("key"), method)(limit=limit)

    assert_groupby_results_equal(expect, got)


@pytest.mark.parametrize("method", ["ffill", "bfill"])
def test_groupby_fill_limit_null_keys(method):
    # null-key rows must stay null under dropna=True even with a limit
    pdf = pd.DataFrame(
        {
            "key": [1.0, 1.0, None, 1.0, None, 1.0],
            "val": [1.0, None, None, None, 2.0, None],
        }
    )
    gdf = cudf.DataFrame(pdf)

    expect = getattr(pdf.groupby("key", dropna=True), method)(limit=1)
    got = getattr(gdf.groupby("key", dropna=True), method)(limit=1)

    assert_groupby_results_equal(expect, got)
