# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_groupby_results_equal


def test_size_as_index_false():
    df = pd.DataFrame({"a": [1, 2, 1], "b": [1, 2, 3]}, columns=["a", "b"])
    expected = df.groupby("a", as_index=False).size()
    result = cudf.from_pandas(df).groupby("a", as_index=False).size()
    assert_groupby_results_equal(result, expected, as_index=False, by="a")


def test_size_series_with_name():
    ser = pd.Series(range(3), name="foo")
    expected = ser.groupby(ser).size()
    result = cudf.from_pandas(ser).groupby(ser).size()
    assert_groupby_results_equal(result, expected)


@pytest.mark.parametrize("dtype", ["Int64", "Float64", "boolean"])
def test_size_series_masked_dtype(dtype):
    # pandas GH#54132: SeriesGroupBy.size on masked dtypes returns Int64
    psr = pd.Series([1, 1, 1], index=["a", "a", "b"], dtype=dtype)
    gsr = cudf.from_pandas(psr)

    expect = psr.groupby(level=0).size()
    got = gsr.groupby(level=0).size()

    assert str(got.dtype) == "Int64"
    assert_groupby_results_equal(expect, got)


def test_apply_sort_false_first_appearance_order():
    # groups are processed in order of first appearance with sort=False,
    # like pandas
    pdf = pd.DataFrame({"k": [3, 1, 3, 2, 1], "v": [1, 2, 3, 4, 5]})
    gdf = cudf.from_pandas(pdf)

    expect = pdf.groupby("k", sort=False)["v"].apply(lambda s: s.sum())
    got = gdf.groupby("k", sort=False)["v"].apply(lambda s: s.sum())

    assert list(got.index.to_pandas()) == [3, 1, 2]
    assert_groupby_results_equal(expect, got)
