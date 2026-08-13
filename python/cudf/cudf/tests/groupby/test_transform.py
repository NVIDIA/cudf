# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import itertools

import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq, assert_groupby_results_equal


@pytest.fixture(params=[False, True], ids=["no-null-keys", "null-keys"])
def keys_null(request):
    return request.param


@pytest.fixture(params=[False, True], ids=["no-null-values", "null-values"])
def values_null(request):
    return request.param


@pytest.fixture
def df(keys_null, values_null):
    keys = ["a", "b", "a", "c", "b", "b", "c", "a"]
    r = range(len(keys))
    if keys_null:
        keys[::3] = itertools.repeat(None, len(r[::3]))
    values = list(range(len(keys)))
    if values_null:
        values[1::3] = itertools.repeat(None, len(r[1::3]))
    return cudf.DataFrame({"key": keys, "values": values})


@pytest.mark.parametrize("agg", ["cumsum", "cumprod", "max", "sum", "prod"])
def test_transform_broadcast(agg, df):
    pf = df.to_pandas()
    got = df.groupby("key").transform(agg)
    expect = pf.groupby("key").transform(agg)
    assert_eq(got, expect, check_dtype=False)


def test_transform_invalid():
    df = cudf.DataFrame({"key": [1, 1], "values": [4, 5]})
    with pytest.raises(TypeError):
        df.groupby("key").transform({"values": "cumprod"})


@pytest.mark.parametrize(
    "by",
    [
        "a",
        ["a", "b"],
        pd.Series([2, 1, 1, 2, 2]),
        pd.Series(["b", "a", "a", "b", "b"]),
    ],
)
@pytest.mark.parametrize("agg", ["sum", lambda df: df.mean()])
def test_groupby_transform_aggregation(by, agg):
    gdf = cudf.DataFrame(
        {"a": [2, 2, 1, 2, 1], "b": [1, 1, 1, 2, 2], "c": [1, 2, 3, 4, 5]}
    )
    pdf = gdf.to_pandas()

    expected = pdf.groupby(by).transform(agg)
    actual = gdf.groupby(by).transform(agg)

    assert_groupby_results_equal(expected, actual)


@pytest.mark.parametrize("by", ["a", ["a", "b"], pd.Series([1, 2, 1, 3])])
def test_groupby_transform_maintain_index(by):
    # test that we maintain the index after a groupby transform
    gdf = cudf.DataFrame(
        {"a": [1, 1, 1, 2], "b": [1, 2, 1, 2]}, index=[3, 2, 1, 0]
    )
    pdf = gdf.to_pandas()
    assert_groupby_results_equal(
        pdf.groupby(by).transform("max"), gdf.groupby(by).transform("max")
    )


def test_transform_size_returns_series():
    # pandas broadcasts GroupBy.size() as a single Series: unnamed for
    # DataFrameGroupBy, keeping the source name for SeriesGroupBy.
    pdf = pd.DataFrame({"A": [1, 1, 2], "B": [4.0, 5.0, 6.0]})
    gdf = cudf.DataFrame(pdf)

    assert_eq(
        pdf.groupby("A").transform("size"),
        gdf.groupby("A").transform("size"),
    )
    assert_eq(
        pdf.groupby("A")["B"].transform("size"),
        gdf.groupby("A")["B"].transform("size"),
    )


@pytest.mark.parametrize("as_index", [True, False])
def test_transform_as_index_no_change(as_index):
    # as_index has no effect on transform (pandas GH#49834)
    pdf = pd.DataFrame({"A": [1, 1, 2], "B": [4, 5, 6]})
    gdf = cudf.DataFrame(pdf)

    expect = pdf.groupby("A", as_index=as_index).transform("size")
    got = gdf.groupby("A", as_index=as_index).transform("size")

    assert_eq(expect, got)


def test_transform_cumcount_series(dropna):
    # transform("cumcount") is an unnamed Series over the original index,
    # never a per-value-column result
    pdf = pd.DataFrame(
        {"A": [1, 1, None, 2], "B": [4.0, 5.0, 6.0, 7.0]},
        index=[3, 2, 1, 0],
    )
    gdf = cudf.DataFrame(pdf)

    expect = pdf.groupby("A", dropna=dropna).transform("cumcount")
    got = gdf.groupby("A", dropna=dropna).transform("cumcount")
    assert_eq(expect, got)


def test_transform_scan_lambda():
    # a named-aggregation lambda resolving to a scan must scan per group,
    # not broadcast the group total
    pdf = pd.DataFrame({"key": [0, 0, 1, 1], "val": [1.0, 2.0, 3.0, 4.0]})
    gdf = cudf.DataFrame(pdf)

    expect = pdf.groupby("key")["val"].transform(lambda x: x.cumsum())
    got = gdf.groupby("key")["val"].transform(lambda x: x.cumsum())

    assert_eq(expect, got)


def test_transform_singleton_groups_index():
    # every group a singleton with sort=True: the result must still be
    # relabeled with the original index, not the group keys
    pdf = pd.DataFrame(
        {"id": [3.0, 1.0, 2.0], "val": [10.0, 20.0, 30.0]},
        index=["r0", "r1", "r2"],
    )
    gdf = cudf.DataFrame(pdf)

    expect = pdf.groupby("id", sort=True)["val"].transform("mean")
    got = gdf.groupby("id", sort=True)["val"].transform("mean")

    assert_eq(expect, got)
