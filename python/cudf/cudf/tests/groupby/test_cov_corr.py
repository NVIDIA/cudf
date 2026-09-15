# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import cudf
from cudf.testing import assert_eq
from cudf.testing._utils import expect_warning_if


@pytest.mark.parametrize(
    "data",
    [
        ["v", "n", "k", "l", "m", "i", "y", "r", "w"],
        [1, 1, 1, 2, 2, 2, 3, 3, 3],
    ],
)
@pytest.mark.parametrize("gkey", ["id", "val1", "val2"])
def test_pearson_corr_invalid_column_types(data, gkey):
    gdf = cudf.DataFrame(
        {
            "id": ["a", "a", "a", "b", "b", "b", "c", "c", "c"],
            "val1": data,
            "val2": ["d", "d", "d", "e", "e", "e", "f", "f", "f"],
        }
    )
    with pytest.raises(
        TypeError,
        match="Correlation accepts only numerical column-pairs",
    ):
        gdf.groupby(gkey).corr("pearson")


def test_pearson_corr_multiindex_dataframe():
    gdf = cudf.DataFrame(
        {"a": [1, 1, 2, 2], "b": [1, 1, 2, 3], "c": [2, 3, 4, 5]}
    ).set_index(["a", "b"])

    actual = gdf.groupby(level="a").corr("pearson")
    expected = gdf.to_pandas().groupby(level="a").corr("pearson")

    assert_eq(expected, actual)


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
                "a": [10.0, 3, 4, 2.0, -3.0, 9.0, 10.0],
                "b": [10.0, 23, -4.0, 2, -3.0, 9, 19.0],
            },
            ["id", "a"],
        ),
    ],
)
@pytest.mark.parametrize("min_periods", [0, 3])
@pytest.mark.parametrize("ddof", [1, 2])
def test_groupby_covariance(data, gkey, min_periods, ddof):
    gdf = cudf.DataFrame(data)
    pdf = gdf.to_pandas()

    actual = gdf.groupby(gkey).cov(min_periods=min_periods, ddof=ddof)
    # We observe a warning if there are too few observations to generate a
    # non-singular covariance matrix _and_ there are enough that pandas will
    # actually attempt to compute a value. Groups with fewer than min_periods
    # inputs will be skipped altogether, so no warning occurs.
    with expect_warning_if(
        (pdf.groupby(gkey).count() < 2).all().all()
        and (pdf.groupby(gkey).count() > min_periods).all().all(),
        RuntimeWarning,
    ):
        expected = pdf.groupby(gkey).cov(min_periods=min_periods, ddof=ddof)

    assert_eq(expected, actual)


def test_groupby_covariance_multiindex_dataframe():
    gdf = cudf.DataFrame(
        {
            "a": [1, 1, 2, 2],
            "b": [1, 1, 2, 2],
            "c": [2, 3, 4, 5],
            "d": [6, 8, 9, 1],
        }
    ).set_index(["a", "b"])

    actual = gdf.groupby(level=["a", "b"]).cov()
    expected = gdf.to_pandas().groupby(level=["a", "b"]).cov()

    assert_eq(expected, actual)


def test_groupby_covariance_empty_columns():
    gdf = cudf.DataFrame(columns=["id", "val1", "val2"])
    pdf = gdf.to_pandas()

    actual = gdf.groupby("id").cov()
    expected = pdf.groupby("id").cov()

    assert_eq(
        expected,
        actual,
        check_dtype=False,
        check_index_type=False,
    )


def test_groupby_cov_invalid_column_types():
    gdf = cudf.DataFrame(
        {
            "id": ["a", "a", "a", "b", "b", "b", "c", "c", "c"],
            "val1": ["v", "n", "k", "l", "m", "i", "y", "r", "w"],
            "val2": ["d", "d", "d", "e", "e", "e", "f", "f", "f"],
        },
    )
    with pytest.raises(
        TypeError,
        match="Covariance accepts only numerical column-pairs",
    ):
        gdf.groupby("id").cov()


def test_groupby_cov_positive_semidefinite_matrix():
    # Refer to discussions in PR #9889 re "pair-wise deletion" strategy
    # being used in pandas to compute the covariance of a dataframe with
    # rows containing missing values.
    # Note: cuDF currently matches pandas behavior in that the covariance
    # matrices are not guaranteed PSD (positive semi definite).
    # https://github.com/NVIDIA/cudf/pull/9889#discussion_r794158358
    gdf = cudf.DataFrame(
        [[1, 2], [None, 4], [5, None], [7, 8]], columns=["v0", "v1"]
    )
    actual = gdf.groupby(by=cudf.Series([1, 1, 1, 1])).cov()
    actual.reset_index(drop=True, inplace=True)

    pdf = gdf.to_pandas()
    expected = pdf.groupby(by=pd.Series([1, 1, 1, 1])).cov()
    expected.reset_index(drop=True, inplace=True)

    assert_eq(
        expected,
        actual,
        check_dtype=False,
    )


@pytest.mark.xfail
def test_groupby_cov_for_pandas_bug_case():
    # Handles case: pandas bug using ddof with missing data.
    # Filed an issue in Pandas on GH, link below:
    # https://github.com/pandas-dev/pandas/issues/45814
    pdf = pd.DataFrame(
        {"id": ["a", "a"], "val1": [1.0, 2.0], "val2": [np.nan, np.nan]}
    )
    expected = pdf.groupby("id").cov(ddof=2)

    gdf = cudf.from_pandas(pdf)
    actual = gdf.groupby("id").cov(ddof=2)

    assert_eq(expected, actual)


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
            ["id", "val1", "val2"],
        ),
        (
            {
                "id": [0] * 4 + [1] * 3,
                "a": [10, 3, 4, 2, -3, 9, 10],
                "b": [10, 23, -4, 2, -3, 9, 19],
            },
            ["id", "a"],
        ),
        (
            {
                "id": ["a", "a", "b", "b", "c", "c"],
                "val": pa.array(
                    [None, None, None, None, None, None], type=pa.float64()
                ),
            },
            ["id"],
        ),
        (
            {
                "id": ["a", "a", "b", "b", "c", "c"],
                "val1": [None, 4, 6, 8, None, 2],
                "val2": [4, 5, None, 2, 9, None],
            },
            ["id"],
        ),
        ({"id": [1.0], "val1": [2.0], "val2": [3.0]}, ["id"]),
    ],
)
@pytest.mark.parametrize("min_per", [0, 1, 2])
def test_pearson_corr_passing(data, gkey, min_per):
    gdf = cudf.DataFrame(data)
    pdf = gdf.to_pandas()

    actual = gdf.groupby(gkey).corr(method="pearson", min_periods=min_per)
    expected = pdf.groupby(gkey).corr(method="pearson", min_periods=min_per)

    assert_eq(expected, actual)


@pytest.mark.parametrize("method", ["kendall", "spearman"])
def test_pearson_corr_unsupported_methods(method):
    gdf = cudf.DataFrame(
        {
            "id": ["a", "a", "a", "b", "b", "b", "c", "c", "c"],
            "val1": [5, 4, 6, 4, 8, 7, 4, 5, 2],
            "val2": [4, 5, 6, 1, 2, 9, 8, 5, 1],
            "val3": [4, 5, 6, 1, 2, 9, 8, 5, 1],
        }
    )

    with pytest.raises(
        NotImplementedError,
        match="Only pearson correlation is currently supported",
    ):
        gdf.groupby("id").corr(method)


def test_pearson_corr_empty_columns():
    gdf = cudf.DataFrame(columns=["id", "val1", "val2"])
    pdf = gdf.to_pandas()

    actual = gdf.groupby("id").corr("pearson")
    expected = pdf.groupby("id").corr("pearson")

    assert_eq(
        expected,
        actual,
        check_dtype=False,
        check_index_type=False,
    )
