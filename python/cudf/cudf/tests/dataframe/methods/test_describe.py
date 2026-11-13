# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


def test_dataframe_describe_exclude():
    rng = np.random.default_rng(seed=12)
    data_length = 10

    df = cudf.DataFrame(
        {
            "x": rng.normal(10, 1, data_length).astype("int64"),
            "y": rng.normal(10, 1, data_length),
        }
    )
    pdf = df.to_pandas()

    gdf_results = df.describe(exclude=["float"])
    pdf_results = pdf.describe(exclude=["float"])

    assert_eq(gdf_results, pdf_results)


def test_dataframe_describe_include():
    rng = np.random.default_rng(seed=12)
    data_length = 10

    df = cudf.DataFrame(
        {
            "x": rng.normal(10, 1, data_length).astype("int64"),
            "y": rng.normal(10, 1, data_length),
        }
    )
    pdf = df.to_pandas()
    gdf_results = df.describe(include=["int"])
    pdf_results = pdf.describe(include=["int"])

    assert_eq(gdf_results, pdf_results)


def test_dataframe_describe_default():
    rng = np.random.default_rng(seed=12)
    data_length = 10

    df = cudf.DataFrame(
        {
            "x": rng.normal(10, 1, data_length),
            "y": rng.normal(10, 1, data_length),
        }
    )
    pdf = df.to_pandas()
    gdf_results = df.describe()
    pdf_results = pdf.describe()

    assert_eq(pdf_results, gdf_results)


def test_dataframe_describe_percentiles():
    rng = np.random.default_rng(seed=12)
    data_length = 100
    sample_percentiles = [0.0, 0.1, 0.33, 0.84, 0.4, 0.99]

    df = cudf.DataFrame(
        {
            "x": rng.normal(10, 1, data_length),
            "y": rng.normal(10, 1, data_length),
        }
    )
    pdf = df.to_pandas()
    gdf_results = df.describe(percentiles=sample_percentiles)
    pdf_results = pdf.describe(percentiles=sample_percentiles)

    assert_eq(pdf_results, gdf_results)


def test_dataframe_describe_include_all():
    rng = np.random.default_rng(seed=12)
    data_length = 10

    df = cudf.DataFrame(
        {
            "x": rng.normal(10, 1, data_length).astype("int64"),
            "y": rng.normal(10, 1, data_length),
            "animal": rng.choice(["dog", "cat", "bird"], data_length),
        }
    )

    pdf = df.to_pandas()
    gdf_results = df.describe(include="all")
    pdf_results = pdf.describe(include="all")

    assert_eq(gdf_results[["x", "y"]], pdf_results[["x", "y"]])
    assert_eq(gdf_results.index, pdf_results.index)
    assert_eq(gdf_results.columns, pdf_results.columns)
    assert_eq(
        gdf_results[["animal"]].fillna(-1).astype("str"),
        pdf_results[["animal"]].fillna(-1).astype("str"),
    )


@pytest.fixture(
    scope="module",
    params=[
        pd.DataFrame(
            {
                "a": [1, 2, 3],
                "b": [10, 22, 33],
                "c": [0.3234, 0.23432, 0.0],
                "d": ["hello", "world", "hello"],
            }
        ),
        pd.DataFrame(
            {
                "a": [1, 2, 3],
                "b": ["hello", "world", "hello"],
                "c": [0.3234, 0.23432, 0.0],
            }
        ),
        pd.DataFrame(
            {
                "int_data": [1, 2, 3],
                "str_data": ["hello", "world", "hello"],
                "float_data": [0.3234, 0.23432, 0.0],
                "timedelta_data": pd.Series(
                    [1, 2, 1], dtype="timedelta64[ns]"
                ),
                "datetime_data": pd.Series([1, 2, 1], dtype="datetime64[ns]"),
            }
        ),
        pd.DataFrame(
            {
                "int_data": [1, 2, 3],
                "str_data": ["hello", "world", "hello"],
                "float_data": [0.3234, 0.23432, 0.0],
                "timedelta_data": pd.Series(
                    [1, 2, 1], dtype="timedelta64[ns]"
                ),
                "datetime_data": pd.Series([1, 2, 1], dtype="datetime64[ns]"),
                "category_data": pd.Series(["a", "a", "b"], dtype="category"),
            }
        ),
    ],
)
def describe_frame(request):
    pdf = request.param
    return pdf, cudf.DataFrame(pdf)


def _assert_describe_equal(pdf, df, **kwargs):
    expected = pdf.describe(**kwargs)
    actual = df.describe(**kwargs)

    for col in expected.columns:
        if expected[col].dtype == np.dtype("object"):
            expected[col] = expected[col].fillna(-1).astype("str")
            actual[col] = actual[col].fillna(-1).astype("str")

    assert_eq(expected, actual)


def test_describe_misc_include(describe_frame):
    pdf, df = describe_frame
    for include in (None, "all", ["str"], ["int"], ["str", "int", "category"]):
        _assert_describe_equal(pdf, df, include=include)


def test_describe_misc_exclude(describe_frame):
    pdf, df = describe_frame
    for exclude in (None, ["str"], ["int"], ["str", "int", "category"]):
        _assert_describe_equal(pdf, df, exclude=exclude)


def test_empty_dataframe_describe():
    pdf = pd.DataFrame({"a": [], "b": []})
    gdf = cudf.from_pandas(pdf)

    expected = pdf.describe()
    actual = gdf.describe()

    assert_eq(expected, actual)
