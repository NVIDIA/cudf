# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from copy import copy, deepcopy

import cupy as cp
import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq, assert_neq

"""
DataFrame copy expectations
* A shallow copy constructs a new compound object and then (to the extent
  possible) inserts references into it to the objects found in the original.
* A deep copy constructs a new compound object and then, recursively, inserts
  copies into it of the objects found in the original.

  A cuDF DataFrame is a compound object containing a few members:
  _index, _size, _cols, where _cols is an OrderedDict
"""


@pytest.fixture(
    params=[
        lambda x: x.copy(deep=False),
        lambda x: x.copy(deep=True),
        copy,
        deepcopy,
    ],
    ids=[
        "DatFrame.copy(deep=False)",
        "DataFrame.copy(deep=True)",
        "copy.copy()",
        "copy.deepcopy()",
    ],
)
def copy_fn(request):
    return request.param


def test_dataframe_deep_copy(copy_fn):
    pdf = pd.DataFrame(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=["a", "b", "c"]
    )
    gdf = cudf.DataFrame(pdf)
    copy_pdf = copy_fn(pdf)
    copy_gdf = copy_fn(gdf)
    copy_pdf["b"] = [0, 0, 0]
    copy_gdf["b"] = [0, 0, 0]
    pdf_is_equal = np.array_equal(pdf["b"].values, copy_pdf["b"].values)
    gdf_is_equal = np.array_equal(
        gdf["b"].to_numpy(), copy_gdf["b"].to_numpy()
    )
    assert not pdf_is_equal
    assert not gdf_is_equal


@pytest.mark.parametrize("ncols", [0, 2])
def test_cudf_dataframe_copy(copy_fn, ncols, all_supported_types_as_str):
    rng = np.random.default_rng(seed=0)
    pdf = pd.DataFrame(
        {
            chr(i + ord("a")): pd.Series(rng.integers(0, 1000, 20)).astype(
                all_supported_types_as_str
            )
            for i in range(ncols)
        }
    )
    df = cudf.DataFrame(pdf)
    copy_df = copy_fn(df)
    assert_eq(df, copy_df)


@pytest.mark.parametrize("ncols", [0, 2])
def test_cudf_dataframe_copy_then_insert(
    copy_fn, ncols, all_supported_types_as_str
):
    rng = np.random.default_rng(seed=0)
    pdf = pd.DataFrame(
        {
            chr(i + ord("a")): pd.Series(rng.integers(0, 1000, 20)).astype(
                all_supported_types_as_str
            )
            for i in range(ncols)
        }
    )
    df = cudf.DataFrame(pdf)
    copy_df = copy_fn(df)
    copy_pdf = copy_fn(pdf)
    copy_df["aa"] = pd.Series(rng.integers(0, 1000, 20)).astype(
        all_supported_types_as_str
    )
    copy_pdf["aa"] = pd.Series(rng.integers(0, 1000, 20)).astype(
        all_supported_types_as_str
    )
    assert not copy_pdf.to_string().split() == pdf.to_string().split()
    assert not copy_df.to_string().split() == df.to_string().split()


def test_deep_copy_write_in_place():
    pdf = pd.DataFrame(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=["a", "b", "c"]
    )
    gdf = cudf.DataFrame(pdf)
    cdf = gdf.copy(deep=True)
    sr = gdf["b"]

    # Write a value in-place on the deep copy.
    # This should only affect the copy and not the original.
    cp.asarray(sr._column)[1] = 42

    assert_neq(gdf, cdf)


@pytest.mark.xfail(reason="cudf column-wise shallow copy is immutable")
def test_dataframe_copy_shallow():
    pdf = pd.DataFrame(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=["a", "b", "c"]
    )
    gdf = cudf.DataFrame(pdf)
    copy_pdf = pdf.copy(deep=False)
    copy_gdf = gdf.copy(deep=False)
    copy_pdf["b"] = [0, 0, 0]
    copy_gdf["b"] = [0, 0, 0]
    assert_eq(pdf["b"], copy_pdf["b"])
    assert_eq(gdf["b"], copy_gdf["b"])


def test_categorical_dataframe_slice_copy():
    pdf = pd.DataFrame({"g": pd.Series(["a", "b", "z"], dtype="category")})
    gdf = cudf.from_pandas(pdf)

    exp = pdf[1:].copy()
    gdf = gdf[1:].copy()

    assert_eq(exp, gdf)
