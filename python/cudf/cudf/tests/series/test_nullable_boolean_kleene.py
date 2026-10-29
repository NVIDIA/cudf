# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import operator

import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "op, scalar",
    [(operator.or_, True), (operator.and_, False)],
)
@pytest.mark.parametrize("reflected", [False, True])
def test_nullable_boolean_kleene_scalar(op, scalar, reflected):
    ps = pd.Series([None, True, False], dtype="boolean")
    gs = cudf.from_pandas(ps)

    if reflected:
        expected = op(scalar, ps)
        actual = op(scalar, gs)
    else:
        expected = op(ps, scalar)
        actual = op(gs, scalar)

    assert_eq(expected, actual)


@pytest.mark.parametrize("op", [operator.or_, operator.and_])
def test_nullable_boolean_kleene_columns(op):
    left = pd.Series([None, True, False, None], dtype="boolean")
    right = pd.Series([True, None, None, False], dtype="boolean")

    expected = op(left, right)
    actual = op(cudf.from_pandas(left), cudf.from_pandas(right))

    assert_eq(expected, actual)


@pytest.mark.parametrize("op", [operator.and_, operator.or_, operator.xor])
def test_nullable_boolean_kleene_truth_table(op):
    left = pd.Series(
        [True, True, True, False, False, False, None, None, None],
        dtype="boolean",
    )
    right = pd.Series(
        [True, False, None, True, False, None, True, False, None],
        dtype="boolean",
    )

    expected = op(left, right)
    actual = op(cudf.from_pandas(left), cudf.from_pandas(right))

    assert_eq(expected, actual)


@pytest.mark.parametrize("op", [operator.and_, operator.or_, operator.xor])
@pytest.mark.parametrize(
    "dtype, values",
    [
        ("bool", [True, False, True]),
        ("int64", [1, 3, 7]),
    ],
)
def test_existing_bitwise_behavior(op, dtype, values):
    left = pd.Series(values, dtype=dtype)
    right = pd.Series(values[::-1], dtype=dtype)

    expected = op(left, right)
    actual = op(cudf.from_pandas(left), cudf.from_pandas(right))

    assert_eq(expected, actual)


def test_nullable_boolean_dataframe_filter():
    pdf = pd.DataFrame({"c0": [None]}, dtype="boolean")
    gdf = cudf.from_pandas(pdf)

    expected = pdf[True | pdf["c0"]]
    actual = gdf[True | gdf["c0"]]

    assert_eq(expected, actual)
