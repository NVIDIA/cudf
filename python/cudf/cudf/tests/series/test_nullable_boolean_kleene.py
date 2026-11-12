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


@pytest.mark.parametrize("op", [operator.and_, operator.or_])
@pytest.mark.parametrize("nullable_on_left", [False, True])
def test_nullable_boolean_kleene_mixed_dtypes(op, nullable_on_left):
    ordinary = pd.Series([True, False, True, False], dtype="bool")
    nullable = pd.Series([None, None, True, False], dtype="boolean")
    if nullable_on_left:
        left, right = nullable, ordinary
    else:
        left, right = ordinary, nullable

    expected = op(left, right)
    actual = op(cudf.from_pandas(left), cudf.from_pandas(right))

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


def test_mixed_nullable_integer_boolean_bitwise_and():
    left = cudf.Series([2, 3, 0], dtype="Int64")
    right = cudf.Series([True, True, True], dtype="bool")

    result = left & right

    assert result.to_pandas().tolist() == [False, True, False]
    assert result.dtype == right.dtype
