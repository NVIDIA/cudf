# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pyarrow as pa
import pytest

import cudf
from cudf.core._compat import PANDAS_GE_31
from cudf.testing import assert_eq


@pytest.mark.skipif(
    not PANDAS_GE_31, reason="pandas 3.1 empty-frame duplicated semantics"
)
@pytest.mark.parametrize(
    "index",
    [
        pd.Index([0, 1], name="rows"),
        pd.Index(["a", "a"], name="rows"),
        pd.Index([], dtype="int64", name="rows"),
    ],
)
@pytest.mark.parametrize("keep", ["first", "last", False])
def test_duplicated_empty_columns_preserves_index(index, keep):
    expected = pd.DataFrame(index=index)
    actual = cudf.from_pandas(expected)
    assert_eq(expected.duplicated(keep=keep), actual.duplicated(keep=keep))
    assert_eq(
        expected[expected.duplicated(keep=keep)],
        actual[actual.duplicated(keep=keep)],
    )


@pytest.mark.parametrize(
    "dtype", ["Float32", "Float64", "Int64", pd.ArrowDtype(pa.float64())]
)
@pytest.mark.parametrize(
    "mapper", [{1.0: 9.0}, {1.0: None}, lambda value: f"label_{value}"]
)
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.filterwarnings(
    "ignore:The inplace keyword in DataFrame.rename is deprecated:pandas.errors.Pandas4Warning"
)
def test_rename_extension_columns_matches_pandas(dtype, mapper, inplace):
    columns = pd.Index([1, 2, 3], dtype=dtype, name="columns")
    source = pd.DataFrame([[1, 2, 3]], columns=columns)
    expected = source.rename(columns=mapper)
    actual = cudf.from_pandas(source)
    result = actual.rename(columns=mapper, inplace=inplace)
    if inplace:
        result = actual
    pd.testing.assert_index_equal(expected.columns, result.columns, exact=True)
    assert_eq(expected, result)


def test_rename_calls_column_mapper_once_per_label():
    source = pd.DataFrame(
        [[1, 2]], columns=pd.Index([1.0, 2.0], dtype="Float64")
    )
    calls = []

    def mapper(label):
        calls.append(label)
        return label + 1

    actual = cudf.from_pandas(source).rename(columns=mapper)
    assert len(calls) == len(source.columns)
    assert_eq(source.rename(columns=lambda value: value + 1), actual)


def test_rename_missing_nullable_column_to_string():
    source = pd.DataFrame(
        [[1, 2]], columns=pd.Index([1.0, pd.NA], dtype="Float64")
    )

    def mapper(label):
        return "missing" if pd.isna(label) else label

    expected = source.rename(columns=mapper)
    actual = cudf.from_pandas(source).rename(columns=mapper)
    pd.testing.assert_index_equal(expected.columns, actual.columns, exact=True)
