# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq
from cudf.testing._utils import expect_warning_if


def test_list_struct_list_memory_usage():
    df = cudf.DataFrame({"a": [[{"b": [1]}]]})
    assert df.memory_usage().sum() == 24


@pytest.mark.parametrize("index", [False, True])
def test_memory_usage_index_preserve_types(index):
    data = [[1, 2, 3]]
    columns = pd.Index(np.array([1, 2, 3], dtype=np.int8), name="a")
    result = (
        cudf.DataFrame(data, columns=columns).memory_usage(index=index).index
    )
    expected = (
        pd.DataFrame(data, columns=columns).memory_usage(index=index).index
    )
    if index:
        # pandas returns an Index[object] with int and string elements
        expected = expected.astype(str)
    assert_eq(result, expected)


@pytest.mark.parametrize("set_index", [None, "A", "C", "D"])
@pytest.mark.parametrize("index", [True, False])
@pytest.mark.parametrize("deep", [True, False])
def test_memory_usage(deep, index, set_index):
    # Testing numerical/datetime by comparing with pandas
    # (string and categorical columns will be different)
    rows = 100
    df = pd.DataFrame(
        {
            "A": np.arange(rows, dtype="int64"),
            "B": np.arange(rows, dtype="int32"),
            "C": np.arange(rows, dtype="float64"),
        }
    )
    df["D"] = pd.to_datetime(df.A)
    if set_index:
        df = df.set_index(set_index)

    gdf = cudf.from_pandas(df)

    if index and isinstance(gdf.index, cudf.RangeIndex):
        # cudf RangeIndex always reports 0 bytes (ignores deep).
        # pandas may return non-zero (e.g. set_index on an arange column
        # is optimised to a RangeIndex in pandas 3), so compare separately.
        with expect_warning_if(deep, UserWarning):
            assert gdf.index.memory_usage(deep=deep) == 0

    else:
        # Check for Series only
        assert df["B"].memory_usage(index=index, deep=deep) == gdf[
            "B"
        ].memory_usage(index=index, deep=deep)

        # Check for entire DataFrame
        assert_eq(
            df.memory_usage(index=index, deep=deep).sort_index(),
            gdf.memory_usage(index=index, deep=deep).sort_index(),
        )


@pytest.mark.xfail
def test_memory_usage_string():
    rows = 100
    rng = np.random.default_rng(seed=0)
    df = pd.DataFrame(
        {
            "A": np.arange(rows, dtype="int32"),
            "B": rng.choice(["apple", "banana", "orange"], rows),
        }
    )
    gdf = cudf.from_pandas(df)

    # Check deep=False (should match pandas)
    assert gdf.B.memory_usage(deep=False, index=False) == df.B.memory_usage(
        deep=False, index=False
    )

    # Check string column
    assert gdf.B.memory_usage(deep=True, index=False) == df.B.memory_usage(
        deep=True, index=False
    )

    # Check string index
    assert gdf.set_index("B").index.memory_usage(
        deep=True
    ) == df.B.memory_usage(deep=True, index=False)


def test_memory_usage_cat():
    rows = 100
    rng = np.random.default_rng(seed=0)
    df = pd.DataFrame(
        {
            "A": np.arange(rows, dtype="int32"),
            "B": rng.choice(["apple", "banana", "orange"], rows),
        }
    )
    df["B"] = df.B.astype("category")
    gdf = cudf.from_pandas(df)

    expected = (
        gdf.B._column.categories.memory_usage
        + gdf.B._column.codes.memory_usage
    )

    # Check cat column
    assert gdf.B.memory_usage(deep=True, index=False) == expected

    # Check cat index
    assert gdf.set_index("B").index.memory_usage(deep=True) == expected
