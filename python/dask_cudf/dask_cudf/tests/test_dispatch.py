# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from dask.base import tokenize
from dask.dataframe import assert_eq
from dask.dataframe.methods import is_categorical_dtype

import cudf


def test_is_categorical_dispatch():
    assert is_categorical_dtype(pd.CategoricalDtype([1, 2, 3]))
    assert is_categorical_dtype(cudf.CategoricalDtype([1, 2, 3]))

    assert is_categorical_dtype(cudf.Series([1, 2, 3], dtype="category"))
    assert is_categorical_dtype(pd.Series([1, 2, 3], dtype="category"))

    assert is_categorical_dtype(pd.Index([1, 2, 3], dtype="category"))
    assert is_categorical_dtype(cudf.Index([1, 2, 3], dtype="category"))


@pytest.mark.parametrize("preserve_index", [True, False])
@pytest.mark.parametrize("index", [None, cudf.RangeIndex(10, name="foo")])
def test_pyarrow_conversion_dispatch(preserve_index, index):
    from dask.dataframe.dispatch import (
        from_pyarrow_table_dispatch,
        to_pyarrow_table_dispatch,
    )

    rng = np.random.default_rng(seed=0)
    df1 = cudf.DataFrame(
        rng.standard_normal(size=(10, 3)), columns=list("abc"), index=index
    )
    df2 = from_pyarrow_table_dispatch(
        df1, to_pyarrow_table_dispatch(df1, preserve_index=preserve_index)
    )

    # preserve_index=False doesn't retain index metadata
    if not preserve_index and index is not None:
        df1.index.name = None

    assert type(df1) is type(df2)
    assert_eq(df1, df2)

    # Check that preserve_index does not produce a RangeIndex
    if preserve_index:
        assert not isinstance(df2.index, cudf.RangeIndex)


@pytest.mark.parametrize("index", [None, [1, 2] * 5])
def test_deterministic_tokenize(index):
    # Checks that `dask.base.normalize_token` correctly
    # dispatches to the logic defined in `backends.py`
    # (making `tokenize(<cudf-data>)` deterministic).
    df = cudf.DataFrame(
        {"A": range(10), "B": ["dog", "cat"] * 5, "C": range(10, 0, -1)},
        index=index,
    )

    # Matching data should produce the same token
    assert tokenize(df) == tokenize(df)
    assert tokenize(df.A) == tokenize(df.A)
    assert tokenize(df.index) == tokenize(df.index)
    assert tokenize(df) == tokenize(df.copy(deep=True))
    assert tokenize(df.A) == tokenize(df.A.copy(deep=True))
    assert tokenize(df.index) == tokenize(df.index.copy(deep=True))

    # Modifying a column element should change the token
    original_token = tokenize(df)
    original_token_a = tokenize(df.A)
    df.iloc[2, df.columns.get_loc("A")] = 10
    assert original_token != tokenize(df)
    assert original_token_a != tokenize(df.A)

    # Modifying an index element should change the token
    original_token = tokenize(df)
    original_token_index = tokenize(df.index)
    new_index = df.index.values
    new_index[2] = 10
    df.index = new_index
    assert original_token != tokenize(df)
    assert original_token_index != tokenize(df.index)

    # Check MultiIndex case
    df2 = df.set_index(["B", "C"], drop=False)
    assert tokenize(df) != tokenize(df2)
    assert tokenize(df2) == tokenize(df2)


def test_deterministic_tokenize_multiindex():
    dt = datetime.strptime("1995-03-15", "%Y-%m-%d")
    index = cudf.MultiIndex(
        levels=[[1, 2], [dt]],
        codes=[[0, 1], [0, 0]],
    )
    df = cudf.DataFrame(index=index)
    assert tokenize(df) == tokenize(df)


@pytest.mark.parametrize("preserve_index", [True, False])
def test_pyarrow_schema_dispatch(preserve_index):
    from dask.dataframe.dispatch import (
        pyarrow_schema_dispatch,
        to_pyarrow_table_dispatch,
    )

    rng = np.random.default_rng(seed=0)
    df = cudf.DataFrame(rng.standard_normal(size=(10, 3)), columns=list("abc"))
    df["d"] = cudf.Series(["cat", "dog"] * 5)
    table = to_pyarrow_table_dispatch(df, preserve_index=preserve_index)
    schema = pyarrow_schema_dispatch(df, preserve_index=preserve_index)

    assert schema.equals(table.schema)
