# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq
from cudf.testing._utils import assert_exceptions_equal


@pytest.mark.parametrize("n", [10, 5])
@pytest.mark.parametrize("op", ["nsmallest", "nlargest"])
@pytest.mark.parametrize("columns", ["a", ["b", "a"]])
def test_dataframe_nlargest_nsmallest(n, op, columns):
    nelem = 10
    rng = np.random.default_rng(seed=0)
    aa = rng.random(nelem)
    bb = rng.random(nelem)

    df = cudf.DataFrame({"a": aa, "b": bb})
    pdf = df.to_pandas()
    assert_eq(getattr(df, op)(n, columns), getattr(pdf, op)(n, columns))


@pytest.mark.parametrize(
    "sliceobj", [slice(1, None), slice(None, -1), slice(1, -1)]
)
def test_dataframe_nlargest_sliced(sliceobj):
    nelem = 20
    n = 10
    rng = np.random.default_rng(seed=0)
    df = pd.DataFrame()
    df["a"] = rng.random(nelem)
    df["b"] = rng.random(nelem)

    expect = df[sliceobj].nlargest(n, "a")
    gdf = cudf.DataFrame(df)
    got = gdf[sliceobj].nlargest(n, "a")
    assert (got.to_pandas() == expect).all().all()


@pytest.mark.parametrize(
    "sliceobj", [slice(1, None), slice(None, -1), slice(1, -1)]
)
def test_dataframe_nsmallest_sliced(sliceobj):
    nelem = 20
    n = 10
    rng = np.random.default_rng(seed=0)
    df = pd.DataFrame()
    df["a"] = rng.random(nelem)
    df["b"] = rng.random(nelem)

    expect = df[sliceobj].nsmallest(n, "a")
    gdf = cudf.DataFrame(df)
    got = gdf[sliceobj].nsmallest(n, "a")
    assert (got.to_pandas() == expect).all().all()


@pytest.mark.parametrize("attr", ["nlargest", "nsmallest"])
def test_dataframe_nlargest_nsmallest_str_error(attr):
    gdf = cudf.DataFrame({"a": [1, 2, 3, 4], "b": ["a", "b", "c", "d"]})
    pdf = gdf.to_pandas()

    assert_exceptions_equal(
        getattr(gdf, attr),
        getattr(pdf, attr),
        ([], {"n": 1, "columns": ["a", "b"]}),
        ([], {"n": 1, "columns": ["a", "b"]}),
    )


@pytest.mark.parametrize("op", ["nsmallest", "nlargest"])
@pytest.mark.parametrize("keep", ["first", "last"])
def test_dataframe_nlargest_nsmallest_with_nulls(op, keep):
    pdf = pd.DataFrame(
        {"a": [None, 3.0, 2.0, None, 5.0], "b": [1, 2, 3, 4, 5]}
    )
    gdf = cudf.DataFrame(pdf)
    assert_eq(
        getattr(gdf, op)(2, "a", keep=keep),
        getattr(pdf, op)(2, "a", keep=keep),
    )
