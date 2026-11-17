# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

import dask
from dask import dataframe as dd

import cudf

import dask_cudf
from dask_cudf.tests.utils import _make_random_frame

_reducers = ["sum", "count", "mean", "var", "std", "min", "max"]


def _get_reduce_fn(name):
    def wrapped(series):
        fn = getattr(series, name)
        return fn()

    return wrapped


@pytest.mark.parametrize("reducer", _reducers)
def test_series_reduce(reducer):
    reducer = _get_reduce_fn(reducer)
    size = 10
    df, gdf = _make_random_frame(size)

    got = reducer(gdf.x)
    exp = reducer(df.x)
    dd.assert_eq(got, exp)


@pytest.mark.parametrize(
    "data",
    [
        cudf.datasets.randomdata(
            nrows=10000,
            dtypes={"a": "category", "b": int, "c": float, "d": int},
        ),
        cudf.datasets.randomdata(
            nrows=10000,
            dtypes={"a": "category", "b": int, "c": float, "d": str},
        ),
        cudf.datasets.randomdata(
            nrows=10000, dtypes={"a": bool, "b": int, "c": float, "d": str}
        ),
    ],
)
@pytest.mark.parametrize(
    "op", ["max", "min", "sum", "prod", "mean", "var", "std"]
)
def test_rowwise_reductions(data, op):
    gddf = dask_cudf.from_cudf(data, npartitions=10)
    pddf = gddf.to_backend("pandas")

    with dask.config.set({"dataframe.convert-string": False}):
        if op in ("var", "std"):
            expected = getattr(pddf, op)(axis=1, numeric_only=True, ddof=0)
            got = getattr(gddf, op)(axis=1, numeric_only=True, ddof=0)
        else:
            expected = getattr(pddf, op)(numeric_only=True, axis=1)
            got = getattr(pddf, op)(numeric_only=True, axis=1)

        dd.assert_eq(
            expected,
            got,
            check_exact=False,
            check_dtype=op not in ("var", "std"),
        )


@pytest.mark.parametrize("skipna", [True, False])
def test_var_nulls(skipna):
    # Copied from 10min example notebook
    # See: https://github.com/NVIDIA/cudf/pull/15347
    s = cudf.Series([1, 2, 3, None, 4])
    ds = dask_cudf.from_cudf(s, npartitions=2)
    dd.assert_eq(s.var(skipna=skipna), ds.var(skipna=skipna))
    dd.assert_eq(s.std(skipna=skipna), ds.std(skipna=skipna))
