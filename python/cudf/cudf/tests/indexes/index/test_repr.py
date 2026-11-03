# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import pandas as pd
import pytest

import cudf


@pytest.mark.parametrize("length", [0, 1, 10, 100, 1000])
def test_numeric_index_repr(length, numeric_types_as_str):
    rng = np.random.default_rng(seed=0)
    data = rng.integers(0, high=100, size=length).astype(numeric_types_as_str)
    pidx = pd.Index(data)
    gidx = cudf.Index(data)

    assert repr(pidx) == repr(gidx)


@pytest.mark.parametrize(
    "index,expected_repr",
    [
        (
            lambda: cudf.Index([1, 2, 3, None]),
            "Index([1, 2, 3, <NA>], dtype='int64')",
        ),
        (
            lambda: cudf.Index([None, 2.2, 3.324342, None]),
            "Index([NaN, 2.2, 3.324342, NaN], dtype='float64')",
        ),
        (
            lambda: cudf.Index([None, None, None], name="hello"),
            "Index([None, None, None], dtype='object', name='hello')",
        ),
        (
            lambda: cudf.Index(
                [None, None, None], dtype="float", name="hello"
            ),
            "Index([NaN, NaN, NaN], dtype='float64', name='hello')",
        ),
        (
            lambda: cudf.Index([None], dtype="float64", name="hello"),
            "Index([NaN], dtype='float64', name='hello')",
        ),
        (
            lambda: cudf.Index([None], dtype="int8", name="hello"),
            "Index([<NA>], dtype='int8', name='hello')",
        ),
        (
            lambda: cudf.Index([None] * 50, dtype="object"),
            "Index([None, None, None, None, None, None, None, None, None, None"
            ", None, None,\n       None, None, None, None, None, None, None, "
            "None, None, None, None, None,\n       None, None, None, None, "
            "None, None, None, None, None, None, None, None,\n       None, "
            "None, None, None, None, None, None, None, None, None, None, "
            "None,\n       None, None],\n      dtype='object')",
        ),
        (
            lambda: cudf.Index([None] * 20, dtype="uint32"),
            "Index([<NA>, <NA>, <NA>, <NA>, <NA>, <NA>, <NA>, <NA>, "
            "<NA>,\n       <NA>, <NA>, <NA>, <NA>, <NA>, <NA>, <NA>, <NA>, "
            "<NA>,\n       <NA>, <NA>],\n      dtype='uint32')",
        ),
        (
            lambda: cudf.Index(
                [None, 111, 22, 33, None, 23, 34, 2343, None], dtype="int16"
            ),
            "Index([<NA>, 111, 22, 33, <NA>, 23, 34, 2343, <NA>], "
            "dtype='int16')",
        ),
        (
            lambda: cudf.Index([1, 2, 3, None], dtype="category"),
            "CategoricalIndex([1, 2, 3, <NA>], categories=[1, 2, 3], "
            "ordered=False, dtype='category')",
        ),
        (
            lambda: cudf.Index([None, None], dtype="category"),
            "CategoricalIndex([<NA>, <NA>], categories=[], ordered=False, "
            "dtype='category')",
        ),
        (
            lambda: cudf.Index(
                np.array([10, 20, 30, None], dtype="datetime64[ns]")
            ),
            "DatetimeIndex([1970-01-01 00:00:00.000000010, "
            "1970-01-01 00:00:00.000000020,"
            "\n       1970-01-01 00:00:00.000000030, NaT],\n      "
            "dtype='datetime64[ns]')",
        ),
        (
            lambda: cudf.Index(
                np.array([10, 20, 30, None], dtype="datetime64[s]")
            ),
            "DatetimeIndex([1970-01-01 00:00:10, "
            "1970-01-01 00:00:20, 1970-01-01 00:00:30,\n"
            "       NaT],\n      dtype='datetime64[s]')",
        ),
        (
            lambda: cudf.Index(
                np.array([10, 20, 30, None], dtype="datetime64[us]")
            ),
            "DatetimeIndex([1970-01-01 00:00:00.000010, "
            "1970-01-01 00:00:00.000020,\n       "
            "1970-01-01 00:00:00.000030, NaT],\n      "
            "dtype='datetime64[us]')",
        ),
        (
            lambda: cudf.Index(
                np.array([10, 20, 30, None], dtype="datetime64[ms]")
            ),
            "DatetimeIndex([1970-01-01 00:00:00.010, "
            "1970-01-01 00:00:00.020,\n       "
            "1970-01-01 00:00:00.030, NaT],\n      "
            "dtype='datetime64[ms]')",
        ),
        (
            lambda: cudf.Index(np.array([None] * 10, dtype="datetime64[ms]")),
            "DatetimeIndex([NaT, NaT, NaT, NaT, NaT, NaT, NaT, NaT, "
            "NaT, NaT], dtype='datetime64[ms]')",
        ),
    ],
)
def test_index_null(index, expected_repr):
    index = index()
    actual_repr = repr(index)

    assert expected_repr == actual_repr
