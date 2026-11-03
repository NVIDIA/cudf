# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "in_dtype,expect",
    [
        (np.dtype("int8"), np.dtype("int8")),
        (np.int8, np.dtype("int8")),
        (pd.Int8Dtype(), pd.Int8Dtype()),
        (pd.StringDtype(), pd.StringDtype()),
        ("int8", np.dtype("int8")),
        ("boolean", pd.BooleanDtype()),
        ("bool_", np.dtype("bool")),
        (np.bool_, np.dtype("bool")),
        (int, np.dtype("int64")),
        (float, np.dtype("float64")),
        (cudf.ListDtype("int64"), cudf.ListDtype("int64")),
        (np.dtype("U"), np.dtype("U")),
        ("timedelta64[ns]", np.dtype("<m8[ns]")),
        ("timedelta64[ms]", np.dtype("<m8[ms]")),
        ("<m8[s]", np.dtype("<m8[s]")),
        ("datetime64[ns]", np.dtype("<M8[ns]")),
        ("datetime64[ms]", np.dtype("<M8[ms]")),
        ("<M8[s]", np.dtype("<M8[s]")),
        (cudf.ListDtype("int64"), cudf.ListDtype("int64")),
        ("category", cudf.CategoricalDtype()),
        (
            cudf.CategoricalDtype(categories=("a", "b", "c")),
            cudf.CategoricalDtype(categories=("a", "b", "c")),
        ),
        (
            pd.CategoricalDtype(categories=("a", "b", "c")),
            cudf.CategoricalDtype(categories=("a", "b", "c")),
        ),
        (
            # this is a pandas.core.arrays.numpy_.PandasDtype...
            pd.array([1], dtype="int16").dtype,
            np.dtype("int16"),
        ),
        (pd.IntervalDtype("int"), cudf.IntervalDtype("int64")),
        (cudf.IntervalDtype("int"), cudf.IntervalDtype("int64")),
        (pd.IntervalDtype("int64"), cudf.IntervalDtype("int64")),
    ],
    ids=repr,
)
def test_dtype(in_dtype, expect):
    assert_eq(cudf.dtype(in_dtype), expect)


@pytest.mark.parametrize(
    "in_dtype",
    [
        "complex",
        np.complex128,
        complex,
        "S",
        "V",
        "float16",
        np.float16,
        "timedelta64",
        "timedelta64[D]",
        "datetime64[D]",
        "datetime64",
    ],
)
def test_dtype_raise(in_dtype):
    with pytest.raises(TypeError):
        cudf.dtype(in_dtype)
