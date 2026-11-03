# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "data",
    [[[1, 2, 3], pd.NA, [4], [], [5, 6]], [1, 2, 3, 4, 5]],
)
@pytest.mark.parametrize(
    "p_index",
    [
        None,
        ["ia", "ib", "ic", "id", "ie"],
        pd.MultiIndex.from_tuples(
            [(0, "a"), (0, "b"), (0, "c"), (1, "a"), (1, "b")]
        ),
    ],
)
def test_explode(data, ignore_index, p_index):
    gdf = cudf.Series(data, index=p_index, name="someseries")
    pdf = gdf.to_pandas(arrow_type=True)

    expect = pdf.explode(ignore_index)
    got = gdf.explode(ignore_index)
    assert_eq(expect, got, check_dtype=False)


@pytest.fixture(params=["int", "float", "datetime", "timedelta"])
def leaf_value(request):
    if request.param == "int":
        return np.int32(1)
    elif request.param == "float":
        return np.float64(1)
    elif request.param == "datetime":
        return pd.to_datetime("1900-01-01")
    elif request.param == "timedelta":
        return pd.to_timedelta("10D")
    else:
        raise ValueError("Unhandled data type")


@pytest.fixture(params=["list", "struct"])
def list_or_struct(request, leaf_value):
    if request.param == "list":
        return [[leaf_value], [leaf_value]]
    elif request.param == "struct":
        return {"a": leaf_value, "b": [leaf_value], "c": {"d": [leaf_value]}}
    else:
        raise ValueError("Unhandled data type")


@pytest.fixture(params=["list", "struct"])
def nested_list(request, list_or_struct, leaf_value):
    if request.param == "list":
        return [list_or_struct, list_or_struct]
    elif request.param == "struct":
        return [
            {
                "a": list_or_struct,
                "b": leaf_value,
                "c": {"d": list_or_struct, "e": leaf_value},
            }
        ]
    else:
        raise ValueError("Unhandled data type")


def test_list_dtype_explode(nested_list):
    sr = cudf.Series([nested_list])
    assert sr.dtype.element_type == sr.explode().dtype
