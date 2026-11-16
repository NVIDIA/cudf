# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warnings
from collections import UserDict

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


def _call_with_warnings(call):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = call()
    return result, [(type(w.message), str(w.message)) for w in caught]


@pytest.mark.parametrize("row_type", [list, tuple, np.array])
@pytest.mark.parametrize("ragged", [False, True])
@pytest.mark.parametrize("as_iterator", [False, True])
def test_ragged_constructor_warning(row_type, ragged, as_iterator):
    values = [[1, 2], [3] if ragged else [3, 4]]

    def construct(module):
        data = [row_type(row) for row in values]
        return module.DataFrame(iter(data) if as_iterator else data)

    expected, expected_warnings = _call_with_warnings(lambda: construct(pd))
    actual, actual_warnings = _call_with_warnings(lambda: construct(cudf))
    assert actual_warnings == expected_warnings
    assert_eq(actual, expected, check_dtype=False)


@pytest.mark.parametrize("box", [list, tuple, np.array, pd.DataFrame])
def test_from_dict_non_dict_warning(box):
    data = box([[1, 2], [3, 4]])
    expected, expected_warnings = _call_with_warnings(
        lambda: pd.DataFrame.from_dict(data)
    )
    actual, actual_warnings = _call_with_warnings(
        lambda: cudf.DataFrame.from_dict(data)
    )
    assert actual_warnings == expected_warnings
    assert_eq(actual, expected)


@pytest.mark.parametrize("orient", ["columns", "index", "tight"])
@pytest.mark.parametrize("dict_type", [dict, UserDict])
def test_from_dict_mapping_warning(orient, dict_type):
    if orient == "tight":
        data = dict_type(
            index=[0, 1],
            columns=["a"],
            data=[[1], [2]],
            index_names=[None],
            column_names=[None],
        )
    else:
        data = dict_type(a=[1, 2])
    expected, expected_warnings = _call_with_warnings(
        lambda: pd.DataFrame.from_dict(data, orient=orient)
    )
    actual, actual_warnings = _call_with_warnings(
        lambda: cudf.DataFrame.from_dict(data, orient=orient)
    )
    assert actual_warnings == expected_warnings
    # Native from_dict treats a general Mapping as column data, unlike
    # pandas' deprecated non-dict input path.
    if not (orient == "columns" and dict_type is UserDict):
        assert_eq(actual, expected)
    else:
        assert_eq(actual, cudf.DataFrame(data))


def test_from_dict_mapping_series_index_warning():
    expected, expected_warnings = _call_with_warnings(
        lambda: pd.DataFrame.from_dict(
            UserDict(a=pd.Series([1, 2])), orient="index"
        )
    )
    actual, actual_warnings = _call_with_warnings(
        lambda: cudf.DataFrame.from_dict(
            UserDict(a=cudf.Series([1, 2])), orient="index"
        )
    )
    assert actual_warnings == expected_warnings
    assert_eq(actual, expected)
