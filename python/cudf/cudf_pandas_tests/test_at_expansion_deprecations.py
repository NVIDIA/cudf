# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warnings

import pandas as pd
import pytest

from cudf.pandas.fast_slow_proxy import _slow_arg
from cudf.pandas.module_accelerator import disable_module_accelerator


@pytest.mark.parametrize("kind", ["series", "multiindex", "dataframe"])
def test_at_expansion_warns_once_before_fallback(kind):
    def make_frame():
        if kind == "dataframe":
            return pd.DataFrame({"a": [1, 2]}), (2, "a"), "x"
        if kind == "multiindex":
            index = pd.MultiIndex.from_tuples([("a", 1), ("a", 2)])
            return pd.Series([1, 2], index=index), ("b", 1), 3
        return pd.Series([1, 2]), 2, "x"

    def insert(frame, key, value):
        try:
            frame.at[key] = value
        except (TypeError, ValueError) as error:
            return type(error), str(error)
        return None

    with (
        disable_module_accelerator(),
        warnings.catch_warnings(record=True) as expected_warnings,
    ):
        warnings.simplefilter("always")
        expected, key, value = make_frame()
        expected_error = insert(expected, key, value)
    with warnings.catch_warnings(record=True) as actual_warnings:
        warnings.simplefilter("always")
        actual, key, value = make_frame()
        actual_error = insert(actual, key, value)
    assert actual_error == expected_error
    assert [
        (warning.category, str(warning.message)) for warning in actual_warnings
    ] == [
        (warning.category, str(warning.message))
        for warning in expected_warnings
    ]
    with disable_module_accelerator():
        if kind == "dataframe":
            pd.testing.assert_frame_equal(_slow_arg(actual), expected)
        else:
            pd.testing.assert_series_equal(_slow_arg(actual), expected)


@pytest.mark.parametrize("operation", ["setitem", "insert", "assign"])
def test_multiindex_expansion_warns_once_before_fallback(operation):
    def expand():
        frame = pd.DataFrame(
            [[1]], columns=pd.MultiIndex.from_tuples([("a", "b")])
        )
        if operation == "setitem":
            frame["new"] = [1 + 2j]
        elif operation == "insert":
            frame.insert(1, "new", [1 + 2j])
        else:
            frame = frame.assign(new=[1 + 2j])
        return frame

    with (
        disable_module_accelerator(),
        warnings.catch_warnings(record=True) as expected_warnings,
    ):
        warnings.simplefilter("always")
        expected = expand()
    with warnings.catch_warnings(record=True) as actual_warnings:
        warnings.simplefilter("always")
        actual = expand()
    assert [
        (warning.category, str(warning.message)) for warning in actual_warnings
    ] == [
        (warning.category, str(warning.message))
        for warning in expected_warnings
    ]
    with disable_module_accelerator():
        pd.testing.assert_frame_equal(_slow_arg(actual), expected)
