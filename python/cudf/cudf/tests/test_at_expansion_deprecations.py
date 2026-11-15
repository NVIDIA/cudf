# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warnings

import pandas as pd
import pytest

import cudf
from cudf.core._compat import PANDAS_GE_31


@pytest.mark.parametrize(
    "kind,empty,key",
    [
        ("series", False, "a"),
        ("series", False, "b"),
        ("series", True, 0),
        ("dataframe", False, ("a", "x")),
        ("dataframe", False, ("b", "x")),
        ("dataframe", False, ("a", "y")),
        ("dataframe", False, ("b", "y")),
        ("dataframe", True, (0, "x")),
    ],
)
def test_at_expansion_warns_like_pandas(kind, empty, key):
    if kind == "series":
        expected = (
            pd.Series([], dtype="int64")
            if empty
            else pd.Series([1], index=["a"])
        )
        expands = key not in expected.index
    else:
        expected = (
            pd.DataFrame() if empty else pd.DataFrame({"x": [1]}, index=["a"])
        )
        expands = (
            key[0] not in expected.index or key[1] not in expected.columns
        )
    actual = cudf.from_pandas(expected)
    with warnings.catch_warnings(record=True) as expected_warnings:
        warnings.simplefilter("always")
        expected.at[key] = 7
    with warnings.catch_warnings(record=True) as actual_warnings:
        warnings.simplefilter("always")
        actual.at[key] = 7
    assert [warning.category for warning in actual_warnings] == [
        warning.category for warning in expected_warnings
    ]
    assert [str(warning.message) for warning in actual_warnings] == [
        str(warning.message) for warning in expected_warnings
    ]
    assert len(actual_warnings) == int(PANDAS_GE_31 and expands)
