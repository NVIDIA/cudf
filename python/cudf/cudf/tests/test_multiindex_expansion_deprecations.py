# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warnings

import pandas as pd
import pytest

import cudf
from cudf.core._compat import PANDAS_GE_31
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "operation,key",
    [
        ("setitem", "new"),
        ("insert", "new"),
        ("assign", "new"),
        ("setitem", ("new", "")),
        ("insert", ("new", "")),
        ("setitem", ("new",)),
        ("insert", ("new",)),
    ],
)
def test_multiindex_column_expansion_warning(operation, key):
    expected = pd.DataFrame(
        [[1], [2]], columns=pd.MultiIndex.from_tuples([("a", "x")])
    )
    actual = cudf.from_pandas(expected)

    def expand(frame):
        if operation == "assign":
            return frame.assign(new=[2, 3])
        if operation == "insert":
            frame.insert(1, key, [2, 3])
        else:
            frame[key] = [2, 3]
        return frame

    if isinstance(key, tuple) and len(key) == 1:
        with pytest.raises(
            ValueError, match="length equal to number of levels"
        ):
            expand(expected)
        with pytest.raises(
            ValueError, match="length equal to number of levels"
        ):
            expand(actual)
        return

    with warnings.catch_warnings(record=True) as expected_warnings:
        warnings.simplefilter("always")
        expected = expand(expected)
    with warnings.catch_warnings(record=True) as actual_warnings:
        warnings.simplefilter("always")
        actual = expand(actual)
    assert [warning.category for warning in actual_warnings] == [
        warning.category for warning in expected_warnings
    ]
    assert [str(warning.message) for warning in actual_warnings] == [
        str(warning.message) for warning in expected_warnings
    ]
    assert len(actual_warnings) == int(PANDAS_GE_31 and isinstance(key, str))
    assert_eq(actual, expected)
