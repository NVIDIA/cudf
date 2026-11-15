# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warnings
from contextlib import nullcontext

import pandas as pd
import pytest

import cudf
from cudf.core._compat import PANDAS_GE_31
from cudf.testing import assert_eq


def _join_with_warnings(frame, other, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = frame.join(other, **kwargs)
    return result, [
        (warning.category, str(warning.message)) for warning in caught
    ]


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"lsuffix": ""},
        {"rsuffix": ""},
        {"lsuffix": "_left"},
        {"rsuffix": "_right"},
        {"lsuffix": "_left", "rsuffix": "_right"},
    ],
)
def test_join_legacy_suffix_warning(kwargs):
    left = pd.DataFrame({"a": [1.0, 2.0]})
    right = pd.DataFrame({"b": [3.0, 4.0]})
    expected, expected_warnings = _join_with_warnings(left, right, **kwargs)
    actual, actual_warnings = _join_with_warnings(
        cudf.from_pandas(left), cudf.from_pandas(right), **kwargs
    )
    assert_eq(actual, expected)
    assert actual_warnings == expected_warnings
    assert len(actual_warnings) == int(PANDAS_GE_31 and bool(kwargs))


@pytest.mark.parametrize("how", ["left", "right", "inner", "outer", "cross"])
@pytest.mark.parametrize("preferred", [False, True])
@pytest.mark.parametrize("suffixes", [("_left", "_right"), (None, "_right")])
def test_join_suffix_values(how, preferred, suffixes):
    left = pd.DataFrame({"a": [1.0, 2.0]})
    right = pd.DataFrame({"a": [3.0, 4.0]})
    kwargs = {"how": how, "sort": True}
    if preferred:
        kwargs["suffixes"] = suffixes
    else:
        kwargs.update(zip(("lsuffix", "rsuffix"), suffixes, strict=True))
    if preferred and not PANDAS_GE_31:
        with pytest.raises(
            TypeError, match="unexpected keyword argument 'suffixes'"
        ):
            cudf.from_pandas(left).join(cudf.from_pandas(right), **kwargs)
        return
    expected, expected_warnings = _join_with_warnings(left, right, **kwargs)
    actual, actual_warnings = _join_with_warnings(
        cudf.from_pandas(left), cudf.from_pandas(right), **kwargs
    )
    assert_eq(actual, expected)
    assert actual_warnings == expected_warnings


@pytest.mark.parametrize("legacy", [{"lsuffix": ""}, {"rsuffix": "_right"}])
def test_join_suffix_conflict_has_no_warning(legacy):
    frame = cudf.DataFrame({"a": [1.0]})
    error = ValueError if PANDAS_GE_31 else TypeError
    message = (
        "Cannot specify both 'suffixes' and 'lsuffix'/'rsuffix'"
        if PANDAS_GE_31
        else "unexpected keyword argument 'suffixes'"
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(error, match=message):
            frame.join(frame, suffixes=("_left", "_right"), **legacy)
    assert not caught


@pytest.mark.parametrize(
    "suffixes", [None, "_right", {}, (), ("_left",), ("a", "b", "c")]
)
def test_join_invalid_preferred_suffixes(suffixes):
    left = pd.DataFrame({"a": [1.0]})
    error = (
        TypeError
        if not PANDAS_GE_31 or not isinstance(suffixes, tuple)
        else ValueError
    )
    with pytest.raises(error):
        left.join(left, suffixes=suffixes)
    frame = cudf.from_pandas(left)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(error):
            frame.join(frame, suffixes=suffixes)
    assert not caught


def test_join_legacy_warning_error_preserves_inputs():
    left = pd.DataFrame({"a": [1.0, 2.0]}, index=pd.Index([0, 1], name="rows"))
    right = pd.DataFrame(
        {"a": [3.0, 4.0]}, index=pd.Index([0, 1], name="rows")
    )
    left.columns.name = "left_columns"
    right.columns.name = "right_columns"
    actual_left = cudf.from_pandas(left)
    actual_right = cudf.from_pandas(right)
    with warnings.catch_warnings():
        warnings.simplefilter("error", pd.errors.Pandas4Warning)
        context = (
            pytest.raises(pd.errors.Pandas4Warning, match=r"lsuffix.*rsuffix")
            if PANDAS_GE_31
            else nullcontext()
        )
        with context:
            actual_left.join(actual_right, lsuffix="_left", rsuffix="_right")
    assert_eq(actual_left, left)
    assert_eq(actual_right, right)
    assert actual_left.index.name == actual_right.index.name == "rows"
    assert actual_left.columns.name == "left_columns"
    assert actual_right.columns.name == "right_columns"
