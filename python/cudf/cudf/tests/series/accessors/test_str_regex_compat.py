# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.core._compat import PANDAS_GE_31
from cudf.testing import assert_eq


@pytest.mark.parametrize("storage", ["python", "pyarrow"])
@pytest.mark.parametrize("na_value", [pd.NA, np.nan])
@pytest.mark.parametrize("data", [[], ["ab", "cd"], ["ab", None, "cd"]])
@pytest.mark.parametrize("box", ["series", "index"])
@pytest.mark.parametrize("method", ["contains", "match"])
@pytest.mark.parametrize("kwargs", [{}, {"na": False}, {"na": True}])
def test_string_regex_predicate_dtype(
    storage, na_value, data, box, method, kwargs
):
    dtype = pd.StringDtype(storage=storage, na_value=na_value)
    expected = pd.Series(data, dtype=dtype, name="values")
    actual = cudf.from_pandas(expected)
    if box == "index":
        expected = pd.Index(expected, name="values")
        actual = cudf.Index(actual, name="values")
    expected_result = getattr(expected.str, method)("ab", **kwargs)
    actual_result = getattr(actual.str, method)("ab", **kwargs)
    if box == "index":
        expected_result = pd.Index(expected_result, name="values")
    assert_eq(actual_result, expected_result)


@pytest.mark.parametrize(
    "pattern",
    [
        r"\u20AC",
        r"\U000020AC",
        r"\N{EURO SIGN}",
        r"[\u0e00-\u0e7f]",
        r"\\\u20AC",
    ],
)
def test_unicode_regex_escape_requires_fallback(pattern):
    source = cudf.Series(["a", "€", "ก"])
    with pytest.raises(NotImplementedError, match="Unicode regex escapes"):
        source.str.contains(pattern)


def test_unicode_regex_escape_validation_and_literals():
    source = cudf.Series([r"\u20AC", "€"])
    expected = source.to_pandas()
    for pattern, regex in [(r"\\u20AC", True), (r"\u20AC", False)]:
        assert_eq(
            source.str.contains(pattern, regex=regex),
            expected.str.contains(pattern, regex=regex),
        )
    with pytest.raises(re.error, match="incomplete escape"):
        source.str.contains(r"\u20")


@pytest.mark.skipif(
    not PANDAS_GE_31, reason="pandas 3.1 explicit match case validation"
)
@pytest.mark.parametrize(
    "pattern,kwargs",
    [
        ("ab", {"flags": re.IGNORECASE, "case": True}),
        (re.compile("ab", re.IGNORECASE), {"case": True}),
    ],
)
def test_match_rejects_explicit_conflicting_case(pattern, kwargs):
    expected = pd.Series(["AB", "cd"])
    actual = cudf.from_pandas(expected)
    for source in (expected, actual):
        with pytest.raises(ValueError, match="Cannot both specify"):
            source.str.match(pattern, **kwargs)


@pytest.mark.parametrize(
    "pattern,kwargs",
    [
        ("ab", {"flags": re.IGNORECASE}),
        (re.compile("ab", re.IGNORECASE), {}),
        ("AB", {"case": True}),
    ],
)
def test_match_default_case_preserves_flags(pattern, kwargs):
    expected = pd.Series(["AB", "cd"])
    actual = cudf.from_pandas(expected)
    assert_eq(
        actual.str.match(pattern, **kwargs),
        expected.str.match(pattern, **kwargs),
    )
