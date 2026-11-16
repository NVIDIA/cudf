# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warnings

import pandas as xpd
import pytest

from cudf.core._compat import PANDAS_GE_31
from cudf.pandas import LOADED
from cudf.pandas.module_accelerator import disable_module_accelerator
from cudf.testing import assert_eq

if not LOADED:
    raise ImportError("These tests must be run with cudf.pandas loaded")

pd = xpd._fsproxy_slow


def test_join_omitted_suffixes_is_quiet():
    left = xpd.DataFrame({"a": [1.0]})
    right = xpd.DataFrame({"b": [2.0]})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = left.join(right)
    assert not caught
    assert result.to_dict() == {"a": {0: 1.0}, "b": {0: 2.0}}


def _join_with_warnings(module, kwargs):
    left = module.DataFrame({"key": [0, 1], "a": [1.0, 2.0]})
    right_name = "b" if kwargs.get("other_list") else "a"
    right = module.DataFrame({right_name: [3.0, 4.0]})
    if kwargs.get("other_list"):
        other = [right]
        kwargs = {
            key: value for key, value in kwargs.items() if key != "other_list"
        }
    else:
        other = right
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = left.join(other, **kwargs)
    return result, [
        (warning.category, str(warning.message)) for warning in caught
    ]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lsuffix": "_left", "rsuffix": "_right"},
        {"lsuffix": "_left", "on": "key"},
        {"lsuffix": "_left", "validate": "one_to_one"},
        {"other_list": True, "lsuffix": ""},
    ],
)
def test_join_legacy_warning_count_including_cpu_fallback(kwargs):
    with disable_module_accelerator():
        expected, expected_warnings = _join_with_warnings(pd, kwargs)
    actual, actual_warnings = _join_with_warnings(xpd, kwargs)
    assert_eq(actual._fsproxy_slow, expected)
    assert actual_warnings == expected_warnings
    assert len(actual_warnings) == int(PANDAS_GE_31)


@pytest.mark.parametrize("suffixes", [("_left", "_right"), (None, "_right")])
def test_join_preferred_suffixes(suffixes):
    kwargs = {"suffixes": suffixes}
    if not PANDAS_GE_31:
        with pytest.raises(
            TypeError, match="unexpected keyword argument 'suffixes'"
        ):
            _join_with_warnings(xpd, kwargs)
        return
    with disable_module_accelerator():
        expected, expected_warnings = _join_with_warnings(pd, kwargs)
    actual, actual_warnings = _join_with_warnings(xpd, kwargs)
    assert_eq(actual._fsproxy_slow, expected)
    assert actual_warnings == expected_warnings == []


@pytest.mark.parametrize("legacy", [{"lsuffix": ""}, {"rsuffix": "_right"}])
def test_join_suffix_conflict_has_no_warning(legacy):
    error = ValueError if PANDAS_GE_31 else TypeError
    frame = xpd.DataFrame({"a": [1.0]})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(error):
            frame.join(frame, suffixes=("_left", "_right"), **legacy)
    assert not caught
