# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warnings

import pandas as xpd
import pytest

from cudf.core._compat import PANDAS_GE_31
from cudf.pandas import LOADED
from cudf.pandas.module_accelerator import disable_module_accelerator

if not LOADED:
    raise ImportError("These tests must be run with cudf.pandas loaded")

pd = xpd._fsproxy_slow
pytestmark = pytest.mark.skipif(
    not PANDAS_GE_31, reason="pandas 3.1 inferred-frequency deprecation"
)


def _capture(index):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        values = [index.inferred_freq, index.inferred_freq]
    return values, [(type(w.message), str(w.message)) for w in caught]


def _index(module, kind, frequency):
    if kind == "datetime":
        return module.date_range("2020-01-01", periods=5, freq=frequency)
    return module.timedelta_range("0 days", periods=5, freq=frequency)


@pytest.mark.parametrize("kind", ["datetime", "timedelta"])
@pytest.mark.parametrize("frequency", ["D", "24h", "2h", "7D"])
@pytest.mark.parametrize("option", [None, False, True])
def test_inferred_frequency_return_type_and_warning(kind, frequency, option):
    with pd.option_context("future.infer_freq_returns_offset", option):
        with disable_module_accelerator():
            expected, expected_warnings = _capture(_index(pd, kind, frequency))
        actual, actual_warnings = _capture(_index(xpd, kind, frequency))
    assert actual == expected
    assert [isinstance(value, str) for value in actual] == [
        isinstance(value, str) for value in expected
    ]
    assert actual_warnings == expected_warnings


@pytest.mark.parametrize("option", [None, False, True])
@pytest.mark.parametrize(
    "values", [[], ["2020-01-01"], ["2020-01-01", "2020-01-03", "2020-01-08"]]
)
def test_uninferable_frequency_is_quiet(option, values):
    with pd.option_context("future.infer_freq_returns_offset", option):
        with disable_module_accelerator():
            expected, expected_warnings = _capture(pd.DatetimeIndex(values))
        actual, actual_warnings = _capture(xpd.DatetimeIndex(values))
    assert actual == expected == [None, None]
    assert actual_warnings == expected_warnings == []


@pytest.mark.parametrize("kind", ["datetime", "timedelta"])
def test_inferred_frequency_warning_as_error_and_option_change(kind):
    with disable_module_accelerator():
        expected_index = _index(pd, kind, "D")
    index = _index(xpd, kind, "D")
    with pd.option_context("future.infer_freq_returns_offset", None):
        with warnings.catch_warnings():
            warnings.simplefilter("error", pd.errors.Pandas4Warning)
            with pytest.raises(pd.errors.Pandas4Warning, match="BaseOffset"):
                _ = index.inferred_freq
            with disable_module_accelerator():
                with pytest.raises(
                    pd.errors.Pandas4Warning, match="BaseOffset"
                ):
                    _ = expected_index.inferred_freq
    for option in (True, False):
        with pd.option_context("future.infer_freq_returns_offset", option):
            with disable_module_accelerator():
                expected, expected_warnings = _capture(expected_index)
            actual, caught = _capture(index)
            assert actual == expected
            assert caught == expected_warnings == []


@pytest.mark.parametrize("frequency", ["W-MON", "ME"])
@pytest.mark.parametrize("option", [None, False, True])
def test_anchored_inferred_frequency(frequency, option):
    with pd.option_context("future.infer_freq_returns_offset", option):
        with disable_module_accelerator():
            expected_index = _index(pd, "datetime", frequency)
            expected, expected_warnings = _capture(expected_index)
        actual, actual_warnings = _capture(xpd.DatetimeIndex(expected_index))
    assert actual == expected
    assert actual_warnings == expected_warnings
