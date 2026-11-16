# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
import warnings

import pandas as xpd
import pytest

from cudf.core._compat import PANDAS_GE_31
from cudf.pandas import LOADED
from cudf.testing import assert_eq

if not LOADED:
    raise ImportError("These tests must be run with cudf.pandas loaded")

pd = xpd._fsproxy_slow


@pytest.mark.parametrize(
    "api,kwargs",
    [
        ("from_records", {}),
        ("read_csv", {"float_precision": "high"}),
        ("read_json", {"convert_dates": False}),
        ("read_json", {"keep_default_dates": False}),
    ],
)
def test_constructor_io_deprecations_through_fallback(api, kwargs):
    results = []
    recorded = []
    for module in (pd, xpd):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if api == "from_records":
                result = module.DataFrame.from_records({"a": [1, 2]})
            elif api == "read_csv":
                result = module.read_csv(io.StringIO("a\n1.25\n"), **kwargs)
            else:
                result = module.read_json(
                    io.StringIO('{"a": 1}\n'), lines=True, **kwargs
                )
        results.append(result)
        recorded.append([(type(w.message), str(w.message)) for w in caught])
    assert recorded[0] == recorded[1]
    assert_eq(results[0], results[1]._fsproxy_slow)


@pytest.mark.skipif(
    not PANDAS_GE_31, reason="pandas 3.1 reindex dtype deprecation"
)
@pytest.mark.parametrize(
    "dtype,value",
    [
        ("datetime64[ns]", "2001-01-01"),
        ("datetime64[us]", "2001-01-01"),
        ("datetime64[us]", "3000-01-01"),
        ("timedelta64[us]", "1 day"),
    ],
)
@pytest.mark.parametrize("frame", [False, True])
def test_temporal_string_reindex_warning_and_dtype(dtype, value, frame):
    results = []
    recorded = []
    for module in (pd, xpd):
        source = module.Series([None], dtype=dtype)
        if frame:
            source = source.to_frame()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = source.reindex([0, 1], fill_value=value)
        results.append(result)
        recorded.append([(type(w.message), str(w.message)) for w in caught])
    assert recorded[0] == recorded[1]
    assert_eq(results[0], results[1]._fsproxy_slow)


@pytest.mark.parametrize(
    "tuple_column",
    [
        False,
        pytest.param(
            True,
            marks=pytest.mark.skipif(
                not PANDAS_GE_31,
                reason="pandas 3.1 Resampler tuple validation",
            ),
        ),
    ],
)
def test_resampler_tuple_warning(tuple_column):
    recorded = []
    results = []
    for module in (pd, xpd):
        data = (
            {("a", "b"): [1, 2]}
            if tuple_column
            else {"a": [1, 2], "b": [3, 4]}
        )
        source = module.DataFrame(
            data, index=module.date_range("2020-01-01", periods=2)
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                result = source.resample("2D")[("a", "b")].sum()
            except TypeError as error:
                result = (type(error), str(error))
        results.append(result)
        recorded.append([(type(w.message), str(w.message)) for w in caught])
    assert recorded[0] == recorded[1]
    if tuple_column:
        assert results[0] == results[1]
    else:
        assert_eq(results[0], results[1]._fsproxy_slow)


def test_groupby_duplicate_level_name_warning():
    recorded = []
    results = []
    for module in (pd, xpd):
        index = module.MultiIndex.from_tuples(
            [(1, 1), (1, 2), (1, 3)], names=["f", "f"]
        )
        source = module.DataFrame(
            {"a": [1, 1, 2], "b": [3, 4, 5]}, index=index
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = source.groupby("f").sum()
        results.append(result)
        recorded.append([(type(w.message), str(w.message)) for w in caught])
    assert recorded[0] == recorded[1]
    assert_eq(results[0], results[1]._fsproxy_slow)
