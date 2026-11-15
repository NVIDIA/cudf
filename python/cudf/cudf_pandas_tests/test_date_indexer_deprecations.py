# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import datetime
import warnings

import numpy as np
import pandas as pd
import pytest

from cudf.pandas.fast_slow_proxy import _slow_arg
from cudf.pandas.module_accelerator import disable_module_accelerator


@pytest.mark.parametrize("container", ["list", "array", "index"])
@pytest.mark.parametrize(
    "operation", ["get_indexer", "series_loc", "dataframe_loc"]
)
def test_date_indexer_warning_matches_pandas(container, operation):
    def select():
        values = [datetime.date(2020, 1, 1), datetime.date(2020, 1, 2)]
        keys = (
            values
            if container == "list"
            else np.array(values)
            if container == "array"
            else pd.Index(values)
        )
        index = pd.DatetimeIndex(["2020-01-01", "2020-01-02"])
        if operation == "get_indexer":
            return index.get_indexer(keys)
        if operation == "series_loc":
            return pd.Series([1, 2], index=index).loc[keys]
        return pd.DataFrame({"a": [1, 2]}, index=index).loc[keys]

    with (
        disable_module_accelerator(),
        warnings.catch_warnings(record=True) as expected_warnings,
    ):
        warnings.simplefilter("always")
        expected = select()
    with warnings.catch_warnings(record=True) as actual_warnings:
        warnings.simplefilter("always")
        actual = select()
    assert [warning.category for warning in actual_warnings] == [
        warning.category for warning in expected_warnings
    ]
    assert [str(warning.message) for warning in actual_warnings] == [
        str(warning.message) for warning in expected_warnings
    ]
    with disable_module_accelerator():
        if operation == "get_indexer":
            np.testing.assert_array_equal(_slow_arg(actual), expected)
        elif operation == "series_loc":
            pd.testing.assert_series_equal(_slow_arg(actual), expected)
        else:
            pd.testing.assert_frame_equal(_slow_arg(actual), expected)


@pytest.mark.parametrize("kind", ["series", "dataframe"])
def test_missing_date_labels_warn_once(kind):
    keys = [datetime.date(2020, 1, 1), datetime.date(2020, 1, 3)]

    def make_frame():
        index = pd.DatetimeIndex(["2020-01-01", "2020-01-02"])
        if kind == "series":
            return pd.Series([1, 2], index=index)
        return pd.DataFrame({"a": [1, 2]}, index=index)

    with (
        disable_module_accelerator(),
        warnings.catch_warnings(record=True) as expected_warnings,
    ):
        warnings.simplefilter("always")
        with pytest.raises(KeyError) as expected_error:
            make_frame().loc[keys]
    with warnings.catch_warnings(record=True) as actual_warnings:
        warnings.simplefilter("always")
        with pytest.raises(KeyError) as actual_error:
            make_frame().loc[keys]
    assert str(actual_error.value) == str(expected_error.value)
    assert [
        (warning.category, str(warning.message)) for warning in actual_warnings
    ] == [
        (warning.category, str(warning.message))
        for warning in expected_warnings
    ]
