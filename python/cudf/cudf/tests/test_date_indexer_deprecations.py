# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import datetime
import warnings

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "operation,kind,container",
    [
        (operation, kind, container)
        for operation in ["get_indexer", "series_loc", "dataframe_loc"]
        for kind in ["date", "datetime", "timestamp", "datetime64"]
        for container in [list, tuple, np.array]
        if (container is not tuple or operation == "get_indexer")
        and (container is not np.array or kind != "date")
    ],
)
def test_date_indexer_deprecation(container, kind, operation):
    values = [datetime.date(2020, 1, 1), datetime.date(2020, 1, 2)]
    if kind == "datetime":
        values = [
            datetime.datetime.combine(value, datetime.time())
            for value in values
        ]
    elif kind == "timestamp":
        values = [pd.Timestamp(value) for value in values]
    elif kind == "datetime64":
        values = [np.datetime64(value) for value in values]
    keys = container(values)

    def select(mod):
        index = mod.DatetimeIndex(["2020-01-01", "2020-01-02"])
        if operation == "get_indexer":
            return index.get_indexer(keys)
        if operation == "series_loc":
            return mod.Series([1, 2], index=index).loc[keys]
        return mod.DataFrame({"a": [1, 2]}, index=index).loc[keys]

    with warnings.catch_warnings(record=True) as expected_warnings:
        warnings.simplefilter("always")
        expected = select(pd)
    with warnings.catch_warnings(record=True) as actual_warnings:
        warnings.simplefilter("always")
        actual = select(cudf)
    assert [warning.category for warning in actual_warnings] == [
        warning.category for warning in expected_warnings
    ]
    assert [str(warning.message) for warning in actual_warnings] == [
        str(warning.message) for warning in expected_warnings
    ]
    assert_eq(actual, expected)
