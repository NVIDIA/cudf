# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


def _index(kind):
    if kind == "range":
        return pd.RangeIndex(2, name="before")
    if kind == "datetime":
        return pd.DatetimeIndex(["2020-01-01", "2020-01-02"], name="before")
    if kind == "timedelta":
        return pd.TimedeltaIndex(["0D", "1D"], name="before")
    if kind == "categorical":
        return pd.CategoricalIndex(["a", "b"], name="before")
    if kind == "interval":
        return pd.IntervalIndex.from_breaks([0, 1, 2], name="before")
    return pd.Index(["a", "b"], name="before")


@pytest.mark.parametrize(
    "kind",
    ["base", "range", "datetime", "timedelta", "categorical", "interval"],
)
@pytest.mark.parametrize("inplace", [False, True])
def test_index_rename_preserves_owner_alias(kind, inplace):
    expected = pd.Series([1, 2], index=_index(kind))
    actual = cudf.from_pandas(expected)
    index = actual.index
    expected_result = expected.index.rename(None, inplace=inplace)
    result = index.rename(None, inplace=inplace)
    assert_eq(expected, actual)
    if inplace:
        assert result is None
        assert actual.index is index
    else:
        assert_eq(expected_result, result)
        assert actual.index.name == "before"
