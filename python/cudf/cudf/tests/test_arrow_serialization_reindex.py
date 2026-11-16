# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pickle

import pandas as pd
import pyarrow as pa
import pytest

import cudf
from cudf.core._compat import PANDAS_GE_31
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "dtype",
    ["object", pd.ArrowDtype(pa.string()), pd.ArrowDtype(pa.large_string())],
)
@pytest.mark.parametrize(
    "rows", [slice(0, 2), slice(20, 22), slice(-2, None), slice(0, 0)]
)
def test_string_pickle_compacts_slice(dtype, rows):
    values = [None if i % 7 == 0 else f"{i}:" + "x" * 100 for i in range(1000)]
    original = cudf.Series(values, dtype=dtype)
    assert not original._column._is_sliced()
    sliced = original.iloc[rows]
    full_pickled = pickle.dumps(original)
    sliced_pickled = pickle.dumps(sliced)

    assert len(sliced_pickled) < len(full_pickled) / 4
    assert_eq(pickle.loads(full_pickled), original)
    assert_eq(pickle.loads(sliced_pickled), sliced)


@pytest.mark.skipif(
    not PANDAS_GE_31, reason="Arrow timezone validation changed in pandas 3.1"
)
@pytest.mark.parametrize("as_pydatetime", [False, True])
@pytest.mark.parametrize("box", ["series", "dataframe"])
@pytest.mark.parametrize("add_row", [False, True])
def test_arrow_timestamp_reindex_fill_awareness(as_pydatetime, box, add_row):
    expected = pd.Series([1, None], dtype=pd.ArrowDtype(pa.timestamp("ns")))
    value = pd.Timestamp("2016-01-01", tz="US/Eastern")
    if as_pydatetime:
        value = value.to_pydatetime()
    if box == "dataframe":
        expected = expected.to_frame()
    actual = cudf.from_pandas(expected)
    index = [0, 1, 2] if add_row else [0, 1]

    if add_row:
        for source in (expected, actual):
            with pytest.raises(TypeError, match="Invalid value"):
                source.reindex(index, fill_value=value)
    else:
        assert_eq(
            actual.reindex(index, fill_value=value),
            expected.reindex(index, fill_value=value),
        )
