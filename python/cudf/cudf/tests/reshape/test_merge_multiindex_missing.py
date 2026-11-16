# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.core._compat import PANDAS_GE_31
from cudf.testing import assert_eq


@pytest.mark.skipif(
    not PANDAS_GE_31, reason="pandas 3.1 matches missing MultiIndex merge keys"
)
@pytest.mark.parametrize("index_side", ["left", "right"])
@pytest.mark.parametrize("how", ["inner", "left", "outer"])
def test_merge_multiindex_matches_numpy_nan_and_null(index_side, how):
    data = {
        "key1": [1.0, np.nan, 3.0],
        "key2": ["a", "b", "c"],
        "val": [10, 20, 30],
    }
    index = pd.MultiIndex.from_arrays(
        [data["key1"], data["key2"]], names=["key1", "key2"]
    )
    column_frame = pd.DataFrame(data)
    index_frame = pd.DataFrame({"data": [100, 200, 300]}, index=index)
    column_gpu = cudf.DataFrame(column_frame, nan_as_null=False)
    index_gpu = cudf.from_pandas(index_frame)
    original_column_null_count = column_gpu._data["key1"].null_count
    if index_side == "right":
        args = {"left_on": ["key1", "key2"], "right_index": True, "how": how}
        expected = column_frame.merge(index_frame, **args)
        actual = column_gpu.merge(index_gpu, **args)
    else:
        args = {"left_index": True, "right_on": ["key1", "key2"], "how": how}
        expected = index_frame.merge(column_frame, **args)
        actual = index_gpu.merge(column_gpu, **args)
    assert_eq(expected, actual, check_like=True)
    assert column_gpu._data["key1"].null_count == original_column_null_count
