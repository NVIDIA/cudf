# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


def test_struct_with_datetime_and_timedelta(temporal_types_as_str):
    df = cudf.DataFrame(
        {
            "a": [12, 232, 2334],
            "datetime": cudf.Series(
                [23432, 3432423, 324324], dtype=temporal_types_as_str
            ),
        }
    )
    series = df.to_struct()
    a_array = np.array([12, 232, 2334])
    datetime_array = np.array([23432, 3432423, 324324]).astype(
        temporal_types_as_str
    )

    actual = series.to_pandas()
    values_list = []
    for i, val in enumerate(a_array):
        values_list.append({"a": val, "datetime": datetime_array[i]})

    expected = pd.Series(values_list)
    assert_eq(expected, actual)


def test_dataframe_to_struct():
    df = cudf.DataFrame()
    expect = cudf.Series(dtype=cudf.StructDtype({}))
    got = df.to_struct()
    assert_eq(expect, got)

    df = cudf.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    expect = cudf.Series(
        [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}, {"a": 3, "b": "z"}]
    )
    got = df.to_struct()
    assert_eq(expect, got)

    # check that a copy was made:
    df.loc[0, "a"] = 5
    assert_eq(got, expect)

    # check that a non-string (but convertible to string) named column can be
    # converted to struct
    df = cudf.DataFrame([[1, 2], [3, 4]], columns=[(1, "b"), 0])
    expect = cudf.Series([{"(1, 'b')": 1, "0": 2}, {"(1, 'b')": 3, "0": 4}])
    with pytest.warns(UserWarning, match="will be casted"):
        got = df.to_struct()
    assert_eq(got, expect)
