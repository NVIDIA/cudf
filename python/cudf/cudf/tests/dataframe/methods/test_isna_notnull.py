# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "na_data",
    [
        pd.DataFrame(
            {
                "a": [0, 1, 2, np.nan, 4, None, 6],
                "b": [np.nan, None, "u", "h", "d", "a", "m"],
            },
            index=["q", "w", "e", "r", "t", "y", "u"],
        ),
        pd.DataFrame({"a": [0, 1, 2, 3, 4], "b": ["a", "b", "u", "h", "d"]}),
        pd.DataFrame(
            {
                "a": [None, None, np.nan, None],
                "b": [np.nan, None, np.nan, None],
            }
        ),
        pd.DataFrame({"a": []}),
        pd.DataFrame({"a": [np.nan], "b": [None]}),
        pd.DataFrame({"a": ["a", "b", "c", None, "e"]}),
        pd.DataFrame({"a": ["a", "b", "c", "d", "e"]}),
    ],
)
@pytest.mark.parametrize("api_call", ["isnull", "isna", "notna", "notnull"])
def test_dataframe_isnull_isna_and_reverse(na_data, nan_as_null, api_call):
    gdf = cudf.DataFrame(na_data, nan_as_null=nan_as_null)

    assert_eq(getattr(na_data, api_call)(), getattr(gdf, api_call)())

    # Test individual columns
    for col in na_data:
        assert_eq(
            getattr(na_data[col], api_call)(), getattr(gdf[col], api_call)()
        )
