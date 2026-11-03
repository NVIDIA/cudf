# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame({"A": [1, 2, 3, np.nan, None, 6]}),
        pd.Series([1, 2, 3, None, np.nan, 5, 6, np.nan]),
    ],
)
def test_dataframe_bfill(df):
    gdf = cudf.from_pandas(df)

    actual = df.bfill()
    expected = gdf.bfill()
    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame({"A": [1, 2, 3, np.nan, None, 6]}),
        pd.Series([1, 2, 3, None, np.nan, 5, 6, np.nan]),
    ],
)
def test_dataframe_ffill(df):
    gdf = cudf.from_pandas(df)

    actual = df.ffill()
    expected = gdf.ffill()
    assert_eq(expected, actual)
