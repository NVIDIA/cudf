# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest

import cudf


@pytest.mark.parametrize(
    "psr",
    [
        pd.Series([None, 3, 4, None], dtype="Int64"),
        pd.Series([1, 2, 3], index=["a", "b", "c"]),
        pd.Series([np.nan, 1.5, np.nan], index=[10, 20, 30]),
        pd.Series([None, None, "x", None]),
        pd.Series([None, None], dtype="float64"),
        pd.Series([np.nan, np.nan]),
        pd.Series([], dtype="float64"),
        pd.Series(
            [None, 1, None],
            index=pd.date_range("2020-01-01", periods=3, freq="D"),
            dtype="Int64",
        ),
    ],
)
@pytest.mark.parametrize("method", ["first_valid_index", "last_valid_index"])
@pytest.mark.parametrize("nan_as_null", [True, False])
def test_series_first_last_valid_index(psr, method, nan_as_null):
    gsr = cudf.from_pandas(psr, nan_as_null=nan_as_null)
    assert getattr(gsr, method)() == getattr(psr, method)()
