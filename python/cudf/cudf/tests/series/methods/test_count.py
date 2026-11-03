# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


def test_series_count_invalid_param():
    s = cudf.Series([], dtype="float64")
    with pytest.raises(TypeError):
        s.count(skipna=True)


def test_series_dataframe_count_float():
    gs = cudf.Series([1, 2, 3, None, np.nan, 10], nan_as_null=False)
    ps = pd.Series([1, 2, 3, None, np.nan, 10])

    assert_eq(ps.count(), gs.count())
    assert_eq(ps.to_frame().count(), gs.to_frame().count())


def test_series_dataframe_count_nullable_int():
    gs = cudf.Series([1, 2, 3, None, np.nan, 10], dtype="Float64")
    assert gs.count() == gs.to_pandas().count()
    assert_eq(
        gs.to_frame().count(),
        gs.to_frame().to_pandas().count(),
    )
