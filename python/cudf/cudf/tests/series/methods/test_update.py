# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize("index", [None, [1, 2, 3]])
@pytest.mark.parametrize(
    "other",
    [
        pd.Series([4, 5, 6]),
        pd.Series([4, 5, 6, 7, 8]),
        pd.Series([4, np.nan, 6]),
        [4, np.nan, 6],
        {1: 9},
    ],
)
def test_series_update(index, other):
    ps = pd.Series([1, 2, 3], index=index)
    data = cudf.Series(ps)
    gs = data.copy(deep=True)
    if isinstance(other, pd.Series):
        other = cudf.Series(other, nan_as_null=False)
        g_other = other.copy(deep=True)
        p_other = g_other.to_pandas()
    else:
        g_other = other
        p_other = other

    ps.update(p_other)
    gs.update(g_other)
    assert_eq(gs, ps)


@pytest.mark.parametrize("dtype", ["int8", "int16", "int32", "int64"])
def test_series_update_lossy_float_to_int_pandas_compat(dtype):
    # pandas raises TypeError when Series.update would cast float values
    # with non-zero fractional parts into an integer column. cuDF must
    # match this behavior under pandas-compatible mode.
    from cudf.testing._utils import assert_exceptions_equal

    psr = pd.Series([10, 11, 12], dtype=dtype)
    gsr = cudf.from_pandas(psr)
    pother = pd.Series([61.1, 63.1], index=[1, 3])
    gother = cudf.from_pandas(pother)
    with cudf.option_context("mode.pandas_compatible", True):
        assert_exceptions_equal(
            lfunc=psr.update,
            rfunc=gsr.update,
            lfunc_args_and_kwargs=([pother], {}),
            rfunc_args_and_kwargs=([gother], {}),
        )
