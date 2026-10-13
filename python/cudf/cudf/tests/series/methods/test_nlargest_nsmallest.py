# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq
from cudf.testing._utils import (
    assert_exceptions_equal,
)


@pytest.mark.parametrize("attr", ["nlargest", "nsmallest"])
@pytest.mark.parametrize(
    "data,dtype",
    [
        (["a", "b", "c", "d", "e"], None),
        (["a", "b", "c", "d", "e"], "category"),
    ],
)
def test_series_nlargest_nsmallest_unsupported_dtype_error(attr, data, dtype):
    # pandas rejects nlargest/nsmallest on string and categorical dtypes.
    if dtype is None:
        ps = pd.Series(data)
    else:
        ps = pd.Series(data, dtype=dtype)
    gs = cudf.from_pandas(ps)

    assert_exceptions_equal(
        getattr(ps, attr), getattr(gs, attr), ([], {"n": 1}), ([], {"n": 1})
    )


@pytest.mark.parametrize("data", [[0, 1, 1, 2, 2, 2, 3, 3], [0], [1, 2, 3]])
@pytest.mark.parametrize("n", [-100, -2, 0, 1, 4])
def test_series_nlargest(data, n):
    """Indirectly tests Series.sort_values()"""
    sr = cudf.Series(data)
    psr = pd.Series(data)
    assert_eq(sr.nlargest(n), psr.nlargest(n))
    assert_eq(sr.nlargest(n, keep="last"), psr.nlargest(n, keep="last"))

    assert_exceptions_equal(
        lfunc=psr.nlargest,
        rfunc=sr.nlargest,
        lfunc_args_and_kwargs=([], {"n": 3, "keep": "what"}),
        rfunc_args_and_kwargs=([], {"n": 3, "keep": "what"}),
    )


@pytest.mark.parametrize("data", [[0, 1, 1, 2, 2, 2, 3, 3], [0], [1, 2, 3]])
@pytest.mark.parametrize("n", [-100, -2, 0, 1, 4])
def test_series_nsmallest(data, n):
    """Indirectly tests Series.sort_values()"""
    sr = cudf.Series(data)
    psr = pd.Series(data)
    assert_eq(sr.nsmallest(n), psr.nsmallest(n))
    assert_eq(
        sr.nsmallest(n, keep="last").sort_index(),
        psr.nsmallest(n, keep="last").sort_index(),
    )

    assert_exceptions_equal(
        lfunc=psr.nsmallest,
        rfunc=sr.nsmallest,
        lfunc_args_and_kwargs=([], {"n": 3, "keep": "what"}),
        rfunc_args_and_kwargs=([], {"n": 3, "keep": "what"}),
    )


@pytest.mark.parametrize("attr", ["nlargest", "nsmallest"])
def test_series_nlargest_nsmallest_keep_last_with_nulls(attr):
    data = [3.0, None, 5.0, 2.0]
    psr = pd.Series(data)
    sr = cudf.Series(data)
    # n covers every row, so the null has to come after the valid values.
    assert_eq(
        getattr(sr, attr)(4, keep="last"), getattr(psr, attr)(4, keep="last")
    )
