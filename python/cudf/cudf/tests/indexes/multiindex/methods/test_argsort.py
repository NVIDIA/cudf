# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "pdi",
    [
        pd.MultiIndex(
            levels=[[1, 3.0, 4, 5], [1, 2.3, 5]],
            codes=[[0, 0, 1, 2, 3], [0, 2, 1, 1, 0]],
            names=["x", "y"],
        ),
        pd.MultiIndex(
            levels=[[1, 3, 4, -10], [1, 11, 5]],
            codes=[[0, 0, 1, 2, 3], [0, 2, 1, 1, 0]],
            names=["x", "y"],
        ),
        pd.MultiIndex(
            levels=[["a", "b", "c", "100"], ["1", "100", "5"]],
            codes=[[0, 0, 1, 2, 3], [0, 2, 1, 1, 0]],
            names=["x", "y"],
        ),
        pd.MultiIndex(
            levels=[[None, "b", "c", "a"], ["1", None, "5"]],
            codes=[[0, 0, 1, 2, 3], [0, 2, 1, 1, 0]],
            names=["x", "y"],
        ),
    ],
)
@pytest.mark.parametrize("ascending", [True, False])
def test_multiindex_argsort(pdi, ascending, request):
    request.applymarker(
        pytest.mark.xfail(
            not ascending
            and any(pd.isna(level).any() for level in pdi.levels),
            reason=f"argsort with {ascending=} with NA levels is incorrect.",
        )
    )
    gdi = cudf.from_pandas(pdi)

    if not ascending:
        expected = pdi.argsort()[::-1]
    else:
        expected = pdi.argsort()

    actual = gdi.argsort(ascending=ascending)

    assert_eq(expected, actual)


def test_multiindex_argsort_returns_intp():
    # argsort returns np.intp (int64) positional indexers, matching
    # numpy/pandas.
    pmi = pd.MultiIndex.from_arrays([[2, 1, 3], [1, 2, 3]])
    gmi = cudf.from_pandas(pmi)

    result = gmi.argsort()
    assert result.dtype == np.dtype(np.intp)
    assert_eq(result, pmi.argsort(), check_dtype=True)
