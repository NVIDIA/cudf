# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize("data", [[1, 2, 3], []])
@pytest.mark.parametrize("categories", [[1, 2, 3], None])
@pytest.mark.parametrize(
    "dtype",
    [
        pd.CategoricalDtype([1, 2, 3], ordered=True),
        pd.CategoricalDtype([1, 2, 3], ordered=False),
        None,
    ],
)
@pytest.mark.parametrize("ordered", [True, False])
@pytest.mark.parametrize("name", [1, "a", None])
def test_categorical_index_basic(data, categories, dtype, ordered, name):
    # can't have both dtype and categories/ordered
    if dtype is not None:
        categories = None
        ordered = None
    if data == [] and categories is None and dtype is None:
        # pandas otherwise returns Index[object] which cuDF does not support
        categories = pd.Index([], dtype=pd.StringDtype(na_value=np.nan))
    pindex = pd.CategoricalIndex(
        data=data,
        categories=categories,
        dtype=dtype,
        ordered=ordered,
        name=name,
    )
    gindex = cudf.CategoricalIndex(
        data=data,
        categories=categories,
        dtype=dtype,
        ordered=ordered,
        name=name,
    )

    assert_eq(pindex, gindex)


@pytest.mark.parametrize("ordered", [True, False])
@pytest.mark.parametrize("name", [None, "test"])
def test_categoricalindex_from_codes(ordered, name):
    codes = [0, 1, 2, 3, 4]
    categories = ["a", "b", "c", "d", "e"]
    result = cudf.CategoricalIndex.from_codes(codes, categories, ordered, name)
    expected = pd.CategoricalIndex(
        pd.Categorical.from_codes(codes, categories, ordered=ordered),
        name=name,
    )
    assert_eq(result, expected)
