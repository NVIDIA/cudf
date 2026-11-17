# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import pandas as pd
import pytest

import cudf


def test_categorical_index_with_nan_repr():
    cat_index = cudf.Index(
        cudf.Series(
            [1, 2, np.nan, 10, np.nan, None], nan_as_null=False
        ).astype("category")
    )

    expected_repr = (
        "CategoricalIndex([1.0, 2.0, <NA>, 10.0, <NA>, <NA>], "
        "categories=[1.0, 2.0, 10.0], ordered=False, dtype='category')"
    )

    assert repr(cat_index) == expected_repr

    sliced_expected_repr = (
        "CategoricalIndex([<NA>, 10.0, <NA>, <NA>], "
        "categories=[1.0, 2.0, 10.0], ordered=False, dtype='category')"
    )

    assert repr(cat_index[2:]) == sliced_expected_repr


def test_unique_categories_repr():
    pi = pd.CategoricalIndex(range(10_000))
    gi = cudf.CategoricalIndex(range(10_000))
    expected_repr = repr(pi)
    actual_repr = repr(gi)
    assert expected_repr == actual_repr


@pytest.mark.parametrize("ordered", [True, False])
def test_categorical_index_ordered(ordered):
    pi = pd.CategoricalIndex(range(10), ordered=ordered)
    gi = cudf.CategoricalIndex(range(10), ordered=ordered)

    assert repr(pi) == repr(gi)
