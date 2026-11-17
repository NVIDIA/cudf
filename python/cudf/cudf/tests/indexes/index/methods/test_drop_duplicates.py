# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize("data", [[1, 2, 3, 1, 2, 3, 4], [], [1], [1, 2, 3]])
def test_index_drop_duplicates(data, all_supported_types_as_str, request):
    pdi = pd.Index(data, dtype=all_supported_types_as_str)
    gdi = cudf.Index(data, dtype=all_supported_types_as_str)

    result = gdi.drop_duplicates()
    expected = pdi.drop_duplicates()
    if data == [] and all_supported_types_as_str == "category":
        # As of pandas 3.0, empty default type of object isn't
        # necessarily equivalent to cuDF's empty default type of
        # pandas.StringDtype
        expected = expected.set_categories(
            pd.Index([], dtype=result.categories.dtype)
        )
    assert_eq(expected, result)
