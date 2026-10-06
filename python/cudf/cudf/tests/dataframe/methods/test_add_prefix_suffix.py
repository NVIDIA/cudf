# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize("axis", [None, 0, "index", 1, "columns"])
def test_dataframe_add_prefix(axis):
    cdf = cudf.DataFrame({"A": [1, 2, 3, 4], "B": [3, 4, 5, 6]})
    pdf = cdf.to_pandas()

    got = cdf.add_prefix("item_", axis=axis)
    expected = pdf.add_prefix("item_", axis=axis)

    assert_eq(got, expected, check_index_type=axis not in (0, "index"))


@pytest.mark.parametrize("axis", [None, 0, "index", 1, "columns"])
def test_dataframe_add_suffix(axis):
    cdf = cudf.DataFrame({"A": [1, 2, 3, 4], "B": [3, 4, 5, 6]})
    pdf = cdf.to_pandas()

    got = cdf.add_suffix("_item", axis=axis)
    expected = pdf.add_suffix("_item", axis=axis)

    assert_eq(got, expected, check_index_type=axis not in (0, "index"))
