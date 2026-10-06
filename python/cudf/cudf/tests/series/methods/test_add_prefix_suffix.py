# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize("axis", [None, 0, "index"])
def test_series_add_prefix(axis):
    cd_s = cudf.Series([1, 2, 3, 4])
    pd_s = cd_s.to_pandas()

    got = cd_s.add_prefix("item_", axis=axis)
    expected = pd_s.add_prefix("item_", axis=axis)

    # Pandas still returns "object" dtype for the index, while cuDF returns "string" dtype. Ignore index type for this test.
    assert_eq(got, expected, check_index_type=False)


@pytest.mark.parametrize("axis", [None, 0, "index"])
def test_series_add_suffix(axis):
    cd_s = cudf.Series([1, 2, 3, 4])
    pd_s = cd_s.to_pandas()

    got = cd_s.add_suffix("_item", axis=axis)
    expected = pd_s.add_suffix("_item", axis=axis)

    # Pandas still returns "object" dtype for the index, while cuDF returns "string" dtype. Ignore index type for this test.
    assert_eq(got, expected, check_index_type=False)
