# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

import cudf


@pytest.fixture(scope="module")
def sr_with_index():
    # All consumers except the scalar setter only read this Series.
    return cudf.Series([1, 2, 3], index=["x", "y", "z"])


@pytest.fixture(scope="module")
def sr_without_index():
    return cudf.Series([1, 2, 3])


def test_series_at_scalar_getitem(sr_with_index):
    assert sr_with_index.at["x"] == 1
    assert sr_with_index.at["y"] == 2


def test_series_at_scalar_setitem(sr_with_index):
    sr_with_index = sr_with_index.copy(deep=True)
    sr_with_index.at["x"] = 10
    assert sr_with_index.at["x"] == 10


@pytest.mark.parametrize("key", [[["x"]], [["x", "y"]]])
def test_series_at_rejects_list_like(sr_with_index, key):
    with pytest.raises(ValueError, match="Invalid call for scalar access"):
        sr_with_index.at[key[0]]


def test_series_at_setitem_rejects_list_like(sr_with_index):
    with pytest.raises(ValueError, match="Invalid call for scalar access"):
        sr_with_index.at[["x"]] = 10


def test_series_iat_scalar_getitem(sr_without_index):
    assert sr_without_index.iat[0] == 1
    assert sr_without_index.iat[1] == 2


def test_series_iat_scalar_setitem(sr_without_index):
    sr_without_index = sr_without_index.copy(deep=True)
    sr_without_index.iat[0] = 10
    assert sr_without_index.iat[0] == 10


@pytest.mark.parametrize("key", [[[0]], [[0, 1]]])
def test_series_iat_rejects_list_like(sr_without_index, key):
    with pytest.raises(
        ValueError, match="iAt based indexing can only have integer indexers"
    ):
        sr_without_index.iat[key[0]]


def test_series_iat_setitem_rejects_list_like(sr_without_index):
    with pytest.raises(
        ValueError, match="iAt based indexing can only have integer indexers"
    ):
        sr_without_index.iat[[0]] = 10
