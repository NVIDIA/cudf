# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

import cudf


@pytest.fixture(scope="module")
def df_with_index():
    # All consumers except the scalar setter only read this dataframe.
    return cudf.DataFrame(
        {"A": [1, 2, 3], "B": [4, 5, 6]}, index=["x", "y", "z"]
    )


@pytest.fixture(scope="module")
def df_without_index():
    return cudf.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})


def test_dataframe_at_scalar_getitem(df_with_index):
    assert df_with_index.at["x", "A"] == 1
    assert df_with_index.at["y", "B"] == 5


def test_dataframe_at_scalar_setitem(df_with_index):
    df_with_index = df_with_index.copy(deep=True)
    df_with_index.at["x", "A"] = 10
    assert df_with_index.at["x", "A"] == 10


@pytest.mark.parametrize("key", [[["x"], "A"], ["x", ["A"]], [["x"], ["A"]]])
def test_dataframe_at_rejects_list_like(df_with_index, key):
    with pytest.raises(ValueError, match="Invalid call for scalar access"):
        df_with_index.at[key[0], key[1]]


@pytest.mark.parametrize("key", [[["x"], "A"], ["x", ["A"]]])
def test_dataframe_at_setitem_rejects_list_like(df_with_index, key):
    with pytest.raises(ValueError, match="Invalid call for scalar access"):
        df_with_index.at[key[0], key[1]] = 10


def test_dataframe_iat_scalar_getitem(df_without_index):
    assert df_without_index.iat[0, 0] == 1
    assert df_without_index.iat[1, 1] == 5


def test_dataframe_iat_scalar_setitem(df_without_index):
    df_without_index = df_without_index.copy(deep=True)
    df_without_index.iat[0, 0] = 10
    assert df_without_index.iat[0, 0] == 10


@pytest.mark.parametrize("key", [[[0], 0], [0, [0]], [[0], [0]]])
def test_dataframe_iat_rejects_list_like(df_without_index, key):
    with pytest.raises(
        ValueError, match="iAt based indexing can only have integer indexers"
    ):
        df_without_index.iat[key[0], key[1]]


@pytest.mark.parametrize("key", [[[0], 0], [0, [0]]])
def test_dataframe_iat_setitem_rejects_list_like(df_without_index, key):
    with pytest.raises(
        ValueError, match="iAt based indexing can only have integer indexers"
    ):
        df_without_index.iat[key[0], key[1]] = 10
