# SPDX-FileCopyrightText: Copyright (c) 2019, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pytest

import cudf

is_dataframe_like = pytest.importorskip(
    "dask.dataframe.utils"
).is_dataframe_like
is_index_like = pytest.importorskip("dask.dataframe.utils").is_index_like
is_series_like = pytest.importorskip("dask.dataframe.utils").is_series_like


def test_is_dataframe_like():
    df = cudf.DataFrame({"x": [1, 2, 3]})
    assert is_dataframe_like(df)
    assert is_series_like(df.x)
    assert is_index_like(df.index)
    assert not is_dataframe_like(df.x)
    assert not is_series_like(df)
    assert not is_index_like(df)
