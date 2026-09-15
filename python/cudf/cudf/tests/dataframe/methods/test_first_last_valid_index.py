# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

import cudf


@pytest.mark.parametrize(
    "data,index",
    [
        ([None, 3, 4], None),
        ([None, None], None),
        ([1, 2, 3, 4], None),
        ([], None),
        ([None, 3, 4], ["x", "y", "z"]),
    ],
)
def test_series_first_last_valid_index(data, index):
    ps = pd.Series(data, index=index, dtype="float64" if data else "object")
    gs = cudf.from_pandas(ps)

    assert gs.first_valid_index() == ps.first_valid_index()
    assert gs.last_valid_index() == ps.last_valid_index()


@pytest.mark.parametrize(
    "data",
    [
        {"A": [None, None, 2], "B": [None, 3, 4]},
        {"A": [None, None, None], "B": [None, None, None]},
        {"A": [1, 2, 3], "B": [4, 5, 6]},
        {},
    ],
)
def test_dataframe_first_last_valid_index(data):
    pdf = pd.DataFrame(data)
    gdf = cudf.from_pandas(pdf)

    assert gdf.first_valid_index() == pdf.first_valid_index()
    assert gdf.last_valid_index() == pdf.last_valid_index()
