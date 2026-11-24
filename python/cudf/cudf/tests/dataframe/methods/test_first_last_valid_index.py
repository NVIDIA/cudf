# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest

import cudf


@pytest.mark.parametrize(
    "pdf",
    [
        pd.DataFrame(
            {"a": [None, None, 2, None], "b": [None, 3, None, None]},
            dtype="Int64",
        ),
        pd.DataFrame(
            {"a": [np.nan, 1.0, np.nan], "b": ["x", None, None]},
            index=["p", "q", "r"],
        ),
        pd.DataFrame({"a": [None, None], "b": [None, None]}, dtype="float64"),
        pd.DataFrame({"a": [], "b": []}, dtype="float64"),
        pd.DataFrame(index=range(3)),
    ],
)
@pytest.mark.parametrize("method", ["first_valid_index", "last_valid_index"])
def test_dataframe_first_last_valid_index(pdf, method):
    gdf = cudf.from_pandas(pdf)
    assert getattr(gdf, method)() == getattr(pdf, method)()
