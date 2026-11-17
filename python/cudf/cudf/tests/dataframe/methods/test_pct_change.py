# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "data",
    [
        [1, 4, 6, 1],
        np.array([1.123, 2.343, 5.890, 0.0]),
        {"a": [1.123, 2.343, np.nan, np.nan], "b": [None, 3, 9.08, None]},
    ],
)
@pytest.mark.parametrize("periods", [-5, 0, 2])
def test_dataframe_pct_change(data, periods):
    gdf = cudf.DataFrame(data)
    pdf = gdf.to_pandas()
    actual = gdf.pct_change(periods=periods)
    expected = pdf.pct_change(periods=periods)

    assert_eq(expected, actual)
