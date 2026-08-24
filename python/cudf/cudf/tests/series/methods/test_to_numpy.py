# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import cupy as cp
import numpy as np
import pandas as pd
import pytest

import cudf


@pytest.mark.parametrize(
    "data",
    [
        [1000000, 200000, 3000000],
        [1000000, 200000, None],
        [],
        [None],
        [None, None, None, None, None],
        [12, 12, 22, 343, 4353534, 435342],
        np.array([10, 20, 30, None, 100]),
        cp.asarray([10, 20, 30, 100]),
    ],
)
def test_timedelta_series_to_numpy(data, timedelta_types_as_str):
    gsr = cudf.Series(data, dtype=timedelta_types_as_str)

    expected = np.array(
        cp.asnumpy(data) if isinstance(data, cp.ndarray) else data,
        dtype=timedelta_types_as_str,
    )
    expected = expected[~np.isnan(expected)]

    actual = gsr.dropna().to_numpy()

    np.testing.assert_array_equal(expected, actual)


@pytest.mark.parametrize(
    "data",
    [
        [1, 2, 4],
        [],
        [5.0, 7.0, 8.0],
        pd.Categorical(["a", "b", "c"]),
        ["m", "a", "d", "v"],
    ],
)
def test_series_to_numpy(data):
    pds = pd.Series(data=data, dtype=None if data else float)
    gds = cudf.Series(data=data, dtype=None if data else float)

    np.testing.assert_array_equal(pds.values, gds.to_numpy())
