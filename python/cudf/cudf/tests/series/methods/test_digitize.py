# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import cudf


@pytest.mark.parametrize("right", [True, False])
@pytest.mark.parametrize("bins_box", [np.asarray, cudf.Series])
def test_series_digitize(right, numeric_and_bool_types_as_str, bins_box):
    num_rows = 20
    num_bins = 5
    rng = np.random.default_rng(seed=0)
    data = rng.integers(0, 100, num_rows).astype(numeric_and_bool_types_as_str)
    bins = np.unique(
        np.sort(
            rng.integers(2, 95, num_bins).astype(numeric_and_bool_types_as_str)
        )
    )
    s = cudf.Series(data)
    indices = s.digitize(bins_box(bins), right)
    np.testing.assert_array_equal(
        np.digitize(data, bins, right), indices.to_numpy()
    )


def test_series_digitize_invalid_bins():
    rng = np.random.default_rng(seed=0)
    s = cudf.Series(rng.integers(0, 30, 80), dtype="int32")
    bins = cudf.Series([2, None, None, 50, 90], dtype="int32")

    with pytest.raises(
        ValueError, match=r"`bins` cannot contain null entries."
    ):
        s.digitize(bins)
