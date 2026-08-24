# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import cudf
from cudf.testing import assert_eq


def test_multiindex_to_numpy():
    midx = cudf.MultiIndex(
        levels=[[1, 3, 4, 5], [1, 2, 5]],
        codes=[[0, 0, 1, 2, 3], [0, 2, 1, 1, 0]],
        names=["x", "y"],
    )
    pmidx = midx.to_pandas()

    assert_eq(midx.to_numpy(), pmidx.values)
