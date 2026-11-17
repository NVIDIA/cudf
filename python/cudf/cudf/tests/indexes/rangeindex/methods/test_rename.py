# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd

import cudf


def test_rename_shallow_copy():
    idx = pd.Index([1])
    result = idx.rename("a")
    assert np.shares_memory(
        idx.to_numpy(copy=False), result.to_numpy(copy=False)
    )

    idx = cudf.Index([1])
    result = idx.rename("a")
    assert idx._column.data.ptr == result._column.data.ptr
