# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest

import cudf


@pytest.mark.parametrize("data", [[], [1]])
def test_index_to_numpy(data, all_supported_types_as_str):
    gdi = cudf.Index(data, dtype=all_supported_types_as_str)
    pdi = pd.Index(data, dtype=all_supported_types_as_str)

    np.testing.assert_array_equal(gdi.to_numpy(), pdi.values)
