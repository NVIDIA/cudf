# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest

from cudf.pandas.fast_slow_proxy import _slow_arg
from cudf.pandas.module_accelerator import disable_module_accelerator


@pytest.mark.parametrize(
    "values",
    [
        [np.datetime64(1, "10s")],
        [np.datetime64(1, "s"), np.datetime64(1, "10s")],
        ["2020-01-01", "invalid"],
    ],
)
def test_datetime_coerce_preserves_valid_values_and_shape(values):
    with disable_module_accelerator():
        expected = pd.to_datetime(values, errors="coerce")
    actual = pd.to_datetime(values, errors="coerce")

    with disable_module_accelerator():
        pd.testing.assert_index_equal(_slow_arg(actual), expected)
