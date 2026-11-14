# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

import cudf


@pytest.mark.parametrize("width", [40, 100])
@pytest.mark.parametrize(
    "name", [None, "my_long_index_name", ("tuple", "name")]
)
def test_rangeindex_repr_display_width(width, name):
    with pd.option_context("display.width", width):
        expected = repr(pd.RangeIndex(100, name=name))
        actual = repr(cudf.RangeIndex(100, name=name))
    assert actual == expected
