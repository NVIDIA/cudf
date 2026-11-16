# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pyarrow as pa
import pytest

import cudf
from cudf.core._compat import PANDAS_GE_31
from cudf.testing import assert_eq

pytestmark = pytest.mark.skipif(
    not PANDAS_GE_31, reason="pandas 3.1 Arrow duration string casts"
)


@pytest.mark.parametrize(
    "source_dtype",
    [
        "string[pyarrow]",
        pd.ArrowDtype(pa.string()),
        pd.ArrowDtype(pa.large_string()),
    ],
)
@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
@pytest.mark.parametrize(
    "values", [["33", None, "444444444"], [], [None, None]]
)
def test_string_to_arrow_duration_preserves_ticks_and_mask(
    source_dtype, unit, values
):
    expected_source = pd.Series(
        ["11", "22", *values], dtype=source_dtype
    ).iloc[2:]
    actual_source = cudf.from_pandas(
        pd.Series(["11", "22", *values], dtype=source_dtype)
    ).iloc[2:]
    dtype = pd.ArrowDtype(pa.duration(unit))
    assert_eq(expected_source.astype(dtype), actual_source.astype(dtype))
