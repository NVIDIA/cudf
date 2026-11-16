# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pyarrow as pa
import pytest

import cudf
from cudf.core._compat import PANDAS_GE_31
from cudf.testing import assert_eq


@pytest.mark.skipif(not PANDAS_GE_31, reason="Changed in pandas 3.1")
@pytest.mark.parametrize("pa_type", [pa.string(), pa.large_string()])
@pytest.mark.parametrize("method", ["findall", "index", "rindex"])
@pytest.mark.parametrize("data", [[], [None], ["abcba", None]])
def test_arrow_string_result_dtype(data, method, pa_type):
    source = pd.Series(data, dtype=pd.ArrowDtype(pa_type))
    expected = getattr(source.str, method)("b")
    result = getattr(cudf.from_pandas(source).str, method)("b")

    assert_eq(result, expected)
