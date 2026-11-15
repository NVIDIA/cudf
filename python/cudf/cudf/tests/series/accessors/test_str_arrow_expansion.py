# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re

import pandas as pd
import pyarrow as pa
import pytest

import cudf
from cudf.core._compat import PANDAS_GE_31
from cudf.testing import assert_eq


@pytest.mark.skipif(
    not PANDAS_GE_31, reason="pandas 3.1 Arrow string expansion"
)
@pytest.mark.parametrize("pa_type", [pa.string(), pa.large_string()])
@pytest.mark.parametrize("data", [[], [None, None]])
@pytest.mark.parametrize(
    "method", ["split", "rsplit", "partition", "rpartition"]
)
def test_arrow_expanded_string_without_width(data, pa_type, method):
    source = pd.Series(data, dtype=pd.ArrowDtype(pa_type), name="strings")
    expected = getattr(source.str, method)("b", expand=True)
    actual = getattr(cudf.from_pandas(source).str, method)("b", expand=True)
    assert_eq(expected, actual)
    pd.testing.assert_index_equal(expected.columns, actual.columns, exact=True)


@pytest.mark.parametrize(
    "dtype", [pd.StringDtype("python"), pd.StringDtype("pyarrow")]
)
@pytest.mark.parametrize("expand", [False, True])
def test_extract_mixed_capture_group_labels(dtype, expand):
    source = pd.Series(["a1", "b2", None, "c3"], dtype=dtype)
    pattern = r"(?P<letter>[ab])(\d)"
    assert_eq(
        source.str.extract(pattern, expand=expand),
        cudf.from_pandas(source).str.extract(pattern, expand=expand),
    )


@pytest.mark.skipif(
    not PANDAS_GE_31,
    reason="pandas 3.1 Arrow extraction accepts unnamed groups",
)
@pytest.mark.parametrize("pattern", [r"([ab])(\d)", r"(?P<letter>[ab])(\d)"])
def test_arrow_case_insensitive_extract_defers_unsupported_backend(pattern):
    source = cudf.Series(["A1", "b2", "c3"], dtype=pd.ArrowDtype(pa.string()))
    with pytest.raises(NotImplementedError, match="Case-insensitive"):
        source.str.extract(pattern, flags=re.IGNORECASE)
