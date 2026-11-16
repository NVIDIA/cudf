# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pyarrow as pa
import pytest

from cudf.core._compat import PANDAS_GE_31
from cudf.pandas.fast_slow_proxy import _slow_arg
from cudf.pandas.module_accelerator import disable_module_accelerator


@pytest.mark.skipif(
    not PANDAS_GE_31, reason="pandas 3.1 Arrow duration parsing"
)
@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
@pytest.mark.parametrize(
    "value",
    [
        "1 day",
        "1.5",
        "1_000",
        str(2**53 + 1),
        str(2**63 - 1),
        str(-(2**63)),
        str(2**63),
    ],
)
def test_arrow_duration_parser_boundaries_match_pandas(unit, value):
    def convert():
        return (
            pd.Series(["11", value, None], dtype="string[pyarrow]")
            .iloc[1:]
            .astype(pd.ArrowDtype(pa.duration(unit)))
        )

    try:
        with disable_module_accelerator():
            expected = convert()
    except Exception as error:
        with pytest.raises(type(error)):
            convert()
        return

    actual = convert()
    with disable_module_accelerator():
        pd.testing.assert_series_equal(_slow_arg(actual), expected)
