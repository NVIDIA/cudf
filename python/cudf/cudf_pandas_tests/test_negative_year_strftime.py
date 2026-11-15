# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest

from cudf.core._compat import PANDAS_GE_31
from cudf.pandas.fast_slow_proxy import _slow_arg
from cudf.pandas.module_accelerator import disable_module_accelerator


@pytest.mark.parametrize("unit", ["s", "ms", "us"])
@pytest.mark.parametrize("format", [None, "%Y-%m-%d %H:%M:%S"])
def test_negative_year_strftime_matches_pandas(unit, format):
    values = np.array(
        ["-0020-01-01", None, "0020-01-01", "2024-03-02"], dtype=f"M8[{unit}]"
    )
    with disable_module_accelerator():
        expected = pd.Series(values).dt.strftime(format)
    actual = pd.Series(values).dt.strftime(format)

    with disable_module_accelerator():
        pd.testing.assert_series_equal(_slow_arg(actual), expected)


def test_negative_local_year_strftime_matches_pandas():
    values = np.array(["0000-01-01T00:30:00"], dtype="M8[s]")

    def format_values():
        return (
            pd.Series(values)
            .dt.tz_localize("UTC")
            .dt.tz_convert("Etc/GMT+1")
            .dt.strftime("%Y-%m-%d")
        )

    if not PANDAS_GE_31:
        with pytest.raises(
            NotImplementedError, match="strftime not yet supported"
        ):
            format_values()
        return

    with disable_module_accelerator():
        expected = format_values()
    actual = format_values()
    with disable_module_accelerator():
        pd.testing.assert_series_equal(_slow_arg(actual), expected)
