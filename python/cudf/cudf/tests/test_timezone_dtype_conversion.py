# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from zoneinfo import ZoneInfo

import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize("timezone", ["Europe/Paris", "US/Eastern"])
@pytest.mark.parametrize("kind", ["Series", "DataFrame"])
def test_pandas_timezone_constructor_in_compatible_mode(timezone, kind):
    values = pd.Series(
        pd.DatetimeIndex(
            ["2020-01-01", None, "2020-02-01"], tz=ZoneInfo(timezone)
        ),
        name="date",
    )
    expected = values if kind == "Series" else values.to_frame()
    with cudf.option_context("mode.pandas_compatible", True):
        actual = getattr(cudf, kind)(expected)
    assert_eq(expected, actual)
