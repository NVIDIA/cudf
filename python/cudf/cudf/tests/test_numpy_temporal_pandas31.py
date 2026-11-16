# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.core._compat import PANDAS_GE_31
from cudf.testing import assert_eq


@pytest.mark.skipif(not PANDAS_GE_31, reason="pandas 3.1 temporal validation")
@pytest.mark.parametrize("kind", ["M", "m"])
@pytest.mark.parametrize("multiplier", [2, 10])
@pytest.mark.parametrize(
    "box", [list, tuple, lambda x: np.array(x, dtype=object)]
)
def test_numpy_scalar_unit_multiplier_rejected(kind, multiplier, box):
    scalar_type = np.datetime64 if kind == "M" else np.timedelta64
    values = box([scalar_type(1, f"{multiplier}s")])
    message = "objects with units containing a multiplier"
    with pytest.raises(ValueError, match=message):
        pd.Series(values)
    with pytest.raises(ValueError, match=message):
        cudf.Series(values)


@pytest.mark.skipif(
    not PANDAS_GE_31, reason="pandas 3.1 calendar-unit alignment"
)
@pytest.mark.parametrize("unit", ["M", "Y"])
def test_timedelta_calendar_units_match_pandas(unit):
    values = np.array([1, 2, -1, 0, 12], dtype=f"m8[{unit}]")
    values[-1] = "NaT"
    assert_eq(pd.Series(values), cudf.Series(values))


@pytest.mark.parametrize("kind", ["M", "m"])
@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
def test_numpy_scalar_supported_units(kind, unit):
    scalar_type = np.datetime64 if kind == "M" else np.timedelta64
    values = [scalar_type(1, unit), scalar_type(2, unit)]
    assert_eq(pd.Series(values), cudf.Series(values))


@pytest.mark.skipif(not PANDAS_GE_31, reason="pandas 3.1 temporal validation")
@pytest.mark.parametrize("kind", ["M", "m"])
@pytest.mark.parametrize("unit", ["D", "s"])
@pytest.mark.parametrize("target_unit", ["s", "us", "ns"])
def test_numpy_scalar_unit_multiplier_with_explicit_dtype(
    kind, unit, target_unit
):
    scalar_type = np.datetime64 if kind == "M" else np.timedelta64
    value = scalar_type(1, f"2{unit}")
    dtype = np.dtype(f"{kind}8[{target_unit}]")
    assert_eq(
        pd.Series(value, index=[0], dtype=dtype),
        cudf.Series(value, index=[0], dtype=dtype),
    )


@pytest.mark.skipif(
    not PANDAS_GE_31, reason="pandas 3.1 calendar-unit alignment"
)
@pytest.mark.parametrize("unit,value", [("Y", 1000), ("M", 12000)])
def test_timedelta_calendar_units_validate_explicit_resolution(unit, value):
    values = np.array([value], dtype=f"m8[{unit}]")
    dtype = np.dtype("m8[ns]")
    with pytest.raises(pd.errors.OutOfBoundsTimedelta):
        pd.Series(values, dtype=dtype)
    with pytest.raises(pd.errors.OutOfBoundsTimedelta):
        cudf.Series(values, dtype=dtype)
