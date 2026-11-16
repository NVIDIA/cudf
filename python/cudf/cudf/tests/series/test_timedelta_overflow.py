# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.core._compat import PANDAS_GE_31
from cudf.testing import assert_eq


@pytest.mark.skipif(
    not PANDAS_GE_31,
    reason="Checked timedelta-array integer multiplication starts in pandas 3.1",
)
@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
@pytest.mark.parametrize("array_operand", [False, True])
@pytest.mark.parametrize("reflect", [False, True])
@pytest.mark.parametrize(
    "ticks,factor",
    [
        (2**62, 2),
        (-(2**62), 2),
        (1, np.uint64(2**63 + 5)),
        (0, np.uint64(2**63 + 5)),
        (2**62 - 1, 2),
        (-(2**63) + 1, -1),
        (None, np.uint64(2**63 + 5)),
    ],
)
def test_timedelta_integer_multiplication_bounds(
    unit, array_operand, reflect, ticks, factor
):
    values = np.array([0, ticks or 0, 0], dtype=np.int64).view(
        f"timedelta64[{unit}]"
    )
    values[[0, 2]] = "NaT"
    if ticks is None:
        values[:] = "NaT"
    expected_source = pd.Series(values).iloc[1:]
    source = cudf.from_pandas(pd.Series(values)).iloc[1:]
    other = (
        pd.Series([factor, factor], index=expected_source.index)
        if array_operand
        else factor
    )
    device_other = cudf.from_pandas(other) if array_operand else other

    def multiply(obj, operand):
        return obj.rmul(operand) if reflect else obj.mul(operand)

    try:
        expected = multiply(expected_source, other)
    except pd.errors.OutOfBoundsTimedelta:
        with pytest.raises(
            pd.errors.OutOfBoundsTimedelta,
            match="Overflow in int64 multiplication",
        ):
            multiply(source, device_other)
    else:
        assert_eq(expected, multiply(source, device_other))
