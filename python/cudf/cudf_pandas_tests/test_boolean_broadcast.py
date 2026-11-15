# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import operator

import pandas as xpd
import pyarrow as pa
import pytest

from cudf.pandas import LOADED
from cudf.testing import assert_eq

if not LOADED:
    raise ImportError("These tests must be run with cudf.pandas loaded")

pd = xpd._fsproxy_slow


@pytest.mark.parametrize(
    "left_dtype,right_dtype",
    [
        ("boolean", "boolean"),
        ("Int64", "boolean"),
        ("int64", "boolean"),
        (pd.ArrowDtype(pa.bool_()), "Int64"),
        (pd.ArrowDtype(pa.int64()), "boolean"),
        ("Int64", pd.ArrowDtype(pa.bool_())),
    ],
)
@pytest.mark.parametrize(
    "op",
    [
        operator.add,
        operator.sub,
        operator.floordiv,
        operator.and_,
        operator.eq,
    ],
)
@pytest.mark.parametrize("labels", [["a", "b"], ["b", "c"]])
def test_nullable_boolean_dataframe_series_broadcast(
    left_dtype, right_dtype, op, labels
):
    left_bool = str(left_dtype) in {"boolean", "bool[pyarrow]"}
    right_bool = str(right_dtype) in {"boolean", "bool[pyarrow]"}
    values = [True, False, None] if left_bool else [1, 0, 2]
    pdf = pd.DataFrame(
        {name: pd.Series(values, dtype=left_dtype) for name in "ab"}
    )
    psr = pd.Series(
        [True if right_bool else 1, None], index=labels, dtype=right_dtype
    )
    actual_frame = xpd.DataFrame(pdf)
    actual_series = xpd.Series(psr)
    try:
        expected = op(pdf, psr)
    except (TypeError, ValueError, NotImplementedError) as error:
        with pytest.raises(type(error)):
            op(actual_frame, actual_series)
    else:
        actual = op(actual_frame, actual_series)
        assert_eq(expected, actual._fsproxy_slow)
