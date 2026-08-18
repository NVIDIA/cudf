# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import datetime

import cupy as cp
import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "data_non_overflow",
    [
        [1000000, 200000, 3000000],
        [1000000, 200000, None],
        [],
        [None],
        [12, 12, 22, 343, 4353534, 435342],
        np.array([10, 20, 30, None, 100]),
        cp.asarray([10, 20, 30, 100]),
        [1000000, 200000, 3000000],
        [1000000, 200000, None],
        [1],
        [12, 11, 232, 223432411, 2343241, 234324, 23234],
        [12, 11, 2.32, 2234.32411, 2343.241, 23432.4, 23234],
        [1.321, 1132.324, 23223231.11, 233.41, 0.2434, 332, 323],
        [12, 11, 2.32, 2234.32411, 2343.241, 23432.4, 23234],
    ],
)
def test_timedelta_index_datetime_index_ops(
    data_non_overflow, datetime_types_as_str, timedelta_types_as_str
):
    gdt = cudf.Index(data_non_overflow, dtype=datetime_types_as_str)
    gtd = cudf.Index(data_non_overflow, dtype=timedelta_types_as_str)

    pdt = gdt.to_pandas()
    ptd = gtd.to_pandas()

    assert_eq(gdt - gtd, pdt - ptd)
    assert_eq(gdt + gtd, pdt + ptd)


@pytest.mark.parametrize(
    "datetime_data,timedelta_data",
    [
        ([1000000, 200000, 3000000], [1000000, 200000, 3000000]),
        ([1000000, 200000, None], [1000000, 200000, None]),
        ([], []),
        ([None], [None]),
        (
            [12, 12, 22, 343, 4353534, 435342],
            [12, 12, 22, 343, 4353534, 435342],
        ),
        (np.array([10, 20, 30, None, 100]), np.array([10, 20, 30, None, 100])),
        (cp.asarray([10, 20, 30, 100]), cp.asarray([10, 20, 30, 100])),
        ([1000000, 200000, 3000000], [200000, 34543, 3000000]),
        ([1000000, 200000, None], [1000000, 200000, 3000000]),
        ([None], [1]),
        (
            [12, 12, 22, 343, 4353534, 435342],
            [None, 1, 220, 3, 34, 4353423287],
        ),
        (
            [12, 11, 232, 223432411, 2343241, 234324, 23234],
            [11, 1132324, 2322323111, 23341, 2434, 332, 323],
        ),
        (
            [12, 11, 2.32, 2234.32411, 2343.241, 23432.4, 23234],
            [11, 1132324, 2322323111, 23341, 2434, 332, 323],
        ),
        (
            [11, 1132324, 2322323111, 23341, 2434, 332, 323],
            [12, 11, 2.32, 2234.32411, 2343.241, 23432.4, 23234],
        ),
        (
            [1.321, 1132.324, 23223231.11, 233.41, 0.2434, 332, 323],
            [12, 11, 2.32, 2234.32411, 2343.241, 23432.4, 23234],
        ),
    ],
)
def test_timedelta_datetime_index_ops_misc(
    datetime_data,
    timedelta_data,
    datetime_types_as_str,
    timedelta_types_as_str,
):
    gdt = cudf.Index(datetime_data, dtype=datetime_types_as_str)
    gtd = cudf.Index(timedelta_data, dtype=timedelta_types_as_str)

    pdt = gdt.to_pandas()
    ptd = gtd.to_pandas()

    assert_eq(gdt - gtd, pdt - ptd)
    assert_eq(gdt + gtd, pdt + ptd)


@pytest.mark.parametrize(
    "data_non_overflow",
    [
        [1000000, 200000, 3000000],
        [1000000, 200000, None],
        [],
        [None],
        [12, 12, 22, 343, 4353534, 435342],
        np.array([10, 20, 30, None, 100]),
        cp.asarray([10, 20, 30, 100]),
        [1000000, 200000, 3000000],
        [1000000, 200000, None],
        [1],
        [12, 11, 232, 223432411, 2343241, 234324, 23234],
        [12, 11, 2.32, 2234.32411, 2343.241, 23432.4, 23234],
        [1.321, 1132.324, 23223231.11, 233.41, 0.2434, 332, 323],
        [12, 11, 2.32, 2234.32411, 2343.241, 23432.4, 23234],
    ],
)
@pytest.mark.parametrize(
    "other_scalars",
    [
        pd.Timedelta(1513393355.5, unit="s"),
        pd.Timedelta(34765, unit="D"),
        datetime.timedelta(days=768),
        datetime.timedelta(seconds=768),
        datetime.timedelta(microseconds=7),
        datetime.timedelta(minutes=447),
        datetime.timedelta(hours=447),
        datetime.timedelta(weeks=734),
        np.timedelta64(4, "s"),
        np.timedelta64(456, "D"),
        np.timedelta64(46, "h"),
        np.timedelta64("nat"),
        np.timedelta64(1, "s"),
        np.timedelta64(1, "ms"),
        np.timedelta64(1, "us"),
        np.timedelta64(1, "ns"),
    ],
)
@pytest.mark.filterwarnings("ignore:divide by zero:RuntimeWarning:pandas")
def test_timedelta_index_ops_with_scalars(
    request,
    data_non_overflow,
    other_scalars,
    timedelta_types_as_str,
    arithmetic_op_method,
):
    if arithmetic_op_method not in ("add", "sub", "truediv", "floordiv"):
        pytest.skip(f"Test not applicable for {arithmetic_op_method}")

    gtdi = cudf.Index(data=data_non_overflow, dtype=timedelta_types_as_str)
    ptdi = gtdi.to_pandas()

    if arithmetic_op_method == "add":
        expected = ptdi + other_scalars
        actual = gtdi + other_scalars
    elif arithmetic_op_method == "sub":
        expected = ptdi - other_scalars
        actual = gtdi - other_scalars
    elif arithmetic_op_method == "truediv":
        expected = ptdi / other_scalars
        actual = gtdi / other_scalars
    elif arithmetic_op_method == "floordiv":
        expected = ptdi // other_scalars
        actual = gtdi // other_scalars

    assert_eq(expected, actual)

    if arithmetic_op_method == "add":
        expected = other_scalars + ptdi
        actual = other_scalars + gtdi
    elif arithmetic_op_method == "sub":
        expected = other_scalars - ptdi
        actual = other_scalars - gtdi
    elif arithmetic_op_method == "truediv":
        expected = other_scalars / ptdi
        actual = other_scalars / gtdi
    elif arithmetic_op_method == "floordiv":
        expected = other_scalars // ptdi
        actual = other_scalars // gtdi

    # Division by zero for datetime or timedelta is
    # dubiously defined in both pandas (Any // 0 -> 0 in
    # pandas) and cuDF (undefined behaviour)
    request.applymarker(
        pytest.mark.xfail(
            condition=(
                arithmetic_op_method == "floordiv"
                and 0 in ptdi.astype("int")
                and np.timedelta64(other_scalars).item() is not None
            ),
            reason="Related to https://github.com/NVIDIA/cudf/issues/5938",
        )
    )
    assert_eq(expected, actual)
