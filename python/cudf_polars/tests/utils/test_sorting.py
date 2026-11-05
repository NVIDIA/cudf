# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

import polars as pl

import pylibcudf as plc

from cudf_polars.containers import Column, DataType
from cudf_polars.utils.sorting import nulls_at_start, sort_order


@pytest.mark.parametrize(
    "descending,nulls_last,num_keys",
    [
        ([True], [False, True], 3),
        ([True, True], [False, True, False], 3),
        ([False, True], [True], 3),
    ],
)
def test_sort_order_raises_mismatch(descending, nulls_last, num_keys):
    with pytest.raises(ValueError):
        _ = sort_order(descending, nulls_last=nulls_last, num_keys=num_keys)


@pytest.mark.parametrize(
    "order,null_order,expected",
    [
        (plc.types.Order.ASCENDING, plc.types.NullOrder.BEFORE, True),
        (plc.types.Order.ASCENDING, plc.types.NullOrder.AFTER, False),
        (plc.types.Order.DESCENDING, plc.types.NullOrder.BEFORE, False),
        (plc.types.Order.DESCENDING, plc.types.NullOrder.AFTER, True),
    ],
)
def test_nulls_at_start(order, null_order, expected):
    dtype = DataType(pl.Int8())
    column = Column(
        plc.Column.from_iterable_of_py([1], dtype.plc_type),
        dtype=dtype,
        is_sorted=plc.types.Sorted.YES,
        order=order,
        null_order=null_order,
    )
    assert nulls_at_start(column) is expected
