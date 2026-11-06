# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import numba
import pyarrow as pa
from numba import cuda
from utils import assert_column_eq, assert_table_eq

import pylibcudf as plc


def test_column_nans_to_nulls(has_nans):
    if has_nans:
        values = [1, float("nan"), float("nan"), None, 3, None]
    else:
        values = [1, 4, 5, None, 3, None]

    replaced = [
        None if (v is None or (v is not None and math.isnan(v))) else v
        for v in values
    ]

    h_input = pa.array(values, type=pa.float32())
    input = plc.Column.from_arrow(h_input)
    expect = pa.array(replaced, type=pa.float32())

    got = plc.transform.column_nans_to_nulls(input)

    assert_column_eq(expect, got)


def test_column_nans_to_nulls_sliced():
    values = [1, float("nan"), 3, float("nan"), 5, float("nan"), 7]
    h_input = pa.array(values, type=pa.float64())
    input = plc.Column.from_arrow(h_input)

    # Slice to get elements [1:5] = [nan, 3, nan, 5]
    sliced = plc.copying.slice(input, [1, 5])[0]

    # Expected result with nans converted to nulls
    expect = pa.array([None, 3, None, 5], type=pa.float64())

    got = plc.transform.column_nans_to_nulls(sliced)

    assert_column_eq(expect, got)


def test_bools_to_mask_roundtrip():
    pa_array = pa.array([True, None, False])
    plc_input = plc.Column.from_arrow(pa_array)
    mask, result_null_count = plc.transform.bools_to_mask(plc_input)

    assert result_null_count == 2
    got = plc_input.with_mask(mask, result_null_count)
    assert_column_eq(pa.array([True, None, None]), got)

    got = plc.transform.mask_to_bools(mask.ptr, 0, len(pa_array))
    expect = pa.array([True, False, False])
    assert_column_eq(expect, got)


def test_encode():
    got_tbl, got_col = plc.transform.encode(
        plc.Table.from_arrow(pa.table({"a": [1, 3, 4], "b": [1, 2, 4]}))
    )

    expect = pa.table(
        [[1, 3, 4], [1, 2, 4]],
        schema=pa.schema(
            [
                pa.field("", pa.int64(), nullable=False),
                pa.field("", pa.int64(), nullable=False),
            ]
        ),
    )
    assert_table_eq(expect, got_tbl)

    expect = pa.array([0, 1, 2], type=pa.int32())
    assert_column_eq(expect, got_col)


def test_one_hot_encode():
    plc_input = plc.Column.from_arrow(pa.array([1, 2, 3]))
    plc_categories = plc.Column.from_arrow(pa.array([0, 0, 0]))
    got = plc.transform.one_hot_encode(plc_input, plc_categories)
    expect = pa.table(
        [[False] * 3] * 3,
        schema=pa.schema([pa.field("", pa.bool_(), nullable=False)] * 3),
    )
    assert_table_eq(expect, got)


def test_transform_udf():
    @cuda.jit(device=True)
    def op(a, b, c):
        return (a + b) * c

    ptx, _ = cuda.compile_ptx_for_current_device(
        op, (numba.float64, numba.float64, numba.float64), device=True
    )

    A = 5.0
    B = 20.0
    C = 0.5

    expect = pa.array([(A + B) * C] * 100)
    got = plc.transform.transform(
        [
            plc.Column.from_arrow(pa.array([A] * 100)),
            plc.Column.from_arrow(pa.array([B] * 100)),
            plc.Column.from_arrow(pa.array([C])),
        ],
        transform_udf=ptx,
        output_type=plc.DataType(plc.TypeId.FLOAT64),
        is_ptx=True,
        is_null_aware=plc.types.NullAware.NO,
        null_policy=plc.types.OutputNullability.PRESERVE,
    )
    assert_column_eq(expect, got)
