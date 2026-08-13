# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

import polars as pl
from polars.testing.asserts import assert_frame_equal

import pylibcudf as plc

from cudf_polars.containers import Column, DataFrame, DataType
from cudf_polars.testing.asserts import assert_gpu_result_equal
from cudf_polars.utils.cuda_stream import get_cuda_stream


def test_select_missing_raises():
    dtype = DataType(pl.Int8())
    df = DataFrame(
        [
            Column(
                plc.column_factories.make_numeric_column(
                    dtype.plc_type, 2, plc.MaskState.ALL_VALID
                ),
                dtype=dtype,
                name="a",
            )
        ],
        stream=get_cuda_stream(),
    )
    with pytest.raises(ValueError):
        df.select(["b", "a"])


def test_replace_missing_raises():
    dtype = DataType(pl.Int8())
    df = DataFrame(
        [
            Column(
                plc.column_factories.make_numeric_column(
                    dtype.plc_type, 2, plc.MaskState.ALL_VALID
                ),
                dtype=dtype,
                name="a",
            )
        ],
        stream=get_cuda_stream(),
    )
    replacement = df.column_map["a"].copy().rename("b")
    with pytest.raises(ValueError):
        df.with_columns([replacement], replace_only=True, stream=df.stream)


def test_from_table_wrong_names():
    stream = get_cuda_stream()
    table = plc.Table(
        [
            plc.column_factories.make_numeric_column(
                plc.DataType(plc.TypeId.INT8), 1, plc.MaskState.ALL_VALID, stream=stream
            )
        ]
    )
    with pytest.raises(ValueError):
        DataFrame.from_table(table, ["a", "b"], [DataType(pl.Int8())], stream=stream)


def test_unnamed_column_raise():
    stream = get_cuda_stream()
    dtype = DataType(pl.Int8())
    payload = plc.column_factories.make_numeric_column(
        dtype.plc_type, 0, plc.MaskState.ALL_VALID, stream=stream
    )

    with pytest.raises(ValueError):
        DataFrame(
            [Column(payload, name="a", dtype=dtype), Column(payload, dtype=dtype)],
            stream=stream,
        )


def test_sorted_like_raises_mismatching_names():
    stream = get_cuda_stream()
    dtype = DataType(pl.Int8())
    df = DataFrame(
        [
            Column(
                plc.column_factories.make_numeric_column(
                    dtype.plc_type, 2, plc.MaskState.ALL_VALID, stream=stream
                ),
                dtype=dtype,
                name="a",
            )
        ],
        stream=stream,
    )
    like = df.copy().rename_columns({"a": "b"})
    with pytest.raises(ValueError):
        df.sorted_like(like)


def test_shallow_copy():
    stream = get_cuda_stream()
    dtype = DataType(pl.Int8())
    column = Column(
        plc.column_factories.make_numeric_column(
            dtype.plc_type, 2, plc.MaskState.ALL_VALID, stream=stream
        ),
        dtype=dtype,
        name="a",
    )
    column.set_sorted(
        is_sorted=plc.types.Sorted.YES,
        order=plc.types.Order.ASCENDING,
        null_order=plc.types.NullOrder.AFTER,
    )
    df = DataFrame([column], stream=stream)
    copy = df.copy()
    copy.column_map["a"].set_sorted(
        is_sorted=plc.types.Sorted.NO,
        order=plc.types.Order.ASCENDING,
        null_order=plc.types.NullOrder.AFTER,
    )
    assert df.column_map["a"].is_sorted == plc.types.Sorted.YES
    assert copy.column_map["a"].is_sorted == plc.types.Sorted.NO


def test_with_columns_preserves_zero_column_row_count():
    stream = get_cuda_stream()
    dtype = DataType(pl.Int8())
    column = Column(
        plc.column_factories.make_numeric_column(
            dtype.plc_type, 5, plc.MaskState.ALL_VALID, stream=stream
        ),
        dtype=dtype,
        name="a",
    )
    zero_col = DataFrame([column], stream=stream).discard_columns({"a"})
    assert zero_col.num_columns == 0
    assert zero_col.num_rows == 5
    # Adding no columns to a zero-column frame must keep the row count.
    result = zero_col.with_columns([], stream=stream)
    assert result.num_columns == 0
    assert result.num_rows == 5


def test_sorted_flags_preserved_empty():
    stream = get_cuda_stream()
    df = pl.DataFrame({"a": pl.Series([], dtype=pl.Int8())})
    df.select(pl.col("a").sort())

    gf = DataFrame.from_polars(df, stream=stream)

    a = gf.column_map["a"]

    assert a.is_sorted == plc.types.Sorted.YES

    assert df.flags == gf.to_polars().flags


@pytest.mark.parametrize("nulls_last", [True, False])
def test_sorted_flags_preserved(with_nulls, nulls_last):
    stream = get_cuda_stream()
    values = [1, 2, -1, 2, 4, 5]
    if with_nulls:
        values[4] = None
    df = pl.DataFrame({"a": values, "b": values, "c": values})

    df = df.select(
        pl.col("a").sort(descending=False, nulls_last=nulls_last),
        pl.col("b").sort(descending=True, nulls_last=nulls_last),
        pl.col("c"),
    )

    gf = DataFrame.from_polars(df, stream=stream)

    a_null_order = (
        plc.types.NullOrder.AFTER
        if nulls_last and with_nulls
        else plc.types.NullOrder.BEFORE
    )
    b_null_order = (
        plc.types.NullOrder.AFTER
        if not nulls_last and with_nulls
        else plc.types.NullOrder.BEFORE
    )
    a, b, c = gf.columns
    assert a.is_sorted == plc.types.Sorted.YES
    assert a.order == plc.types.Order.ASCENDING
    assert a.null_order == a_null_order
    assert b.is_sorted == plc.types.Sorted.YES
    assert b.order == plc.types.Order.DESCENDING
    assert b.null_order == b_null_order
    assert c.is_sorted == plc.types.Sorted.NO
    assert df.flags == gf.to_polars().flags


def test_empty_name_roundtrips_overlap(engine: pl.GPUEngine):
    df = pl.LazyFrame({"": [1, 2, 3], "column_0": [4, 5, 6]})
    assert_gpu_result_equal(df, engine=engine)


def test_empty_name_roundtrips_no_overlap(engine: pl.GPUEngine):
    df = pl.LazyFrame({"": [1, 2, 3], "b": [4, 5, 6]})
    assert_gpu_result_equal(df, engine=engine)


@pytest.mark.parametrize(
    "polars_tbl",
    [
        pl.DataFrame(),
        pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}),
        pl.DataFrame({"a": [1, 2, 3]}),
        pl.DataFrame({"a": [1], "b": [2], "c": [3]}),
        pl.DataFrame({"a": ["a", "bb", "ccc"]}),
        pl.DataFrame({"a": [1, 2, None], "b": [None, 3, 4]}),
    ],
)
def test_serialization_roundtrip(polars_tbl):
    stream = get_cuda_stream()
    df = DataFrame.from_polars(polars_tbl, stream=stream)

    header, frames = df.serialize()
    res = DataFrame.deserialize(header, frames, stream=stream)

    assert_frame_equal(df.to_polars(), res.to_polars())
