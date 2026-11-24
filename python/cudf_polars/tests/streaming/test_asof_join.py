# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Streaming as-of join behavior across chunk boundaries."""

from __future__ import annotations

import pytest

import polars as pl
from polars.testing import assert_frame_equal

from cudf_polars.engine.options import StreamingOptions
from cudf_polars.testing.asserts import assert_gpu_result_equal


@pytest.mark.engine_params(["spmd-small", "ray"])
@pytest.mark.parametrize("by", [None, "symbol"])
@pytest.mark.filterwarnings(
    "ignore:Sortedness of columns cannot be checked when 'by' groups provided"
)
def test_asof_join_time_major_chunks(streaming_engine_factory, by):
    left = pl.LazyFrame(
        {
            "time": [1, 2, 2, 3, 4, 5, 6],
            "symbol": ["A", "A", "B", "A", "B", "A", "B"],
        }
    ).sort("time")
    right = pl.LazyFrame(
        {
            "time": [1, 2, 2, 2, 3, 4, 4, 6],
            "symbol": ["A", "B", "A", "A", "B", "A", "B", "A"],
            "quote": [10, 20, 30, 40, 50, None, 70, 80],
        }
    ).sort("time")
    query = left.join_asof(right, on="time", by=by)
    engine = streaming_engine_factory(StreamingOptions(max_rows_per_partition=2))
    assert_gpu_result_equal(query, engine=engine, check_row_order=True)


@pytest.mark.engine_params(["spmd-small"])
@pytest.mark.filterwarnings(
    "ignore:Sortedness of columns cannot be checked when 'by' groups provided"
)
def test_asof_join_sparse_left(streaming_engine_factory):
    left = pl.LazyFrame(
        {"time": [0, 4, 8, 11, 13], "symbol": ["A", "B", "A", "B", "A"]}
    ).sort("time")
    right = pl.LazyFrame(
        {
            "time": list(range(1, 13)),
            "symbol": ["A", "B", "B", "B", "A", "B", "B", "B", "B", "B", "A", "B"],
            "quote": list(range(12)),
        }
    ).sort("time")
    query = left.join_asof(right, on="time", by="symbol")
    engine = streaming_engine_factory(StreamingOptions(max_rows_per_partition=2))
    assert_gpu_result_equal(query, engine=engine, check_row_order=True)


@pytest.mark.engine_params(["spmd-small", "ray"])
def test_asof_join_without_ordering_metadata(streaming_engine_factory):
    left = pl.LazyFrame({"time": [1, 2, 3, 4], "left": [10, 20, 30, 40]})
    right = pl.LazyFrame({"time": [1, 2, 3], "right": [100, 200, 300]})
    query = left.join_asof(right, on="time")
    engine = streaming_engine_factory(StreamingOptions(max_rows_per_partition=2))
    assert_gpu_result_equal(query, engine=engine, check_row_order=True)


@pytest.mark.engine_params(["spmd-small", "ray"])
def test_asof_join_expression_on(streaming_engine_factory):
    left = pl.LazyFrame({"time": pl.Series([1, 2, 3, 4], dtype=pl.Int32)})
    right = pl.LazyFrame(
        {
            "time": pl.Series([1, 3, 4], dtype=pl.Int32),
            "quote": [10, 30, 40],
        }
    )
    query = left.join_asof(
        right,
        left_on=pl.col("time").cast(pl.Int64),
        right_on=pl.col("time").cast(pl.Int64),
        coalesce=False,
    )
    engine = streaming_engine_factory(
        StreamingOptions(max_rows_per_partition=2, fallback_mode="silent")
    )
    assert_gpu_result_equal(query, engine=engine, check_row_order=True)


@pytest.mark.engine_params(["spmd-small"])
def test_asof_join_rejects_unordered_left(streaming_engine_factory):
    left = pl.LazyFrame({"time": [3, 1, 4, 2]})
    right = pl.LazyFrame({"time": [1, 2, 3], "right": [100, 200, 300]})
    query = left.join_asof(right, on="time", check_sortedness=False)
    engine = streaming_engine_factory(StreamingOptions(max_rows_per_partition=2))
    with pytest.RaisesGroup(pl.exceptions.InvalidOperationError):
        query.collect(engine=engine)


@pytest.mark.engine_params(["spmd-small"])
def test_asof_join_null_right_time(streaming_engine_factory):
    left = pl.LazyFrame({"time": [1, 2, 3, 4]}).sort("time")
    right = pl.LazyFrame({"time": [None, 1, 2, 4], "quote": [99, 10, 20, 40]}).sort(
        "time"
    )
    engine = streaming_engine_factory(StreamingOptions(max_rows_per_partition=2))
    assert_gpu_result_equal(
        left.join_asof(right, on="time"), engine=engine, check_row_order=True
    )


@pytest.mark.engine_params(["spmd-small"])
def test_asof_join_interleaved_null_right_time(streaming_engine_factory):
    left = pl.LazyFrame({"time": [1, 2, 3]})
    right = pl.LazyFrame({"time": [1, None, 2, 3], "quote": [10, 99, 20, 30]})
    engine = streaming_engine_factory(StreamingOptions(max_rows_per_partition=2))
    assert_gpu_result_equal(
        left.join_asof(right, on="time", check_sortedness=False),
        engine=engine,
        check_row_order=True,
    )


@pytest.mark.engine_params(["spmd-small"])
@pytest.mark.filterwarnings(
    "ignore:Sortedness of columns cannot be checked when 'by' groups provided"
)
def test_asof_join_composite_by_and_slice(streaming_engine_factory):
    left = pl.LazyFrame(
        {
            "time": [1, 2, 3, 4, 5, 6],
            "symbol": ["A", "B", "A", "B", "A", "B"],
            "venue": ["X", "X", "Y", "X", "X", "Y"],
        }
    ).sort("time")
    right = pl.LazyFrame(
        {
            "time": [1, 2, 2, 3, 4, 5, 6],
            "symbol": ["A", "A", "B", "A", "B", "A", "B"],
            "venue": ["X", "Y", "X", "X", "X", "X", "Y"],
            "quote": [10, 20, 30, 40, 50, 60, 70],
        }
    ).sort("time")
    query = left.join_asof(right, on="time", by=["symbol", "venue"]).slice(1, 4)
    engine = streaming_engine_factory(StreamingOptions(max_rows_per_partition=2))
    assert_gpu_result_equal(query, engine=engine, check_row_order=True)


@pytest.mark.engine_params(["spmd-small"])
def test_asof_join_nulls_last(streaming_engine_factory):
    left = pl.LazyFrame({"time": [1, 3, None]}).sort("time", nulls_last=True)
    right = pl.LazyFrame({"time": [1, 2, None], "quote": [10, 20, 99]}).sort(
        "time", nulls_last=True
    )
    engine = streaming_engine_factory(StreamingOptions(max_rows_per_partition=2))
    assert_frame_equal(
        left.join_asof(right, on="time").collect(engine=engine),
        pl.DataFrame({"time": [1, 3, None], "quote": [10, 20, None]}),
    )


@pytest.mark.engine_params(["spmd-small", "ray"])
def test_asof_join_empty_right_chunks(streaming_engine_factory):
    left = pl.LazyFrame({"time": [1, 2, 3]}).sort("time")
    right = pl.LazyFrame(
        {
            "time": pl.Series([], dtype=pl.Int64),
            "quote": pl.Series([], dtype=pl.Int64),
        }
    ).sort("time")
    engine = streaming_engine_factory(StreamingOptions(max_rows_per_partition=2))
    assert_gpu_result_equal(
        left.join_asof(right, on="time"), engine=engine, check_row_order=True
    )


@pytest.mark.engine_params(["spmd-small"])
def test_asof_join_no_right_payload(streaming_engine_factory):
    left = pl.LazyFrame({"time": [1, 2, 3]}).sort("time")
    right = pl.LazyFrame({"time": [1, 2]}).sort("time")
    engine = streaming_engine_factory(StreamingOptions(max_rows_per_partition=2))
    assert_gpu_result_equal(
        left.join_asof(right, on="time"), engine=engine, check_row_order=True
    )


@pytest.mark.engine_params(["spmd-small"])
def test_asof_join_drains_unused_right_chunks(streaming_engine_factory):
    left = pl.LazyFrame({"time": [1, 2]}).sort("time")
    right = pl.LazyFrame(
        {"time": list(range(1, 11)), "quote": list(range(1, 11))}
    ).sort("time")
    engine = streaming_engine_factory(StreamingOptions(max_rows_per_partition=2))
    assert_gpu_result_equal(
        left.join_asof(right, on="time"), engine=engine, check_row_order=True
    )


@pytest.mark.engine_params(["spmd-small"])
def test_asof_join_validates_unused_right_chunks(streaming_engine_factory):
    left = pl.LazyFrame({"time": [1]})
    right = pl.LazyFrame({"time": [1, 3, 2], "quote": [10, 30, 20]})
    engine = streaming_engine_factory(StreamingOptions(max_rows_per_partition=2))
    with pytest.RaisesGroup(pl.exceptions.InvalidOperationError):
        left.join_asof(right, on="time", check_sortedness=False).collect(engine=engine)
