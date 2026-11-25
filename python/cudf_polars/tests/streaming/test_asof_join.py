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
    ).sort("time", maintain_order=True)
    right = pl.LazyFrame(
        {
            "time": [1, 2, 2, 2, 3, 4, 4, 6],
            "symbol": ["A", "B", "A", "A", "B", "A", "B", "A"],
            "quote": [10, 20, 30, 40, 50, None, 70, 80],
        }
    ).sort("time", maintain_order=True)
    query = left.join_asof(right, on="time", by=by)
    engine = streaming_engine_factory(StreamingOptions(max_rows_per_partition=2))
    assert_gpu_result_equal(query, engine=engine, check_row_order=True)


@pytest.mark.engine_params(["spmd-small", "ray"])
@pytest.mark.parametrize("nulls_last", [False, True])
@pytest.mark.filterwarnings(
    "ignore:Sortedness of columns cannot be checked when 'by' groups provided"
)
def test_asof_join_grouped_key_order(streaming_engine_factory, nulls_last):
    left = pl.LazyFrame(
        {
            "symbol": ["A", "A", "B", "A", "B", "C", None],
            "time": [1, 5, 1, 2, 3, 2, 1],
        }
    ).sort(["symbol", "time"], maintain_order=True, nulls_last=nulls_last)
    right = pl.LazyFrame(
        {
            "ticker": ["A", "B", "A", "C", "A", "B", "B", None],
            "time": [1, 1, 2, 1, 2, 2, 4, 1],
            "quote": [10, 20, 30, 40, 50, 60, 70, 99],
        }
    ).sort(["ticker", "time"], maintain_order=True, nulls_last=nulls_last)
    engine = streaming_engine_factory(StreamingOptions(max_rows_per_partition=2))
    assert_gpu_result_equal(
        left.join_asof(right, on="time", by_left="symbol", by_right="ticker"),
        engine=engine,
        check_row_order=True,
    )


@pytest.mark.engine_params(["spmd-small", "ray"])
@pytest.mark.filterwarnings(
    "ignore:Sortedness of columns cannot be checked when 'by' groups provided"
)
def test_asof_join_group_local_order(streaming_engine_factory):
    left = pl.LazyFrame({"symbol": ["B", "B", "A", "A"], "time": [1, 3, 1, 2]})
    right = pl.LazyFrame(
        {
            "symbol": ["B", "B", "A", "A"],
            "time": [1, 2, 1, 2],
            "quote": [10, 20, 30, 40],
        }
    )
    engine = streaming_engine_factory(StreamingOptions(max_rows_per_partition=2))
    assert_gpu_result_equal(
        left.join_asof(right, on="time", by="symbol"),
        engine=engine,
        check_row_order=True,
    )


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
def test_asof_join_skips_unchecked_left_validation(streaming_engine_factory):
    left = pl.LazyFrame({"time": [3, 1, 4, 2]})
    right = pl.LazyFrame({"time": [1, 2, 3], "right": [100, 200, 300]})
    query = left.join_asof(right, on="time", check_sortedness=False)
    engine = streaming_engine_factory(StreamingOptions(max_rows_per_partition=2))
    assert query.collect(engine=engine)["time"].to_list() == [3, 1, 4, 2]


@pytest.mark.engine_params(["spmd-small", "ray"])
def test_asof_join_null_right_and_unused_tail(streaming_engine_factory):
    left = pl.LazyFrame({"time": [1, 2]})
    right = pl.LazyFrame(
        {
            "time": [None, None, 1, 2, 3, 4, 5, 6],
            "quote": [99, 98, 10, 20, 30, 40, 50, 60],
        }
    )
    engine = streaming_engine_factory(StreamingOptions(max_rows_per_partition=2))
    assert_gpu_result_equal(
        left.join_asof(right, on="time"), engine=engine, check_row_order=True
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
@pytest.mark.parametrize("check_sortedness", [False, True])
def test_asof_join_nulls_last(streaming_engine_factory, *, check_sortedness):
    left = pl.LazyFrame({"time": [1, 3, None]}).sort("time", nulls_last=True)
    right = pl.LazyFrame({"time": [1, 2, None], "quote": [10, 20, 99]}).sort(
        "time", nulls_last=True
    )
    engine = streaming_engine_factory(StreamingOptions(max_rows_per_partition=2))
    assert_frame_equal(
        left.join_asof(right, on="time", check_sortedness=check_sortedness).collect(
            engine=engine
        ),
        pl.DataFrame({"time": [1, 3, None], "quote": [10, 20, None]}),
    )


@pytest.mark.engine_params(["spmd-small", "ray"])
@pytest.mark.parametrize("by", [None, "symbol"])
@pytest.mark.filterwarnings(
    "ignore:Sortedness of columns cannot be checked when 'by' groups provided"
)
def test_asof_join_interleaved_left_nulls(streaming_engine_factory, by):
    left = pl.LazyFrame(
        {
            "time": [None, 1, None, None, 3, None, 5],
            "symbol": ["A", "A", "B", "B", "B", "A", "A"],
        }
    )
    if by is not None:
        left = left.set_sorted("time")
    right = pl.LazyFrame(
        {"time": [1, 2, 4], "symbol": ["A", "B", "A"], "quote": [10, 20, 40]}
    ).sort("time")
    engine = streaming_engine_factory(StreamingOptions(max_rows_per_partition=2))
    assert_frame_equal(
        left.join_asof(right, on="time", by=by).collect(engine=engine),
        left.set_sorted("time").join_asof(right, on="time", by=by).collect(),
    )


@pytest.mark.engine_params(["spmd-small", "ray"])
@pytest.mark.filterwarnings(
    "ignore:Sortedness of columns cannot be checked when 'by' groups provided"
)
def test_asof_join_key_major_interleaved_left_nulls(streaming_engine_factory):
    left = pl.LazyFrame(
        {
            "symbol": ["A", "A", "A", "B", "B", "B"],
            "time": [None, 1, None, None, 1, None],
        }
    ).set_sorted(["symbol", "time"])
    right = pl.LazyFrame(
        {
            "symbol": ["A", "A", "B", "B"],
            "time": [1, 2, 1, 3],
            "quote": [10, 20, 30, 40],
        }
    ).sort(["symbol", "time"])
    engine = streaming_engine_factory(StreamingOptions(max_rows_per_partition=2))
    assert_gpu_result_equal(
        left.join_asof(right, on="time", by="symbol"),
        engine=engine,
        check_row_order=True,
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
@pytest.mark.parametrize("check_sortedness", [False, True])
def test_asof_join_no_right_payload(
    streaming_engine_factory, monkeypatch, *, check_sortedness
):
    monkeypatch.setattr("cudf_polars.streaming.actor_graph.asof_join.CUDF_ROW_LIMIT", 2)
    left = pl.LazyFrame({"time": [5]}).sort("time")
    right_times = [1, 2, 3, 4] if check_sortedness else [3, 1, 2]
    right = pl.LazyFrame({"time": right_times})
    engine = streaming_engine_factory(StreamingOptions(max_rows_per_partition=2))
    assert_gpu_result_equal(
        left.join_asof(right, on="time", check_sortedness=check_sortedness),
        engine=engine,
        check_row_order=True,
    )


@pytest.mark.engine_params(["spmd-small"])
@pytest.mark.parametrize("has_payload", [False, True])
def test_asof_join_grouped_sortedness_warning(streaming_engine_factory, *, has_payload):
    left = pl.LazyFrame({"time": [1, 2], "group": ["A", "A"]})
    right = pl.LazyFrame(
        {"time": [1, 2], "group": ["A", "A"]}
        | ({"quote": [10, 20]} if has_payload else {})
    )
    engine = streaming_engine_factory(StreamingOptions(max_rows_per_partition=1))
    with pytest.warns(UserWarning, match="Sortedness of columns cannot be checked"):
        left.join_asof(right, on="time", by="group").collect(engine=engine)


@pytest.mark.engine_params(["ray"])
def test_asof_join_checks_rank_boundary(streaming_engine_factory):
    left = pl.LazyFrame({"time": [3, 4, 1, 2]})
    right = pl.LazyFrame({"time": [1, 2, 3, 4], "quote": [10, 20, 30, 40]}).sort("time")
    engine = streaming_engine_factory(StreamingOptions(max_rows_per_partition=2))
    with pytest.RaisesGroup(pl.exceptions.InvalidOperationError):
        left.join_asof(right, on="time").collect(engine=engine)


@pytest.mark.engine_params(["ray"])
@pytest.mark.parametrize("left_times", [[], [1]])
def test_asof_join_empty_rank(streaming_engine_factory, left_times):
    left = pl.LazyFrame({"time": pl.Series(left_times, dtype=pl.Int64)})
    right = pl.LazyFrame({"time": [1, 2], "quote": [10, 20]})
    engine = streaming_engine_factory(StreamingOptions(max_rows_per_partition=1))
    assert_gpu_result_equal(
        left.join_asof(right, on="time"), engine=engine, check_row_order=True
    )


@pytest.mark.engine_params(["spmd-small"])
@pytest.mark.parametrize("check_sortedness", [False, True])
@pytest.mark.parametrize("has_payload", [False, True])
def test_asof_join_unused_right_validation(
    streaming_engine_factory, *, check_sortedness, has_payload
):
    left = pl.LazyFrame({"time": [1]})
    right = pl.LazyFrame(
        {"time": [1, 3, 2]} | ({"quote": [10, 30, 20]} if has_payload else {})
    )
    engine = streaming_engine_factory(StreamingOptions(max_rows_per_partition=2))
    query = left.join_asof(right, on="time", check_sortedness=check_sortedness)
    if check_sortedness:
        with pytest.RaisesGroup(pl.exceptions.InvalidOperationError):
            query.collect(engine=engine)
    else:
        assert_frame_equal(
            query.collect(engine=engine),
            pl.DataFrame({"time": [1]} | ({"quote": [10]} if has_payload else {})),
        )


@pytest.mark.engine_params(["spmd-small"])
def test_asof_join_unchecked_sorted_keys(streaming_engine_factory):
    left = pl.LazyFrame({"time": [1, 2, 3, 4]}).sort("time")
    right = pl.LazyFrame({"time": [1, 3], "quote": [10, 30]}).sort("time")
    engine = streaming_engine_factory(StreamingOptions(max_rows_per_partition=2))
    assert_gpu_result_equal(
        left.join_asof(right, on="time", check_sortedness=False),
        engine=engine,
        check_row_order=True,
    )


@pytest.mark.engine_params(["spmd-small"])
@pytest.mark.parametrize("time_ordered", [False, True])
@pytest.mark.filterwarnings(
    "ignore:Sortedness of columns cannot be checked when 'by' groups provided"
)
def test_asof_join_right_row_limit(
    streaming_engine_factory, monkeypatch, *, time_ordered
):
    monkeypatch.setattr("cudf_polars.streaming.actor_graph.asof_join.CUDF_ROW_LIMIT", 2)
    left = pl.LazyFrame({"time": [4], "symbol": ["A"]})
    right = pl.LazyFrame(
        {"time": [1, 2, 3], "symbol": ["A", "B", "C"], "quote": [10, 20, 30]}
    )
    if time_ordered:
        left = left.sort("time")
        right = right.sort("time")
    engine = streaming_engine_factory(StreamingOptions(max_rows_per_partition=1))
    with pytest.RaisesGroup(NotImplementedError):
        left.join_asof(right, on="time", by="symbol").collect(engine=engine)
