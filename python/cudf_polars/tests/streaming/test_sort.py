# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor

import pytest

import polars as pl

import pylibcudf as plc
from cudf_streaming.channel_metadata import OrderKey, OrderScheme
from cudf_streaming.table_chunk import TableChunk
from rapidsmpf.streaming.core.message import Message

from cudf_polars.containers import DataFrame, DataType
from cudf_polars.dsl.ir import Empty, IRExecutionContext
from cudf_polars.engine.options import StreamingOptions
from cudf_polars.streaming.actor_graph.collectives.common import reserve_op_id
from cudf_polars.streaming.actor_graph.collectives.sort import (
    extract_orderscheme_partitioning,
)
from cudf_polars.streaming.actor_graph.utils import gather_in_task_group
from cudf_polars.testing.asserts import assert_gpu_result_equal


@pytest.fixture
def engine(streaming_engine_factory):
    return streaming_engine_factory(
        StreamingOptions(
            max_rows_per_partition=3,
            fallback_mode="raise",
            raise_on_fail=True,
        ),
    )


@pytest.fixture
def engine_large(streaming_engine_factory):
    return streaming_engine_factory(
        StreamingOptions(
            max_rows_per_partition=2_100,
            fallback_mode="raise",
            raise_on_fail=True,
        ),
    )


@pytest.fixture(scope="module")
def df():
    return pl.LazyFrame(
        {
            "x": [1, 2, 3, 4, 5, 6, 7],
            "y": [1, 6, 7, 2, 5, 4, 3],
            "z": ["e", "c", "b", "g", "a", "f", "d"],
        }
    )


def large_frames():
    x = [1.0] * 10_000
    x[-1] = float("nan")
    y = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 1000

    yield pytest.param(
        pl.LazyFrame(
            {
                "x": x,
            }
        ),
        ["x"],
        False,
        id="all_equal_one_nan",
    )

    yield pytest.param(
        pl.LazyFrame(
            {
                "x": x,
                "y": y,
            }
        ),
        ["x", "y"],
        False,
        id="two_cols",
    )

    idx = list(range(10_000))
    yield pytest.param(
        pl.LazyFrame(
            {
                "x": x,
                "y": y,
                "idx": idx,
            }
        ),
        ["x", "y"],
        True,
        id="two_col_stable",
    )


def test_sort(df, engine):
    q = df.sort(by=["y", "z"])
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("large_df,by,stable", list(large_frames()))
@pytest.mark.parametrize(
    "nulls_last,descending", [(True, False), (True, True), (False, True)]
)
def test_large_sort(large_df, by, engine_large, stable, nulls_last, descending):
    q = large_df.sort(
        by=by, nulls_last=nulls_last, maintain_order=stable, descending=descending
    )
    assert_gpu_result_equal(q, engine=engine_large)


def test_sort_head(df, engine):
    q = df.sort(by=["y", "z"]).head(2)
    assert_gpu_result_equal(q, engine=engine)


def test_sort_tail(df, engine):
    q = df.sort(by=["y", "z"]).tail(2)
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("offset", [1, -4])
def test_sort_slice(df, engine, offset):
    # Slice in the middle, which distributed sorts need to be careful with
    q = df.sort(by=["y", "z"]).slice(offset, 2)
    with pytest.raises(
        NotImplementedError,
        match=r"This slice not supported for multiple partitions.",
    ):
        assert_gpu_result_equal(q, engine=engine)


def test_sort_after_sparse_join(streaming_engine_factory):
    engine = streaming_engine_factory(
        StreamingOptions(max_rows_per_partition=4, raise_on_fail=True),
    )
    left = pl.LazyFrame({"foo": list(range(5)), "bar": list(range(5))})
    right = pl.LazyFrame({"foo": list(range(1))})
    q = left.join(right, on="foo", how="inner").sort(by=["foo"])
    assert_gpu_result_equal(q, engine=engine)


async def _send_frames(context, ch, frames) -> None:
    stream = context.br().stream_pool.get_stream()
    for i, frame in enumerate(frames):
        tbl = DataFrame.from_polars(frame, stream).table
        await ch.send(
            context,
            Message(
                i,
                TableChunk.from_pylibcudf_table(
                    tbl, stream, exclusive_view=True, br=context.br()
                ),
            ),
        )
    await ch.drain(context)


def _run_extract_orderscheme_partitioning(
    spmd_engine,
    schema_ir,
    order_keys,
    frames,
):
    context = spmd_engine.context
    comm = spmd_engine.comm

    async def _run():
        ch = context.create_channel()
        with (
            ThreadPoolExecutor(max_workers=1) as executor,
            reserve_op_id() as op_id,
        ):
            ir_context = IRExecutionContext(
                executor, get_cuda_stream=context.br().stream_pool.get_stream
            )
            _, result = await gather_in_task_group(
                _send_frames(context, ch, frames),
                extract_orderscheme_partitioning(
                    context, comm, schema_ir, ir_context, ch, order_keys, op_id
                ),
            )
        return result

    return asyncio.run(_run())


def _chunk_data(
    context, result, schema: dict[str, DataType]
) -> list[dict[str, list[object]]]:
    """Return buffered chunk values in replay order."""
    _, rows = _chunk_sequences_and_data(context, result, schema)
    return rows


def _chunk_sequences_and_data(
    context, result, schema: dict[str, DataType]
) -> tuple[list[int], list[dict[str, list[object]]]]:
    """Return buffered chunk sequence numbers and values in replay order."""
    names = list(schema)
    dtypes = list(schema.values())
    sequence_numbers = []
    rows = []
    for msg in result.chunks:
        sequence_numbers.append(msg.sequence_number)
        chunk = TableChunk.from_message(msg, br=context.br())
        df = DataFrame.from_table(
            chunk.table_view(),
            names,
            dtypes,
            stream=chunk.stream,
        ).to_polars()
        rows.append({name: df[name].to_list() for name in names})
    return sequence_numbers, rows


def _key_frame(keys) -> pl.DataFrame:
    return pl.DataFrame({"key": pl.Series(keys, dtype=pl.Int32())})


def _only_ordering(partitioning, order_keys):
    assert partitioning is not None
    assert partitioning.local == "inherit"
    inter_rank = partitioning.inter_rank
    assert isinstance(inter_rank, OrderScheme)
    (ordering,) = inter_rank.orderings
    assert ordering.keys == tuple(order_keys)
    return ordering


def _boundary_values(context, ordering, name: str = "key") -> list[object]:
    chunk = ordering.get_boundaries(context.br())
    return (
        DataFrame.from_table(
            chunk.table_view(), [name], [DataType(pl.Int32())], stream=chunk.stream
        )
        .to_polars()[name]
        .to_list()
    )


@pytest.mark.spmd
@pytest.mark.parametrize("n_chunks", [2, 4])
def test_extract_orderscheme_partitioning(spmd_engine, n_chunks) -> None:
    context = spmd_engine.context
    comm = spmd_engine.comm

    n_rows = 4
    key_start = comm.rank * n_chunks * n_rows
    order_keys = [OrderKey(0, plc.types.Order.ASCENDING, plc.types.NullOrder.BEFORE)]
    schema_ir = Empty({"key": DataType(pl.Int32())})
    frames = [
        _key_frame(range(start, start + n_rows))
        for i in range(n_chunks)
        for start in [key_start + i * n_rows]
    ]

    result = _run_extract_orderscheme_partitioning(
        spmd_engine, schema_ir, order_keys, frames
    )

    assert len(result.chunks) == n_chunks
    ordering = _only_ordering(result.partitioning, order_keys)
    assert ordering.strict_boundaries  # all keys are distinct integers
    assert ordering.num_boundaries == comm.nranks * n_chunks - 1

    # Verify actual boundary values: start of each partition except the first
    expected_keys = [i * n_rows for i in range(1, comm.nranks * n_chunks)]
    assert _boundary_values(context, ordering) == expected_keys
    assert _chunk_data(context, result, {"key": DataType(pl.Int32())}) == [
        {"key": list(range(key_start + i * n_rows, key_start + (i + 1) * n_rows))}
        for i in range(n_chunks)
    ]


@pytest.mark.spmd
def test_extract_orderscheme_partitioning_projects_order_keys(spmd_engine) -> None:
    context = spmd_engine.context
    comm = spmd_engine.comm

    n_chunks = 2
    n_rows = 4
    key_start = comm.rank * n_chunks * n_rows
    order_keys = [OrderKey(1, plc.types.Order.ASCENDING, plc.types.NullOrder.BEFORE)]
    schema_ir = Empty({"payload": DataType(pl.Int64()), "key": DataType(pl.Int32())})
    frames = [
        pl.DataFrame(
            {
                # Deliberately not sorted by payload.
                "payload": pl.Series(
                    range(-start, -start - n_rows, -1), dtype=pl.Int64()
                ),
                "key": pl.Series(range(start, start + n_rows), dtype=pl.Int32()),
            }
        )
        for i in range(n_chunks)
        for start in [key_start + i * n_rows]
    ]

    result = _run_extract_orderscheme_partitioning(
        spmd_engine, schema_ir, order_keys, frames
    )

    assert len(result.chunks) == n_chunks
    ordering = _only_ordering(result.partitioning, order_keys)
    assert ordering.strict_boundaries
    assert ordering.num_boundaries == comm.nranks * n_chunks - 1
    chunk = ordering.get_boundaries(context.br())
    assert chunk.table_view().num_columns() == 1
    expected_keys = [i * n_rows for i in range(1, comm.nranks * n_chunks)]
    assert _boundary_values(context, ordering) == expected_keys
    assert _chunk_data(
        context,
        result,
        {"payload": DataType(pl.Int64()), "key": DataType(pl.Int32())},
    ) == [
        {
            "payload": list(range(-start, -start - n_rows, -1)),
            "key": list(range(start, start + n_rows)),
        }
        for start in (key_start + i * n_rows for i in range(n_chunks))
    ]


@pytest.mark.spmd
def test_extract_orderscheme_partitioning_unsorted(spmd_engine) -> None:
    context = spmd_engine.context

    order_keys = [OrderKey(0, plc.types.Order.ASCENDING, plc.types.NullOrder.BEFORE)]
    schema_ir = Empty({"key": DataType(pl.Int32())})
    result = _run_extract_orderscheme_partitioning(
        spmd_engine,
        schema_ir,
        order_keys,
        [_key_frame([10, 11, 12, 13]), _key_frame([0, 1, 2, 3])],
    )

    assert result.partitioning is None
    assert len(result.chunks) == 2
    assert _chunk_data(context, result, {"key": DataType(pl.Int32())}) == [
        {"key": [10, 11, 12, 13]},
        {"key": [0, 1, 2, 3]},
    ]


@pytest.mark.spmd
def test_extract_orderscheme_partitioning_single_chunk(spmd_engine) -> None:
    """One chunk on a single rank → num_partitions == 1 < 2 → None."""
    context = spmd_engine.context
    comm = spmd_engine.comm

    if comm.nranks != 1:
        pytest.skip("single-partition None path only applies when nranks == 1")

    order_keys = [OrderKey(0, plc.types.Order.ASCENDING, plc.types.NullOrder.BEFORE)]
    schema_ir = Empty({"key": DataType(pl.Int32())})
    result = _run_extract_orderscheme_partitioning(
        spmd_engine, schema_ir, order_keys, [_key_frame(range(4))]
    )

    assert result.partitioning is None
    assert len(result.chunks) == 1
    assert _chunk_data(context, result, {"key": DataType(pl.Int32())}) == [
        {"key": [0, 1, 2, 3]},
    ]


@pytest.mark.spmd
def test_extract_orderscheme_partitioning_preserves_empty_replay_chunks(
    spmd_engine,
) -> None:
    """Empty chunks are replayed but excluded from boundary extraction."""
    context = spmd_engine.context
    comm = spmd_engine.comm

    n_rows = 4
    key_start = comm.rank * 2 * n_rows
    order_keys = [OrderKey(0, plc.types.Order.ASCENDING, plc.types.NullOrder.BEFORE)]
    schema_ir = Empty({"key": DataType(pl.Int32())})
    result = _run_extract_orderscheme_partitioning(
        spmd_engine,
        schema_ir,
        order_keys,
        [
            _key_frame(range(key_start, key_start + n_rows)),
            _key_frame([]),
            _key_frame(range(key_start + n_rows, key_start + 2 * n_rows)),
        ],
    )
    sequence_numbers, rows = _chunk_sequences_and_data(
        context, result, {"key": DataType(pl.Int32())}
    )

    assert result.partitioning is not None
    assert sequence_numbers == [0, 1, 2]
    assert rows == [
        {"key": list(range(key_start, key_start + n_rows))},
        {"key": []},
        {"key": list(range(key_start + n_rows, key_start + 2 * n_rows))},
    ]


@pytest.mark.spmd
def test_extract_orderscheme_partitioning_all_empty(spmd_engine) -> None:
    """All-empty input is replayed without extracted partitioning."""
    context = spmd_engine.context
    order_keys = [OrderKey(0, plc.types.Order.ASCENDING, plc.types.NullOrder.BEFORE)]
    schema_ir = Empty({"key": DataType(pl.Int32())})

    result = _run_extract_orderscheme_partitioning(
        spmd_engine,
        schema_ir,
        order_keys,
        [_key_frame([]), _key_frame([])],
    )
    sequence_numbers, rows = _chunk_sequences_and_data(
        context, result, {"key": DataType(pl.Int32())}
    )

    assert result.partitioning is None
    assert sequence_numbers == [0, 1]
    assert rows == [{"key": []}, {"key": []}]


@pytest.mark.spmd
def test_extract_orderscheme_partitioning_descending(spmd_engine) -> None:
    """Boundary values and strictness are correct for descending sort order."""
    context = spmd_engine.context
    comm = spmd_engine.comm

    if comm.nranks != 1:
        pytest.skip("descending boundary value check is clearest on single rank")

    # Two chunks sorted descending: [7,6,5,4] then [3,2,1,0]
    # Expected: 1 boundary at the first row of the second chunk.
    order_keys = [OrderKey(0, plc.types.Order.DESCENDING, plc.types.NullOrder.AFTER)]
    schema_ir = Empty({"key": DataType(pl.Int32())})

    result = _run_extract_orderscheme_partitioning(
        spmd_engine,
        schema_ir,
        order_keys,
        [_key_frame([7, 6, 5, 4]), _key_frame([3, 2, 1, 0])],
    )

    assert len(result.chunks) == 2
    ordering = _only_ordering(result.partitioning, order_keys)
    assert ordering.strict_boundaries
    assert ordering.num_boundaries == 1

    assert _boundary_values(context, ordering) == [3]
    assert _chunk_data(context, result, {"key": DataType(pl.Int32())}) == [
        {"key": [7, 6, 5, 4]},
        {"key": [3, 2, 1, 0]},
    ]


def test_sort_by_renamed_join_column(streaming_engine_factory):
    engine = streaming_engine_factory(
        StreamingOptions(max_rows_per_partition=1, raise_on_fail=True),
    )
    df1 = pl.LazyFrame({"k1": [1, 2], "text": ["A", "B"]})
    df2 = pl.LazyFrame({"k2": [1, 2], "text": ["y", "x"]})
    ctx = pl.SQLContext()
    ctx.register("df1", df1)
    ctx.register("df2", df2)
    q = ctx.execute(
        "SELECT df1.text AS t1, df2.text AS t2 "
        "FROM df1 INNER JOIN df2 ON df1.k1 = df2.k2 "
        "ORDER BY df2.text"
    )
    assert_gpu_result_equal(q, engine=engine)
