# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay Polars' chunk view and wrapping on the archived native stack alone."""

import os
from concurrent.futures import ThreadPoolExecutor

import pyarrow as pa
import pylibcudf as plc
import pytest
import rmm
from cudf_streaming.table_chunk import TableChunk
from rapidsmpf.memory.buffer_resource import BufferResource

_INT64 = plc.DataType(plc.TypeId.INT64)


def _run_chunk(source, consumer, br, mode):
    producer = source.stream
    plc.experimental.join_streams([producer], consumer)
    source_table = source.table_view()
    columns = source_table.columns()
    if mode == "alias":
        result = plc.Table(columns)
    else:
        every = plc.expressions.Literal(
            plc.Scalar.from_py(7_000_000, _INT64, stream=consumer)
        )
        remainder = plc.transform.compute_column(
            source_table,
            plc.expressions.Operation(
                plc.expressions.ASTOperator.PYMOD,
                plc.expressions.ColumnReference(0),
                every,
            ),
            stream=consumer,
        )
        result = plc.Table(
            [
                plc.binaryop.binary_operation(
                    columns[0],
                    remainder,
                    plc.binaryop.BinaryOperator.SUB,
                    _INT64,
                    stream=consumer,
                )
            ]
        )
    fixed = mode == "alias" and os.environ.get("CUDF_POLARS_SANITIZE_PACKED")
    if fixed:
        import polars as pl
        from cudf_polars.containers import DataFrame, DataType
        from cudf_polars.streaming.actor_graph.utils import (
            _copy_shared_output_columns,
        )

        names, dtypes = ["x"], [DataType(pl.Int64())]
        result = _copy_shared_output_columns(
            DataFrame.from_table(result, names, dtypes, consumer),
            DataFrame.from_table(source_table, names, dtypes, producer),
        )
        assert result.columns()[0].data().ptr != columns[0].data().ptr
    wrapped = TableChunk.from_pylibcudf_table(
        result, consumer, exclusive_view=bool(fixed) or mode != "alias", br=br
    )
    # Keep producer-side deallocation after the final consumer access.
    plc.experimental.join_streams([consumer], producer)
    assert wrapped.shape == source.shape
    return wrapped


@pytest.mark.parametrize("mode", ["compute", "alias"])
def test_chunk_handoff(mode):
    previous = rmm.mr.get_current_device_resource()
    mr = rmm.mr.CudaAsyncMemoryResource()
    rmm.mr.set_current_device_resource(mr)
    try:
        br = BufferResource(mr)
        streams = [br.stream_pool.get_stream() for _ in range(8)]
        rounds = int(os.environ.get("CUDF_POLARS_REDUCED_ROUNDS", "500"))
        with ThreadPoolExecutor(max_workers=4) as threads:
            for step in range(rounds):
                rows = (17, 72, 128, 384)[step % 4]
                futures = []
                for producer, consumer in zip(
                    streams[:4], streams[4:], strict=True
                ):
                    table = plc.Table(
                        [
                            plc.Column.from_arrow(
                                pa.array(
                                    [
                                        3_279_111 * i if i % 11 else None
                                        for i in range(rows)
                                    ],
                                    type=pa.int64(),
                                ),
                                stream=producer,
                            )
                        ]
                    )
                    source = TableChunk.from_pylibcudf_table(
                        table, producer, exclusive_view=True, br=br
                    )
                    del table
                    futures.append(
                        threads.submit(_run_chunk, source, consumer, br, mode)
                    )
                for future in futures:
                    future.result()
        del source, futures, future
        for stream in streams:
            stream.synchronize()
    finally:
        rmm.mr.set_current_device_resource(previous)
