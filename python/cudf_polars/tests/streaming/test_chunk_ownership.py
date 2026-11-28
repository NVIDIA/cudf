# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import polars as pl

import pylibcudf as plc
from cudf_streaming.table_chunk import TableChunk

from cudf_polars.containers import DataFrame, DataType
from cudf_polars.engine.options import StreamingOptions
from cudf_polars.streaming.actor_graph import utils


def test_evaluated_chunk_owns_shared_output_columns(spmd_engine_factory, monkeypatch):
    br = spmd_engine_factory(StreamingOptions()).context.br()
    producer, consumer = (br.stream_pool.get_stream() for _ in range(2))
    table = DataFrame.from_polars(pl.DataFrame({"x": [1, None, 3]}), producer).table
    original_ptr = table.columns()[0].data().ptr
    chunk = TableChunk.from_pylibcudf_table(table, producer, exclusive_view=True, br=br)
    del table

    dtype = DataType(pl.Int64())
    fresh_ptr = None

    def evaluate(_ir, source, *, context):
        nonlocal fresh_ptr
        fresh = DataFrame.from_polars(pl.DataFrame({"y": [4, 5, 6]}), consumer)
        new_column = fresh.table.columns()[0]
        fresh_ptr = new_column.data().ptr
        return DataFrame.from_table(
            plc.Table([source.table.columns()[0], new_column]),
            ["x", "y"],
            [dtype, dtype],
            consumer,
        )

    monkeypatch.setattr(utils, "evaluate_ir", evaluate)
    ir = SimpleNamespace(children=[SimpleNamespace(schema={"x": dtype})])
    result = utils._evaluate_chunk_sync(chunk, ir, None, br)
    del chunk

    assert result.is_spillable()
    copied, unchanged = result.table_view().columns()
    assert copied.data().ptr != original_ptr
    assert unchanged.data().ptr == fresh_ptr
    consumer.synchronize()
    assert copied.to_pylist() == [1, None, 3]
