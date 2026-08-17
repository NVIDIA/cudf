# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio

import pytest

import polars as pl

from cudf_streaming.channel_metadata import (
    ChannelMetadata,
    HashScheme,
    Partitioning,
)
from cudf_streaming.table_chunk import TableChunk

from cudf_polars.containers import DataFrame, DataType
from cudf_polars.engine.options import StreamingOptions
from cudf_polars.engine.spmd import allgather_polars_dataframe
from cudf_polars.streaming.actor_graph.collectives.common import reserve_op_id
from cudf_polars.streaming.actor_graph.collectives.shuffle import (
    LocalRepartitioner,
    ShuffleManager,
)
from cudf_polars.streaming.actor_graph.utils import (
    _is_already_partitioned,
)
from cudf_polars.testing.asserts import assert_gpu_result_equal


@pytest.mark.parametrize(
    "options",
    [
        StreamingOptions(max_rows_per_partition=1, broadcast_limit=48),
        StreamingOptions(max_rows_per_partition=5, broadcast_limit=48),
    ],
)
def test_join_rapidsmpf(streaming_engine_factory, options) -> None:
    streaming_engine = streaming_engine_factory(options)
    left = pl.LazyFrame(
        {
            "x": range(15),
            "y": [1, 2, 3] * 5,
            "z": [1.0, 2.0, 3.0, 4.0, 5.0] * 3,
        }
    )
    right = pl.LazyFrame(
        {
            "xx": range(6),
            "y": [2, 4, 3] * 2,
            "zz": [1, 2] * 3,
        }
    )
    q = left.join(right, on="y", how="inner")
    assert_gpu_result_equal(q, engine=streaming_engine, check_row_order=False)


@pytest.mark.parametrize(
    "options",
    [
        StreamingOptions(max_rows_per_partition=1),
        StreamingOptions(max_rows_per_partition=5),
    ],
)
def test_sort_rapidsmpf(streaming_engine_factory, options) -> None:
    streaming_engine = streaming_engine_factory(options)
    df = pl.LazyFrame(
        {
            "x": range(15),
            "y": [1, 2, 3] * 5,
            "z": [1.0, 2.0, 3.0, 4.0, 5.0] * 3,
        }
    )
    q = df.sort(by=["y", "z"])
    assert_gpu_result_equal(q, engine=streaming_engine, check_row_order=True)


@pytest.mark.parametrize(
    "join_expr",
    [
        pl.col("y") * 2,
        [pl.col("y"), pl.col("a") + 1],
    ],
)
@pytest.mark.parametrize(
    "dynamic_planning",
    [None, {}],
    ids=["static-planning", "dynamic-planning"],
)
def test_join_non_col_keys_rapidsmpf(
    streaming_engine_factory, join_expr, dynamic_planning
) -> None:
    """Pointwise expression keys are evaluated transiently for hash shuffling."""
    engine = streaming_engine_factory(
        StreamingOptions(
            max_rows_per_partition=1,
            target_partition_size=1,
            broadcast_limit=1,
            dynamic_planning=dynamic_planning,
            raise_on_fail=True,
        )
    )
    left = pl.LazyFrame(
        {
            "a": [1, 2, 3, 4] * 3,
            "y": [10, 20, 30, 40] * 3,
            "val_l": range(12),
        }
    )
    right = pl.LazyFrame(
        {
            "a": [1, 2, 3, 4] * 2,
            "y": [10, 20, 30, 40] * 2,
            "val_r": range(8),
        }
    )
    q = left.join(right, on=join_expr, how="inner", coalesce=True)
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


@pytest.mark.parametrize(
    "dynamic_planning",
    [None, {}],
    ids=["static-planning", "dynamic-planning"],
)
def test_join_asymmetric_non_col_keys_rapidsmpf(
    streaming_engine_factory, dynamic_planning
) -> None:
    """Asymmetric pointwise expressions can be hash-shuffled without payload columns."""
    engine = streaming_engine_factory(
        StreamingOptions(
            max_rows_per_partition=1,
            target_partition_size=1,
            broadcast_limit=1,
            dynamic_planning=dynamic_planning,
            raise_on_fail=True,
        )
    )
    left = pl.LazyFrame({"a": [1, 2, 3, 4], "val_l": ["x", "y", "z", "w"]})
    right = pl.LazyFrame({"b": [2, 3, 4, 5], "val_r": ["p", "q", "r", "s"]})
    q = left.join(right, left_on=pl.col("a") + 1, right_on=pl.col("b"), how="inner")
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


def test_is_already_partitioned():
    # Unit test for _is_already_partitioned helper
    chunks = 4
    columns = (0, 1)
    modulus = 8
    nranks = 1

    # Exact match: should return True
    metadata_match = ChannelMetadata(
        chunks,
        partitioning=Partitioning(
            inter_rank=HashScheme(columns, modulus),
            local="inherit",
        ),
    )
    assert _is_already_partitioned(metadata_match, columns, modulus, nranks) is True

    # Different columns: should return False
    metadata_diff_cols = ChannelMetadata(
        chunks,
        partitioning=Partitioning(
            inter_rank=HashScheme((0,), modulus),
            local="inherit",
        ),
    )
    assert (
        _is_already_partitioned(metadata_diff_cols, columns, modulus, nranks) is False
    )

    # Different local partitioning: should return False
    metadata_diff_local = ChannelMetadata(
        chunks,
        partitioning=Partitioning(
            inter_rank=HashScheme(columns, modulus),
            local=None,
        ),
    )
    assert (
        _is_already_partitioned(metadata_diff_local, columns, modulus, nranks) is False
    )

    # Different modulus: should return False
    metadata_diff_mod = ChannelMetadata(
        chunks,
        partitioning=Partitioning(
            inter_rank=HashScheme(columns, 16),
            local="inherit",
        ),
    )
    assert _is_already_partitioned(metadata_diff_mod, columns, modulus, nranks) is False

    # No partitioning: should return False
    metadata_none = ChannelMetadata(chunks)
    assert _is_already_partitioned(metadata_none, columns, modulus, nranks) is False

    # Local not "inherit": should return False
    metadata_local = ChannelMetadata(
        chunks,
        partitioning=Partitioning(
            inter_rank=HashScheme(columns, modulus),
            local=HashScheme((0,), 4),
        ),
    )
    assert _is_already_partitioned(metadata_local, columns, modulus, nranks) is False


@pytest.mark.spmd
@pytest.mark.parametrize("local_count", [1, 2, 4])
def test_local_repartitioner_hash(spmd_engine, local_count) -> None:
    context = spmd_engine.context
    comm = spmd_engine.comm

    pl_df = pl.DataFrame({"key": list(range(4)) * 3, "val": list(range(12))})
    col_names = pl_df.columns
    dtypes = [DataType(dt) for dt in pl_df.dtypes]

    results: list[tuple[int, pl.DataFrame]] = []

    async def _run() -> None:
        stream = context.br().stream_pool.get_stream()
        cudf_df = DataFrame.from_polars(pl_df, stream)
        with reserve_op_id() as op_id:
            shuffle = ShuffleManager(
                context, comm, num_partitions=comm.nranks, collective_id=op_id
            )
            async with shuffle.inserting() as inserter:
                inserter.insert_hash(
                    TableChunk.from_pylibcudf_table(
                        cudf_df.table, stream, exclusive_view=True, br=context.br()
                    ),
                    columns_to_hash=(0,),
                )

            local = LocalRepartitioner(shuffle, local_count=local_count)
            await local.repartition_by_hash(columns_to_hash=(0,), stream=stream)

            for pid in local.local_partitions():
                tbl = local.extract_chunk(pid, stream)
                results.append(
                    (
                        pid,
                        DataFrame.from_table(
                            tbl, col_names, dtypes, stream
                        ).to_polars(),
                    )
                )

    asyncio.run(_run())

    assert len(results) == local_count

    # Same key always lands in the same local partition.
    key_to_pid: dict[int, int] = {}
    for pid, df in results:
        for key_val in df["key"].to_list():
            assert key_to_pid.setdefault(key_val, pid) == pid

    # AllGather across ranks: every rank inserts 12 rows, all must survive.
    local_df = pl.concat([df for _, df in results])
    with reserve_op_id() as op_id:
        global_df = allgather_polars_dataframe(
            engine=spmd_engine, local_df=local_df, op_id=op_id
        )
    assert global_df.height == 12 * comm.nranks


@pytest.mark.spmd
@pytest.mark.parametrize("local_count", [1, 2, 4])
def test_local_repartitioner_index(spmd_engine, local_count) -> None:
    context = spmd_engine.context
    comm = spmd_engine.comm

    pl_payload = pl.DataFrame(
        {
            "local_part": [i % local_count for i in range(12)],
            "val": list(range(12)),
        }
    )
    pl_rank_part = pl.DataFrame({"rank_part": [i % comm.nranks for i in range(12)]})
    out_col_names = ["val"]
    out_dtypes = [DataType(pl.Int32())]

    results: list[tuple[int, pl.DataFrame]] = []

    async def _run() -> None:
        stream = context.br().stream_pool.get_stream()
        payload_df = DataFrame.from_polars(pl_payload, stream)
        rank_part_df = DataFrame.from_polars(pl_rank_part, stream)

        with reserve_op_id() as op_id:
            shuffle = ShuffleManager(
                context, comm, num_partitions=comm.nranks, collective_id=op_id
            )
            async with shuffle.inserting() as inserter:
                inserter.insert_index(
                    TableChunk.from_pylibcudf_table(
                        payload_df.table, stream, exclusive_view=True, br=context.br()
                    ),
                    TableChunk.from_pylibcudf_table(
                        rank_part_df.table, stream, exclusive_view=True, br=context.br()
                    ),
                )

            local = LocalRepartitioner(shuffle, local_count=local_count)
            await local.repartition_by_index(partition_col=0, stream=stream)

            for pid in local.local_partitions():
                tbl = local.extract_chunk(pid, stream)
                results.append(
                    (
                        pid,
                        DataFrame.from_table(
                            tbl, out_col_names, out_dtypes, stream
                        ).to_polars(),
                    )
                )

    asyncio.run(_run())

    assert len(results) == local_count

    # Routing: val=v must land in local partition v % local_count (== local_part).
    for pid, df in results:
        assert df.columns == ["val"]
        for val in df["val"].to_list():
            assert val % local_count == pid

    # Global: every inserted row survives.
    local_df = pl.concat([df for _, df in results])
    with reserve_op_id() as op_id:
        global_df = allgather_polars_dataframe(
            engine=spmd_engine, local_df=local_df, op_id=op_id
        )
    assert global_df.height == 12 * comm.nranks
