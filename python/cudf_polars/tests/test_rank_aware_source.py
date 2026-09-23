# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math
import sys

import cloudpickle
import pytest

import polars as pl
from polars.io.plugins import register_io_source
from polars.testing import assert_frame_equal

from cudf_polars.containers import DataFrame
from cudf_polars.streaming.rank_aware_source import RankAwareSource, SizedChunks
from cudf_polars.utils.cuda_stream import get_cuda_stream

# The scan-source classes below are defined in this test module. When shipped to
# Ray/Dask workers they would otherwise be pickled by reference, and the worker
# cannot import this test module. Register the module for pickle-by-value so the
# class definitions travel with the instances. Ray vendors its own cloudpickle,
# so register on that too when Ray is installed.
cloudpickle.register_pickle_by_value(sys.modules[__name__])
try:
    from ray import cloudpickle as ray_cloudpickle
except ImportError:
    pass
else:
    ray_cloudpickle.register_pickle_by_value(sys.modules[__name__])


class Source(RankAwareSource):
    """Rank-aware scan source that emits ``df`` on rank 0."""

    def __init__(self, df: pl.DataFrame) -> None:
        self.df = df

    def __call__(self, with_columns, predicate, n_rows, batch_size, rank=0, nranks=1):
        out = self.df.clear() if rank else self.df
        if with_columns is not None:
            out = out.select(with_columns)
        if predicate is not None:
            out = out.filter(predicate)
        if n_rows is not None:
            out = out.head(n_rows)
        yield out


def test_rank_aware_source(engine: pl.GPUEngine):
    df = pl.DataFrame(
        {"a": pl.Series([1, 2, 3, 4, 5], dtype=pl.Int8()), "b": [10, 20, 30, 40, 50]}
    )
    q = register_io_source(Source(df), schema={"a": pl.Int8, "b": pl.Int64})
    assert_frame_equal(q.collect(engine=engine), df)
    assert_frame_equal(q.select("a").collect(engine=engine), df.select("a"))
    assert_frame_equal(
        q.filter(pl.col("a") > 2).collect(engine=engine), df.filter(pl.col("a") > 2)
    )


def test_rank_aware_source_multi_chunk(engine: pl.GPUEngine):
    class MultiChunkSource(RankAwareSource):
        """Rank-aware scan source whose data lives on rank 0, emitted as two chunks."""

        def __init__(self, chunks: list[pl.DataFrame]) -> None:
            self.chunks = chunks

        def __call__(
            self, with_columns, predicate, n_rows, batch_size, rank=0, nranks=1
        ):
            if rank:
                yield from (c.clear() for c in self.chunks)
            else:
                yield from self.chunks

    chunks = [pl.DataFrame({"a": [1, 2]}), pl.DataFrame({"a": [3, 4]})]
    q = register_io_source(MultiChunkSource(chunks), schema={"a": pl.Int64})
    assert_frame_equal(q.collect(engine=engine), pl.DataFrame({"a": [1, 2, 3, 4]}))


def test_rank_aware_source_gpu_chunks(engine: pl.GPUEngine):
    class GpuSource(RankAwareSource):
        """Rank-aware source that yields GPU-resident cudf-polars DataFrames."""

        def __init__(self, df: pl.DataFrame) -> None:
            self.df = df

        def __call__(
            self, with_columns, predicate, n_rows, batch_size, rank=0, nranks=1
        ):
            df = self.df if not rank else self.df.clear()
            yield DataFrame.from_polars(df, stream=get_cuda_stream())

    df = pl.DataFrame({"a": [1, 2, 3]})
    q = register_io_source(GpuSource(df), schema={"a": pl.Int64})
    assert_frame_equal(q.collect(engine=engine), df)


class PartitioningSource(RankAwareSource):
    """Self-partitions a shared frame across workers using ``rank``/``nranks``."""

    def __init__(self, df: pl.DataFrame) -> None:
        self.df = df

    def __call__(self, with_columns, predicate, n_rows, batch_size, rank=0, nranks=1):
        per_rank = math.ceil(self.df.height / nranks)
        out = self.df.slice(rank * per_rank, per_rank)
        if with_columns is not None:
            out = out.select(with_columns)
        yield out


def test_rank_aware_source_self_partition(engine: pl.GPUEngine):
    df = pl.DataFrame({"a": list(range(10))})
    q = register_io_source(PartitioningSource(df), schema={"a": pl.Int64})
    assert_frame_equal(q.collect(engine=engine), df)


def test_sized_chunks_reports_count_and_iterates():
    # SizedChunks reports its length up front (so the streaming engine can stream
    # lazily) and is a single-pass iterator over the wrapped chunks.
    frames = [pl.DataFrame({"a": [1]}), pl.DataFrame({"a": [2]})]
    sized = SizedChunks(2, iter(frames))
    assert len(sized) == 2
    assert list(sized) == frames  # single-pass iteration yields the chunks
    assert list(sized) == []  # exhausted after one pass


@pytest.mark.spmd
def test_rank_aware_source_collect_twice(spmd_engine: pl.GPUEngine):
    df = pl.DataFrame({"a": list(range(10))})
    q = register_io_source(PartitioningSource(df), schema={"a": pl.Int64})

    # First collect runs the rank-aware source under streaming (binds the rank).
    assert_frame_equal(q.collect(engine=spmd_engine), df)

    # Second collect uses the default engine, which does not bind a rank, so the
    # source runs with its rank=0, nranks=1 defaults and returns everything.
    assert_frame_equal(q.collect(), df)
