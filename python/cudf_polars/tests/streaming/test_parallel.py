# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pickle

import pytest

import polars as pl
from polars import GPUEngine
from polars.testing import assert_frame_equal

from cudf_polars import Translator
from cudf_polars.dsl.ir import Cache, Join
from cudf_polars.dsl.traversal import traversal
from cudf_polars.engine.options import StreamingOptions
from cudf_polars.streaming.base import StatsCollector
from cudf_polars.streaming.parallel import optimize_with_stats
from cudf_polars.testing.asserts import assert_gpu_result_equal
from cudf_polars.utils.config import ConfigOptions


@pytest.mark.parametrize("column", ["a", "b"])
def test_explode_multi(column, streaming_engine):
    df = pl.LazyFrame(
        {
            "a": [[1, 2], [3, 4], None],
            "b": [[5, 6], [7, 8], [9, 10]],
            "c": [None, 11, 12],
        }
    )
    q = df.explode(column)
    assert_gpu_result_equal(q, engine=streaming_engine)


@pytest.mark.parametrize(
    "mapping", [{}, {"b": "c"}, {"b": "a", "a": "b"}, {"a": "c", "b": "d"}]
)
def test_rename_multi(mapping, streaming_engine):
    df = pl.LazyFrame({"a": [1, 2, 3], "b": [3, 4, 5]})
    q = df.rename(mapping)
    assert_gpu_result_equal(q, engine=streaming_engine)


def test_rename_concat(streaming_engine) -> None:
    # https://github.com/rapidsai/cudf/pull/19121#issuecomment-2959305678
    q = pl.concat(
        [
            pl.LazyFrame({"a": [1, 2, 3]}).rename({"a": "A"}),
            pl.LazyFrame({"a": [4, 5, 6]}).rename({"a": "A"}),
        ]
    )
    assert_gpu_result_equal(q, engine=streaming_engine)


def test_fallback_on_concat_zlice(spmd_engine_factory) -> None:
    # Pin ``fallback_mode="warn"`` so the spmd-small baseline (which sets
    # ``SILENT``) doesn't suppress the warning this test asserts on.
    streaming_engine = spmd_engine_factory(StreamingOptions(fallback_mode="warn"))
    q = pl.concat(
        [
            pl.LazyFrame({"a": [1, 2]}),
            pl.LazyFrame({"a": [3, 4]}),
            pl.LazyFrame({"a": [5, 6]}),
        ]
    ).tail(1)

    with pytest.raises(
        UserWarning, match=r"This slice not supported for multiple partitions."
    ):
        assert_gpu_result_equal(q, engine=streaming_engine)


# ---------------------------------------------------------------------------
# Tests migrated from tests/streaming/test_parallel.py
# ---------------------------------------------------------------------------


def test_evaluate_streaming(streaming_engine):
    df = pl.LazyFrame({"a": [1, 2, 3], "b": [3, 4, 5], "c": [5, 6, 7], "d": [7, 9, 8]})
    q = df.select(pl.col("a") - (pl.col("b") + pl.col("c") * 2), pl.col("d")).sort("d")

    expected = q.collect(engine="in-memory")
    # Use the in-memory executor for the GPU comparison: a vanilla
    # ``pl.GPUEngine(raise_on_fail=True)`` defaults to ``executor="streaming"``,
    # which routes through ``DefaultSingletonEngine`` and would fail while the
    # session ``SPMDEngine`` from conftest is alive.
    got_gpu = q.collect(engine=pl.GPUEngine(executor="in-memory", raise_on_fail=True))
    got_streaming = q.collect(engine=streaming_engine)
    assert_frame_equal(expected, got_gpu)
    assert_frame_equal(expected, got_streaming)


def test_optimize_removes_cache_nodes() -> None:
    source = pl.LazyFrame({"key": [1, 1, 2], "value": [10, 20, 30]})
    query = source.join(source, on="key", suffix="_right")
    engine = GPUEngine(
        executor="streaming",
        executor_options={"join_filter_pushdown": None},
    )
    ir = Translator(query._ldf.visit(), engine).translate_ir()

    assert isinstance(ir, Join)
    assert isinstance(ir.children[0], Cache)
    assert ir.children[0] is ir.children[1]

    optimized = optimize_with_stats(
        ir,
        ConfigOptions.from_polars_engine(engine),
        StatsCollector(),
    )

    assert isinstance(optimized, Join)
    assert optimized.children[0] is optimized.children[1]
    assert not any(isinstance(node, Cache) for node in traversal([optimized]))


@pytest.mark.parametrize(
    "agg",
    [
        pl.col("int").max(),
        # Check LiteralColumn serialization
        pl.Series("value", [[4, 5, 6]], dtype=pl.List(pl.Int32)),
    ],
)
def test_pickle_groupby_args(agg):
    df = pl.LazyFrame(
        {
            "key": [1, 1, 1, 2, 3, 1, 4, 6, 7],
            "int": [1, 2, 3, 4, 5, 6, 7, 8, 9],
            "float": [7.0, 1, 2, 3, 4, 5, 6, 7, 8],
        }
    )
    q = df.group_by(pl.col("key")).agg(agg)
    ir = Translator(q._ldf.visit(), GPUEngine()).translate_ir()
    for node in traversal([ir]):
        pickle.loads(pickle.dumps(node._non_child_args))


def test_pickle_conditional_join_args():
    left = pl.LazyFrame(
        {
            "a": [1, 2, 3, 1, None],
            "b": [1, 2, 3, 4, 5],
            "c": [2, 3, 4, 5, 6],
        }
    )
    right = pl.LazyFrame(
        {
            "a": [1, 4, 3, 7, None, None, 1],
            "c": [2, 3, 4, 5, 6, 7, 8],
            "d": [6, None, 7, 8, -1, 2, 4],
        }
    )
    q = left.join_where(right, pl.col("a") < pl.col("a_right"))
    ir = Translator(q._ldf.visit(), GPUEngine()).translate_ir()
    for node in traversal([ir]):
        pickle.loads(pickle.dumps(node._non_child_args))
