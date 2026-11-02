# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import random

import pytest

import polars as pl

from cudf_polars.engine.options import StreamingOptions
from cudf_polars.streaming.actor_graph.union import _OrderedUnionPlan
from cudf_polars.testing.asserts import assert_gpu_result_equal


def test_union_shared_fanout_no_deadlock(streaming_engine):
    # union actor can deadlock when input branches share a fanout.
    # See https://github.com/NVIDIA/cudf/issues/21750
    n = 100
    df = pl.LazyFrame({"key": list(range(50)) * (n // 50), "val": list(range(n))})
    gb = df.group_by("key").agg(pl.col("val").sum())
    project = df.select("key", "val")
    q = pl.concat([gb, project])
    assert_gpu_result_equal(q, engine=streaming_engine, check_row_order=False)


def test_sort_slice_over_union_of_duplicated_streams(streaming_engine):
    # Sort+head over a concat of two group-by branches.
    lf1 = (
        pl.LazyFrame({"name": ["alice"], "score": [1.0]})
        .group_by("name")
        .agg(pl.col("score").sum())
    )
    lf2 = (
        pl.LazyFrame({"name": ["bob"], "score": [2.0]})
        .group_by("name")
        .agg(pl.col("score").sum())
    )
    q = pl.concat([lf1, lf2]).sort("score").head(10)
    assert_gpu_result_equal(q, engine=streaming_engine, check_row_order=False)


def _emitted_order(
    plan: _OrderedUnionPlan, actual: tuple[tuple[int, ...], ...]
) -> dict[int, list[tuple[int, int, int]]]:
    """Simulate routing and return each rank's emitted ``(child, rank, index)``."""
    out: dict[int, list[tuple[int, int, int]]] = {
        rank: [] for rank in range(plan.nranks)
    }
    for child, child_counts in enumerate(actual):
        inbox: dict[int, dict[int, list[tuple[int, int, int]]]] = {
            dst: {src: [] for src in range(plan.nranks)} for dst in out
        }
        for src, count in enumerate(child_counts):
            for index in range(count):
                dst = plan.destination(child, src, index)
                assert 0 <= dst < plan.nranks
                inbox[dst][src].append((child, src, index))
        for dst, emitted in out.items():
            for src in range(plan.nranks):
                emitted.extend(inbox[dst][src])
    return out


def test_ordered_union_plan_two_ranks():
    # A has 2 chunks on each rank, B has 1 chunk on each rank.
    plan = _OrderedUnionPlan(nranks=2, counts=((2, 2), (1, 1)))
    out = _emitted_order(plan, plan.counts)
    assert out[0] == [(0, 0, 0), (0, 0, 1), (0, 1, 0)]
    assert out[1] == [(0, 1, 1), (1, 0, 0), (1, 1, 0)]
    assert [plan.local_count(r) for r in range(2)] == [3, 3]


def test_ordered_union_plan_small_second_child():
    # A large first child balanced across ranks and a single-chunk second
    # child on rank 0: that chunk is appended to the last rank.
    nranks, k = 4, 5
    plan = _OrderedUnionPlan(
        nranks=nranks, counts=((k,) * nranks, (1,) + (0,) * (nranks - 1))
    )
    assert plan.destination(1, 0, 0) == nranks - 1
    out = _emitted_order(plan, plan.counts)
    assert out[nranks - 1][-1] == (1, 0, 0)


@pytest.mark.parametrize("seed", range(5))
def test_ordered_union_plan_preserves_order(seed):
    rng = random.Random(seed)
    for _ in range(500):
        nranks = rng.randint(1, 5)
        nchildren = rng.randint(1, 4)
        estimated = tuple(
            tuple(rng.randint(0, 4) for _ in range(nranks)) for _ in range(nchildren)
        )
        # local_count is only an estimate: include surplus and missing chunks.
        actual = tuple(
            tuple(max(0, c + rng.choice((0, 0, 1, 2, -1))) for c in row)
            for row in estimated
        )
        plan = _OrderedUnionPlan(nranks=nranks, counts=estimated)
        out = _emitted_order(plan, actual)
        emitted = [item for rank in range(nranks) for item in out[rank]]
        expected = [
            (child, src, index)
            for child, row in enumerate(actual)
            for src, count in enumerate(row)
            for index in range(count)
        ]
        assert emitted == expected


@pytest.mark.spmd
@pytest.mark.parametrize("max_rows_per_partition", [1, 7, 1_000])
def test_concat_preserves_order_uneven_children(
    streaming_engine_factory, max_rows_per_partition
):
    engine = streaming_engine_factory(
        StreamingOptions(max_rows_per_partition=max_rows_per_partition)
    )
    big = pl.LazyFrame({"a": range(100), "b": [str(i % 7) for i in range(100)]})
    small = pl.LazyFrame({"a": [-1, -2, -3], "b": ["x", "y", "z"]})
    empty = big.filter(pl.col("a") < 0)
    q = pl.concat([big, small, empty, big.select(pl.col("a") * 2, "b")])
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.spmd
def test_concat_preserves_order_with_duplicated_child(streaming_engine_factory):
    engine = streaming_engine_factory(StreamingOptions(max_rows_per_partition=10))
    # A one-row aggregate (duplicated across ranks) between distributed frames.
    total = (
        pl.LazyFrame({"g": [1, 1, 1], "a": [1, 2, 3]})
        .group_by("g")
        .agg(pl.col("a").sum())
        .select("a")
    )
    lf = pl.LazyFrame({"a": range(50)})
    q = pl.concat([lf, total, lf])
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.spmd
def test_unordered_union(streaming_engine_factory):
    engine = streaming_engine_factory(StreamingOptions(max_rows_per_partition=10))
    lf = pl.LazyFrame({"a": range(50)})
    q = pl.union([lf, lf.select(pl.col("a") + 100)])
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


@pytest.mark.spmd
def test_concat_feeding_sampling_over(streaming_engine_factory):
    # The ordered union must not stall its cross-rank exchange when the
    # consumer (here a non-scalar over(...), which samples a few chunks and
    # then waits on an AllGather) stops reading its input for a while.
    engine = streaming_engine_factory(StreamingOptions(max_rows_per_partition=10))
    n = 400
    a = pl.LazyFrame({"g": [i % 5 for i in range(n)], "x": list(range(n))})
    b = pl.LazyFrame({"g": [i % 5 for i in range(n)], "x": list(range(n, 2 * n))})
    q = pl.concat([a, b]).with_columns(pl.col("x").shift(1).over("g").alias("y"))
    assert_gpu_result_equal(q, engine=engine)
