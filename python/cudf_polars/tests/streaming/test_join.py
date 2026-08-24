# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for dynamic join path in join_actor (including Right and Full joins)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import polars as pl

from cudf_polars import Translator
from cudf_polars.dsl.ir import Cache, Join
from cudf_polars.dsl.traversal import traversal
from cudf_polars.engine.options import StreamingOptions
from cudf_polars.streaming.actor_graph.join import _use_pwise_join
from cudf_polars.streaming.base import PartitionInfo
from cudf_polars.streaming.parallel import lower_ir_graph
from cudf_polars.streaming.shuffle import Shuffle
from cudf_polars.streaming.statistics import collect_statistics
from cudf_polars.testing.asserts import assert_gpu_result_equal
from cudf_polars.testing.engine_utils import warns_on_spmd
from cudf_polars.utils.config import ConfigOptions, StreamingExecutor

if TYPE_CHECKING:
    import concurrent.futures


@pytest.fixture
def left():
    return pl.LazyFrame(
        {
            "x": range(15),
            "y": [1, 2, 3] * 5,
            "z": [1.0, 2.0, 3.0, 4.0, 5.0] * 3,
        }
    )


@pytest.fixture
def right():
    return pl.LazyFrame(
        {
            "xx": range(9),
            "y": [2, 4, 3] * 3,
            "zz": [1, 2, 3] * 3,
        }
    )


@pytest.mark.parametrize("how", ["inner", "left", "right", "full"])
@pytest.mark.parametrize(
    "options",
    [
        StreamingOptions(max_rows_per_partition=3, broadcast_limit=48),
        StreamingOptions(max_rows_per_partition=5, broadcast_limit=48),
    ],
)
def test_dynamic_join_how(left, right, streaming_engine_factory, options, how):
    """Dynamic join path: all join types including Right and Full."""
    streaming_engine = streaming_engine_factory(options)
    q = left.join(right, on="y", how=how)
    assert_gpu_result_equal(q, engine=streaming_engine, check_row_order=False)


@pytest.mark.parametrize("how", ["right", "full"])
def test_dynamic_join_right_full_reverse(left, right, streaming_engine_factory, how):
    """Dynamic join path: Right/Full with reversed left/right (stress ordering)."""
    streaming_engine = streaming_engine_factory(
        StreamingOptions(max_rows_per_partition=3, broadcast_limit=48),
    )
    # Reverse so "right" frame is larger; exercises right-side preservation
    q = right.join(left, on="y", how=how)
    assert_gpu_result_equal(q, engine=streaming_engine, check_row_order=False)


# ---------------------------------------------------------------------------
# Tests migrated from tests/streaming/test_join.py
# ---------------------------------------------------------------------------


def test_join_then_shuffle(left, right, streaming_engine_factory):
    streaming_engine = streaming_engine_factory(
        StreamingOptions(max_rows_per_partition=2, broadcast_limit=24),
    )
    q = left.join(right, on="y", how="inner").select(
        pl.col("x").sum(),
        pl.col("xx").mean(),
        pl.col("y").n_unique(),
        (pl.col("y") * pl.col("y")).n_unique().alias("y2"),
    )
    assert_gpu_result_equal(q, engine=streaming_engine, check_row_order=False)


@pytest.mark.parametrize("reverse", [True, False])
@pytest.mark.parametrize("max_rows_per_partition", [3, 9])
def test_join_conditional(reverse, max_rows_per_partition, streaming_engine_factory):
    streaming_engine = streaming_engine_factory(
        StreamingOptions(
            max_rows_per_partition=max_rows_per_partition,
            fallback_mode="warn",
            dynamic_planning=None,
        ),
    )
    left = pl.LazyFrame({"x": range(15), "y": [1, 2, 3] * 5})
    right = pl.LazyFrame({"xx": range(9), "yy": [2, 4, 3] * 3})
    if reverse:
        left, right = right, left
    q = left.join_where(right, pl.col("y") < pl.col("yy"))
    with warns_on_spmd(
        streaming_engine,
        UserWarning,
        match="ConditionalJoin not supported for multiple partitions.",
    ):
        assert_gpu_result_equal(q, engine=streaming_engine, check_row_order=False)


# ---------------------------------------------------------------------------
# Tests migrated from tests/streaming/test_join.py
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("how", ["inner", "left", "right", "full", "semi", "anti"])
@pytest.mark.parametrize("reverse", [True, False])
@pytest.mark.parametrize(
    "options",
    [
        StreamingOptions(
            max_rows_per_partition=1, target_partition_size=24, broadcast_limit=24
        ),
        StreamingOptions(
            max_rows_per_partition=1, target_partition_size=24, broadcast_limit=384
        ),
        StreamingOptions(
            max_rows_per_partition=5, target_partition_size=24, broadcast_limit=24
        ),
        StreamingOptions(
            max_rows_per_partition=5, target_partition_size=24, broadcast_limit=384
        ),
        StreamingOptions(
            max_rows_per_partition=10, target_partition_size=24, broadcast_limit=24
        ),
        StreamingOptions(
            max_rows_per_partition=10, target_partition_size=24, broadcast_limit=384
        ),
        StreamingOptions(
            max_rows_per_partition=15, target_partition_size=24, broadcast_limit=24
        ),
        StreamingOptions(
            max_rows_per_partition=15, target_partition_size=24, broadcast_limit=384
        ),
    ],
)
def test_join(left, right, how, reverse, streaming_engine_factory, options):
    streaming_engine = streaming_engine_factory(options)
    if reverse:
        left, right = right, left

    q = left.join(right, on="y", how=how)

    assert_gpu_result_equal(q, engine=streaming_engine, check_row_order=False)

    # Join again on the same key.
    # (covers code path that avoids redundant shuffles)
    if how in ("inner", "left", "right"):
        right2 = pl.LazyFrame(
            {
                "xxx": range(6),
                "yyy": [2, 4, 3] * 2,
                "zzz": [3, 4] * 3,
            }
        )
        q2 = q.join(right2, left_on="y", right_on="yyy", how=how)
        assert_gpu_result_equal(q2, engine=streaming_engine, check_row_order=False)


@pytest.mark.parametrize("zlice", [(0, 2), (2, 2), (-2, None)])
def test_join_and_slice(request, zlice, streaming_engine_factory):
    streaming_engine = streaming_engine_factory(
        StreamingOptions(
            max_rows_per_partition=3,
            target_partition_size=64,
            broadcast_limit=6400,
            fallback_mode="warn",
        ),
    )
    if streaming_engine.nranks > 1:
        # The multi-rank fallback for slice doesn't preserve row order
        # within equal-key groups, so the slice can pick different rows
        # than the CPU baseline.
        request.applymarker(
            pytest.mark.xfail(
                reason="https://github.com/NVIDIA/cudf/issues/22405",
                strict=False,
            )
        )
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
    q = left.join(right, on="a", how="inner").slice(*zlice)
    # Check that we get the correct row count
    # See: https://github.com/NVIDIA/cudf/issues/19153
    with warns_on_spmd(
        streaming_engine,
        UserWarning,
        match="This slice not supported for multiple partitions.",
        when=zlice in {(2, 2), (-2, None)},
    ):
        assert q.collect(engine=streaming_engine).height == q.collect().height

    # Need sort to match order after a join
    q = left.join(right, on="a", how="inner").sort(pl.col("a")).slice(*zlice)
    with warns_on_spmd(
        streaming_engine,
        UserWarning,
        match="This slice not supported for multiple partitions.",
        when=zlice == (2, 2),
    ):
        assert_gpu_result_equal(q, engine=streaming_engine)


@pytest.mark.parametrize(
    "maintain_order", ["left_right", "right_left", "left", "right"]
)
def test_join_maintain_order_fallback_streaming(
    left, right, maintain_order, streaming_engine_factory
):
    streaming_engine = streaming_engine_factory(
        StreamingOptions(
            max_rows_per_partition=3,
            target_partition_size=64,
            broadcast_limit=64,
            fallback_mode="warn",
        ),
    )
    q = left.join(right, on="y", how="inner", maintain_order=maintain_order)

    with warns_on_spmd(
        streaming_engine,
        UserWarning,
        match=r"Join\(maintain_order=.*\) not supported for multiple partitions\.",
    ):
        assert_gpu_result_equal(q, engine=streaming_engine)


# ---------------------------------------------------------------------------
# Tests migrated from tests/streaming/test_join.py (IR inspection)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("broadcast_limit", [1, 48, 128, 1024])
def test_broadcast_limit(
    left,
    right,
    broadcast_limit,
    parquet_stats_executor: concurrent.futures.ThreadPoolExecutor,
):
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "max_rows_per_partition": 3,
            # target_partition_size=1 makes broadcast_limit/1 = broadcast_limit
            # partitions, giving a clean partition-count threshold for this test.
            "target_partition_size": 1,
            "broadcast_limit": broadcast_limit,
            "dynamic_planning": None,  # Requires static planning
        },
    )
    left = pl.LazyFrame(
        {
            "x": range(15),
            "y": [1, 2, 3] * 5,
            "z": [1.0, 2.0, 3.0, 4.0, 5.0] * 3,
        }
    )
    right = pl.LazyFrame(
        {
            "xx": range(9),
            "y": [2, 4, 3] * 3,
            "zz": [1, 2, 3] * 3,
        }
    )

    q = left.join(right, on="y", how="inner")
    ir = Translator(q._ldf.visit(), engine).translate_ir()
    config_options = ConfigOptions.from_polars_engine(engine)
    lowering = lower_ir_graph(
        ir,
        config_options,
        collect_statistics(
            ir,
            config_options,
            parquet_stats_executor,
        ),
    )
    shuffle_nodes = [
        type(node) for node in lowering.partition_info if isinstance(node, Shuffle)
    ]

    # NOTE: Expect small table to have 3 partitions (9 / 3).
    # Therefore, we will get a shuffle-based join if our
    # "broadcast_limit" config is less than 3 (with target_partition_size=1).
    if broadcast_limit < 3:
        # Expect shuffle-based join
        assert len(shuffle_nodes) == 2
    else:
        # Expect broadcast join
        assert len(shuffle_nodes) == 0


def test_shared_join_preserves_partitioning(
    parquet_stats_executor: concurrent.futures.ThreadPoolExecutor,
):
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "max_rows_per_partition": 3,
            "dynamic_planning": None,  # Requires static planning
        },
    )

    left = pl.LazyFrame({"key": list(range(20)) * 5, "val_a": range(100)})
    right = pl.LazyFrame({"key": list(range(20)) * 5, "val_b": range(100)})
    joined = left.join(right, on="key")

    # Use joined result twice to trigger Cache (CSE)
    q = pl.concat(
        [
            joined.group_by("key").agg(pl.col("val_a").sum()),
            joined.group_by("key").agg(pl.col("val_b").sum()),
        ]
    )

    config_options = ConfigOptions.from_polars_engine(engine)
    ir = Translator(q._ldf.visit(), engine).translate_ir()
    lowering = lower_ir_graph(
        ir,
        config_options,
        collect_statistics(ir, config_options, parquet_stats_executor),
    )
    lowered_ir = lowering.lowered
    partition_info = lowering.partition_info

    assert not any(isinstance(node, Cache) for node in traversal([lowered_ir]))

    # Removing Cache should preserve the shared join's partitioning on 'key'.
    join_partitioning = [
        [ne.name for ne in partition_info[node].partitioned_on]
        for node in traversal([lowered_ir])
        if isinstance(node, Join)
    ]
    assert join_partitioning == [["key"]], (
        f"Shared join should be partitioned on 'key', got {join_partitioning}"
    )

    # Only 2 shuffles needed (for join sides, not for groupby)
    num_shuffles = sum(
        1 for node in traversal([lowered_ir]) if isinstance(node, Shuffle)
    )
    assert num_shuffles == 2, f"Expected 2 shuffles, got {num_shuffles}"


def test_dynamic_planning_skips_compile_time_partition_wise_join():
    lf = pl.LazyFrame({"y": [1, 2, 3]})
    q = lf.join(lf, on="y", how="inner")
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={"dynamic_planning": {}},
    )
    ir = Translator(q._ldf.visit(), engine).translate_ir()
    join_ir = next(n for n in traversal([ir]) if isinstance(n, Join))
    left_ir, right_ir = join_ir.children
    executor = StreamingExecutor(dynamic_planning={})
    partition_info = {
        join_ir: PartitionInfo(1, partitioned_on=()),
        left_ir: PartitionInfo(1, partitioned_on=()),
        right_ir: PartitionInfo(1, partitioned_on=()),
    }
    assert not _use_pwise_join(executor, partition_info, join_ir)


def test_join_computed_expr_right_key(streaming_engine_factory) -> None:
    """Join on a computed key expression."""
    engine = streaming_engine_factory(
        StreamingOptions(
            target_partition_size=1,
            max_rows_per_partition=4,
            broadcast_limit=1,  # Disable broadcast joins
            raise_on_fail=True,
        ),
    )
    if engine.nranks < 2:
        pytest.skip("bug only manifests on 2+ ranks")

    zip_prefixes = ["10", "20", "30", "40"]
    full_zips = ["10001", "20001", "30001", "40001"]
    reps = 4

    # Start with joins on concrete column references
    # to establish left and right partitioning metadata.
    left_a = pl.LazyFrame(
        {
            "zip_prefix": zip_prefixes * reps,
            "val_a": list(range(len(zip_prefixes) * reps)),
        }
    )
    left_b = pl.LazyFrame(
        {
            "zip_prefix": zip_prefixes * reps,
            "val_b": list(range(100, 100 + len(zip_prefixes) * reps)),
        }
    )
    left = left_a.join(left_b, on="zip_prefix", how="inner")

    right_a = pl.LazyFrame(
        {
            "full_zip": full_zips * reps,
            "val_c": list(range(200, 200 + len(full_zips) * reps)),
        }
    )
    right_b = pl.LazyFrame(
        {
            "full_zip": full_zips * reps,
            "val_d": list(range(300, 300 + len(full_zips) * reps)),
        }
    )
    right = right_a.join(right_b, on="full_zip", how="inner")

    # Now join on a computed key expression.
    # This should not silently drop rows across ranks
    q = left.join(
        right,
        left_on="zip_prefix",
        right_on=pl.col("full_zip").str.slice(0, 2),
    )
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)
