# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pickle
from typing import TYPE_CHECKING

import pytest

import polars as pl

from cudf_polars import Translator
from cudf_polars.dsl.traversal import traversal
from cudf_polars.engine.options import StreamingOptions
from cudf_polars.streaming.actor_graph import core
from cudf_polars.streaming.parallel import lower_ir_graph
from cudf_polars.streaming.statistics import collect_statistics
from cudf_polars.testing.asserts import assert_gpu_result_equal
from cudf_polars.utils.config import ConfigOptions

if TYPE_CHECKING:
    import concurrent.futures


def _assert_stable_ids_match(orig, loaded) -> None:
    for a, b in zip(traversal([orig]), traversal([loaded]), strict=True):
        assert a.get_stable_id() == b.get_stable_id()


@pytest.fixture(scope="module")
def df():
    return pl.LazyFrame(
        {
            "x": range(3_000),
            "y": [1, 2, 3] * 1_000,
            "z": [1.0, 2.0, 3.0, 4.0, 5.0] * 600,
        }
    )


@pytest.mark.parametrize("max_rows_per_partition", [1_000, 1_000_000])
def test_parallel_dataframescan(
    df,
    streaming_engine_factory,
    max_rows_per_partition,
    parquet_stats_executor: concurrent.futures.ThreadPoolExecutor,
):
    streaming_engine = streaming_engine_factory(
        StreamingOptions(max_rows_per_partition=max_rows_per_partition),
    )
    total_row_count = len(df.collect(engine=streaming_engine))
    assert_gpu_result_equal(df, engine=streaming_engine)

    # Check partitioning (throwaway engine — no cluster/runtime needed)
    _engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={"max_rows_per_partition": max_rows_per_partition},
    )
    qir = Translator(df._ldf.visit(), _engine).translate_ir()
    config_options = ConfigOptions.from_polars_engine(_engine)
    lowering = lower_ir_graph(
        qir,
        config_options,
        collect_statistics(
            qir,
            config_options,
            parquet_stats_executor,
        ),
    )
    ir = lowering.lowered
    info = lowering.partition_info
    count = info[ir].count
    if max_rows_per_partition < total_row_count:
        assert count > 1
    else:
        assert count == 1


def test_nullable_array_dataframescan(streaming_engine_factory):
    streaming_engine = streaming_engine_factory(
        StreamingOptions(max_rows_per_partition=2, fallback_mode="raise"),
    )
    q = pl.LazyFrame(
        {
            "embedding": pl.Series(
                # The outer null is in the nonzero-offset second partition.
                [
                    [0.0, 1.0],
                    [2.0, None],
                    None,
                    [3.0, 4.0],
                ],
                dtype=pl.Array(pl.Float32, 2),
            )
        }
    )

    assert_gpu_result_equal(q, engine=streaming_engine)


def test_dataframescan_concat(df, streaming_engine_factory):
    streaming_engine = streaming_engine_factory(
        StreamingOptions(max_rows_per_partition=1_000),
    )
    df2 = pl.concat([df, df])
    assert_gpu_result_equal(df2, engine=streaming_engine)


def test_dataframescan_fanout_three_consumers(request, df, streaming_engine_factory):
    # One scan feeding three consumers: exercises the rapidsmpf fanout actor
    # (data) together with the metadata-forwarding actor, with more than two
    # outputs.
    streaming_engine = streaming_engine_factory(
        StreamingOptions(max_rows_per_partition=1_000),
    )
    if streaming_engine.nranks > 1:
        request.applymarker(
            pytest.mark.xfail(
                reason="https://github.com/NVIDIA/cudf/issues/22376",
                strict=False,
            )
        )
    assert_gpu_result_equal(pl.concat([df, df, df]), engine=streaming_engine)


def test_dataframescan_fanout_self_join(df, streaming_engine_factory):
    # Self-join: one scan feeds both sides of the join.
    streaming_engine = streaming_engine_factory(
        StreamingOptions(max_rows_per_partition=1_000),
    )
    q = df.join(df, on=df.collect_schema().names()[0], how="inner")
    assert_gpu_result_equal(q, engine=streaming_engine, check_row_order=False)


def test_dataframescan_fanout_bounded(df, spmd_engine_factory, monkeypatch):
    # Horizontal concat of two pointwise transforms of one scan. No Join,
    # Union, Over or broadcast consumes the scan, so its fanout is bounded.
    # Uses the in-process SPMD engine so the spy sees the planning call, and a
    # single partition because HConcat falls back for multiple partitions.
    streaming_engine = spmd_engine_factory(
        StreamingOptions(max_rows_per_partition=10_000),
    )
    seen = []
    orig = core.determine_fanout_nodes

    def spy(*args, **kwargs):
        result = orig(*args, **kwargs)
        seen.append(result)
        return result

    monkeypatch.setattr(core, "determine_fanout_nodes", spy)
    q = pl.concat(
        [df.select(a=pl.col("x") + 1), df.select(b=pl.col("y") * 2)],
        how="horizontal",
    )
    assert_gpu_result_equal(q, engine=streaming_engine)
    infos = [info for result in seen for info in result.values()]
    assert any(not info.unbounded and info.num_consumers == 2 for info in infos)


def test_join_in_memory_lazy_stable_id_pickle(
    streaming_engine_factory,
    parquet_stats_executor: concurrent.futures.ThreadPoolExecutor,
):
    engine = streaming_engine_factory(
        StreamingOptions(max_rows_per_partition=1_000, raise_on_fail=True),
    )
    left = (
        pl.LazyFrame({"k": [1, 2, 3], "x": [10, 20, 30]}).collect(engine=engine).lazy()
    )
    right = pl.LazyFrame({"k": [2, 3, 4], "y": [1, 2, 3]}).collect(engine=engine).lazy()
    qir = Translator(left.join(right, on="k")._ldf.visit(), engine).translate_ir()
    config_options = ConfigOptions.from_polars_engine(engine)
    lowering = lower_ir_graph(
        qir,
        config_options,
        collect_statistics(
            qir,
            config_options,
            parquet_stats_executor,
        ),
    )
    ir = lowering.lowered
    _assert_stable_ids_match(ir, pickle.loads(pickle.dumps(ir)))


def test_dataframescan_pickle(
    df, parquet_stats_executor: concurrent.futures.ThreadPoolExecutor
):
    _engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={"max_rows_per_partition": 1_000},
    )
    qir = Translator(df._ldf.visit(), _engine).translate_ir()
    config_options = ConfigOptions.from_polars_engine(_engine)
    lowering = lower_ir_graph(
        qir,
        config_options,
        collect_statistics(
            qir,
            config_options,
            parquet_stats_executor,
        ),
    )
    ir = lowering.lowered

    # Pickle and unpickle the IR (which contains DataFrameScan)
    pickled = pickle.dumps(ir)
    unpickled_ir = pickle.loads(pickled)

    # Verify the unpickled IR is equivalent
    assert type(unpickled_ir) is type(ir)
    assert unpickled_ir.schema == ir.schema
    _assert_stable_ids_match(ir, unpickled_ir)
