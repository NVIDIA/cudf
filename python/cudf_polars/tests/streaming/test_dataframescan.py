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


def test_dataframescan_concat(request, df, streaming_engine_factory):
    streaming_engine = streaming_engine_factory(
        StreamingOptions(max_rows_per_partition=1_000),
    )
    if streaming_engine.nranks > 1:
        # Multi-rank Union interleaves child outputs across ranks: client
        # receives [rank0_A, rank0_B, rank1_A, rank1_B] instead of the
        # polars-CPU [A, B].
        request.applymarker(
            pytest.mark.xfail(
                reason="https://github.com/rapidsai/cudf/issues/22376",
                strict=False,
            )
        )
    df2 = pl.concat([df, df])
    assert_gpu_result_equal(df2, engine=streaming_engine)


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
