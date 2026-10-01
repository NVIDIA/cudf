# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pickle
from typing import TYPE_CHECKING

import pytest

import polars as pl

from cudf_polars import Translator
from cudf_polars.dsl.ir import DataFrameScan
from cudf_polars.dsl.traversal import traversal
from cudf_polars.engine.options import StreamingOptions
from cudf_polars.streaming.parallel import lower_ir_graph
from cudf_polars.streaming.statistics import collect_statistics
from cudf_polars.testing.asserts import assert_gpu_result_equal
from cudf_polars.utils.config import ConfigOptions
from cudf_polars.utils.versions import POLARS_VERSION_LT_143

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
                reason="https://github.com/NVIDIA/cudf/issues/22376",
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


@pytest.mark.skipif(
    POLARS_VERSION_LT_143,
    reason="Polars < 1.43 does not push len() below the union",
)
def test_len_over_dataframescan_normal_frame_is_evaluated(streaming_engine_factory):
    # Ordinary small frames: neither branch is near the row limit, so this
    # never touches the fast path -- a sanity check of normal evaluation.
    streaming_engine = streaming_engine_factory(
        StreamingOptions(max_rows_per_partition=2, fallback_mode="raise"),
    )
    lf = pl.Series([{}], dtype=pl.Struct({})).new_from_index(0, 3).to_frame().lazy()
    q = pl.concat([lf, lf]).select(pl.len())
    assert_gpu_result_equal(q, engine=streaming_engine)


@pytest.mark.skipif(
    POLARS_VERSION_LT_143,
    reason="Polars < 1.43 does not push len() below the union",
)
@pytest.mark.parametrize("sizes", [(5, 5), (6, 2)], ids=["at-limit", "over-limit"])
def test_len_over_dataframescan_avoids_materializing_oversized_frames(
    streaming_engine_factory, monkeypatch, sizes
):
    # The real row limit is ~2.1 billion (cudf::size_type). Testing at that
    # scale for real is impractically slow even for a branch that DOES fit:
    # confirmed directly -- unlike plain CPU polars, which stays lazy and
    # collects a query like this in under a millisecond regardless of row
    # count, cudf-polars' streaming engine has to actually partition and
    # evaluate a DataFrameScan that falls through the fast path, which for
    # a real ~2.1-billion-row frame ran for 35+ minutes and 85+ GB resident
    # before being killed. So the row limit is patched down to a small
    # value here, and the same ">" comparison in streaming/select.py is
    # exercised with real, fast execution at that smaller scale instead --
    # still fully behavioral (a real collect and value check), not
    # introspecting the lowered IR's node types.
    monkeypatch.setattr("cudf_polars.streaming.select.CUDF_ROW_LIMIT", 5)
    streaming_engine = streaming_engine_factory(
        StreamingOptions(fallback_mode="raise"),
    )
    a, b = sizes
    lf_a = pl.Series([{}], dtype=pl.Struct({})).new_from_index(0, a).to_frame().lazy()
    lf_b = pl.Series([{}], dtype=pl.Struct({})).new_from_index(0, b).to_frame().lazy()
    q = pl.concat([lf_a, lf_b]).select(pl.len())
    assert_gpu_result_equal(q, engine=streaming_engine)


@pytest.mark.skipif(
    POLARS_VERSION_LT_143,
    reason="Polars < 1.43 does not push len() below the union",
)
def test_len_over_dataframescan_oversized_branch_is_never_evaluated(
    spmd_engine_factory, monkeypatch
):
    # The test above confirms the final count is correct, but a broken fast
    # path could in principle still produce the right value through some
    # other route. This confirms the mechanism directly: the oversized
    # branch's DataFrameScan must never actually run, not just that the
    # total happens to come out right. Counting calls to
    # DataFrameScan.do_evaluate -- the one place this engine materializes a
    # scan -- is a behavioural check, not node-type introspection of the
    # lowered IR.
    #
    # dask and ray execute DataFrameScan in separate worker processes, so a
    # monkeypatch here (which only affects this process) can't observe
    # their calls; spmd runs scans in-process via a thread pool, where it
    # can. Hence spmd_engine_factory rather than streaming_engine_factory.
    monkeypatch.setattr("cudf_polars.streaming.select.CUDF_ROW_LIMIT", 5)
    streaming_engine = spmd_engine_factory(StreamingOptions(fallback_mode="raise"))

    def count_do_evaluate_calls(a: int, b: int) -> int:
        calls: list[None] = []
        # Called twice per test; monkeypatch doesn't revert between calls
        # within one test, so the second call's "orig" is really the first
        # call's spy. That's fine -- each spy only ever appends to its own
        # fresh `calls` list before delegating further down the chain, so
        # counts stay correct either way.
        orig = DataFrameScan.do_evaluate.__func__

        def spy(cls, *args, **kwargs):
            calls.append(None)
            return orig(cls, *args, **kwargs)

        monkeypatch.setattr(DataFrameScan, "do_evaluate", classmethod(spy))
        lf_a = (
            pl.Series([{}], dtype=pl.Struct({})).new_from_index(0, a).to_frame().lazy()
        )
        lf_b = (
            pl.Series([{}], dtype=pl.Struct({})).new_from_index(0, b).to_frame().lazy()
        )
        q = pl.concat([lf_a, lf_b]).select(pl.len())
        assert q.collect(engine=streaming_engine).item() == a + b
        return len(calls)

    # Both branches within the (patched) limit: both DataFrameScans
    # evaluate normally, giving a baseline call count for this config.
    both_small = count_do_evaluate_calls(5, 2)
    # One branch exceeds the limit: its DataFrameScan must be skipped, so
    # strictly fewer calls than the baseline -- regardless of exactly how
    # many partitions the one remaining branch is split into.
    one_oversized = count_do_evaluate_calls(6, 2)
    assert one_oversized < both_small
