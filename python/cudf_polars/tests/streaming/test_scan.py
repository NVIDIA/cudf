# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import math
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, cast

import pytest

import polars as pl

import pylibcudf as plc

from cudf_polars import Translator
from cudf_polars.containers import DataFrame, DataType
from cudf_polars.dsl.ir import (
    Empty,
    IRExecutionContext,
    Scan,
)
from cudf_polars.dsl.utils.io import (
    CachedParquetInfo,
    _prefetch_parquet_footers_for_paths,
    attach_cached_parquet_metadata,
    prefetch_parquet_file_metadata_for_ir,
)
from cudf_polars.engine.options import StreamingOptions
from cudf_polars.streaming.actor_graph.io import resolve_max_concurrent_io_tasks
from cudf_polars.streaming.actor_graph.scan_ordering import (
    parquet_metadata_ordering,
)
from cudf_polars.streaming.base import (
    DataSourceInfo,
    IOPartitionFlavor,
    IOPartitionPlan,
    StatsCollector,
)
from cudf_polars.streaming.io import (
    ParquetScanTask,
    ScanTask,
    StreamingScan,
    expand_scan_for_rank,
    scan_partition_plan,
)
from cudf_polars.streaming.parallel import lower_ir_graph
from cudf_polars.streaming.partitioning_requests import (
    NamedOrderKey,
    OrderPartitioningRequest,
)
from cudf_polars.streaming.statistics import collect_statistics
from cudf_polars.testing.asserts import assert_gpu_result_equal
from cudf_polars.testing.engine_utils import SMALL_MAX_ROWS_PER_PARTITION
from cudf_polars.testing.io import make_partitioned_source
from cudf_polars.utils.config import (
    ConfigOptions,
    MaxConcurrentIOTasks,
    ParquetOptions,
)

if TYPE_CHECKING:
    import concurrent.futures
    from collections.abc import Callable
    from pathlib import Path
    from typing import Any, Literal

    import cudf_polars.engine.core
    from cudf_polars.engine.core import StreamingEngine


@pytest.fixture(scope="module")
def df():
    return pl.DataFrame(
        {
            "x": range(3_000),
            "y": ["cat", "dog", "fish"] * 1_000,
            "z": [1.0, 2.0, 3.0, 4.0, 5.0] * 600,
        }
    )


@pytest.mark.parametrize(
    "fmt, scan_fn",
    [
        ("csv", pl.scan_csv),
        ("ndjson", pl.scan_ndjson),
        ("parquet", pl.scan_parquet),
    ],
)
def test_parallel_scan(
    tmp_path: Path,
    df: pl.DataFrame,
    fmt: Literal["csv", "ndjson", "parquet", "chunked_parquet"],
    scan_fn: Any,
    streaming_engine: cudf_polars.engine.core.StreamingEngine,
) -> None:
    # The spmd-small case creates *many* partitions with the length-3000 df.
    # A smaller dataframe gives us sufficient test coverage, and runs much faster.
    if (
        streaming_engine.config["executor_options"]["max_rows_per_partition"]
        == SMALL_MAX_ROWS_PER_PARTITION
    ):
        df = df.head(40)
    make_partitioned_source(df, tmp_path, fmt, n_files=3)
    q = scan_fn(tmp_path)
    assert_gpu_result_equal(q, engine=streaming_engine)


@pytest.mark.parametrize(
    "target_partition_size_and_n_files", [(1_000, 1), (1_000, 2), (1_000_000, 5)]
)
def test_scan_parquet_prefetch_file_metadata(
    tmp_path: Path,
    target_partition_size_and_n_files: tuple[int, int],
    df: pl.DataFrame,
    streaming_engine_factory: Callable[..., StreamingEngine],
):
    target_partition_size, n_files = target_partition_size_and_n_files
    streaming_engine = streaming_engine_factory(
        StreamingOptions(
            target_partition_size=target_partition_size,
            parquet_options={"prefetch_file_metadata": True},
        ),
    )
    make_partitioned_source(df, tmp_path, "parquet", n_files=n_files)
    assert_gpu_result_equal(pl.scan_parquet(tmp_path), engine=streaming_engine)


def test_prefetch_file_metadata_non_parquet_scan(df, streaming_engine_factory) -> None:
    streaming_engine = streaming_engine_factory(
        StreamingOptions(parquet_options={"prefetch_file_metadata": True}),
    )
    assert_gpu_result_equal(df.lazy().select("x"), engine=streaming_engine)


def test_prefetch_parquet_file_metadata_no_parquet_scans() -> None:
    result = prefetch_parquet_file_metadata_for_ir(
        Empty({}), py_executor=None, stats=None
    )
    assert result == {}


def test_attach_cached_parquet_metadata_skips_incomplete_scan() -> None:
    scan = _make_parquet_scan(["first.parquet", "second.parquet"])
    task = ParquetScanTask(scan, scan.paths, 0, 1, scan.parquet_options)
    streaming_scan = StreamingScan([task], scan)

    attach_cached_parquet_metadata(streaming_scan, {})

    assert scan.cached_parquet_info is None


def test_prefetch_skips_paths_cached_by_stats_collection(
    tmp_path,
    df: pl.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    parquet_stats_executor,
) -> None:
    import cudf_polars.dsl.utils.io as io_module
    from cudf_polars.streaming.io import _clear_source_info_cache

    _clear_source_info_cache()
    n_files = 5
    max_footer_samples = 2
    make_partitioned_source(df, tmp_path, "parquet", n_files=n_files)
    paths = sorted(str(p) for p in tmp_path.glob("*.parquet"))

    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        parquet_options={"max_footer_samples": max_footer_samples},
    )
    q = pl.scan_parquet(tmp_path)
    from cudf_polars import Translator

    ir = Translator(q._ldf.visit(), engine).translate_ir()
    config = ConfigOptions.from_polars_engine(engine)
    stats = collect_statistics(ir, config, parquet_stats_executor)

    source = stats.scan_stats[ir]
    assert source.cached_parquet_info is not None
    sampled_paths = {info.path for info in source.cached_parquet_info}
    assert len(sampled_paths) == max_footer_samples

    fetched_paths: list[str] = []
    real_prefetch = io_module._prefetch_parquet_footers_for_paths

    def recording_prefetch(
        paths_arg: list[str], *, parse_hybrid_metadata: bool = False
    ) -> list:
        fetched_paths.extend(paths_arg)
        return real_prefetch(paths_arg, parse_hybrid_metadata=parse_hybrid_metadata)

    monkeypatch.setattr(
        io_module, "_prefetch_parquet_footers_for_paths", recording_prefetch
    )

    scan = _make_parquet_scan(paths)
    task = ParquetScanTask(scan, paths, 0, 1, scan.parquet_options)
    streaming_scan = StreamingScan([task], scan)

    result = prefetch_parquet_file_metadata_for_ir(
        streaming_scan, py_executor=None, stats=stats
    )

    assert set(result) == set(paths)
    assert set(fetched_paths) == set(paths) - sampled_paths


def test_prefetch_parquet_file_metadata_remote_only(tmp_path, df) -> None:
    make_partitioned_source(df, tmp_path, "parquet", n_files=1)
    local_path = str(next(tmp_path.glob("*.parquet")))

    scan = _make_parquet_scan([local_path])
    task = ParquetScanTask(scan, scan.paths, 0, 1, scan.parquet_options)
    streaming_scan = StreamingScan([task], scan)

    # Local paths are skipped entirely when remote_only=True.
    result = prefetch_parquet_file_metadata_for_ir(
        streaming_scan, py_executor=None, stats=None, remote_only=True
    )
    assert result == {}
    attach_cached_parquet_metadata(streaming_scan, result)
    assert scan.cached_parquet_info is None

    # The same local path is prefetched when remote_only=False (the default).
    result = prefetch_parquet_file_metadata_for_ir(
        streaming_scan, py_executor=None, stats=None
    )
    assert set(result) == {local_path}


def test_cached_parquet_info_hybrid_scan_reader_lazy(tmp_path, df) -> None:
    make_partitioned_source(df, tmp_path, "parquet", n_files=1)
    local_path = str(next(tmp_path.glob("*.parquet")))

    [info] = _prefetch_parquet_footers_for_paths([local_path])
    assert info._hybrid_scan_metadata is None

    info.hybrid_scan_reader(info.default_reader_options())
    assert info._hybrid_scan_metadata is not None


@pytest.mark.parametrize(
    "paths,expected",
    [
        ([], 2),
        (["file.parquet"], 2),
        (["file.parquet", "s3://bucket/file.parquet"], 8),
        (["s3://bucket/file.parquet"], 8),
    ],
)
def test_resolve_max_concurrent_io_tasks_default(
    paths: list[str], expected: int
) -> None:
    assert resolve_max_concurrent_io_tasks(MaxConcurrentIOTasks(), paths) == expected


def test_resolve_max_concurrent_io_tasks_explicit() -> None:
    assert (
        resolve_max_concurrent_io_tasks(
            MaxConcurrentIOTasks(local=6, remote=6), ["s3://bucket/file.parquet"]
        )
        == 6
    )


@pytest.mark.parametrize(
    "paths,expected",
    [
        (["file.parquet"], 3),
        (["s3://bucket/file.parquet"], 7),
    ],
)
def test_resolve_max_concurrent_io_tasks_local_remote_policy(
    paths: list[str], expected: int
) -> None:
    assert (
        resolve_max_concurrent_io_tasks(MaxConcurrentIOTasks(local=3, remote=7), paths)
        == expected
    )


def test_resolve_max_concurrent_io_tasks_partial_override() -> None:
    max_concurrent_io_tasks = MaxConcurrentIOTasks(remote=7)
    assert (
        resolve_max_concurrent_io_tasks(max_concurrent_io_tasks, ["file.parquet"]) == 2
    )
    assert (
        resolve_max_concurrent_io_tasks(
            max_concurrent_io_tasks, ["s3://bucket/file.parquet"]
        )
        == 7
    )


@pytest.mark.parametrize("use_hybrid_scan", [True, False])
def test_prefetch_file_metadata_select_fast_count(
    df: pl.DataFrame,
    streaming_engine_factory: Callable[..., StreamingEngine],
    tmp_path: Path,
    *,
    use_hybrid_scan: bool,
) -> None:
    streaming_engine = streaming_engine_factory(
        StreamingOptions(
            parquet_options={
                "prefetch_file_metadata": True,
                "use_hybrid_scan": use_hybrid_scan,
            }
        ),
    )
    source = tmp_path / "data.parquet"
    df.write_parquet(source)
    q = pl.scan_parquet(source).select(pl.len())
    assert_gpu_result_equal(q, engine=streaming_engine)


# ---------------------------------------------------------------------------
# Tests migrated from tests/streaming/test_scan.py
# ---------------------------------------------------------------------------


def test_split_scan_aligns_to_row_group_boundaries(
    tmp_path, df, streaming_engine_factory
):
    streaming_engine = streaming_engine_factory(
        StreamingOptions(target_partition_size=1_000),
    )
    make_partitioned_source(df, tmp_path, "parquet", n_files=1, row_group_size=10)
    q = pl.scan_parquet(tmp_path)
    assert_gpu_result_equal(q, engine=streaming_engine)


@pytest.mark.parametrize("mask", [None, pl.col("x") < 1_000])
def test_split_scan_predicate(tmp_path, df, mask, streaming_engine_factory):
    streaming_engine = streaming_engine_factory(
        StreamingOptions(target_partition_size=1_000),
    )
    make_partitioned_source(df, tmp_path, "parquet", n_files=1)
    q = pl.scan_parquet(tmp_path)
    if mask is not None:
        q = q.filter(mask)
    assert_gpu_result_equal(q, engine=streaming_engine)


@pytest.mark.parametrize("n_files", [2, 3])
@pytest.mark.parametrize("blocksize", [1_000, 10_000, 1_000_000])
def test_target_partition_size(
    tmp_path,
    df,
    blocksize,
    n_files,
    streaming_engine_factory,
    parquet_stats_executor: concurrent.futures.ThreadPoolExecutor,
):
    streaming_engine = streaming_engine_factory(
        StreamingOptions(target_partition_size=blocksize),
    )
    make_partitioned_source(df, tmp_path, "parquet", n_files=n_files)
    q = pl.scan_parquet(tmp_path)
    assert_gpu_result_equal(q, engine=streaming_engine)

    # Check partitioning (throwaway engine — no cluster/runtime needed)
    _engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={"target_partition_size": blocksize},
    )
    qir = Translator(q._ldf.visit(), _engine).translate_ir()
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
    if blocksize <= 12_000:
        assert count > n_files
    else:
        assert count < n_files


def test_scan_join(engine: pl.GPUEngine, tmp_path: Path) -> None:
    # This test exercises some logic on nodes with multiple children (join)
    # where one or more of the children are Scan nodes.
    left = pl.DataFrame({"a": ["a", "b", "c", "d"], "b": [1, 2, 3, 4]})
    right = pl.DataFrame({"a": ["a", "b", "c", "d"], "c": [10, 20, 30, 40]})

    left.write_parquet(tmp_path / "left.parquet")
    right.write_parquet(tmp_path / "right.parquet")

    left_q = pl.scan_parquet(tmp_path / "left.parquet")
    right_q = pl.scan_parquet(tmp_path / "right.parquet")
    q = left_q.join(right_q, on="a", how="inner")
    assert_gpu_result_equal(q, engine=engine)


def test_scan_union(engine: pl.GPUEngine, tmp_path: Path) -> None:
    # This test exercises some logic on nodes with a Union[Scan, ...]
    df = pl.DataFrame({"a": ["a", "b", "c", "d"], "b": [1, 2, 3, 4]})
    df.write_parquet(tmp_path / "data.parquet")

    df_q = pl.scan_parquet(tmp_path / "data.parquet")

    q = pl.concat([df_q, df_q])
    assert_gpu_result_equal(q, engine=engine)


def _make_parquet_scan(
    paths: list[str],
    parquet_options: ParquetOptions | None = None,
    *,
    schema: dict[str, DataType] | None = None,
    skip_rows: int = 0,
    n_rows: int = -1,
    row_index: tuple[str, int] | None = None,
) -> Scan:
    parquet_options = parquet_options or ParquetOptions()
    return Scan(
        schema or {"x": DataType(pl.Int64())},
        "parquet",
        {},
        None,
        paths,
        None,
        skip_rows,
        n_rows,
        row_index,
        None,
        None,
        parquet_options,
        None,
    )


def _make_csv_scan(paths: list[str]) -> Scan:
    return Scan(
        {"x": DataType(pl.Int64())},
        "csv",
        {
            "has_header": True,
            "schema": None,
            "skip_rows": 0,
            "skip_rows_after_header": 0,
            "parse_options": {
                "separator": ord(","),
                "quote_char": ord('"'),
                "eol_char": ord("\n"),
                "null_values": None,
                "comment_prefix": None,
                "decimal_comma": False,
            },
        },
        None,
        paths,
        None,
        0,
        -1,
        None,
        None,
        None,
        ParquetOptions(),
        None,
    )


@pytest.mark.parametrize(
    "plan,paths,rank,nranks,expected_path_groups",
    [
        (
            IOPartitionPlan(2, IOPartitionFlavor.FUSED_FILES),
            [f"f{i}" for i in range(6)],
            0,
            1,
            [["f0", "f1"], ["f2", "f3"], ["f4", "f5"]],
        ),
        (
            IOPartitionPlan(2, IOPartitionFlavor.FUSED_FILES),
            [f"f{i}" for i in range(6)],
            0,
            2,
            [["f0", "f1"], ["f2", "f3"]],
        ),
        (
            IOPartitionPlan(2, IOPartitionFlavor.FUSED_FILES),
            [f"f{i}" for i in range(6)],
            1,
            2,
            [["f4", "f5"]],
        ),
        (IOPartitionPlan(3, IOPartitionFlavor.SINGLE_READ), ["a", "b", "c"], 1, 2, []),
    ],
)
def test_expand_scan_for_rank_fused_and_single_read(
    plan: IOPartitionPlan,
    paths: list[str],
    rank: int,
    nranks: int,
    expected_path_groups: list[list[str]],
) -> None:
    partition_count = math.ceil(len(paths) / plan.factor)
    streaming_scan = expand_scan_for_rank(
        _make_parquet_scan(paths),
        plan,
        partition_count,
        rank=rank,
        nranks=nranks,
        parquet_options=ParquetOptions(),
    )
    for scan, expected_paths in zip(
        streaming_scan.tasks, expected_path_groups, strict=True
    ):
        assert isinstance(scan, ParquetScanTask)
        assert scan.split_index == 0
        assert scan.total_splits == 1
        assert scan.paths == expected_paths


@pytest.mark.parametrize(
    "rank,expected_splits",
    [
        (0, [(0, 4), (1, 4)]),
        (1, [(2, 4), (3, 4)]),
    ],
)
def test_expand_scan_for_rank_split_files(
    rank: int,
    expected_splits: list[tuple[int, int]],
) -> None:
    plan = IOPartitionPlan(4, IOPartitionFlavor.SPLIT_FILES)
    paths = ["file.parquet"]
    partition_count = plan.factor * len(paths)
    streaming_scan = expand_scan_for_rank(
        _make_parquet_scan(paths),
        plan,
        partition_count,
        rank=rank,
        nranks=2,
        parquet_options=ParquetOptions(),
    )
    assert len(streaming_scan.tasks) == len(expected_splits)
    for scan, (split_index, total_splits) in zip(
        streaming_scan.tasks, expected_splits, strict=True
    ):
        assert isinstance(scan, ParquetScanTask)
        assert scan.split_index == split_index
        assert scan.total_splits == total_splits
        assert scan.paths == ["file.parquet"]


def test_attach_cached_parquet_metadata_resolves_row_groups(
    tmp_path: Path,
) -> None:
    source = tmp_path / "data.parquet"
    pl.DataFrame({"x": range(4)}).write_parquet(source, row_group_size=2)

    base = _make_parquet_scan([str(source)])
    streaming_scan = expand_scan_for_rank(
        base,
        IOPartitionPlan(2, IOPartitionFlavor.SPLIT_FILES),
        partition_count=2,
        rank=0,
        nranks=1,
        parquet_options=base.parquet_options,
    )

    cached = prefetch_parquet_file_metadata_for_ir(streaming_scan, None)
    attach_cached_parquet_metadata(streaming_scan, cached)

    row_groups = []
    for scan in streaming_scan.tasks:
        assert isinstance(scan, ParquetScanTask)
        bounds = scan.get_task_bounds()
        row_groups.append(bounds.row_groups)
    assert row_groups == [[[0]], [[1]]]


def test_attach_cached_parquet_metadata_uses_rank_local_tasks(
    tmp_path: Path,
) -> None:
    paths = [str(tmp_path / f"part-{i}.parquet") for i in range(4)]
    for path in paths:
        pl.DataFrame({"x": range(4)}).write_parquet(path, row_group_size=2)

    base = _make_parquet_scan(paths)
    streaming_scan = expand_scan_for_rank(
        base,
        IOPartitionPlan(1, IOPartitionFlavor.SINGLE_FILE),
        partition_count=4,
        rank=0,
        nranks=2,
        parquet_options=base.parquet_options,
    )

    cached = prefetch_parquet_file_metadata_for_ir(streaming_scan, None)
    attach_cached_parquet_metadata(streaming_scan, cached)

    assert base.cached_parquet_info is not None
    assert [info.path for info in base.cached_parquet_info] == paths[:2]
    for scan in streaming_scan.tasks:
        assert isinstance(scan, ParquetScanTask)
        bounds = scan.get_task_bounds()
        assert bounds.row_groups == [[0, 1]]


def test_attach_cached_parquet_metadata_leaves_sub_row_group_split_unaligned(
    tmp_path: Path,
) -> None:
    source = tmp_path / "data.parquet"
    pl.DataFrame({"x": range(4)}).write_parquet(source, row_group_size=2)

    base = _make_parquet_scan([str(source)])
    streaming_scan = expand_scan_for_rank(
        base,
        IOPartitionPlan(4, IOPartitionFlavor.SPLIT_FILES),
        partition_count=4,
        rank=0,
        nranks=1,
        parquet_options=base.parquet_options,
    )

    cached = prefetch_parquet_file_metadata_for_ir(streaming_scan, None)
    attach_cached_parquet_metadata(streaming_scan, cached)

    for scan in streaming_scan.tasks:
        assert isinstance(scan, ParquetScanTask)
        assert scan.total_splits > 1
        bounds = scan.get_task_bounds()
        assert bounds.row_groups is None


@pytest.mark.parametrize(
    "skip_rows,n_rows,row_index",
    [(1, -1, None), (0, 2, None), (0, -1, ("index", 0))],
)
def test_attach_cached_parquet_metadata_leaves_sliced_single_read_unaligned(
    tmp_path: Path,
    skip_rows: int,
    n_rows: int,
    row_index: tuple[str, int] | None,
) -> None:
    source = tmp_path / "data.parquet"
    pl.DataFrame({"x": range(4)}).write_parquet(source, row_group_size=2)

    base = _make_parquet_scan(
        [str(source)], skip_rows=skip_rows, n_rows=n_rows, row_index=row_index
    )
    streaming_scan = expand_scan_for_rank(
        base,
        IOPartitionPlan(1, IOPartitionFlavor.SINGLE_READ),
        partition_count=1,
        rank=0,
        nranks=1,
        parquet_options=base.parquet_options,
    )

    cached = prefetch_parquet_file_metadata_for_ir(streaming_scan, None)
    attach_cached_parquet_metadata(streaming_scan, cached)

    for scan in streaming_scan.tasks:
        assert isinstance(scan, ParquetScanTask)
        assert scan.total_splits == 1
        bounds = scan.get_task_bounds()
        assert bounds.row_groups is None
        assert bounds.skip_rows == skip_rows
        assert bounds.n_rows == n_rows


def test_streaming_scan_raises() -> None:
    # This isn't reachable by normal cudf-polars usage.
    scan = _make_csv_scan(["file.csv"])
    task = ScanTask(scan, scan.paths, 0, 1)
    ctx = IRExecutionContext()
    with pytest.raises(NotImplementedError, match=r"StreamingScan.do_evaluate"):
        StreamingScan.do_evaluate([task], scan, context=ctx)


@pytest.mark.parametrize(
    "predicate,use_columns",
    [
        # uses hybrid scan reader
        (pl.col("x") < 1_000, None),
        (pl.col("x") < 1_000, ["x", "z"]),
        (pl.col("x") < 1_000, ["z"]),
        (pl.col("x") < 1_000, ["x"]),
        # falls back to default parquet reader
        (pl.col("y").str.contains("cat"), None),
        (None, None),
    ],
)
def test_split_scan_hybrid(
    tmp_path: Path,
    df: pl.DataFrame,
    predicate: pl.Expr | None,
    use_columns: list[str] | None,
    streaming_engine_factory: Callable[..., StreamingEngine],
) -> None:
    streaming_engine = streaming_engine_factory(
        StreamingOptions(
            target_partition_size=1_000,
            parquet_options={
                "use_hybrid_scan": True,
                "prefetch_file_metadata": True,
            },
        ),
    )
    make_partitioned_source(df, tmp_path, "parquet", n_files=1, row_group_size=100)
    q = pl.scan_parquet(tmp_path)
    if predicate is not None:
        q = q.filter(predicate)
    if use_columns is not None:
        q = q.select(use_columns)
    assert_gpu_result_equal(q, engine=streaming_engine)


def test_scan_path_mismatch_raises() -> None:
    # This isn't reachable by polars' public API, so we test it directly.
    scan = _make_parquet_scan(
        ["file.parquet"], parquet_options=ParquetOptions(prefetch_file_metadata=True)
    )
    ctx = IRExecutionContext()

    with pytest.raises(
        AssertionError,
        match=r"Paths do not match cached parquet info",
    ):
        Scan.do_evaluate(
            scan.schema,
            scan.typ,
            scan.reader_options,
            scan.paths,
            scan.with_columns,
            scan.skip_rows,
            scan.n_rows,
            scan.row_index,
            scan.include_file_paths,
            scan.predicate,
            scan.parquet_options,
            [],
            context=ctx,
        )


def test_parquet_split_task_fetches_missing_metadata(tmp_path: Path) -> None:
    source = tmp_path / "data.parquet"
    pl.DataFrame({"x": range(4)}).write_parquet(source, row_group_size=2)

    scan = _make_parquet_scan(
        [str(source)], parquet_options=ParquetOptions(prefetch_file_metadata=True)
    )

    result = ParquetScanTask.do_evaluate(
        scan,
        scan.paths,
        0,
        2,
        scan.parquet_options,
        context=IRExecutionContext(),
    )

    assert result.to_polars().to_dict(as_series=False) == {"x": [0, 1]}


def test_prefetch_file_metadata_join(
    tmp_path: Path, streaming_engine_factory: Callable[..., StreamingEngine]
) -> None:
    p1 = tmp_path / "f1.parquet"
    p2 = tmp_path / "f2.parquet"
    pl.DataFrame({"k": [1, 2, 3], "a": [4, 5, 6]}).write_parquet(p1)
    pl.DataFrame({"k": [1, 2, 3], "b": [7, 8, 9]}).write_parquet(p2)

    engine = streaming_engine_factory(
        StreamingOptions(parquet_options={"prefetch_file_metadata": True}),
    )

    q = pl.scan_parquet(p1).join(pl.scan_parquet(p2), on="k")
    q.collect(engine=engine)


def _make_cached_parquet_info(
    paths: list[str], size: int = 10
) -> list[CachedParquetInfo]:
    return [
        # `file_metadata` is not used by identity/hash tests.
        # It only needs to be a stable value for equality checks.
        CachedParquetInfo(
            path=path,
            size=size,
            file_metadata=cast("plc.io.parquet_metadata.FileMetaData", path),
        )
        for path in paths
    ]


def test_prefetch_file_metadata_with_cached_scan_parent_nodes(
    tmp_path: Path, streaming_engine_factory: Callable[..., StreamingEngine]
) -> None:
    # Regression test for replace not replacing StreamingScan nodes with their prefetched variants.
    source = tmp_path / "data.parquet"
    pl.DataFrame(
        {
            "k": [1, 1, 2, 2, 3, 3],
            "v": [10, 11, 20, 21, 30, 31],
        }
    ).write_parquet(source)

    engine = streaming_engine_factory(
        StreamingOptions(parquet_options={"prefetch_file_metadata": True}),
    )

    cached_scan = pl.scan_parquet(source).cache()
    left = cached_scan.group_by("k").agg(pl.col("v").sum().alias("sum_v"))
    right = cached_scan.group_by("k").agg(pl.len().alias("n"))
    q = left.join(right, on="k").sort("k")

    assert_gpu_result_equal(q, engine=engine)


def test_scan_task_identity_equality() -> None:
    base = _make_csv_scan(["a.csv", "b.csv"])
    paths = ["a.csv"]

    a = ScanTask(base, paths, 0, 1)
    b = ScanTask(base, paths, 0, 1)
    c = ScanTask(base, ["b.csv"], 0, 1)

    assert a == b
    assert hash(a) == hash(b)
    assert a != c


def test_scan_task_validates_split_bounds() -> None:
    base = _make_csv_scan(["a.csv"])

    with pytest.raises(ValueError, match=r"Expected at least one split"):
        ScanTask(base, ["a.csv"], 0, 0)

    with pytest.raises(ValueError, match=r"Expected split_index in"):
        ScanTask(base, ["a.csv"], 1, 1)


def test_parquet_scan_task_validates_inputs() -> None:
    csv_scan = _make_csv_scan(["a.csv"])
    with pytest.raises(ValueError, match=r"Expected a parquet scan"):
        ParquetScanTask(csv_scan, csv_scan.paths, 0, 1, csv_scan.parquet_options)

    parquet_scan = _make_parquet_scan(["a.parquet", "b.parquet"])
    with pytest.raises(ValueError, match=r"Expected a single path for a split task"):
        ParquetScanTask(
            parquet_scan,
            parquet_scan.paths,
            0,
            2,
            parquet_scan.parquet_options,
        )


def test_parquet_split_task_identity_equality() -> None:
    base = _make_parquet_scan(["a.parquet"])

    a = ParquetScanTask(base, base.paths, 0, 4, base.parquet_options)
    b = ParquetScanTask(base, base.paths, 0, 4, base.parquet_options)
    c = ParquetScanTask(base, base.paths, 1, 4, base.parquet_options)

    assert a == b
    assert hash(a) == hash(b)
    assert a != c


def test_streaming_scan_identity_equality() -> None:
    base = _make_parquet_scan(["a.parquet"])
    split = ParquetScanTask(
        base,
        base.paths,
        0,
        2,
        base.parquet_options,
    )
    split_same = ParquetScanTask(
        base,
        base.paths,
        0,
        2,
        base.parquet_options,
    )
    split_diff = ParquetScanTask(
        base,
        base.paths,
        1,
        2,
        base.parquet_options,
    )

    a = StreamingScan([split], base)
    b = StreamingScan([split_same], base)
    c = StreamingScan([split_diff], base)

    assert a == b
    assert hash(a) == hash(b)
    assert a != c


def test_cached_parquet_info_excluded_from_identity() -> None:
    base = _make_parquet_scan(["a.parquet"])
    info = _make_cached_parquet_info(base.paths)

    scan_without = _make_parquet_scan(base.paths)
    scan_with = Scan(
        base.schema,
        "parquet",
        {},
        None,
        base.paths,
        None,
        0,
        -1,
        None,
        None,
        None,
        base.parquet_options,
        info,
    )
    assert scan_without == scan_with
    assert hash(scan_without) == hash(scan_with)


def _run_parquet_metadata_ordering(
    streaming_scan: StreamingScan,
    requests: tuple[OrderPartitioningRequest, ...],
    spmd_engine,
    global_chunk_count: int,
):
    async def _run():
        with ThreadPoolExecutor(max_workers=1) as executor:
            ir_context = IRExecutionContext(
                executor,
                get_cuda_stream=spmd_engine.context.br().stream_pool.get_stream,
            )
            return await parquet_metadata_ordering(
                spmd_engine.context,
                spmd_engine.comm,
                streaming_scan,
                global_chunk_count,
                requests,
                ir_context,
                collective_id=0,
            )

    return asyncio.run(_run())


def _order_request(*names: str) -> OrderPartitioningRequest:
    return OrderPartitioningRequest(
        tuple(
            NamedOrderKey(
                name,
                plc.types.Order.ASCENDING,
                plc.types.NullOrder.AFTER,
            )
            for name in names
        )
    )


def _ordering_boundary_values(
    ordering,
    spmd_engine,
    *,
    name: str,
    dtype: DataType,
) -> list[object]:
    boundaries = ordering.get_boundaries(spmd_engine.context.br())
    return (
        DataFrame.from_table(
            boundaries.table_view(),
            [name],
            [dtype],
            stream=boundaries.stream,
        )
        .to_polars()[name]
        .to_list()
    )


def test_parquet_scan_ordering_partitioning_from_footer_metadata(
    tmp_path: Path, spmd_engine
) -> None:
    from cudf_streaming.channel_metadata import OrderScheme

    paths = []
    for i in range(2):
        path = tmp_path / f"part-{i}.parquet"
        pl.DataFrame(
            {
                "x": [i] * 6,
                "y": range(i * 6, (i + 1) * 6),
            }
        ).write_parquet(path, row_group_size=3)
        paths.append(str(path))

    scan = _make_parquet_scan(
        paths,
        schema={
            "x": DataType(pl.Int64()),
            "y": DataType(pl.Int64()),
        },
    )
    streaming_scan = StreamingScan(
        [ParquetScanTask(scan, [path], 0, 1, scan.parquet_options) for path in paths],
        scan,
    )
    unproven_request = OrderPartitioningRequest(
        (
            NamedOrderKey(
                "x",
                plc.types.Order.DESCENDING,
                plc.types.NullOrder.AFTER,
            ),
        )
    )
    partitioning = _run_parquet_metadata_ordering(
        streaming_scan,
        (unproven_request, _order_request("x", "y")),
        spmd_engine,
        len(paths),
    )

    assert partitioning is not None
    assert partitioning.local == "inherit"
    assert isinstance(partitioning.inter_rank, OrderScheme)
    (ordering,) = partitioning.inter_rank.orderings
    (key,) = ordering.keys
    assert list(scan.schema)[key.column_index] == "x"
    assert ordering.strict_boundaries is True
    assert ordering.locally_ordered is False
    assert _ordering_boundary_values(
        ordering, spmd_engine, name="x", dtype=scan.schema["x"]
    ) == [1]


def test_parquet_scan_ordering_partitioning_rejects_null_chunks(
    tmp_path: Path, spmd_engine
) -> None:
    paths = []
    for i, values in enumerate(([1, None], [2, 3])):
        path = tmp_path / f"part-{i}.parquet"
        pl.DataFrame({"x": values}).write_parquet(path, row_group_size=2)
        paths.append(str(path))

    scan = _make_parquet_scan(paths)
    streaming_scan = StreamingScan(
        [ParquetScanTask(scan, [path], 0, 1, scan.parquet_options) for path in paths],
        scan,
    )

    assert (
        _run_parquet_metadata_ordering(
            streaming_scan, (_order_request("x"),), spmd_engine, len(paths)
        )
        is None
    )


def test_parquet_scan_ordering_partitioning_uses_row_group_splits(
    tmp_path: Path, spmd_engine
) -> None:
    from cudf_streaming.channel_metadata import OrderScheme

    path = tmp_path / "data.parquet"
    pl.DataFrame({"x": range(8)}).write_parquet(path, row_group_size=2)
    scan = _make_parquet_scan([str(path)])
    split_count = 4
    streaming_scan = StreamingScan(
        [
            ParquetScanTask(scan, [str(path)], i, split_count, scan.parquet_options)
            for i in range(split_count)
        ],
        scan,
    )

    partitioning = _run_parquet_metadata_ordering(
        streaming_scan, (_order_request("x"),), spmd_engine, split_count
    )

    assert partitioning is not None
    assert isinstance(partitioning.inter_rank, OrderScheme)
    (ordering,) = partitioning.inter_rank.orderings
    assert ordering.strict_boundaries is True
    assert _ordering_boundary_values(
        ordering, spmd_engine, name="x", dtype=scan.schema["x"]
    ) == [2, 4, 6]


def test_parquet_scan_ordering_partitioning_string_column(
    tmp_path: Path, spmd_engine
) -> None:
    from cudf_streaming.channel_metadata import OrderScheme

    paths = []
    for i, values in enumerate((["a", "b"], ["c", "d"])):
        path = tmp_path / f"part-{i}.parquet"
        pl.DataFrame({"s": values}).write_parquet(path, row_group_size=2)
        paths.append(str(path))

    scan = _make_parquet_scan(paths, schema={"s": DataType(pl.String())})
    streaming_scan = StreamingScan(
        [ParquetScanTask(scan, [path], 0, 1, scan.parquet_options) for path in paths],
        scan,
    )

    partitioning = _run_parquet_metadata_ordering(
        streaming_scan, (_order_request("s"),), spmd_engine, len(paths)
    )

    assert partitioning is not None
    assert isinstance(partitioning.inter_rank, OrderScheme)
    (ordering,) = partitioning.inter_rank.orderings
    assert ordering.strict_boundaries is True
    assert _ordering_boundary_values(
        ordering, spmd_engine, name="s", dtype=scan.schema["s"]
    ) == ["c"]


@pytest.mark.parametrize("reverse_paths", [False, True], ids=["ordered", "reversed"])
def test_parquet_scan_ordering_partitioning_fused_files_require_ordered_paths(
    tmp_path: Path, spmd_engine, *, reverse_paths: bool
) -> None:
    paths = []
    for i, values in enumerate((range(4), range(4, 8))):
        path = tmp_path / f"part-{i}.parquet"
        pl.DataFrame({"x": values}).write_parquet(path, row_group_size=2)
        paths.append(str(path))
    if reverse_paths:
        paths.reverse()

    scan = _make_parquet_scan(paths)
    streaming_scan = StreamingScan(
        [ParquetScanTask(scan, paths, 0, 1, scan.parquet_options)], scan
    )
    partitioning = _run_parquet_metadata_ordering(
        streaming_scan,
        (_order_request("x"),),
        spmd_engine,
        global_chunk_count=1,
    )

    if reverse_paths:
        assert partitioning is None
    else:
        assert partitioning is not None


@pytest.mark.spmd
def test_parquet_scan_ordering_partitioning_allgather(
    tmp_path: Path, spmd_engine
) -> None:
    from cudf_streaming.channel_metadata import OrderScheme

    if spmd_engine.comm.nranks < 2:
        pytest.skip("requires multiple ranks")

    rows_per_rank = 4
    rank = spmd_engine.comm.rank
    path = tmp_path / f"part-{rank}.parquet"
    pl.DataFrame(
        {"x": range(rank * rows_per_rank, (rank + 1) * rows_per_rank)}
    ).write_parquet(path, row_group_size=2)

    scan = _make_parquet_scan([str(path)])
    streaming_scan = StreamingScan(
        [ParquetScanTask(scan, scan.paths, 0, 1, scan.parquet_options)], scan
    )
    partitioning = _run_parquet_metadata_ordering(
        streaming_scan,
        (_order_request("x"),),
        spmd_engine,
        global_chunk_count=spmd_engine.comm.nranks,
    )

    assert partitioning is not None
    assert isinstance(partitioning.inter_rank, OrderScheme)
    (ordering,) = partitioning.inter_rank.orderings
    assert _ordering_boundary_values(
        ordering, spmd_engine, name="x", dtype=scan.schema["x"]
    ) == [i * rows_per_rank for i in range(1, spmd_engine.comm.nranks)]


def test_parquet_scan_ordering_partitioning_skips_synthetic_columns(
    tmp_path: Path, spmd_engine
) -> None:
    from cudf_streaming.channel_metadata import OrderScheme

    path = tmp_path / "data.parquet"
    pl.DataFrame({"x": range(4)}).write_parquet(path, row_group_size=2)
    scan = _make_parquet_scan(
        [str(path)],
        schema={
            "x": DataType(pl.Int64()),
            "path": DataType(pl.String()),
        },
    )
    streaming_scan = StreamingScan(
        [ParquetScanTask(scan, [str(path)], 0, 1, scan.parquet_options)],
        scan,
    )

    partitioning = _run_parquet_metadata_ordering(
        streaming_scan,
        (_order_request("path"), _order_request("x")),
        spmd_engine,
        global_chunk_count=1,
    )

    assert partitioning is not None
    assert isinstance(partitioning.inter_rank, OrderScheme)
    (ordering,) = partitioning.inter_rank.orderings
    (key,) = ordering.keys
    assert list(scan.schema)[key.column_index] == "x"


class FooSource(DataSourceInfo):
    def __init__(self, size: int):
        self._size = size

    @property
    def type(self):
        return "parquet"

    @property
    def row_count(self):
        return None

    def column_storage_size(self, _col: str) -> int:
        return self._size

    def serialize(self):
        return {}

    @classmethod
    def deserialize(cls, data):
        return cls(0)


class FooStats(StatsCollector):
    def __init__(self, ir: Scan, size: int):
        super().__init__()
        self.scan_stats = {ir: FooSource(size)}


def _make_config(target: int) -> ConfigOptions:
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={"target_partition_size": target},
    )
    return ConfigOptions.from_polars_engine(engine)


@pytest.mark.parametrize(
    "file_size,n_paths,expected_factor,expected_flavor",
    [
        (12, 1, 1, IOPartitionFlavor.FUSED_FILES),
        (15, 1, 2, IOPartitionFlavor.SPLIT_FILES),
        (20, 1, 2, IOPartitionFlavor.SPLIT_FILES),
        (24, 1, 2, IOPartitionFlavor.SPLIT_FILES),
        (25, 1, 3, IOPartitionFlavor.SPLIT_FILES),
        (7, 3, 1, IOPartitionFlavor.FUSED_FILES),
        (6, 3, 2, IOPartitionFlavor.FUSED_FILES),
        (4, 4, 3, IOPartitionFlavor.FUSED_FILES),
        (3, 4, 3, IOPartitionFlavor.FUSED_FILES),
        (1, 3, 3, IOPartitionFlavor.FUSED_FILES),
    ],
)
def test_scan_partition_plan_nearest(
    file_size: int,
    n_paths: int,
    expected_factor: int,
    expected_flavor: IOPartitionFlavor,
) -> None:
    paths = [f"f{i}.parquet" for i in range(n_paths)]
    scan = _make_parquet_scan(paths)
    plan = scan_partition_plan(scan, FooStats(scan, file_size), _make_config(10))
    assert plan.factor == expected_factor
    assert plan.flavor == expected_flavor
