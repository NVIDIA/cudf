# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import pytest

import polars as pl

from cudf_polars import Translator
from cudf_polars.containers import DataType
from cudf_polars.dsl.ir import (
    Empty,
    IRExecutionContext,
    Scan,
)
from cudf_polars.dsl.utils.io import (
    CachedParquetInfo,
    prefetch_parquet_file_metadata_for_ir,
)
from cudf_polars.engine.options import StreamingOptions
from cudf_polars.streaming.base import (
    DataSourceInfo,
    IOPartitionFlavor,
    IOPartitionPlan,
    StatsCollector,
)
from cudf_polars.streaming.io import (
    FusedScan,
    SplitScan,
    StreamingScan,
    expand_scan_for_rank,
    scan_partition_plan,
)
from cudf_polars.streaming.parallel import lower_ir_graph
from cudf_polars.streaming.statistics import collect_statistics
from cudf_polars.testing.asserts import assert_gpu_result_equal
from cudf_polars.testing.engine_utils import SMALL_MAX_ROWS_PER_PARTITION
from cudf_polars.testing.io import make_partitioned_source
from cudf_polars.utils.config import ConfigOptions, ParquetOptions

if TYPE_CHECKING:
    import concurrent.futures
    from collections.abc import Callable
    from pathlib import Path
    from typing import Any, Literal

    import pylibcudf as plc

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


def test_prefetch_parquet_file_metadata_remote_only(tmp_path, df) -> None:
    make_partitioned_source(df, tmp_path, "parquet", n_files=1)
    local_path = str(next(tmp_path.glob("*.parquet")))

    scan = _make_parquet_scan([local_path])
    fused = FusedScan(scan.schema, scan, scan.paths, scan.parquet_options, [])
    streaming_scan = StreamingScan([fused], scan, "fused")

    # Local paths are skipped entirely when remote_only=True.
    result = prefetch_parquet_file_metadata_for_ir(
        streaming_scan, py_executor=None, stats=None, remote_only=True
    )
    assert result == {}

    # The same local path is prefetched when remote_only=False (the default).
    result = prefetch_parquet_file_metadata_for_ir(
        streaming_scan, py_executor=None, stats=None
    )
    assert set(result) == {local_path}


def test_prefetch_file_metadata_select_fast_count(
    df: pl.DataFrame,
    streaming_engine_factory: Callable[..., StreamingEngine],
    tmp_path: Path,
) -> None:
    streaming_engine = streaming_engine_factory(
        StreamingOptions(parquet_options={"prefetch_file_metadata": True}),
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
    paths: list[str], parquet_options: ParquetOptions | None = None
) -> Scan:
    parquet_options = parquet_options or ParquetOptions()
    return Scan(
        {"x": DataType(pl.Int64())},
        "parquet",
        {},
        None,
        paths,
        None,
        0,
        -1,
        None,
        None,
        None,
        parquet_options,
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
        streaming_scan.scans, expected_path_groups, strict=True
    ):
        assert isinstance(scan, FusedScan)
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
    assert len(streaming_scan.scans) == len(expected_splits)
    for scan, (split_index, total_splits) in zip(
        streaming_scan.scans, expected_splits, strict=True
    ):
        assert isinstance(scan, SplitScan)
        assert scan.split_index == split_index
        assert scan.total_splits == total_splits
        assert scan.paths == ["file.parquet"]


def test_streaming_scan_raises() -> None:
    # This isn't reachable by normal cudf-polars usage.
    scan = _make_parquet_scan(["file.parquet"])
    fused = FusedScan(scan.schema, scan, scan.paths, scan.parquet_options, [])
    ctx = IRExecutionContext()
    with pytest.raises(NotImplementedError, match=r"StreamingScan.do_evaluate"):
        StreamingScan.do_evaluate([fused], scan, context=ctx)


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


def test_streaming_scan_missing_prefetch_metadata_raises() -> None:
    # This isn't reachable by polars' public API, so we test it directly.
    scan = _make_parquet_scan(
        ["file.parquet"], parquet_options=ParquetOptions(prefetch_file_metadata=True)
    )
    fused = FusedScan(scan.schema, scan, scan.paths, scan.parquet_options, [])

    ctx = IRExecutionContext()
    with pytest.raises(NotImplementedError, match=r"StreamingScan.do_evaluate"):
        StreamingScan.do_evaluate([fused], scan, context=ctx)


def test_split_scan_do_evaluate_missing_prefetch_metadata() -> None:
    paths = ["/some/missing/file.parquet"]
    parquet_options = ParquetOptions(prefetch_file_metadata=True)
    context = IRExecutionContext()
    schema = {"x": DataType(pl.Int64())}

    with pytest.raises(
        AssertionError,
        match=(r"Paths do not match cached parquet info."),
    ):
        SplitScan.do_evaluate(
            0,
            4,
            schema,
            "parquet",
            {},
            paths,
            None,
            0,
            -1,
            None,
            None,
            None,
            parquet_options,
            [],
            context=context,
        )


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


def test_fused_scan_identity_equality() -> None:
    base = _make_parquet_scan(["a.parquet", "b.parquet"])
    paths = ["a.parquet"]
    info = _make_cached_parquet_info(paths)

    a = FusedScan(base.schema, base, paths, base.parquet_options, info)
    b = FusedScan(base.schema, base, paths, base.parquet_options, info.copy())
    c = FusedScan(base.schema, base, ["b.parquet"], base.parquet_options, info)

    assert a == b
    assert hash(a) == hash(b)
    assert a != c


def test_split_scan_identity_equality() -> None:
    base = _make_parquet_scan(["a.parquet"])
    info = _make_cached_parquet_info(base.paths)

    a = SplitScan(base.schema, base, base.paths, 0, 4, base.parquet_options, info)
    b = SplitScan(
        base.schema, base, base.paths, 0, 4, base.parquet_options, info.copy()
    )
    c = SplitScan(base.schema, base, base.paths, 1, 4, base.parquet_options, info)

    assert a == b
    assert hash(a) == hash(b)
    assert a != c


def test_streaming_scan_identity_equality() -> None:
    base = _make_parquet_scan(["a.parquet"])
    split = SplitScan(
        base.schema,
        base,
        base.paths,
        0,
        2,
        base.parquet_options,
        _make_cached_parquet_info(base.paths, size=10),
    )
    split_same = SplitScan(
        base.schema,
        base,
        base.paths,
        0,
        2,
        base.parquet_options,
        _make_cached_parquet_info(base.paths, size=11),
    )
    split_diff = SplitScan(
        base.schema,
        base,
        base.paths,
        1,
        2,
        base.parquet_options,
        _make_cached_parquet_info(base.paths, size=10),
    )

    a = StreamingScan([split], base, "split")
    b = StreamingScan([split_same], base, "split")
    c = StreamingScan([split_diff], base, "split")

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

    split_without = SplitScan(
        base.schema, base, base.paths, 0, 4, base.parquet_options, None
    )
    split_with = SplitScan(
        base.schema, base, base.paths, 0, 4, base.parquet_options, info
    )
    assert split_without == split_with
    assert hash(split_without) == hash(split_with)

    fused_without = FusedScan(base.schema, base, base.paths, base.parquet_options, None)
    fused_with = FusedScan(base.schema, base, base.paths, base.parquet_options, info)
    assert fused_without == fused_with
    assert hash(fused_without) == hash(fused_with)


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
