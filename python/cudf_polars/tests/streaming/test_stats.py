# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import pickle
from typing import TYPE_CHECKING, ClassVar, cast

import pytest

import polars as pl

import pylibcudf as plc
import rmm.pylibrmm.stream

import cudf_polars.containers
import cudf_polars.streaming.io as streaming_io
from cudf_polars import Translator
from cudf_polars.containers import DataType
from cudf_polars.dsl.ir import (
    Empty,
    Projection,
)
from cudf_polars.engine.options import StreamingOptions
from cudf_polars.streaming.base import SerializedDataSourceInfo, StatsCollector
from cudf_polars.streaming.io import (
    DataFrameSourceInfo,
    ParquetMetadata,
    ParquetSourceInfo,
    _build_parquet_source,
    _clear_source_info_cache,
)
from cudf_polars.streaming.statistics import collect_statistics
from cudf_polars.testing.asserts import assert_gpu_result_equal
from cudf_polars.testing.io import make_lazy_frame, make_partitioned_source
from cudf_polars.utils.config import ConfigOptions

if TYPE_CHECKING:
    import concurrent.futures
    import pathlib

    from cudf_polars.typing import Schema


@pytest.fixture(scope="module")
def df_and_schema() -> tuple[pl.DataFrame, Schema]:
    stream = rmm.pylibrmm.stream.Stream()
    df = pl.DataFrame(
        {
            "x": range(3_000),
            "y": ["cat", "dog", "fish"] * 1_000,
            "z": [1.0, 2.0, 3.0, 4.0, 5.0] * 600,
        }
    )
    df_ = cudf_polars.containers.DataFrame.from_polars(df, stream=stream)
    schema = {column.name: column.dtype for column in df_.columns}
    return df, schema


# Simple engine for IR translation / stats collection only (no actual GPU execution)
@pytest.fixture(scope="module")
def stats_engine():
    return pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "target_partition_size": 10_000,
            "max_rows_per_partition": 1_000,
        },
    )


def test_base_stats_dataframescan(
    df_and_schema: tuple[pl.DataFrame, Schema],
    stats_engine,
    parquet_stats_executor: concurrent.futures.ThreadPoolExecutor,
):
    df, _schema = df_and_schema
    row_count = df.height
    q = pl.LazyFrame(df)
    ir = Translator(q._ldf.visit(), stats_engine).translate_ir()
    stats = collect_statistics(
        ir, ConfigOptions.from_polars_engine(stats_engine), parquet_stats_executor
    )

    source = stats.scan_stats[ir]
    assert source.row_count == row_count

    # Storage size is not available for in-memory frames
    assert source.column_storage_size("x") is None


@pytest.mark.parametrize("n_files", [1, 3])
@pytest.mark.parametrize("row_group_size", [None, 10_000])
@pytest.mark.parametrize("max_footer_samples", [3, 0])
@pytest.mark.parametrize("max_row_group_samples", [1, 0])
def test_base_stats_parquet(
    tmp_path,
    df_and_schema: tuple[pl.DataFrame, Schema],
    n_files,
    row_group_size,
    max_footer_samples,
    max_row_group_samples,
    parquet_stats_executor: concurrent.futures.ThreadPoolExecutor,
):
    _clear_source_info_cache()
    df, _schema = df_and_schema
    make_partitioned_source(
        df,
        tmp_path,
        "parquet",
        n_files=n_files,
        row_group_size=row_group_size,
    )
    q = pl.scan_parquet(tmp_path)
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={"target_partition_size": 10_000},
        parquet_options={
            "max_footer_samples": max_footer_samples,
            "max_row_group_samples": max_row_group_samples,
        },
    )
    ir = Translator(q._ldf.visit(), engine).translate_ir()
    stats = collect_statistics(
        ir, ConfigOptions.from_polars_engine(engine), parquet_stats_executor
    )
    source = stats.scan_stats[ir]

    if max_footer_samples:
        assert source.row_count == df.height
        assert source.column_storage_size("x") is not None
        assert source.column_storage_size("x") > 0
        assert source.column_storage_size("y") is not None
        assert source.column_storage_size("y") > 0
    else:
        assert source.row_count is None
        assert source.column_storage_size("x") is None
        assert source.column_storage_size("y") is None


def test_parquet_source_info_uses_decoded_dtype_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeDataType:
        def __init__(self, type_id: plc.TypeId) -> None:
            self.plc_type = plc.DataType(type_id)

        def id(self) -> plc.TypeId:
            return self.plc_type.id()

    class FakeParquetMetadata:
        row_count = 2_000
        mean_size_per_file: ClassVar[dict[str, int]] = {
            "i64": 1,
            "dec32": 1,
            "s": 1,
            "already_large": 20_000,
        }
        num_row_groups_per_file = (1, 1)

        def __init__(self, paths: tuple[str, ...], max_footer_samples: int) -> None:
            self.paths = paths
            self.max_footer_samples = max_footer_samples
            self.sampled_file_count = 1
            self.total_file_count = len(paths)

    sampled_cols: list[str] = []

    def fake_sample_rg_sizes(
        _metadata: object,
        target_cols: list[str],
        _max_row_group_samples: int,
    ) -> dict[str, int]:
        sampled_cols.extend(target_cols)
        return {}

    monkeypatch.setattr(streaming_io, "ParquetMetadata", FakeParquetMetadata)
    monkeypatch.setattr(streaming_io, "_sample_rg_sizes", fake_sample_rg_sizes)

    source = ParquetSourceInfo.from_paths(
        ("a.parquet", "b.parquet"),
        frozenset(
            {
                "i64",
                "dec32",
                "s",
                "already_large",
            }
        ),
        (
            ("i64", DataType(pl.Int64())),
            ("dec32", cast("DataType", FakeDataType(plc.TypeId.DECIMAL32))),
            ("s", DataType(pl.String())),
            ("already_large", DataType(pl.Int64())),
        ),
        max_footer_samples=2,
        max_row_group_samples=1,
    )

    rows_per_file = 1_000
    nullmask = 125
    assert source.column_storage_size("i64") == rows_per_file * 8 + nullmask
    assert source.column_storage_size("dec32") == rows_per_file * 4 + nullmask
    assert source.column_storage_size("s") == (rows_per_file + 1) * 4 + nullmask
    assert source.column_storage_size("already_large") == 20_000
    assert sampled_cols == ["s"]


def test_dataframescan_stats_pickle(
    stats_engine, parquet_stats_executor: concurrent.futures.ThreadPoolExecutor
):
    df = pl.DataFrame({"x": range(100), "y": [1, 2] * 50})
    q = pl.LazyFrame(df)
    ir = Translator(q._ldf.visit(), stats_engine).translate_ir()
    stats = collect_statistics(
        ir, ConfigOptions.from_polars_engine(stats_engine), parquet_stats_executor
    )

    # Pickle and unpickle the stats collector
    pickled = pickle.dumps(stats)
    unpickled_stats = pickle.loads(pickled)

    # Verify the unpickled stats are equivalent
    assert type(unpickled_stats) is type(stats)
    assert unpickled_stats.scan_stats[ir].row_count == 100


def test_parquet_round_trip() -> None:
    info = ParquetSourceInfo(1000, {"x": 200, "y": 400})
    data = info.serialize()
    restored = ParquetSourceInfo.deserialize(data)

    assert restored.type == "parquet"
    assert restored.row_count == info.row_count
    assert restored.per_file_means == info.per_file_means


def test_parquet_round_trip_empty() -> None:
    info = ParquetSourceInfo(None, {})
    data = info.serialize()
    restored = ParquetSourceInfo.deserialize(data)

    assert restored.row_count is None
    assert restored.per_file_means == {}


def test_parquet_source_info_stores_footers_when_all_files_sampled(
    tmp_path: pathlib.Path,
    df_and_schema: tuple[pl.DataFrame, Schema],
) -> None:
    _clear_source_info_cache()
    df, schema = df_and_schema
    make_partitioned_source(df, tmp_path, "parquet", n_files=2)
    paths = tuple(str(p) for p in sorted(tmp_path.iterdir()))
    info = _build_parquet_source(
        paths,
        frozenset(df.columns),
        tuple(schema.items()),
        max_footer_samples=10,
        max_row_group_samples=0,
    )

    assert info.cached_parquet_info is not None
    assert len(info.cached_parquet_info) == len(paths)
    assert (
        sum(cached.file_metadata.num_rows for cached in info.cached_parquet_info)
        == df.height
    )


def test_parquet_source_info_omits_footers_when_paths_are_sampled(
    tmp_path: pathlib.Path,
    df_and_schema: tuple[pl.DataFrame, Schema],
) -> None:
    _clear_source_info_cache()
    df, schema = df_and_schema
    make_partitioned_source(df, tmp_path, "parquet", n_files=5)
    paths = tuple(str(p) for p in sorted(tmp_path.iterdir()))
    info = _build_parquet_source(
        paths,
        frozenset(df.columns),
        tuple(schema.items()),
        max_footer_samples=2,
        max_row_group_samples=0,
    )

    assert info.cached_parquet_info is None


def test_parquet_metadata_reads_footers(
    tmp_path: pathlib.Path,
    df_and_schema: tuple[pl.DataFrame, Schema],
) -> None:
    df, _schema = df_and_schema
    make_partitioned_source(df, tmp_path, "parquet", n_files=1)
    path = next(tmp_path.iterdir())
    metadata = ParquetMetadata((str(path),), max_footer_samples=1)

    assert metadata.cached_parquet_info is not None
    assert len(metadata.cached_parquet_info) == 1
    assert metadata.row_count == df.height


def test_dataframe_round_trip() -> None:
    info = DataFrameSourceInfo(2500)
    data = info.serialize()
    restored = DataFrameSourceInfo.deserialize(data)

    assert restored.type == "dataframe"
    assert restored.row_count == info.row_count
    assert restored.column_storage_size("x") is None


@pytest.mark.parametrize("kind", ["parquet", "csv", "frame"])
def test_stats_planning(tmp_path, kind, streaming_engine_factory):
    streaming_engine = streaming_engine_factory(
        StreamingOptions(target_partition_size=10_000, max_rows_per_partition=1_000),
    )
    sales = pl.DataFrame(
        {
            "order_id": [1, 2, 3, 4, 5, 6],
            "customer_id": [101, 102, 101, 103, 102, 101],
            "amount": [50.0, 75.0, 30.0, 120.0, 85.0, 40.0],
            "product": ["A", "B", "A", "C", "B", "A"],
        }
    )
    sales = make_lazy_frame(sales, kind, path=tmp_path / f"sales_{kind}")
    customers = pl.DataFrame(
        {
            "customer_id": [101, 102, 103],
            "customer_name": ["Alice", "Bob", "Charlie"],
            "region": ["North", "South", "North"],
        }
    )
    customers = make_lazy_frame(customers, kind, path=tmp_path / f"customers_{kind}")
    q_join = sales.filter(pl.col("amount") < 100.0).join(
        customers, on="customer_id", how="inner"
    )
    q_gb = q_join.group_by("customer_id").agg(
        [
            pl.col("amount").sum().alias("total_amount"),
            pl.col("order_id").count().alias("order_count"),
            pl.col("customer_name").first().alias("name"),
            pl.col("region").first().alias("region"),
        ]
    )
    assert_gpu_result_equal(q_gb.sort("customer_id"), engine=streaming_engine)


def test_parquet_deserialize_wrong_type() -> None:
    data = DataFrameSourceInfo(100).serialize()
    with pytest.raises(ValueError, match="Expected ParquetSourceInfo"):
        ParquetSourceInfo.deserialize(data)


def test_dataframe_deserialize_wrong_type() -> None:
    data = ParquetSourceInfo(1000, {"x": 200}).serialize()
    with pytest.raises(ValueError, match="Expected DataFrameSourceInfo"):
        DataFrameSourceInfo.deserialize(data)


def test_dataframe_deserialize_missing_row_count() -> None:
    data = SerializedDataSourceInfo(
        type="dataframe", row_count=None, per_file_means=None
    )
    with pytest.raises(
        ValueError, match="Row count is required for DataFrameSourceInfo"
    ):
        DataFrameSourceInfo.deserialize(data)


def test_parquet_empty_per_file_means() -> None:
    per_file_means: dict[str, int] = {}
    info = ParquetSourceInfo(1000, per_file_means)
    assert info.per_file_means is per_file_means

    info = ParquetSourceInfo(1000, None)
    assert info.per_file_means == {}


def test_serialize_stats_roundtrip_dataframescan(
    stats_engine: pl.GPUEngine,
    parquet_stats_executor: concurrent.futures.ThreadPoolExecutor,
) -> None:
    df = pl.DataFrame({"x": range(200), "y": [1, 2] * 100})
    q = pl.LazyFrame(df)
    ir = Translator(q._ldf.visit(), stats_engine).translate_ir()
    config = ConfigOptions.from_polars_engine(stats_engine)
    stats = collect_statistics(ir, config, parquet_stats_executor)

    serialized = stats.serialize(ir)
    wire = json.loads(json.dumps(serialized))
    restored = StatsCollector.deserialize(wire, ir)

    assert set(restored.scan_stats) == set(stats.scan_stats)
    for node, info in stats.scan_stats.items():
        rt = restored.scan_stats[node]
        assert rt.row_count == info.row_count
        assert rt.column_storage_size("x") == info.column_storage_size("x")


def test_serialize_stats_roundtrip_parquet(
    tmp_path: pathlib.Path,
    df_and_schema: tuple[pl.DataFrame, Schema],
    parquet_stats_executor: concurrent.futures.ThreadPoolExecutor,
) -> None:
    _clear_source_info_cache()
    df, _schema = df_and_schema
    make_partitioned_source(df, tmp_path, "parquet", n_files=3)
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={"target_partition_size": 10_000},
        parquet_options={"max_footer_samples": 3, "max_row_group_samples": 1},
    )
    q = pl.scan_parquet(tmp_path)
    ir = Translator(q._ldf.visit(), engine).translate_ir()
    config = ConfigOptions.from_polars_engine(engine)
    stats = collect_statistics(ir, config, parquet_stats_executor)

    serialized = stats.serialize(ir)
    wire = json.loads(json.dumps(serialized))
    restored = StatsCollector.deserialize(wire, ir)

    assert set(restored.scan_stats) == set(stats.scan_stats)
    for node, info in stats.scan_stats.items():
        rt = restored.scan_stats[node]
        assert rt.row_count == info.row_count
        for col in ("x", "y", "z"):
            assert rt.column_storage_size(col) == info.column_storage_size(col)


def test_serialize_uses_value_equality() -> None:
    schema = {"x": DataType(pl.Int64())}
    scan_x = Empty(schema)
    scan_y = Empty(schema)
    assert scan_x == scan_y
    assert scan_x is not scan_y

    root = Projection(schema, scan_y)

    stats = StatsCollector()
    stats.scan_stats[scan_x] = DataFrameSourceInfo(100)

    result = stats.serialize(root)
    assert len(result) == 1
    assert result[0]["index"] >= 0
