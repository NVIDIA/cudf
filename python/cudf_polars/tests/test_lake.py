# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import polars as pl

from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)
from cudf_polars.testing.engine_utils import is_streaming_engine

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

pytest.importorskip("pyiceberg")
deltalake = pytest.importorskip("deltalake")
import pyiceberg.schema  # noqa: E402
from pyiceberg.catalog.sql import SqlCatalog  # noqa: E402
from pyiceberg.partitioning import PartitionField, PartitionSpec  # noqa: E402
from pyiceberg.transforms import IdentityTransform  # noqa: E402
from pyiceberg.types import (  # noqa: E402
    DoubleType,
    LongType,
    NestedField,
    StringType,
)


@pytest.fixture
def iceberg_catalog(tmp_path: Path) -> SqlCatalog:
    with SqlCatalog(
        "default",
        uri=f"sqlite:///{tmp_path}/catalog.db",
        warehouse=f"file://{tmp_path / 'warehouse'}",
    ) as catalog:
        catalog.create_namespace_if_not_exists("ns")
        return catalog


@pytest.fixture
def flat_iceberg(iceberg_catalog: SqlCatalog) -> str:
    first = pl.LazyFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    tbl = iceberg_catalog.create_table(
        "ns.flat", schema=first.collect_schema().to_arrow()
    )
    first.sink_iceberg(tbl, mode="append")
    pl.LazyFrame({"a": [4, 5], "b": ["p", "q"]}).sink_iceberg(tbl, mode="append")
    return tbl.metadata_location


@pytest.fixture
def evolved_iceberg(iceberg_catalog: SqlCatalog) -> str:
    first = pl.LazyFrame(
        {"a": pl.Series([1, 2, 3], dtype=pl.Int32), "b": ["x", "y", "z"]}
    )
    tbl = iceberg_catalog.create_table(
        "ns.evolved", schema=first.collect_schema().to_arrow()
    )
    first.sink_iceberg(tbl, mode="append")

    with tbl.update_schema() as update:
        update.rename_column("b", "b_renamed")
    tbl.refresh()

    pl.LazyFrame(
        {
            "a": pl.Series([4, 5], dtype=pl.Int64),
            "b_renamed": ["p", "q"],
            "c": [1.5, 2.5],
        }
    ).sink_iceberg(tbl, mode="append", schema_mode="merge")
    return tbl.metadata_location


@pytest.fixture
def renamed_iceberg(iceberg_catalog: SqlCatalog) -> str:
    first = pl.LazyFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    tbl = iceberg_catalog.create_table(
        "ns.renamed", schema=first.collect_schema().to_arrow()
    )
    first.sink_iceberg(tbl, mode="append")

    with tbl.update_schema() as update:
        update.rename_column("b", "b_renamed")
    tbl.refresh()

    pl.LazyFrame({"a": [4, 5], "b_renamed": ["p", "q"]}).sink_iceberg(
        tbl, mode="append"
    )
    return tbl.metadata_location


@pytest.fixture
def promoted_iceberg(iceberg_catalog: SqlCatalog) -> str:
    first = pl.LazyFrame(
        {
            "a": pl.Series([1, 2, 3], dtype=pl.Int32),
            "b": ["x", "y", "z"],
            "c": pl.Series([0.5, 1.5, 2.5], dtype=pl.Float32),
        }
    )
    tbl = iceberg_catalog.create_table(
        "ns.promoted", schema=first.collect_schema().to_arrow()
    )
    first.sink_iceberg(tbl, mode="append")

    with tbl.update_schema() as update:
        update.update_column("a", field_type=LongType())
        update.update_column("c", field_type=DoubleType())
    tbl.refresh()

    pl.LazyFrame(
        {
            "a": pl.Series([4, 5], dtype=pl.Int64),
            "b": ["p", "q"],
            "c": pl.Series([3.5, 4.5], dtype=pl.Float64),
        }
    ).sink_iceberg(tbl, mode="append")
    return tbl.metadata_location


@pytest.fixture
def partitioned_iceberg(iceberg_catalog: SqlCatalog) -> str:
    schema = pyiceberg.schema.Schema(
        NestedField(1, "a", LongType(), required=False),
        NestedField(2, "part", StringType(), required=False),
    )
    spec = PartitionSpec(
        PartitionField(
            source_id=2, field_id=1000, transform=IdentityTransform(), name="part"
        )
    )
    tbl = iceberg_catalog.create_table(
        "ns.partitioned", schema=schema, partition_spec=spec
    )
    pl.LazyFrame({"a": [1, 2, 3, 4], "part": ["u", "u", "v", "v"]}).sink_iceberg(
        tbl, mode="append"
    )
    return tbl.metadata_location


@pytest.fixture
def flat_delta(tmp_path: Path) -> str:
    path = tmp_path / "delta_flat"
    pl.LazyFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]}).sink_delta(
        str(path), mode="overwrite"
    )
    pl.LazyFrame({"a": [4, 5], "b": ["p", "q"]}).sink_delta(str(path), mode="append")
    return str(path)


@pytest.fixture
def deleted_rows_delta(tmp_path: Path) -> str:
    path = tmp_path / "delta_deleted"
    pl.LazyFrame({"a": list(range(10)), "b": [f"s{i}" for i in range(10)]}).sink_delta(
        str(path),
        mode="overwrite",
        delta_write_options={"configuration": {"delta.enableDeletionVectors": "true"}},
    )
    deltalake.DeltaTable(str(path)).delete("a % 3 == 0")
    return str(path)


@pytest.fixture
def partitioned_delta(tmp_path: Path) -> str:
    path = tmp_path / "delta_partitioned"
    pl.LazyFrame(
        {"a": [1, 2, 3, 4, 5, 6], "part": ["u", "u", "v", "v", "w", "w"]}
    ).sink_delta(
        str(path),
        mode="overwrite",
        delta_write_options={"partition_by": ["part"]},
    )
    return str(path)


@pytest.fixture
def evolved_hive(tmp_path: Path) -> Path:
    path = tmp_path / "hive"
    pl.LazyFrame({"a": [1, 2], "b": [10, 20]}).sink_parquet(
        path / "part=u" / "data.parquet", mkdir=True
    )
    pl.LazyFrame({"a": [3, 4]}).sink_parquet(
        path / "part=v" / "data.parquet", mkdir=True
    )
    pl.LazyFrame({"a": [5, 6], "b": [50, 60]}).sink_parquet(
        path / "part=w" / "data.parquet", mkdir=True
    )
    return path


@pytest.mark.parametrize(
    "table",
    [
        "flat_iceberg",
        "flat_delta",
        "evolved_iceberg",
        "renamed_iceberg",
        "promoted_iceberg",
        "partitioned_iceberg",
        "partitioned_delta",
        "deleted_rows_delta",
    ],
)
def test_scan_lake(
    request: pytest.FixtureRequest, engine: pl.GPUEngine, table: str
) -> None:
    scan = pl.scan_iceberg if table.endswith("iceberg") else pl.scan_delta
    q = scan(request.getfixturevalue(table))
    assert_gpu_result_equal(q.sort("a"), engine=engine)


FLAT_PREDICATES = [
    pl.col("a") < 3,
    pl.col("b") == "q",
    (pl.col("a") > 1) & (pl.col("b") != "y"),
    pl.col("a").is_between(2, 4),
]
EVOLVED_PREDICATES = [
    pl.col("a") > 2,
    pl.col("b_renamed") == "y",
    pl.col("b_renamed").is_in(["x", "q"]),
    pl.col("c") > 2.0,
    pl.col("c").is_null(),
    (pl.col("b_renamed") != "x") & (pl.col("a") < 5),
    (pl.col("a") > 1) | (pl.col("c") > 2.0),
]
PROMOTED_PREDICATES = [
    pl.col("a") > 2,
    pl.col("c") < 3.0,
    (pl.col("a") > 1) & (pl.col("b") != "q"),
]
PARTITIONED_PREDICATES = [
    pl.col("a") > 1,
    pl.col("a").is_in([2, 3, 6]),
    pl.col("part") == "v",
    (pl.col("a") > 1) & (pl.col("part") == "v"),
    (pl.col("a") == 1) | (pl.col("part") == "v"),
]


@pytest.mark.parametrize(
    "table, predicate",
    [
        *(
            (table, p)
            for table in ("flat_iceberg", "flat_delta")
            for p in FLAT_PREDICATES
        ),
        *(("evolved_iceberg", p) for p in EVOLVED_PREDICATES),
        ("renamed_iceberg", pl.col("b_renamed") != "y"),
        *(("promoted_iceberg", p) for p in PROMOTED_PREDICATES),
        *(
            (table, p)
            for table in ("partitioned_iceberg", "partitioned_delta")
            for p in PARTITIONED_PREDICATES
        ),
        ("deleted_rows_delta", pl.col("a") > 4),
    ],
)
def test_scan_lake_predicate(
    request: pytest.FixtureRequest,
    engine: pl.GPUEngine,
    table: str,
    predicate: pl.Expr,
) -> None:
    scan = pl.scan_iceberg if table.endswith("iceberg") else pl.scan_delta
    q = scan(request.getfixturevalue(table))
    assert_gpu_result_equal(q.filter(predicate).sort("a"), engine=engine)


@pytest.mark.parametrize(
    "query",
    [
        *(
            lambda q, columns=columns: q.select(columns).sort(columns)
            for columns in (
                ["a"],
                ["b_renamed"],
                ["c"],
                ["c", "a"],
                ["a", "b_renamed", "c"],
            )
        ),
        *(lambda q, n=n: q.sort("a").head(n) for n in (1, 3, 5)),
        lambda q: q.with_row_index("idx"),
    ],
    ids=[
        "select_a",
        "select_b_renamed",
        "select_c",
        "select_c_a",
        "select_all",
        "head_1",
        "head_3",
        "head_5",
        "row_index",
    ],
)
def test_scan_iceberg_schema_evolution(
    engine: pl.GPUEngine, evolved_iceberg: str, query: Any
) -> None:
    assert_gpu_result_equal(query(pl.scan_iceberg(evolved_iceberg)), engine=engine)


@pytest.mark.parametrize(
    "table, query",
    [
        ("evolved_iceberg", lambda q: q.filter(pl.col("a") < 3)),
        ("evolved_iceberg", lambda q: q.filter(pl.col("b_renamed") == "y")),
        ("evolved_iceberg", lambda q: q.filter(pl.col("c") > 2.0)),
        (
            "evolved_iceberg",
            lambda q: q.filter(pl.col("b_renamed") == "y").with_row_index("idx"),
        ),
        ("evolved_iceberg", lambda q: q.filter(pl.col("b_renamed") == "y").head(1)),
        (
            "evolved_iceberg",
            lambda q: q.with_row_index("idx").filter(pl.col("b_renamed") == "y"),
        ),
        ("evolved_iceberg", lambda q: q.head(4).filter(pl.col("b_renamed") == "y")),
        ("renamed_iceberg", lambda q: q.filter(pl.col("b_renamed") == "y")),
        ("promoted_iceberg", lambda q: q.filter(pl.col("a") < 3)),
        ("promoted_iceberg", lambda q: q.filter(pl.col("b") == "q")),
        ("partitioned_iceberg", lambda q: q.filter(pl.col("a").is_in([2, 3]))),
        ("partitioned_delta", lambda q: q.filter(pl.col("a").is_in([2, 3]))),
    ],
    ids=[
        "widened",
        "renamed",
        "added",
        "filter_then_row_index",
        "filter_then_slice",
        "row_index_then_filter",
        "slice_then_filter",
        "renamed_only",
        "promoted",
        "promoted_other_column",
        "partitioned_iceberg",
        "partitioned_delta",
    ],
)
def test_scan_lake_predicate_pushdown(
    request: pytest.FixtureRequest,
    in_memory_engine: pl.GPUEngine,
    table: str,
    query: Any,
) -> None:
    scan = pl.scan_iceberg if table.endswith("iceberg") else pl.scan_delta
    q = scan(request.getfixturevalue(table))
    assert_gpu_result_equal(
        query(q),
        engine=in_memory_engine,
        check_row_order=False,
    )


@pytest.mark.parametrize(
    "query",
    [
        lambda q: q.filter(pl.col("a") > 2),
        lambda q: q.filter(pl.col("b") > 30),
        lambda q: q.filter(pl.col("b") > 30).select("b", "path"),
        lambda q: q.filter(pl.col("a").is_in([1, 4, 6])).select("part", "path"),
        lambda q: q.filter((pl.col("a") < 5) & (pl.col("part") != "v")),
    ],
    ids=["data", "added", "added_only", "partition_only", "mixed"],
)
def test_scan_lake_hive_file_paths_predicate(
    engine: pl.GPUEngine, evolved_hive: Path, query: Any
) -> None:
    q = pl.scan_parquet(
        evolved_hive,
        hive_partitioning=True,
        missing_columns="insert",
        include_file_paths="path",
    )
    assert_gpu_result_equal(query(q), engine=engine, check_row_order=False)


@pytest.fixture
def extra_columns(tmp_path: Path) -> Path:
    for name, frames in {
        "later": [{"a": [1, 2]}, {"a": [3], "c": [9]}],
        "first": [{"a": [1, 2], "c": [9, 9]}, {"a": [3]}],
    }.items():
        for index, frame in enumerate(frames):
            pl.LazyFrame(frame).sink_parquet(
                tmp_path / name / f"{index}.parquet", mkdir=True
            )
    pl.LazyFrame({"a": [1]}).sink_parquet(
        tmp_path / "hive" / "p=1" / "0.parquet", mkdir=True
    )
    pl.LazyFrame({"a": [2], "p": [2]}).sink_parquet(
        tmp_path / "hive" / "p=2" / "0.parquet", mkdir=True
    )
    return tmp_path


@pytest.mark.parametrize(
    "directory, kwargs, query",
    [
        ("first", {}, lambda q: q),
        ("hive", {"hive_partitioning": True}, lambda q: q),
    ],
    ids=["first_file", "hive_column"],
)
def test_scan_lake_extra_columns_expected(
    engine: pl.GPUEngine,
    extra_columns: Path,
    directory: str,
    kwargs: dict[str, Any],
    query: Any,
) -> None:
    q = pl.scan_parquet(extra_columns / directory, missing_columns="insert", **kwargs)
    assert_gpu_result_equal(query(q), engine=engine, check_row_order=False)


@pytest.mark.parametrize(
    "directory, kwargs, query",
    [
        ("later", {}, lambda q: q),
        ("later", {}, lambda q: q.select("a")),
        ("later", {}, lambda q: q.filter(pl.col("a") > 1)),
        ("first", {"schema": {"a": pl.Int64}}, lambda q: q),
    ],
    ids=["later_file", "projected", "filtered", "explicit_schema"],
)
def test_scan_lake_extra_columns_raise(
    engine: pl.GPUEngine,
    extra_columns: Path,
    directory: str,
    kwargs: dict[str, Any],
    query: Any,
) -> None:
    q = query(
        pl.scan_parquet(
            extra_columns / directory / "*.parquet", missing_columns="insert", **kwargs
        )
    )
    with pytest.raises(pl.exceptions.SchemaError):
        q.collect()
    if is_streaming_engine(engine):
        with pytest.RaisesGroup(pl.exceptions.SchemaError, flatten_subgroups=True):
            q.collect(engine=engine)
    else:
        with pytest.raises(pl.exceptions.SchemaError):
            q.collect(engine=engine)


@pytest.fixture
def default_files(tmp_path: Path) -> tuple[list[str], pa.Schema]:
    a = pa.field("a", pa.int64(), metadata={b"PARQUET:field_id": b"1"})
    d = pa.field("d", pa.int64(), metadata={b"PARQUET:field_id": b"2"})
    old = pa.schema([a])
    new = pa.schema([a, d])
    paths = [str(tmp_path / "0.parquet"), str(tmp_path / "1.parquet")]
    pq.write_table(pa.table({"a": [1, 2]}, schema=old), paths[0])
    pq.write_table(pa.table({"a": [3], "d": [30]}, schema=new), paths[1])
    return paths, new


@pytest.mark.parametrize(
    "query",
    [
        lambda q: q,
        lambda q: q.filter(pl.col("d") > 5),
        lambda q: q.select("d"),
    ],
    ids=["all", "filter", "select"],
)
def test_scan_iceberg_initial_defaults(
    engine: pl.GPUEngine,
    default_files: tuple[list[str], pa.Schema],
    query: Any,
) -> None:
    paths, schema = default_files
    q = pl.scan_parquet(
        paths,
        schema={"a": pl.Int64, "d": pl.Int64},
        missing_columns="insert",
        extra_columns="ignore",
        _column_mapping=("iceberg-column-mapping", schema),
        _default_values=("iceberg", ({}, {2: pl.Series([7], dtype=pl.Int64)})),
    )
    assert_gpu_result_equal(
        query(q),
        engine=engine,
        check_row_order=False,
    )


def test_scan_lake_missing_columns_raise(
    engine: pl.GPUEngine, default_files: tuple[list[str], pa.Schema]
) -> None:
    paths, schema = default_files
    q = pl.scan_parquet(
        paths,
        schema={"a": pl.Int64, "d": pl.Int64},
        missing_columns="raise",
        extra_columns="ignore",
        _column_mapping=("iceberg-column-mapping", schema),
    )
    with pytest.raises(pl.exceptions.ColumnNotFoundError):
        q.collect()
    if is_streaming_engine(engine):
        with pytest.RaisesGroup(
            pl.exceptions.ColumnNotFoundError, flatten_subgroups=True
        ):
            q.collect(engine=engine)
    else:
        with pytest.raises(pl.exceptions.ColumnNotFoundError):
            q.collect(engine=engine)


def _write_cast_files(
    tmp_path: Path, source: pl.DataType, target: pl.DataType
) -> list[str]:
    paths = [str(tmp_path / "0.parquet"), str(tmp_path / "1.parquet")]
    pl.DataFrame({"a": pl.Series([1, 2]).cast(source)}).write_parquet(paths[0])
    pl.DataFrame({"a": pl.Series([3]).cast(target)}).write_parquet(paths[1])
    return paths


@pytest.mark.parametrize(
    "source, target, cast_options",
    [
        (pl.Int32, pl.Int64, pl.ScanCastOptions(integer_cast="upcast")),
        (pl.UInt16, pl.Int32, pl.ScanCastOptions(integer_cast="upcast")),
        (pl.Int32, pl.Float64, pl.ScanCastOptions(integer_cast="allow-float")),
        (pl.Float32, pl.Float64, pl.ScanCastOptions(float_cast="upcast")),
        (pl.Float64, pl.Float32, pl.ScanCastOptions(float_cast="downcast")),
        (
            pl.Datetime("ns"),
            pl.Datetime("us"),
            pl.ScanCastOptions(datetime_cast="nanosecond-downcast"),
        ),
        (
            pl.Datetime("ms"),
            pl.Datetime("us"),
            pl.ScanCastOptions(datetime_cast="millisecond-upcast"),
        ),
    ],
    ids=[
        "int_upcast",
        "uint_to_wider_int",
        "int_to_float",
        "float_upcast",
        "float_downcast",
        "datetime_ns_downcast",
        "datetime_ms_upcast",
    ],
)
def test_scan_lake_cast_columns_allowed(
    engine: pl.GPUEngine,
    tmp_path: Path,
    source: pl.DataType,
    target: pl.DataType,
    cast_options: pl.ScanCastOptions,
) -> None:
    q = pl.scan_parquet(
        _write_cast_files(tmp_path, source, target),
        schema={"a": target},
        missing_columns="insert",
        cast_options=cast_options,
    )
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


@pytest.mark.parametrize(
    "source, target, cast_options",
    [
        (pl.Int32, pl.Int64, pl.ScanCastOptions()),
        (pl.Int64, pl.Int32, pl.ScanCastOptions(integer_cast="upcast")),
        (pl.Int32, pl.UInt64, pl.ScanCastOptions(integer_cast="upcast")),
        (pl.Int32, pl.Float64, pl.ScanCastOptions(integer_cast="upcast")),
        (pl.Float64, pl.Float32, pl.ScanCastOptions(float_cast="upcast")),
        (pl.Datetime("ns"), pl.Datetime("us"), pl.ScanCastOptions()),
        (
            pl.Datetime("us"),
            pl.Datetime("ms"),
            pl.ScanCastOptions(datetime_cast="upcast"),
        ),
    ],
    ids=[
        "int_forbid",
        "int_narrow",
        "int_to_unsigned",
        "int_to_float",
        "float_downcast",
        "datetime_forbid",
        "datetime_downcast",
    ],
)
def test_scan_lake_cast_columns_forbidden(
    engine: pl.GPUEngine,
    tmp_path: Path,
    source: pl.DataType,
    target: pl.DataType,
    cast_options: pl.ScanCastOptions,
) -> None:
    q = pl.scan_parquet(
        _write_cast_files(tmp_path, source, target),
        schema={"a": target},
        missing_columns="insert",
        cast_options=cast_options,
    )
    with pytest.raises(pl.exceptions.SchemaError):
        q.collect()
    if is_streaming_engine(engine):
        with pytest.RaisesGroup(pl.exceptions.SchemaError, flatten_subgroups=True):
            q.collect(engine=engine)
    else:
        with pytest.raises(pl.exceptions.SchemaError):
            q.collect(engine=engine)


def test_scan_iceberg_without_field_ids(
    in_memory_engine: pl.GPUEngine, tmp_path: Path
) -> None:
    path = str(tmp_path / "0.parquet")
    pq.write_table(pa.table({"a": [1, 2]}), path)
    schema = pa.schema(
        [pa.field("a", pa.int64(), metadata={b"PARQUET:field_id": b"1"})]
    )
    q = pl.scan_parquet(
        path,
        schema={"a": pl.Int64},
        missing_columns="insert",
        _column_mapping=("iceberg-column-mapping", schema),
    )
    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)


def test_scan_lake_partition_values_without_column_mapping(
    in_memory_engine: pl.GPUEngine,
    default_files: tuple[list[str], pa.Schema],
) -> None:
    paths, _ = default_files
    q = pl.scan_parquet(
        paths,
        _default_values=("iceberg", ({1: pl.Series([1, 2])}, {})),
    )
    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)


def _puffin(deleted: dict[str, list[int]]) -> bytes:
    # pyroaring a dependency of pyiceberg
    # Could be removed once https://github.com/apache/iceberg-python/issues/1551
    # is addressed.
    pyroaring = pytest.importorskip("pyroaring")
    magic = b"PFA1"
    body = bytearray(magic)
    blobs = []
    for data_file, positions in deleted.items():
        payload = (
            bytes.fromhex("d1d33964")
            + (1).to_bytes(8, "little")
            + (0).to_bytes(4, "little")
            + pyroaring.BitMap(positions).serialize()
        )
        blob = len(payload).to_bytes(4, "big") + payload + bytes(4)
        blobs.append(
            {
                "type": "deletion-vector-v1",
                "fields": [],
                "snapshot-id": -1,
                "sequence-number": -1,
                "offset": len(body),
                "length": len(blob),
                "properties": {"referenced-data-file": data_file},
            }
        )
        body += blob
    footer = json.dumps({"blobs": blobs}).encode()
    return bytes(
        body + magic + footer + len(footer).to_bytes(4, "little") + bytes(4) + magic
    )


@pytest.fixture(params=["position_deletes", "deletion_vectors", "delta"])
def deletion_files(request: pytest.FixtureRequest, tmp_path: Path) -> tuple[str, Any]:
    data = tmp_path / "data"
    paths = []
    for index in range(3):
        path = data / f"part={index}" / "data.parquet"
        pl.LazyFrame({"a": [4 * index + i for i in range(4)]}).sink_parquet(
            path, mkdir=True
        )
        paths.append(str(path))
    deleted = {0: [0, 3], 2: [1]}

    if request.param == "position_deletes":
        position_deletes = {}
        for index, positions in deleted.items():
            path = tmp_path / "deletes" / f"{index}.parquet"
            pl.LazyFrame({"file_path": [paths[index]] * len(positions)}).with_columns(
                pos=pl.Series(positions, dtype=pl.Int64)
            ).sink_parquet(path, mkdir=True)
            position_deletes[index] = [str(path)]
        return str(data), ("iceberg", (position_deletes, {}))

    if request.param == "deletion_vectors":
        puffin = tmp_path / "deletes.puffin"
        puffin.write_bytes(
            _puffin({paths[index]: positions for index, positions in deleted.items()})
        )
        return str(data), ("iceberg", ({}, dict.fromkeys(deleted, str(puffin))))

    selections = {
        paths[index]: [i not in positions for i in range(max(positions) + 1)]
        for index, positions in deleted.items()
    }

    def selection_vectors(requested: pl.DataFrame) -> pl.DataFrame:
        return pl.DataFrame(
            {"selection_vector": [selections.get(p) for p in requested["path"]]},
            schema={"selection_vector": pl.List(pl.Boolean)},
        )

    return str(data), ("delta-deletion-vector", selection_vectors)


@pytest.mark.parametrize(
    "hive_partitioning, query",
    [
        (False, lambda q: q.sort("a")),
        (False, lambda q: q.with_row_index("idx")),
        (False, lambda q: q.select(pl.len())),
        (True, lambda q: q.sort("a")),
    ],
    ids=["all", "row_index", "count", "hive"],
)
def test_scan_lake_deletions(
    engine: pl.GPUEngine,
    deletion_files: tuple[str, Any],
    hive_partitioning: bool,  # noqa: FBT001
    query: Any,
) -> None:
    data, deletions = deletion_files
    assert_gpu_result_equal(
        query(
            pl.scan_parquet(
                data, hive_partitioning=hive_partitioning, _deletion_files=deletions
            )
        ),
        engine=engine,
    )
