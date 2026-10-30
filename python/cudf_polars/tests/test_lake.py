# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import datetime
import json
import re
from typing import TYPE_CHECKING

import pytest

import polars as pl

from cudf_polars.engine.options import StreamingOptions
from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)
from cudf_polars.testing.engine_utils import is_streaming_engine
from cudf_polars.utils.versions import POLARS_VERSION_LT_21

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path
    from typing import Any, Literal

    from pyiceberg.table import Table
    from pyiceberg.table.update.schema import UpdateSchema

    from cudf_polars.engine.core import StreamingEngine

pytest.importorskip("pyiceberg")
deltalake = pytest.importorskip("deltalake")
import pyarrow as pa  # noqa: E402
import pyiceberg.schema  # noqa: E402
from pyiceberg.catalog.sql import SqlCatalog  # noqa: E402
from pyiceberg.partitioning import PartitionField, PartitionSpec  # noqa: E402
from pyiceberg.transforms import IdentityTransform  # noqa: E402
from pyiceberg.types import (  # noqa: E402
    DoubleType,
    FloatType,
    IntegerType,
    ListType,
    LongType,
    MapType,
    NestedField,
    StringType,
    StructType,
    TimestampType,
)


@pytest.fixture
def iceberg_catalog(tmp_path: Path) -> Iterator[SqlCatalog]:
    with SqlCatalog(
        "default",
        uri=f"sqlite:///{tmp_path}/catalog.db",
        warehouse=f"file://{tmp_path / 'warehouse'}",
    ) as catalog:
        catalog.create_namespace_if_not_exists("ns")
        yield catalog


@pytest.fixture
def flat_iceberg(iceberg_catalog: SqlCatalog) -> str:
    first = pl.LazyFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    tbl = iceberg_catalog.create_table(
        "ns.flat",
        schema=pyiceberg.schema.Schema(
            NestedField(1, "a", LongType(), required=False),
            NestedField(2, "b", StringType(), required=False),
        ),
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
        "ns.evolved",
        schema=pyiceberg.schema.Schema(
            NestedField(1, "a", IntegerType(), required=False),
            NestedField(2, "b", StringType(), required=False),
        ),
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
        "ns.renamed",
        schema=pyiceberg.schema.Schema(
            NestedField(1, "a", LongType(), required=False),
            NestedField(2, "b", StringType(), required=False),
        ),
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
        "ns.promoted",
        schema=pyiceberg.schema.Schema(
            NestedField(1, "a", IntegerType(), required=False),
            NestedField(2, "b", StringType(), required=False),
            NestedField(3, "c", FloatType(), required=False),
        ),
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
def copy_on_write_delta(tmp_path: Path) -> str:
    path = tmp_path / "delta_copy_on_write"
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
        "copy_on_write_delta",
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
        ("copy_on_write_delta", pl.col("a") > 4),
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
    "query",
    [
        lambda q: q.filter(pl.col("b_renamed") == "y").with_row_index("idx"),
        lambda q: q.filter(pl.col("b_renamed") == "y").head(1),
        lambda q: q.with_row_index("idx").filter(pl.col("b_renamed") == "y"),
        lambda q: q.head(4).filter(pl.col("b_renamed") == "y"),
    ],
    ids=[
        "filter_then_row_index",
        "filter_then_slice",
        "row_index_then_filter",
        "slice_then_filter",
    ],
)
def test_scan_lake_predicate_pushdown(
    in_memory_engine: pl.GPUEngine, evolved_iceberg: str, query: Any
) -> None:
    assert_gpu_result_equal(
        query(pl.scan_iceberg(evolved_iceberg)),
        engine=in_memory_engine,
        check_row_order=False,
    )


def test_scan_iceberg_partition_predicate_without_file_columns(
    engine: pl.GPUEngine, partitioned_iceberg: str
) -> None:
    q = pl.scan_iceberg(partitioned_iceberg).filter(pl.col("part") != "w")
    assert_gpu_result_equal(q.select("part"), engine=engine, check_row_order=False)


@pytest.mark.parametrize(
    "query",
    [
        lambda q: q.filter(pl.col("a") > 2),
        lambda q: q.filter(pl.col("b") > 30),
        lambda q: q.filter(pl.col("b") > 30).select("b"),
        lambda q: q.filter(pl.col("a").is_in([1, 4, 6])).select("part"),
        lambda q: q.filter((pl.col("a") < 5) & (pl.col("part") != "v")),
        lambda q: q.select("part"),
    ],
    ids=["data", "added", "added_only", "partition_only", "mixed", "partitions"],
)
def test_scan_delta_evolved_partitions_predicate(
    engine: pl.GPUEngine, tmp_path: Path, query: Any
) -> None:
    path = str(tmp_path / "delta")
    pl.LazyFrame({"a": [1, 2, 3, 4], "part": ["u", "u", "v", "v"]}).sink_delta(
        path, mode="overwrite", delta_write_options={"partition_by": ["part"]}
    )
    pl.LazyFrame({"a": [5, 6], "b": [50, 60], "part": ["w", "w"]}).sink_delta(
        path, mode="append", delta_write_options={"schema_mode": "merge"}
    )
    assert_gpu_result_equal(
        query(pl.scan_delta(path)), engine=engine, check_row_order=False
    )


@pytest.mark.parametrize(
    "first, second",
    [
        (
            (
                {"s": [{"x": 1, "y": 2}, None]},
                {"s": pl.Struct({"x": pl.Int64, "y": pl.Int64})},
            ),
            (
                {"s": [{"y": 4, "x": 3}]},
                {"s": pl.Struct({"y": pl.Int64, "x": pl.Int64})},
            ),
        ),
        (
            ({"s": [{"x": 1}, None]}, {"s": pl.Struct({"x": pl.Int64})}),
            (
                {"s": [{"x": 3, "t": {"u": "p"}, "l": [1, 2]}]},
                {
                    "s": pl.Struct(
                        {
                            "x": pl.Int64,
                            "t": pl.Struct({"u": pl.String}),
                            "l": pl.List(pl.Int64),
                        }
                    )
                },
            ),
        ),
        (
            ({"a": [1]}, {"a": pl.Int64}),
            (
                {"a": [2], "s": [{"x": 3}]},
                {"a": pl.Int64, "s": pl.Struct({"x": pl.Int64})},
            ),
        ),
        (
            (
                {"l": [[{"x": 1, "y": 2}], None, []]},
                {"l": pl.List(pl.Struct({"x": pl.Int64, "y": pl.Int64}))},
            ),
            (
                {"l": [[{"y": 4, "x": 3}]]},
                {"l": pl.List(pl.Struct({"y": pl.Int64, "x": pl.Int64}))},
            ),
        ),
    ],
    ids=[
        "reordered_fields",
        "added_nested_fields",
        "added_struct_column",
        "list_of_structs",
    ],
)
def test_scan_delta_nested_schema_evolution(
    engine: pl.GPUEngine,
    tmp_path: Path,
    first: tuple[dict[str, Any], dict[str, pl.DataType]],
    second: tuple[dict[str, Any], dict[str, pl.DataType]],
) -> None:
    path = str(tmp_path / "delta")
    pl.LazyFrame(*first).sink_delta(path, mode="overwrite")
    pl.LazyFrame(*second).sink_delta(
        path, mode="append", delta_write_options={"schema_mode": "merge"}
    )
    assert_gpu_result_equal(pl.scan_delta(path), engine=engine, check_row_order=False)


def test_scan_iceberg_same_table_twice(engine: pl.GPUEngine, flat_iceberg: str) -> None:
    q = pl.concat([pl.scan_iceberg(flat_iceberg), pl.scan_iceberg(flat_iceberg)])
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


def test_scan_iceberg_readded_column(
    engine: pl.GPUEngine, iceberg_catalog: SqlCatalog
) -> None:
    tbl = iceberg_catalog.create_table(
        "ns.readded",
        schema=pyiceberg.schema.Schema(
            NestedField(1, "a", LongType(), required=False),
            NestedField(2, "b", LongType(), required=False),
        ),
    )
    pl.DataFrame([{"a": 1, "b": 10}]).write_iceberg(tbl, mode="append")
    with tbl.update_schema() as update:
        update.delete_column("a")
    with tbl.update_schema() as update:
        update.add_column("a", LongType())
    pl.DataFrame([{"a": 2, "b": 20}]).write_iceberg(tbl, mode="append")
    assert_gpu_result_equal(
        pl.scan_iceberg(tbl.metadata_location), engine=engine, check_row_order=False
    )


def test_scan_iceberg_without_field_ids(
    in_memory_engine: pl.GPUEngine, iceberg_catalog: SqlCatalog, tmp_path: Path
) -> None:
    tbl = iceberg_catalog.create_table(
        "ns.added_files",
        schema=pyiceberg.schema.Schema(NestedField(1, "a", LongType(), required=False)),
    )
    pl.DataFrame([{"a": 1}]).write_iceberg(tbl, mode="append")
    path = str(tmp_path / "without_field_ids.parquet")
    pl.DataFrame({"a": [2]}).write_parquet(path)
    tbl.add_files([path])
    q = pl.scan_iceberg(tbl.metadata_location)
    with pytest.raises(
        pl.exceptions.ComputeError,
        match=re.escape(
            "NotImplementedError: Iceberg data file without parquet field IDs"
        ),
    ):
        q.collect(engine=in_memory_engine)


@pytest.mark.parametrize(
    "kind, first, evolutions, second",
    [
        (
            "struct",
            [{"s": {"x": 1, "y": 2}}, {"s": None}],
            (lambda update: update.rename_column("s.y", "z"),),
            [{"s": {"x": 3, "z": 4}}],
        ),
        (
            "struct",
            [{"s": {"x": 1, "y": 2}}, {"s": None}],
            (
                lambda update: update.rename_column("s.x", "tmp"),
                lambda update: update.rename_column("s.y", "x"),
                lambda update: update.rename_column("s.tmp", "y"),
            ),
            [{"s": {"y": 3, "x": 4}}],
        ),
        (
            "struct",
            [{"s": {"x": 1, "y": 2}}, {"s": None}],
            (
                lambda update: update.delete_column("s.y"),
                lambda update: update.add_column(("s", "y"), LongType()),
            ),
            [{"s": {"x": 3, "y": 4}}],
        ),
        (
            "list",
            [{"s": [{"u": 1}]}, {"s": None}, {"s": []}],
            (lambda update: update.rename_column("s.element.u", "v"),),
            [{"s": [{"v": 2}, None]}],
        ),
        (
            "string_list",
            [{"s": {"p": ["a", None]}}, {"s": None}],
            (lambda update: update.rename_column("s.p", "q"),),
            [{"s": {"q": []}}, {"s": {"q": ["b"]}}, {"s": {"q": None}}],
        ),
    ],
    ids=[
        "renamed_field",
        "swapped_names",
        "readded_field",
        "list_element_field",
        "string_list_field",
    ],
)
def test_scan_iceberg_nested_by_field_id(
    engine: pl.GPUEngine,
    iceberg_catalog: SqlCatalog,
    kind: Literal["struct", "list", "string_list"],
    first: list[dict[str, Any]],
    evolutions: tuple[Callable[[UpdateSchema], Any], ...],
    second: list[dict[str, Any]],
) -> None:
    if kind == "struct":
        column: StructType | ListType = StructType(
            NestedField(2, "x", LongType(), required=False),
            NestedField(3, "y", LongType(), required=False),
        )
    elif kind == "list":
        column = ListType(
            2,
            StructType(NestedField(3, "u", LongType(), required=False)),
            element_required=False,
        )
    else:
        column = StructType(
            NestedField(
                2,
                "p",
                ListType(3, StringType(), element_required=False),
                required=False,
            )
        )
    tbl = iceberg_catalog.create_table(
        "ns.nested",
        schema=pyiceberg.schema.Schema(NestedField(1, "s", column, required=False)),
    )
    pl.DataFrame(first).write_iceberg(tbl, mode="append")
    for evolve in evolutions:
        with tbl.update_schema() as update:
            evolve(update)
    pl.DataFrame(second).write_iceberg(tbl, mode="append")
    assert_gpu_result_equal(
        pl.scan_iceberg(tbl.metadata_location), engine=engine, check_row_order=False
    )


def test_scan_iceberg_nested_by_field_id_filter(
    engine: pl.GPUEngine, iceberg_catalog: SqlCatalog
) -> None:
    tbl = iceberg_catalog.create_table(
        "ns.nested",
        schema=pyiceberg.schema.Schema(
            NestedField(
                1,
                "s",
                StructType(
                    NestedField(2, "x", LongType(), required=False),
                    NestedField(3, "y", LongType(), required=False),
                ),
                required=False,
            )
        ),
    )
    pl.DataFrame([{"s": {"x": 1, "y": 2}}, {"s": None}]).write_iceberg(
        tbl, mode="append"
    )
    with tbl.update_schema() as update:
        update.rename_column("s.y", "z")
    pl.DataFrame([{"s": {"x": 3, "z": 4}}]).write_iceberg(tbl, mode="append")
    q = pl.scan_iceberg(tbl.metadata_location).filter(pl.col("s").struct.field("z") > 1)
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


@pytest.mark.parametrize(
    "column, projection, match",
    [
        ("map", "t", "Iceberg column mapping of map columns"),
        ("map_in_struct", "t", "Iceberg column mapping of map columns"),
        ("nested_partition", "s", "Iceberg partition values for nested fields"),
    ],
)
def test_scan_iceberg_nested_unsupported(
    in_memory_engine: pl.GPUEngine,
    iceberg_catalog: SqlCatalog,
    column: Literal["map", "map_in_struct", "nested_partition"],
    projection: str,
    match: str,
) -> None:
    if column == "map":
        field_type: MapType | StructType = MapType(
            2, StringType(), 3, LongType(), value_required=False
        )
        row: dict[str, Any] = {"t": 1}
    elif column == "map_in_struct":
        field_type = StructType(
            NestedField(
                2,
                "m",
                MapType(5, StringType(), 6, LongType(), value_required=False),
                required=False,
            )
        )
        row = {"t": 1}
    else:
        field_type = StructType(NestedField(2, "x", LongType(), required=False))
        row = {"s": {"x": 1}, "t": 1}
    tbl = iceberg_catalog.create_table(
        "ns.unsupported",
        schema=pyiceberg.schema.Schema(
            NestedField(1, "s", field_type, required=False),
            NestedField(4, "t", LongType(), required=False),
        ),
    )
    if column == "nested_partition":
        with tbl.update_spec() as update:
            update.add_identity("s.x")
    pl.DataFrame([row]).write_iceberg(tbl, mode="append")
    q = pl.scan_iceberg(tbl.metadata_location).select(projection)
    with pytest.raises(
        pl.exceptions.ComputeError, match=f"NotImplementedError: {re.escape(match)}"
    ):
        q.collect(engine=in_memory_engine)


def _partition_error_scan(
    iceberg_catalog: SqlCatalog, *, file_without_field: bool
) -> pl.LazyFrame:
    tbl = iceberg_catalog.create_table(
        "ns.partition_error",
        schema=pyiceberg.schema.Schema(NestedField(1, "a", LongType(), required=False)),
    )
    if file_without_field:
        pl.DataFrame([{"a": 1}]).write_iceberg(tbl, mode="append")
    with tbl.update_schema() as update:
        update.add_column("d", LongType())
    pl.DataFrame({"a": [2, 3], "d": [20, 30]}).write_iceberg(tbl, mode="append")
    return _scan_iceberg_files(
        tbl,
        [task.file.file_path for task in tbl.scan().plan_files()],
        _default_values=(
            "iceberg",
            ({2: "unsupported type change: from: string, to: binary"}, {}),
        ),
    )


def test_scan_iceberg_partition_error_reads_files(
    engine: pl.GPUEngine, iceberg_catalog: SqlCatalog
) -> None:
    q = _partition_error_scan(iceberg_catalog, file_without_field=False)
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


def test_scan_iceberg_partition_error_missing_field(
    engine: pl.GPUEngine, iceberg_catalog: SqlCatalog
) -> None:
    q = _partition_error_scan(iceberg_catalog, file_without_field=True)
    match = "unsupported type change"
    with pytest.raises(pl.exceptions.ComputeError, match=match):
        q.collect()
    if is_streaming_engine(engine):
        with pytest.RaisesGroup(
            pytest.RaisesExc(pl.exceptions.ComputeError, match=match),
            flatten_subgroups=True,
        ):
            q.collect(engine=engine)
    else:
        with pytest.raises(pl.exceptions.ComputeError, match=match):
            q.collect(engine=engine)


def test_scan_iceberg_partition_error_sliced(
    in_memory_engine: pl.GPUEngine, iceberg_catalog: SqlCatalog
) -> None:
    q = _partition_error_scan(iceberg_catalog, file_without_field=True).head(1)
    with pytest.raises(
        pl.exceptions.ComputeError,
        match=re.escape(
            "NotImplementedError: "
            "Sliced Iceberg scan with partition values polars cannot supply"
        ),
    ):
        q.collect(engine=in_memory_engine)


def _scan_iceberg_files(tbl: Table, paths: list[str], **kwargs: Any) -> pl.LazyFrame:
    # pyiceberg cannot write V3 tables or position deletes, so build the scan
    # that pl.scan_iceberg builds for such a table.
    mapping = tbl.schema().as_arrow()
    return pl.scan_parquet(
        paths,
        schema=pl.DataFrame(mapping.empty_table()).schema,
        missing_columns="insert",
        extra_columns="ignore",
        cast_options=pl.ScanCastOptions._default_iceberg(),
        _column_mapping=("iceberg-column-mapping", mapping),
        **kwargs,
    )


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
    engine: pl.GPUEngine, iceberg_catalog: SqlCatalog, query: Any
) -> None:
    tbl = iceberg_catalog.create_table(
        "ns.defaults",
        schema=pyiceberg.schema.Schema(NestedField(1, "a", LongType(), required=False)),
    )
    pl.DataFrame({"a": [1, 2]}).write_iceberg(tbl, mode="append")
    with tbl.update_schema() as update:
        update.add_column("d", LongType())
    pl.DataFrame({"a": [3], "d": [30]}).write_iceberg(tbl, mode="append")
    q = _scan_iceberg_files(
        tbl,
        [task.file.file_path for task in tbl.scan().plan_files()],
        _default_values=("iceberg", ({}, {2: pl.Series([7], dtype=pl.Int64)})),
    )
    assert_gpu_result_equal(query(q), engine=engine, check_row_order=False)


@pytest.mark.skipif(
    POLARS_VERSION_LT_21,
    reason="Polars 2.0 does not upcast millisecond timestamps in Iceberg scans",
)
def test_scan_iceberg_millisecond_timestamps(
    engine: pl.GPUEngine, iceberg_catalog: SqlCatalog, tmp_path: Path
) -> None:
    tbl = iceberg_catalog.create_table(
        "ns.timestamps",
        schema=pyiceberg.schema.Schema(
            NestedField(1, "t", TimestampType(), required=False)
        ),
    )
    path = str(tmp_path / "milliseconds.parquet")
    pl.DataFrame(
        {"t": [datetime.datetime(2026, 1, 1, 0, 0, 0, 123000), None]},
        schema={"t": pl.Datetime("ms")},
    ).write_parquet(
        path,
        arrow_schema=pa.schema(
            [pa.field("t", pa.timestamp("ms"), metadata={"PARQUET:field_id": "1"})]
        ),
    )
    assert_gpu_result_equal(_scan_iceberg_files(tbl, [path]), engine=engine)


def test_scan_iceberg_nested_initial_default_unsupported(
    in_memory_engine: pl.GPUEngine, iceberg_catalog: SqlCatalog
) -> None:
    tbl = iceberg_catalog.create_table(
        "ns.nested_defaults",
        schema=pyiceberg.schema.Schema(
            NestedField(
                1,
                "s",
                StructType(NestedField(2, "x", LongType(), required=False)),
                required=False,
            )
        ),
    )
    pl.DataFrame({"s": [{"x": 1}]}).write_iceberg(tbl, mode="append")
    q = _scan_iceberg_files(
        tbl,
        [task.file.file_path for task in tbl.scan().plan_files()],
        _default_values=("iceberg", ({}, {2: pl.Series([1])})),
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


@pytest.fixture(params=["within_files", "past_file_ends"])
def deleted_positions(request: pytest.FixtureRequest) -> dict[int, list[int]]:
    if request.param == "within_files":
        return {0: [0, 3], 2: [1]}
    return {0: [0, 100], 1: [4], 2: [1]}


@pytest.fixture(
    params=["position_deletes", "deletion_vectors", "delta", "delta_partitioned"]
)
def deleted_rows_scan(
    request: pytest.FixtureRequest,
    tmp_path: Path,
    iceberg_catalog: SqlCatalog,
    deleted_positions: dict[int, list[int]],
) -> pl.LazyFrame:
    frames = [
        pl.DataFrame({"a": [4 * index + i for i in range(4)]}) for index in range(3)
    ]
    if request.param in ("position_deletes", "deletion_vectors"):
        tbl = iceberg_catalog.create_table(
            "ns.deleted",
            schema=pyiceberg.schema.Schema(
                NestedField(1, "a", LongType(), required=False)
            ),
        )
        for frame in frames:
            frame.write_iceberg(tbl, mode="append")
        paths = [task.file.file_path for task in tbl.scan().plan_files()]
        if request.param == "position_deletes":
            position_deletes = {}
            for index, positions in deleted_positions.items():
                path = tmp_path / "deletes" / f"{index}.parquet"
                pl.LazyFrame(
                    {"file_path": [paths[index]] * len(positions)}
                ).with_columns(pos=pl.Series(positions, dtype=pl.Int64)).sink_parquet(
                    path, mkdir=True
                )
                position_deletes[index] = [str(path)]
            return _scan_iceberg_files(
                tbl, paths, _deletion_files=("iceberg", (position_deletes, {}))
            )
        puffin = tmp_path / "deletes.puffin"
        puffin.write_bytes(
            _puffin(
                {
                    paths[index]: positions
                    for index, positions in deleted_positions.items()
                }
            )
        )
        return _scan_iceberg_files(
            tbl,
            paths,
            _deletion_files=(
                "iceberg",
                ({}, dict.fromkeys(deleted_positions, str(puffin))),
            ),
        )
    paths = []
    for index, frame in enumerate(frames):
        path = tmp_path / "data" / f"part={index}" / "data.parquet"
        frame.write_parquet(path, mkdir=True)
        paths.append(str(path))
    selections = {
        paths[index]: [i not in positions for i in range(max(positions) + 1)]
        for index, positions in deleted_positions.items()
    }

    def selection_vectors(requested: pl.DataFrame) -> pl.DataFrame:
        return pl.DataFrame(
            {"selection_vector": [selections.get(p) for p in requested["path"]]},
            schema={"selection_vector": pl.List(pl.Boolean)},
        )

    partitioned = request.param == "delta_partitioned"
    # deltalake cannot write deletion vectors, so build the scan that
    # pl.scan_delta builds for such a table.
    return pl.scan_parquet(
        paths,
        missing_columns="insert",
        extra_columns="ignore",
        cast_options=pl.ScanCastOptions._default_iceberg(),
        hive_partitioning=partitioned,
        hive_schema={"part": pl.Int64} if partitioned else None,
        _deletion_files=("delta-deletion-vector", selection_vectors),
    )


@pytest.mark.parametrize(
    "query",
    [
        lambda q: q.sort("a"),
        lambda q: q.with_row_index("idx"),
        lambda q: q.select(pl.len()),
        lambda q: q.filter(pl.col("a") > 2).sort("a"),
        lambda q: q.filter(pl.col("a") > 2).head(3),
    ],
    ids=["all", "row_index", "count", "filter", "filter_then_head"],
)
def test_scan_lake_deletions(
    engine: pl.GPUEngine, deleted_rows_scan: pl.LazyFrame, query: Any
) -> None:
    assert_gpu_result_equal(query(deleted_rows_scan), engine=engine)


@pytest.mark.parametrize(
    "query",
    [
        lambda q: q.head(5),
        lambda q: q.slice(2, 5),
        lambda q: q.head(6).filter(pl.col("a") > 2),
    ],
    ids=["head", "slice", "head_then_filter"],
)
def test_scan_lake_deletions_sliced(
    request: pytest.FixtureRequest,
    engine: pl.GPUEngine,
    deleted_rows_scan: pl.LazyFrame,
    query: Any,
) -> None:
    if request.node.callspec.params["deleted_positions"] == "past_file_ends":
        request.applymarker(
            pytest.mark.xfail(
                reason="https://github.com/pola-rs/polars/issues/30407",
                strict=True,
            )
        )
    assert_gpu_result_equal(query(deleted_rows_scan), engine=engine)


@pytest.mark.parametrize(
    "query",
    [
        lambda q: q.sort("a"),
        lambda q: q.select(pl.len()),
        lambda q: q.filter(pl.col("a") > 2).sort("a"),
    ],
    ids=["all", "count", "filter"],
)
def test_scan_lake_deletions_partition_per_file(
    streaming_engine_factory: Callable[[StreamingOptions], StreamingEngine],
    deleted_rows_scan: pl.LazyFrame,
    query: Any,
) -> None:
    engine = streaming_engine_factory(StreamingOptions(target_partition_size=1))
    assert_gpu_result_equal(query(deleted_rows_scan), engine=engine)


@pytest.mark.parametrize(
    "mapped", [(0, 1, 2), (1, 2)], ids=["some_missing", "all_missing"]
)
def test_scan_iceberg_deletion_vector_missing_from_puffin(
    engine: pl.GPUEngine,
    iceberg_catalog: SqlCatalog,
    tmp_path: Path,
    mapped: tuple[int, ...],
) -> None:
    tbl = iceberg_catalog.create_table(
        "ns.deleted",
        schema=pyiceberg.schema.Schema(NestedField(1, "a", LongType(), required=False)),
    )
    for index in range(3):
        pl.DataFrame({"a": [4 * index + i for i in range(4)]}).write_iceberg(
            tbl, mode="append"
        )
    paths = [task.file.file_path for task in tbl.scan().plan_files()]
    puffin = tmp_path / "deletes.puffin"
    puffin.write_bytes(_puffin({paths[0]: [0, 3]}))
    q = _scan_iceberg_files(
        tbl,
        paths,
        _deletion_files=("iceberg", ({}, dict.fromkeys(mapped, str(puffin)))),
    )
    if POLARS_VERSION_LT_21:
        assert_gpu_result_equal(q, engine=engine, check_row_order=False)
        return
    match = "has no deletion vector for data file"
    with pytest.raises(pl.exceptions.ComputeError, match=match):
        q.collect()
    if is_streaming_engine(engine):
        with pytest.RaisesGroup(
            pytest.RaisesExc(pl.exceptions.ComputeError, match=match),
            flatten_subgroups=True,
        ):
            q.collect(engine=engine)
    else:
        with pytest.raises(pl.exceptions.ComputeError, match=match):
            q.collect(engine=engine)


def test_scan_iceberg_deletion_vector_file_uri(
    engine: pl.GPUEngine, iceberg_catalog: SqlCatalog, tmp_path: Path
) -> None:
    tbl = iceberg_catalog.create_table(
        "ns.deleted",
        schema=pyiceberg.schema.Schema(NestedField(1, "a", LongType(), required=False)),
    )
    for index in range(2):
        pl.DataFrame({"a": [4 * index + i for i in range(4)]}).write_iceberg(
            tbl, mode="append"
        )
    paths = [task.file.file_path for task in tbl.scan().plan_files()]
    puffin = tmp_path / "deletes.puffin"
    puffin.write_bytes(_puffin({paths[0]: [0, 3], paths[1]: [1]}))
    q = _scan_iceberg_files(
        tbl,
        paths,
        _deletion_files=("iceberg", ({}, dict.fromkeys((0, 1), f"file://{puffin}"))),
    )
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


def test_scan_iceberg_remote_deletion_vector_unsupported(
    in_memory_engine: pl.GPUEngine, iceberg_catalog: SqlCatalog
) -> None:
    tbl = iceberg_catalog.create_table(
        "ns.deleted",
        schema=pyiceberg.schema.Schema(NestedField(1, "a", LongType(), required=False)),
    )
    pl.DataFrame({"a": [1, 2]}).write_iceberg(tbl, mode="append")
    q = _scan_iceberg_files(
        tbl,
        [task.file.file_path for task in tbl.scan().plan_files()],
        _deletion_files=("iceberg", ({}, {0: "s3://bucket/deletes.puffin"})),
    )
    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)


@pytest.mark.parametrize(
    "single_file_positions", [(), (2,)], ids=["alone", "after_single_file"]
)
def test_scan_iceberg_position_delete_file_references_many_data_files(
    engine: pl.GPUEngine,
    iceberg_catalog: SqlCatalog,
    tmp_path: Path,
    single_file_positions: tuple[int, ...],
) -> None:
    tbl = iceberg_catalog.create_table(
        "ns.deleted",
        schema=pyiceberg.schema.Schema(NestedField(1, "a", LongType(), required=False)),
    )
    for index in range(2):
        pl.DataFrame({"a": [4 * index + i for i in range(4)]}).write_iceberg(
            tbl, mode="append"
        )
    paths = [task.file.file_path for task in tbl.scan().plan_files()]
    delete_files = []
    if single_file_positions:
        delete_files.append(str(tmp_path / "single.parquet"))
        pl.DataFrame(
            {
                "file_path": [paths[0]] * len(single_file_positions),
                "pos": pl.Series(single_file_positions, dtype=pl.Int64),
            }
        ).write_parquet(delete_files[-1])
    delete_files.append(str(tmp_path / "many.parquet"))
    pl.DataFrame(
        {"file_path": paths, "pos": pl.Series([0, 1], dtype=pl.Int64)}
    ).write_parquet(delete_files[-1])
    q = _scan_iceberg_files(
        tbl, paths, _deletion_files=("iceberg", ({0: delete_files}, {}))
    )
    match = "references more than one data file"
    if is_streaming_engine(engine):
        with pytest.RaisesGroup(
            pytest.RaisesExc(pl.exceptions.ComputeError, match=match),
            flatten_subgroups=True,
        ):
            q.collect(engine=engine)
    else:
        with pytest.raises(pl.exceptions.ComputeError, match=match):
            q.collect(engine=engine)
