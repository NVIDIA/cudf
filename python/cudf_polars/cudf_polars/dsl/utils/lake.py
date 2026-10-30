# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Options that ``scan_iceberg`` and ``scan_delta`` attach to a parquet scan."""

from __future__ import annotations

import dataclasses
import itertools
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, cast

import polars as pl

import pylibcudf as plc

from cudf_polars.containers import Column, DataFrame
from cudf_polars.dsl import expr
from cudf_polars.dsl.to_ast import to_parquet_filter
from cudf_polars.dsl.traversal import traversal
from cudf_polars.dsl.utils.replace import replace

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from rmm.pylibrmm.stream import Stream

    from cudf_polars.containers import DataType
    from cudf_polars.dsl.utils.io import CachedParquetInfo
    from cudf_polars.typing import Schema

    _ElementLayout = tuple[
        str, plc.io.parquet_metadata.PhysicalType, int, int, tuple[Any, ...] | None
    ]
    _Layout = tuple[_ElementLayout, ...]
    _FieldIds = tuple[int | None, tuple["_FieldIds", ...]]

__all__ = ["IcebergColumn", "LakehouseScanOptions", "read_lake_files"]


@dataclasses.dataclass(frozen=True, slots=True)
class IcebergColumn:
    """
    A column of an Iceberg schema, identified by its physical field ID.

    Parameters
    ----------
    name
        Name the column has in the table schema, which may differ from the
        name recorded in any given data file.
    physical_id
        Field ID written into the parquet footer for this column.
    kind
        One of ``"primitive"``, ``"list"``, ``"map"`` or ``"struct"``.
    children
        Fields of a nested column, empty for a primitive column.
    """

    name: str
    physical_id: int
    kind: str
    children: tuple[IcebergColumn, ...]
    # Polars 2.1 also upcasts millisecond timestamps to microseconds
    _LAKE_CAST_POLICIES: ClassVar[tuple[Mapping[str, bool | str], ...]] = tuple(
        MappingProxyType(
            {
                "integer_upcast": True,
                "integer_to_float_cast": False,
                "float_upcast": True,
                "float_downcast": True,
                "datetime_nanoseconds_downcast": True,
                "datetime_microseconds_downcast": False,
                "datetime_milliseconds_upcast": milliseconds_upcast,
                "datetime_microseconds_upcast": False,
                "datetime_convert_timezone": True,
                "null_upcast": True,
                "categorical_to_string": True,
                "missing_struct_fields": "insert",
                "extra_struct_fields": "ignore",
            }
        )
        for milliseconds_upcast in (False, True)
    )

    @classmethod
    def from_polars(cls, spec: tuple[str, int, tuple[Any, ...]]) -> IcebergColumn:
        """
        Build from the nested tuples polars uses for an Iceberg column.

        Parameters
        ----------
        spec
            ``(name, physical_id, type_)`` where ``type_`` is one of:
                * ``("primitive", dtype)``
                * ``("list", column)``
                * ``("map", key, value)``
                * ``("struct", {physical_id: column})``

        Returns
        -------
        The parsed column.
        """
        name, physical_id, type_ = spec
        kind, *rest = type_
        if kind == "primitive":
            children: tuple[IcebergColumn, ...] = ()
        elif kind == "struct":
            children = tuple(cls.from_polars(child) for child in rest[0].values())
        elif kind in ("list", "map"):
            children = tuple(cls.from_polars(child) for child in rest)
        else:  # pragma: no cover; Iceberg schemas only produce the kinds above
            raise NotImplementedError(f"Unhandled Iceberg column type {kind!r}")
        return cls(name=name, physical_id=physical_id, kind=kind, children=children)

    def walk(self) -> Iterable[IcebergColumn]:
        """Yield this column and every column nested within it."""
        yield self
        for child in self.children:
            yield from child.walk()


@dataclasses.dataclass(frozen=True)
class LakehouseScanOptions:
    """
    The Iceberg and Delta Lake specific parts of a polars ``Scan``.

    Parameters
    ----------
    columns
        Top-level columns of the Iceberg schema, in table order, or ``None``
        for a Delta scan, which reads columns by name.
    position_deletes
        Maps the index of a path in the scan to the Iceberg position delete
        files that apply to it.
    deletion_vectors
        Maps the index of a path in the scan to the Iceberg puffin file
        holding its deletion vector.
    delta_selections
        Maps the index of a path in the scan to the Delta deletion vector
        that applies to it, as a boolean series that is true for the rows
        to keep.
    partition_values
        Maps a physical field ID to the value its Iceberg identity-transformed
        partition field takes, one row per path.
    partition_errors
        Maps a physical field ID to the reason polars could not supply the
        values of its Iceberg identity-transformed partition field. Such a
        field is read from the files, and a file that lacks it fails the scan.
    initial_defaults
        Maps a physical field ID to the single-element series holding the
        Iceberg V3 initial default for a column absent from a file.
    row_count
        ``(total_rows, deleted_rows)`` for the whole scan when the Iceberg
        table metadata could supply it.

    Notes
    -----
    ``scan_delta`` only supplies ``delta_selections``. Every other field
    comes from ``scan_iceberg``. A Delta scan gets its partition values from
    the hive partitions of the scan instead.


    A file may lack columns of the schema, which are filled with their
    initial default or with nulls, and may hold columns or struct fields
    the schema does not, which are ignored. Iceberg and Delta scans always
    read files this way.
    """

    columns: tuple[IcebergColumn, ...] | None
    position_deletes: Mapping[int, tuple[str, ...]]
    deletion_vectors: Mapping[int, str]
    delta_selections: Mapping[int, pl.Series]
    partition_values: Mapping[int, pl.Series]
    partition_errors: Mapping[int, str]
    initial_defaults: Mapping[int, pl.Series]
    row_count: tuple[int, int] | None

    def __post_init__(self) -> None:  # noqa: D105
        if any(
            nested.kind == "map"
            for column in self.columns or ()
            for nested in column.walk()
        ):
            raise NotImplementedError("Iceberg column mapping of map columns")
        top_level_ids = {column.physical_id for column in self.columns or ()}
        unknown_defaults = set(self.initial_defaults) - top_level_ids
        if unknown_defaults:
            raise NotImplementedError(
                f"Iceberg defaults for nested or unmapped fields {sorted(unknown_defaults)}"
            )
        unknown_partitions = (
            set(self.partition_values).union(self.partition_errors) - top_level_ids
        )
        if unknown_partitions:
            raise NotImplementedError(
                f"Iceberg partition values for nested fields {sorted(unknown_partitions)}"
            )

    @classmethod
    def from_file_options(
        cls, file_options: Any, paths: Sequence[str]
    ) -> LakehouseScanOptions | None:
        """
        Build from the ``file_options`` of a polars ``Scan`` node.

        Parameters
        ----------
        file_options
            The ``PyFileOptions`` of the scan.
        paths
            Data files of the scan, named as polars names them.

        Returns
        -------
        The parsed options, or ``None`` for a scan that is not a lake scan.

        Raises
        ------
        NotImplementedError
            If polars supplies a variant we do not understand, or the scan
            inserts missing columns but resolves the schema differently from
            an Iceberg or Delta scan. Failing here keeps us from silently
            ignoring information that changes the result. Also raised for
            Iceberg deletion vectors in remote storage, which cannot be read.

        Notes
        -----
        A Delta scan carries no column mapping, deletions or defaults, but a
        table that has evolved still has files that are missing a column of
        the schema, which a plain parquet read cannot express. Such a scan is
        recognised by the policies polars gives every Iceberg and Delta scan
        for missing columns, extra columns and casts.

        A Delta deletion vector arrives as a callback holding a handle to the
        table, which cannot be sent to another process, so it is resolved
        here rather than in the tasks that read the files.
        """
        is_lake_scan = (
            file_options.column_mapping is not None
            or file_options.deletion_files is not None
            or file_options.default_values is not None
        )
        if not is_lake_scan and file_options.missing_columns_policy == "raise":
            return None
        if (
            file_options.missing_columns_policy != "insert"
            or file_options.extra_columns_policy != "ignore"
            or file_options.cast_columns_policy not in IcebergColumn._LAKE_CAST_POLICIES
        ):
            raise NotImplementedError(
                "Inserting missing columns is only supported for Iceberg and Delta scans"
            )
        column_mapping = file_options.column_mapping
        columns: tuple[IcebergColumn, ...] | None = None
        if column_mapping is not None:
            mapping_kind, mapping = column_mapping
            if (
                mapping_kind != "iceberg-column-mapping"
            ):  # pragma: no cover; only kind polars emits
                raise NotImplementedError(f"Unhandled column mapping {mapping_kind!r}")
            columns = tuple(
                IcebergColumn.from_polars(spec) for spec in mapping.values()
            )

        position_deletes: dict[int, tuple[str, ...]] = {}
        deletion_vectors: dict[int, str] = {}
        delta_selections: dict[int, pl.Series] = {}
        if file_options.deletion_files is not None:
            deletion_kind, deletion_payload = file_options.deletion_files
            if deletion_kind == "iceberg":
                raw_position_deletes, raw_deletion_vectors = deletion_payload
                position_deletes = {
                    index: tuple(delete_files)
                    for index, delete_files in raw_position_deletes.items()
                }
                deletion_vectors = {
                    index: puffin.removeprefix("file://")
                    for index, puffin in raw_deletion_vectors.items()
                }
                if any(
                    plc.io.SourceInfo._is_remote_uri(puffin)
                    for puffin in deletion_vectors.values()
                ):
                    raise NotImplementedError(
                        "Iceberg deletion vectors in remote storage"
                    )
            elif deletion_kind == "delta-deletion-vector":
                frame = deletion_payload(pl.DataFrame({"path": list(paths)}))
                if (
                    frame is None
                ):  # pragma: no cover; the polars callback returns a frame
                    delta_selections = {}
                else:
                    delta_selections = {
                        index: selection
                        for index, selection in enumerate(
                            frame.get_column("selection_vector")
                        )
                        if selection is not None
                    }
            else:
                raise NotImplementedError(  # pragma: no cover; only kinds polars emits
                    f"Unhandled deletion files {deletion_kind!r}"
                )

        partition_values: dict[int, pl.Series] = {}
        partition_errors: dict[int, str] = {}
        initial_defaults: dict[int, pl.Series] = {}
        if file_options.default_values is not None:
            defaults_kind, defaults_payload = file_options.default_values
            if defaults_kind != "iceberg":  # pragma: no cover; only kind polars emits
                raise NotImplementedError(f"Unhandled default values {defaults_kind!r}")
            partition_fields, raw_initial_defaults = defaults_payload
            for physical_id, value in partition_fields.items():
                if isinstance(value, str):
                    partition_errors[physical_id] = value
                else:
                    partition_values[physical_id] = pl.Series._from_pyseries(value)
            initial_defaults = {
                physical_id: pl.Series._from_pyseries(value)
                for physical_id, value in raw_initial_defaults.items()
            }
        return cls(
            columns=columns,
            position_deletes=position_deletes,
            deletion_vectors=deletion_vectors,
            delta_selections=delta_selections,
            partition_values=partition_values,
            partition_errors=partition_errors,
            initial_defaults=initial_defaults,
            row_count=file_options.row_count,
        )

    def check_field_ids(self, paths: Sequence[str]) -> None:
        """
        Check that every data file can be read by field ID.

        Parameters
        ----------
        paths
            Data files of the scan.

        Raises
        ------
        NotImplementedError
            If the scan reads columns by field ID and any of ``paths`` holds
            no parquet field IDs.
        """
        if self.columns is None:
            return
        metadatas = plc.io.parquet_metadata.read_parquet_footers(
            plc.io.SourceInfo(list(paths))
        )
        for path, metadata in zip(paths, metadatas, strict=True):
            _footer(path, metadata, by_field_id=True)

    @property
    def has_deletions(self) -> bool:
        """Whether any rows have to be dropped after reading."""
        return bool(
            self.position_deletes or self.deletion_vectors or self.delta_selections
        )

    def slice(self, start: int, stop: int) -> LakehouseScanOptions:
        """
        Restrict to the paths in ``range(start, stop)``.

        Deletions and partition values are keyed by the position of a path
        within the scan, so both have to be re-keyed when a task takes only
        part of the paths.

        Parameters
        ----------
        start
            Index of the first path to keep.
        stop
            Index one past the last path to keep.

        Returns
        -------
        Options covering the selected paths.
        """
        return dataclasses.replace(
            self,
            position_deletes={
                i - start: v
                for i, v in self.position_deletes.items()
                if start <= i < stop
            },
            deletion_vectors={
                i - start: v
                for i, v in self.deletion_vectors.items()
                if start <= i < stop
            },
            delta_selections={
                i - start: v
                for i, v in self.delta_selections.items()
                if start <= i < stop
            },
            partition_values={
                physical_id: series.slice(start, stop - start)
                for physical_id, series in self.partition_values.items()
            },
            row_count=None,
        )

    def __hash__(self) -> int:
        """Hash of every option."""
        return hash(
            (
                self.columns,
                tuple(sorted(self.position_deletes.items())),
                tuple(sorted(self.deletion_vectors.items())),
                tuple(sorted(self.delta_selections)),
                tuple(sorted(self.partition_values)),
                tuple(sorted(self.partition_errors.items())),
                tuple(sorted(self.initial_defaults)),
                self.row_count,
            )
        )

    def __eq__(self, other: Any) -> bool:
        """Whether two sets of options agree."""
        return (
            isinstance(other, LakehouseScanOptions)
            and self.columns == other.columns
            and self.position_deletes == other.position_deletes
            and self.deletion_vectors == other.deletion_vectors
            and _series_mapping_eq(self.delta_selections, other.delta_selections)
            and _series_mapping_eq(self.partition_values, other.partition_values)
            and self.partition_errors == other.partition_errors
            and _series_mapping_eq(self.initial_defaults, other.initial_defaults)
            and self.row_count == other.row_count
        )


def _series_mapping_eq(
    left: Mapping[int, pl.Series], right: Mapping[int, pl.Series]
) -> bool:
    return left.keys() == right.keys() and all(
        left_series.equals(right_series)
        for left_series, right_series in zip(left.values(), right.values(), strict=True)
    )


def read_lake_files(
    paths: Sequence[str],
    lake_options: LakehouseScanOptions,
    schema: Schema,
    with_columns: Sequence[str] | None,
    cached_parquet_info: Sequence[CachedParquetInfo] | None = None,
    predicate: expr.Expr | None = None,
    *,
    with_source_index: bool = False,
    stream: Stream,
) -> tuple[DataFrame, list[int], plc.Column | None, bool]:
    """
    Read the data files of an Iceberg or Delta scan onto the table schema.

    Parameters
    ----------
    paths
        Data files of the scan, in scan order.
    lake_options
        Options of the scan.
    schema
        Schema the scan must produce.
    with_columns
        Columns to project, or ``None`` for all of them.
    cached_parquet_info
        Footers already read for ``paths``, in the same order, or ``None`` to
        read them here.
    predicate
        Predicate, in terms of the table schema, to filter the files with
        while reading them, or ``None`` to read every row.
    with_source_index
        Whether to return the index into ``paths`` of the file each row was
        read from when ``predicate`` is given.
    stream
        CUDA stream used for device memory operations and kernel launches.

    Returns
    -------
    tuple of:
        - The concatenated frame.
        - The number of rows each path holds before ``predicate`` is applied.
        - The source index of each row, or ``None`` if ``with_source_index`` is ``False``.
        - Whether ``predicate`` was applied exactly.
    """
    by_field_id = lake_options.columns is not None
    projected = (
        frozenset(schema)
        if with_columns is None
        else frozenset(schema).intersection(with_columns)
    )
    wanted: dict[Any, str]
    if lake_options.columns is not None:
        wanted = {
            column.physical_id: column.name
            for column in lake_options.columns
            if column.name in projected
            and column.physical_id not in lake_options.partition_values
        }
    else:
        wanted = {name: name for name in schema if name in projected}

    metadatas: Sequence[plc.io.parquet_metadata.FileMetaData | None] = (
        [None] * len(paths)
        if cached_parquet_info is None
        else [info.file_metadata for info in cached_parquet_info]
    )
    footers = [
        _footer(path, metadata, by_field_id=by_field_id)
        for path, metadata in zip(paths, metadatas, strict=True)
    ]
    rows_per_path = [footer.num_rows for footer in footers]
    predicate_columns = (
        {
            node.name: node
            for node in traversal([predicate])
            if isinstance(node, expr.Col)
        }
        if predicate is not None
        else {}
    )
    track_source = predicate is not None and with_source_index
    frames: list[DataFrame] = []
    source_indices: list[plc.Column] = []
    exact = True
    start = 0
    for (keys, _), group in itertools.groupby(
        footers, key=lambda footer: footer.group_key(wanted)
    ):
        stop = start + len(list(group))
        present = frozenset(keys)
        read_keys = [key for key in wanted if key in present]
        filters = None
        if predicate is not None:
            filters, group_exact = _group_filter(
                predicate,
                predicate_columns,
                paths[start],
                footers[start],
                read_keys,
                wanted,
                schema,
                stream=stream,
            )
            exact = exact and group_exact
        frame, source_index = _read_group(
            paths[start:stop],
            read_keys,
            footers[start],
            rows_per_path[start:stop],
            wanted,
            schema,
            lake_options,
            filters,
            start if track_source else None,
            by_field_id=by_field_id,
            stream=stream,
        )
        frames.append(frame)
        if source_index is not None:
            source_indices.append(source_index)
        start = stop

    source_index = (
        None
        if not track_source
        else source_indices[0]
        if len(source_indices) == 1
        else plc.concatenate.concatenate(source_indices, stream=stream)
    )
    if len(frames) == 1:
        return frames[0], rows_per_path, source_index, exact
    names = list(frames[0].column_map)
    if not names:
        return (
            DataFrame(
                [], num_rows=sum(frame.num_rows for frame in frames), stream=stream
            ),
            rows_per_path,
            source_index,
            exact,
        )
    return (
        DataFrame.from_table(
            plc.concatenate.concatenate(
                [
                    plc.Table([frame.column_map[name].obj for name in names])
                    for frame in frames
                ],
                stream=stream,
            ),
            names,
            [schema[name] for name in names],
            stream=stream,
        ),
        rows_per_path,
        source_index,
        exact,
    )


@dataclasses.dataclass(frozen=True, slots=True)
class _Footer:
    """What the parquet footer of one data file says about its columns."""

    names: tuple[str, ...]
    keys: tuple[Any, ...]
    signatures: tuple[_Layout, ...]
    field_ids: tuple[_FieldIds | None, ...]
    num_rows: int

    def group_key(
        self, wanted: Mapping[Any, str]
    ) -> tuple[tuple[Any, ...], tuple[tuple[_Layout, _FieldIds | None], ...]]:
        """
        Key on which consecutive files can be read together.

        Parameters
        ----------
        wanted
            Maps the key of each column the scan must produce to its name in
            the table schema.

        Returns
        -------
        The top-level keys of the file, and the names, physical and logical
        types and field IDs of each wanted column it holds and of its nested
        fields. The parquet reader requires the selected columns of its
        sources to agree on their types and on the names of their nested
        fields. Files that agree on the field IDs as well can be read by
        name even when the scan is keyed by field ID.
        """
        return self.keys, tuple(
            (signature, field_ids)
            for key, signature, field_ids in zip(
                self.keys, self.signatures, self.field_ids, strict=True
            )
            if key in wanted
        )


def _element_layout(
    element: plc.io.parquet_metadata.SchemaElement,
) -> _ElementLayout:
    """Name, physical and logical type of a parquet schema element."""
    logical_type = element.logical_type
    if logical_type is None:
        return (
            element.name,
            element.type,
            element.type_length,
            element.num_children,
            None,
        )
    decimal_type = logical_type.decimal_type
    temporal_type = (
        logical_type.time_type
        if logical_type.time_type is not None
        else logical_type.timestamp_type
    )
    int_type = logical_type.int_type
    return (
        element.name,
        element.type,
        element.type_length,
        element.num_children,
        (
            logical_type.type,
            None
            if decimal_type is None
            else (decimal_type.scale, decimal_type.precision),
            None if temporal_type is None else temporal_type.unit,
            None if int_type is None else (int_type.bit_width, int_type.is_signed),
        ),
    )


def _footer(
    path: str,
    metadata: plc.io.parquet_metadata.FileMetaData | None,
    *,
    by_field_id: bool,
) -> _Footer:
    """
    Top-level columns and row count of a parquet file.

    Raises
    ------
    NotImplementedError
        If ``by_field_id`` and a field that polars identifies by its field ID
        has none, or a list uses a legacy layout that the parquet reader and
        polars read differently.
    """
    if metadata is None:
        metadata = plc.io.parquet_metadata.read_parquet_footers(
            plc.io.SourceInfo([path])
        )[0]
    schema = metadata.schema

    def nested_field_ids(index: int, *, list_parent: str | None) -> _FieldIds:
        element = schema[index]
        if element.is_stub():
            if list_parent is not None and element.name in (
                "array",
                f"{list_parent}_tuple",
            ):  # pragma: no cover; pyarrow cannot write this layout
                raise NotImplementedError(
                    f"Iceberg data file with a legacy list layout: {path}"
                )
            return nested_field_ids(element.children_idx[0], list_parent=list_parent)
        if element.field_id is None and list_parent is None:
            raise NotImplementedError(
                f"Iceberg data file without parquet field IDs: {path}"
            )
        logical_type = element.logical_type
        is_list = logical_type is not None and logical_type.type in (
            plc.io.parquet_metadata.LogicalTypeId.LIST,
            plc.io.parquet_metadata.LogicalTypeId.MAP,
        )
        children = tuple(
            nested_field_ids(child, list_parent=element.name if is_list else None)
            for child in element.children_idx
        )
        return element.field_id, children

    def subtree(index: int) -> Iterable[plc.io.parquet_metadata.SchemaElement]:
        element = schema[index]
        yield element
        for child in element.children_idx:
            yield from subtree(child)

    top_level = schema[0].children_idx
    names = tuple(schema[index].name for index in top_level)
    signatures = tuple(
        tuple(_element_layout(element) for element in subtree(index))
        for index in top_level
    )
    if not by_field_id:
        return _Footer(
            names=names,
            keys=names,
            signatures=signatures,
            field_ids=(None,) * len(top_level),
            num_rows=metadata.num_rows,
        )
    field_ids = tuple(nested_field_ids(index, list_parent=None) for index in top_level)
    return _Footer(
        names=names,
        keys=tuple(field_id for field_id, _ in field_ids),
        signatures=signatures,
        field_ids=field_ids,
        num_rows=metadata.num_rows,
    )


def _group_filter(
    predicate: expr.Expr,
    predicate_columns: Mapping[str, expr.Col],
    path: str,
    footer: _Footer,
    read_keys: Sequence[Any],
    wanted: Mapping[Any, str],
    schema: Schema,
    *,
    stream: Stream,
) -> tuple[plc.expressions.Expression | None, bool]:
    """
    Translate a predicate into a parquet filter for one run of files.

    Parameters
    ----------
    predicate
        Predicate in terms of the table schema.
    predicate_columns
        Maps the name of each column ``predicate`` refers to onto its
        ``Col`` node in ``predicate``.
    path
        First file of the run, whose metadata gives the column types of the
        whole run. It is only read when ``predicate`` refers to a column in
        ``read_keys``.
    footer
        Footer of ``path``, used to map keys to the names the files hold.
    read_keys
        Keys of the wanted columns that the files hold: field IDs for an
        Iceberg scan read by field ID, otherwise column names.
    wanted
        Maps the key of each column the scan must produce to its name in the
        table schema.
    schema
        Schema the scan must produce.
    stream
        CUDA stream used for device memory operations and kernel launches.

    Returns
    -------
    filters
        Parquet filter in terms of the names the files hold, or ``None`` when
        no part of ``predicate`` can be applied while reading. The scalars
        it holds are created on ``stream``, and its data is valid on
        ``stream``.
    exact
        Whether ``filters`` is equivalent to ``predicate``. When it is not,
        the rows read are a superset of the rows that satisfy ``predicate``,
        which must be applied again after reading.
    """
    name_by_key = dict(zip(footer.keys, footer.names, strict=True))
    file_names = {
        wanted[key]: name_by_key[key]
        for key in read_keys
        if wanted[key] in predicate_columns
    }
    file_types = (
        plc.io.parquet_metadata.read_parquet_metadata(plc.io.SourceInfo([path]))
        .schema()
        .column_types()
        if file_names
        else {}
    )
    readable = {
        name: file_name
        for name, file_name in file_names.items()
        if schema[name].id() not in _NESTED_TYPES
        and file_types.get(file_name) == schema[name].plc_type
    }
    renames: dict[expr.Expr, expr.Expr] = {
        predicate_columns[name]: expr.Col(predicate_columns[name].dtype, file_name)
        for name, file_name in readable.items()
        if file_name != name
    }
    (renamed,) = replace([predicate], renames)
    filters, residual = to_parquet_filter(
        renamed,
        stream=stream,
        unreadable_columns=frozenset(predicate_columns).difference(readable),
    )
    return filters, filters is not None and residual is None


_NESTED_TYPES = frozenset({plc.TypeId.STRUCT, plc.TypeId.LIST})


def _cast_to_table_type(
    column: plc.Column,
    children: Sequence[plc.io.types.ColumnNameSpec],
    target: DataType,
    mapping: tuple[_FieldIds, IcebergColumn] | None,
    *,
    stream: Stream,
) -> plc.Column:
    """
    Cast a column read from a file to its type in the table schema.

    Parameters
    ----------
    column
        Column as read from the file.
    children
        Names of the children of ``column``, as the parquet reader
        reports them.
    target
        Type of the column in the table schema.
    mapping
        For a scan read by field ID, the field IDs of ``column`` and its
        nested fields in the file, and the Iceberg column that ``target``
        was built from. ``None`` to match nested fields by name.
    stream
        CUDA stream used for device memory operations and kernel launches.

    Returns
    -------
    The column with the type of ``target``. Struct fields are matched by
    field ID when ``mapping`` is given and by name otherwise, put in the
    order of ``target`` and cast field by field. Fields of ``target`` that
    ``column`` lacks are filled with nulls, and fields of ``column`` that
    ``target`` lacks are dropped.
    """
    if target.id() not in _NESTED_TYPES:
        return Column(column, dtype=target).astype(target, stream).obj
    if target.id() == plc.TypeId.LIST:
        (_, (_, element_children)) = children
        element_mapping = None
        if mapping is not None:
            (_, file_elements), iceberg_column = mapping
            (iceberg_element,) = iceberg_column.children
            file_element = file_elements[0] if file_elements else (None, ())
            element_mapping = (file_element, iceberg_element)
        new_children = [
            column.child(0),
            _cast_to_table_type(
                column.child(1),
                element_children,
                target.children[0],
                element_mapping,
                stream=stream,
            ),
        ]
    else:
        file_fields: Sequence[_FieldIds | None]
        incoming_keys: list[Any]
        target_fields: list[tuple[Any, IcebergColumn | None]]
        if mapping is None:
            file_fields = [None] * len(children)
            incoming_keys = [field_name for field_name, _ in children]
            target_fields = [
                (field.name, None)
                for field in cast("pl.Struct", target.polars_type).fields
            ]
        else:
            (_, file_fields), iceberg_column = mapping
            incoming_keys = [field_id for field_id, _ in file_fields]
            target_fields = [
                (iceberg_field.physical_id, iceberg_field)
                for iceberg_field in iceberg_column.children
            ]
        position = {key: index for index, key in enumerate(incoming_keys)}
        new_children = []
        for field_dtype, (key, iceberg_field) in zip(
            target.children, target_fields, strict=True
        ):
            index = position.get(key)
            if index is None:
                new_children.append(
                    _null_column(
                        field_dtype, column.offset() + column.size(), stream=stream
                    )
                )
                continue
            _, field_children = children[index]
            file_field = file_fields[index]
            new_children.append(
                _cast_to_table_type(
                    column.child(index),
                    field_children,
                    field_dtype,
                    None
                    if file_field is None or iceberg_field is None
                    else (file_field, iceberg_field),
                    stream=stream,
                )
            )
    return plc.Column(
        column.type(),
        column.size(),
        None,
        column.null_mask(),
        column.null_count(),
        column.offset(),
        new_children,
    )


def _null_column(dtype: DataType, size: int, *, stream: Stream) -> plc.Column:
    """An all-null column of any type, including nested types."""
    type_id = dtype.id()
    if type_id not in _NESTED_TYPES:
        return plc.Column.from_scalar(
            plc.Scalar.from_py(None, dtype.plc_type, stream=stream),
            size,
            stream=stream,
        )
    if type_id == plc.TypeId.LIST:
        children = [
            plc.Column.from_scalar(
                plc.Scalar.from_py(0, plc.DataType(plc.TypeId.INT32), stream=stream),
                size + 1,
                stream=stream,
            ),
            _null_column(dtype.children[0], 0, stream=stream),
        ]
    else:
        children = [
            _null_column(child, size, stream=stream) for child in dtype.children
        ]
    return plc.Column(
        dtype.plc_type,
        size,
        None,
        plc.null_mask.create_null_mask(
            size, plc.types.MaskState.ALL_NULL, stream=stream
        ),
        size,
        0,
        children,
    )


def _read_group(
    paths: Sequence[str],
    present: Sequence[Any],
    footer: _Footer,
    rows_per_path: Sequence[int],
    wanted: Mapping[Any, str],
    schema: Schema,
    lake_options: LakehouseScanOptions,
    filters: plc.expressions.Expression | None,
    source_offset: int | None,
    *,
    by_field_id: bool,
    stream: Stream,
) -> tuple[DataFrame, plc.Column | None]:
    """
    Read one run of same-shaped files and map it onto the table schema.

    Parameters
    ----------
    paths
        Data files of the run, in scan order. They hold the same top-level
        columns, and the wanted ones have the same types and the same names
        and field IDs throughout, so the run is read by column name.
    present
        Keys of the wanted columns that the files hold: field IDs when
        ``by_field_id``, otherwise column names. The files are only read
        when this is non-empty.
    footer
        Footer of the first file of the run, used to map the column names
        the reader returns back to their keys.
    rows_per_path
        Number of rows each file holds before ``filters`` is applied.
    wanted
        Maps the key of each column the scan must produce to its name in the
        table schema, in output order.
    schema
        Schema the scan must produce.
    lake_options
        Options of the scan, giving the defaults for columns the files do
        not hold.
    filters
        Parquet filter to apply while reading, in terms of the names the
        files hold, or ``None`` to read every row.
    source_offset
        Index within the scan of the first file of the run, or ``None`` when
        the source of each row is not needed.
    by_field_id
        Whether ``present`` and ``wanted`` are keyed by Iceberg field ID
        rather than by column name.
    stream
        CUDA stream used for device memory operations and kernel launches.

    Returns
    -------
    frame
        The rows read, with the columns of ``wanted`` renamed and cast to
        the table schema. Columns the files do not hold are filled with
        their initial default, or with nulls.
    source_index
        For each row of ``frame``, the index within the scan of the file it
        was read from, or ``None`` when ``source_offset`` is ``None``. Its
        data is valid on ``stream``.
    """
    read: dict[Any, tuple[plc.Column, Sequence[plc.io.types.ColumnNameSpec]]] = {}
    num_rows = sum(rows_per_path)
    source_index: plc.Column | None = None
    if present:
        options = (
            plc.io.parquet.ParquetReaderOptions.builder(plc.io.SourceInfo(list(paths)))
            .allow_mismatched_pq_schemas(val=True)
            .build()
        )
        name_by_key = dict(zip(footer.keys, footer.names, strict=True))
        options.set_column_names([name_by_key[key] for key in present])
        if filters is not None:
            options.set_filter(filters)
        if source_offset is not None:
            options.enable_prepend_source_index_column(val=True)
        table = plc.io.parquet.read_parquet(options, stream=stream)
        num_rows = table.tbl.num_rows()
        names = table.column_names(include_children=True)
        columns = table.tbl.columns()
        if source_offset is not None:
            source_index = (
                columns[0]
                if source_offset == 0
                else plc.binaryop.binary_operation(
                    columns[0],
                    plc.Scalar.from_py(
                        source_offset, plc.DataType(plc.TypeId.INT32), stream=stream
                    ),
                    plc.binaryop.BinaryOperator.ADD,
                    plc.DataType(plc.TypeId.INT32),
                    stream=stream,
                )
            )
            names = names[1:]
            columns = columns[1:]
        key_by_name = dict(zip(footer.names, footer.keys, strict=True))
        read = {
            key_by_name[name]: (column, children)
            for (name, children), column in zip(names, columns, strict=True)
        }
    elif source_offset is not None:
        (source_index,) = plc.filling.repeat(
            plc.Table(
                [
                    plc.filling.sequence(
                        len(rows_per_path),
                        plc.Scalar.from_py(
                            source_offset, plc.DataType(plc.TypeId.INT32), stream=stream
                        ),
                        plc.Scalar.from_py(
                            1, plc.DataType(plc.TypeId.INT32), stream=stream
                        ),
                        stream=stream,
                    )
                ]
            ),
            plc.Column.from_arrow(
                pl.Series(values=rows_per_path, dtype=pl.Int32()), stream=stream
            ),
            stream=stream,
        ).columns()

    file_field_ids = dict(zip(footer.keys, footer.field_ids, strict=True))
    iceberg_columns = {
        column.physical_id: column for column in lake_options.columns or ()
    }
    frame_columns = []
    for key, name in wanted.items():
        dtype = schema[name]
        if key in read:
            column, children = read[key]
            field_ids = file_field_ids[key]
            obj = _cast_to_table_type(
                column,
                children,
                dtype,
                None if field_ids is None else (field_ids, iceberg_columns[key]),
                stream=stream,
            )
            frame_columns.append(Column(obj, name=name, dtype=dtype))
        elif by_field_id and key in lake_options.partition_errors:
            raise pl.exceptions.ComputeError(
                "failed to resolve Iceberg column mapping: "
                "error loading identity transform value from metadata for "
                f"missing field: {lake_options.partition_errors[key]}"
            )
        else:
            default = lake_options.initial_defaults.get(key) if by_field_id else None
            if default is None:
                obj = _null_column(dtype, num_rows, stream=stream)
                frame_columns.append(Column(obj, name=name, dtype=dtype))
            else:
                (obj,) = plc.filling.repeat(
                    plc.Table([plc.Column.from_arrow(default, stream=stream)]),
                    num_rows,
                    stream=stream,
                ).columns()
                frame_columns.append(
                    Column(obj, name=name, dtype=dtype).astype(dtype, stream)
                )
    if not frame_columns:
        return DataFrame([], num_rows=num_rows, stream=stream), source_index
    return DataFrame(frame_columns, stream=stream), source_index
