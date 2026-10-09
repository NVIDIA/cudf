# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Options that ``scan_iceberg`` and ``scan_delta`` attach to a parquet scan."""

from __future__ import annotations

import dataclasses
import itertools
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

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
        plc.io.parquet_metadata.PhysicalType, int, int, tuple[Any, ...] | None
    ]
    _Layout = tuple[_ElementLayout, ...]

__all__ = ["IcebergColumn", "LakeScanOptions", "read_lake_files"]

_INTEGER_TYPES = MappingProxyType(
    {
        plc.TypeId.INT8: (8, True),
        plc.TypeId.INT16: (16, True),
        plc.TypeId.INT32: (32, True),
        plc.TypeId.INT64: (64, True),
        plc.TypeId.UINT8: (8, False),
        plc.TypeId.UINT16: (16, False),
        plc.TypeId.UINT32: (32, False),
        plc.TypeId.UINT64: (64, False),
    }
)
_FLOAT_WIDTHS = MappingProxyType({plc.TypeId.FLOAT32: 32, plc.TypeId.FLOAT64: 64})
_TIMESTAMP_DIGITS = MappingProxyType(
    {
        plc.TypeId.TIMESTAMP_MILLISECONDS: 3,
        plc.TypeId.TIMESTAMP_MICROSECONDS: 6,
        plc.TypeId.TIMESTAMP_NANOSECONDS: 9,
    }
)


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
    children
        Fields of a nested column, empty for a primitive column.
    """

    name: str
    physical_id: int
    children: tuple[IcebergColumn, ...]

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
                * ``("fixed-size-list", column, width)``
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
        elif kind in ("list", "fixed-size-list", "map"):
            children = tuple(
                cls.from_polars(child) for child in rest if isinstance(child, tuple)
            )
        else:  # pragma: no cover; polars only produces the kinds above
            raise NotImplementedError(f"Unhandled Iceberg column type {kind!r}")
        return cls(name=name, physical_id=physical_id, children=children)


@dataclasses.dataclass(frozen=True)
class LakeScanOptions:
    """
    The Iceberg and Delta Lake specific parts of a polars ``Scan``.

    Parameters
    ----------
    columns
        Top-level columns of the Iceberg schema, in table order, or ``None``
        when the scan reads columns by name (Delta, and Iceberg tables that
        never evolved).
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
        Maps a physical field ID to the value its identity-transformed
        partition field takes, one row per path.
    initial_defaults
        Maps a physical field ID to the single-element series holding the
        Iceberg V3 initial default for a column absent from a file.
    missing_columns_policy
        ``"insert"`` to fill columns absent from a file, ``"raise"`` to fail.
    extra_columns_policy
        ``"ignore"`` to drop columns a file has but the schema does not,
        ``"raise"`` to fail.
    cast_columns_policy
        Which casts from the physical to the table type are permitted.
    row_count
        ``(total_rows, deleted_rows)`` for the whole scan when the table
        metadata could supply it.
    expected_columns
        Names a data file may hold when ``extra_columns_policy`` is
        ``"raise"``, or ``None`` to allow any.
    """

    columns: tuple[IcebergColumn, ...] | None
    position_deletes: Mapping[int, tuple[str, ...]]
    deletion_vectors: Mapping[int, str]
    delta_selections: Mapping[int, pl.Series]
    partition_values: Mapping[int, pl.Series]
    initial_defaults: Mapping[int, pl.Series]
    missing_columns_policy: str
    extra_columns_policy: str
    cast_columns_policy: Mapping[str, Any]
    row_count: tuple[int, int] | None
    expected_columns: frozenset[str] | None = None

    def __post_init__(self) -> None:  # noqa: D105
        if self.columns is not None and any(column.children for column in self.columns):
            raise NotImplementedError("Iceberg column mapping of nested columns")
        if self.partition_values and self.columns is None:
            raise NotImplementedError("Iceberg partition values without column mapping")
        if self.extra_columns_policy not in ("ignore", "raise"):
            raise NotImplementedError(  # pragma: no cover; only two policies exist
                f"Extra columns policy {self.extra_columns_policy!r}"
            )
        if self.missing_columns_policy not in ("insert", "raise"):
            raise NotImplementedError(  # pragma: no cover; only two policies exist
                f"Missing columns policy {self.missing_columns_policy!r}"
            )
        unknown_defaults = set(self.initial_defaults) - {
            column.physical_id for column in self.columns or ()
        }
        if unknown_defaults:  # pragma: no cover; polars keys defaults by schema id
            raise NotImplementedError(
                f"Iceberg defaults for unmapped fields {sorted(unknown_defaults)}"
            )

    @classmethod
    def from_file_options(
        cls, file_options: Any, paths: Sequence[str]
    ) -> LakeScanOptions | None:
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
            If polars supplies a variant we do not understand. Failing here
            keeps us from silently ignoring information that changes the
            result.

        Notes
        -----
        A Delta scan carries no column mapping, deletions or defaults, but a
        table that has evolved still has files that are missing a column of
        the schema, which a plain parquet read cannot express. Such a scan is
        recognised by its missing columns policy.

        A Delta deletion vector arrives as a callback holding a handle to the
        table, which cannot be sent to another process, so it is resolved
        here rather than in the tasks that read the files.
        """
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
                deletion_vectors = dict(raw_deletion_vectors)
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
        initial_defaults: dict[int, pl.Series] = {}
        if file_options.default_values is not None:
            defaults_kind, defaults_payload = file_options.default_values
            if defaults_kind != "iceberg":  # pragma: no cover; only kind polars emits
                raise NotImplementedError(f"Unhandled default values {defaults_kind!r}")
            partition_fields, raw_initial_defaults = defaults_payload
            for physical_id, value in partition_fields.items():
                if isinstance(value, str):
                    raise NotImplementedError(
                        f"Iceberg partition field {physical_id}: {value}"
                    )
                partition_values[physical_id] = pl.Series._from_pyseries(value)
            initial_defaults = {
                physical_id: pl.Series._from_pyseries(value)
                for physical_id, value in raw_initial_defaults.items()
            }
        if (
            columns is None
            and not position_deletes
            and not deletion_vectors
            and not delta_selections
            and not partition_values
            and not initial_defaults
            and file_options.missing_columns_policy == "raise"
        ):
            return None
        return cls(
            columns=columns,
            position_deletes=position_deletes,
            deletion_vectors=deletion_vectors,
            delta_selections=delta_selections,
            partition_values=partition_values,
            initial_defaults=initial_defaults,
            missing_columns_policy=file_options.missing_columns_policy,
            extra_columns_policy=file_options.extra_columns_policy,
            cast_columns_policy=file_options.cast_columns_policy,
            row_count=file_options.row_count,
        )

    def expect_columns(
        self,
        reader_schema: Mapping[str, Any] | None,
        first_path: str,
        hive_names: Iterable[str],
    ) -> LakeScanOptions:
        """
        Record the names a data file may hold.

        Parameters
        ----------
        reader_schema
            Schema given to the scan, as polars serializes it, or ``None``
            when the scan infers its schema from the first file.
        first_path
            First data file of the scan.
        hive_names
            Names of the hive partition columns, which a file may also hold.

        Returns
        -------
        Options whose ``expected_columns`` are set.
        """
        names = (
            reader_schema["fields"]
            if reader_schema is not None
            else _footer(first_path, None, by_field_id=False).names
        )
        return dataclasses.replace(
            self, expected_columns=frozenset(names).union(hive_names)
        )

    def check_field_ids(self, first_path: str) -> None:
        """
        Check that the first data file can be read by field ID.

        Parameters
        ----------
        first_path
            First data file of the scan.

        Raises
        ------
        NotImplementedError
            If the scan reads columns by field ID and ``first_path`` holds
            no parquet field IDs. Other files are checked when they are read.
        """
        if self.columns is not None:
            _footer(first_path, None, by_field_id=True)

    @property
    def has_deletions(self) -> bool:
        """Whether any rows have to be dropped after reading."""
        return bool(
            self.position_deletes or self.deletion_vectors or self.delta_selections
        )

    def slice(self, start: int, stop: int) -> LakeScanOptions:
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
                tuple(sorted(self.initial_defaults)),
                self.missing_columns_policy,
                self.extra_columns_policy,
                self.row_count,
                self.expected_columns,
            )
        )

    def __eq__(self, other: Any) -> bool:
        """Whether two sets of options agree."""
        return (
            isinstance(other, LakeScanOptions)
            and self.columns == other.columns
            and self.position_deletes == other.position_deletes
            and self.deletion_vectors == other.deletion_vectors
            and _series_mapping_eq(self.delta_selections, other.delta_selections)
            and _series_mapping_eq(self.partition_values, other.partition_values)
            and _series_mapping_eq(self.initial_defaults, other.initial_defaults)
            and self.missing_columns_policy == other.missing_columns_policy
            and self.extra_columns_policy == other.extra_columns_policy
            and self.cast_columns_policy == other.cast_columns_policy
            and self.row_count == other.row_count
            and self.expected_columns == other.expected_columns
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
    lake_options: LakeScanOptions,
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
    if lake_options.expected_columns is not None:
        for footer in footers:
            extra = [
                name
                for name in footer.names
                if name not in lake_options.expected_columns
            ]
            if extra:
                raise pl.exceptions.SchemaError(
                    f"extra column in file outside of expected schema: {extra[0]}"
                )

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
    num_rows: int

    def group_key(
        self, wanted: Mapping[Any, str]
    ) -> tuple[tuple[Any, ...], tuple[_Layout, ...]]:
        """
        Key on which consecutive files can be read together.

        Parameters
        ----------
        wanted
            Maps the key of each column the scan must produce to its name in
            the table schema.

        Returns
        -------
        The top-level keys of the file, and the physical and logical types
        of each wanted column it holds. The parquet reader requires the
        selected columns of its sources to agree on both.
        """
        return self.keys, tuple(
            signature
            for key, signature in zip(self.keys, self.signatures, strict=True)
            if key in wanted
        )


def _element_layout(
    element: plc.io.parquet_metadata.SchemaElement,
) -> _ElementLayout:
    """Physical and logical type of a parquet schema element."""
    logical_type = element.logical_type
    if logical_type is None:
        return element.type, element.type_length, element.num_children, None
    decimal_type = logical_type.decimal_type
    temporal_type = (
        logical_type.time_type
        if logical_type.time_type is not None
        else logical_type.timestamp_type
    )
    int_type = logical_type.int_type
    return (
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
    """Top-level columns and row count of a parquet file."""
    if metadata is None:
        metadata = plc.io.parquet_metadata.read_parquet_footers(
            plc.io.SourceInfo([path])
        )[0]
    elements = iter(metadata.schema)
    root = next(elements)
    names = []
    field_ids = []
    signatures = []
    for _ in range(root.num_children):
        element = next(elements)
        if by_field_id and element.field_id is None:
            raise NotImplementedError(
                f"Iceberg data file without parquet field IDs: {path}"
            )
        names.append(element.name)
        field_ids.append(element.field_id)
        signature = [_element_layout(element)]
        remaining = element.num_children
        while remaining:
            child = next(elements)
            signature.append(_element_layout(child))
            remaining += child.num_children - 1
        signatures.append(tuple(signature))
    return _Footer(
        names=tuple(names),
        keys=tuple(field_ids) if by_field_id else tuple(names),
        signatures=tuple(signatures),
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
        if file_types.get(file_name) == schema[name].plc_type
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


def _check_cast(
    name: str,
    incoming: plc.DataType,
    target: DataType,
    cast_columns_policy: Mapping[str, Any],
) -> None:
    """
    Check that a column read from a file may be cast to the table type.

    Parameters
    ----------
    name
        Name of the column in the table schema.
    incoming
        Type the column was read with.
    target
        Type of the column in the table schema.
    cast_columns_policy
        Which casts from the physical to the table type are permitted.

    Raises
    ------
    polars.exceptions.SchemaError
        If ``cast_columns_policy`` forbids an integer, float or datetime
        unit cast that the column needs. Other type differences are cast
        without checking, since cudf and polars do not represent every
        type the same way.
    """
    source = incoming.id()
    dest = target.plc_type.id()
    if source == dest:
        return
    if source in _INTEGER_TYPES and dest in _INTEGER_TYPES:
        source_bits, source_signed = _INTEGER_TYPES[source]
        dest_bits, dest_signed = _INTEGER_TYPES[dest]
        allowed = (
            cast_columns_policy["integer_upcast"]
            and dest_bits > source_bits
            and (dest_signed or not source_signed)
        )
    elif source in _FLOAT_WIDTHS and dest in _FLOAT_WIDTHS:
        allowed = cast_columns_policy[
            "float_upcast"
            if _FLOAT_WIDTHS[dest] > _FLOAT_WIDTHS[source]
            else "float_downcast"
        ]
    elif source in _INTEGER_TYPES and dest in _FLOAT_WIDTHS:
        allowed = cast_columns_policy["integer_to_float_cast"]
    elif source in _TIMESTAMP_DIGITS and dest in _TIMESTAMP_DIGITS:
        source_digits = _TIMESTAMP_DIGITS[source]
        dest_digits = _TIMESTAMP_DIGITS[dest]
        if source_digits == 9:
            allowed = cast_columns_policy["datetime_nanoseconds_downcast"]
        elif source_digits == 6:
            allowed = cast_columns_policy[
                "datetime_microseconds_downcast"
                if dest_digits == 3
                else "datetime_microseconds_upcast"
            ]
        else:
            allowed = cast_columns_policy["datetime_milliseconds_upcast"]
    else:
        return
    if not allowed:
        raise pl.exceptions.SchemaError(
            f"data type mismatch for column {name}: "
            f"incoming: {source.name} != target: {target.polars_type}"
        )


def _read_group(
    paths: Sequence[str],
    present: Sequence[Any],
    footer: _Footer,
    rows_per_path: Sequence[int],
    wanted: Mapping[Any, str],
    schema: Schema,
    lake_options: LakeScanOptions,
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
        columns, with the same physical types for the wanted ones, as the
        parquet reader requires of the sources of one read.
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
        Options of the scan, giving the policy and defaults for columns the
        files do not hold.
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
    read: dict[Any, plc.Column] = {}
    num_rows = sum(rows_per_path)
    source_index: plc.Column | None = None
    if present:
        builder = plc.io.parquet.ParquetReaderOptions.builder(
            plc.io.SourceInfo(list(paths))
        ).allow_mismatched_pq_schemas(val=True)
        if by_field_id:
            builder = builder.column_field_ids(list(present))
        options = builder.build()
        if not by_field_id:
            options.set_column_names(list(present))
        if filters is not None:
            options.set_filter(filters)
        if source_offset is not None:
            options.enable_prepend_source_index_column(val=True)
        table = plc.io.parquet.read_parquet(options, stream=stream)
        num_rows = table.tbl.num_rows()
        names = table.column_names(include_children=False)
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
            key_by_name[name]: column
            for name, column in zip(names, columns, strict=True)
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

    frame_columns = []
    for key, name in wanted.items():
        dtype = schema[name]
        if key in read:
            _check_cast(name, read[key].type(), dtype, lake_options.cast_columns_policy)
            frame_columns.append(
                Column(read[key], name=name, dtype=dtype).astype(dtype, stream)
            )
        elif lake_options.missing_columns_policy == "raise":
            raise pl.exceptions.ColumnNotFoundError(f"did not find column {name}")
        else:
            default = lake_options.initial_defaults.get(key) if by_field_id else None
            if default is None:
                obj = plc.Column.from_scalar(
                    plc.Scalar.from_py(None, dtype.plc_type, stream=stream),
                    num_rows,
                    stream=stream,
                )
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
