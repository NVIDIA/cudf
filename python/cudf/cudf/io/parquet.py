# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import datetime
import io
import itertools
import json
import math
import operator
import shutil
import tempfile
import warnings
from collections import defaultdict
from collections.abc import Hashable, Sequence
from contextlib import ExitStack
from functools import partial, reduce
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

import numpy as np
import pandas as pd

import pylibcudf as plc
from pylibcudf import expressions as plc_expr

from cudf.api.types import is_list_like
from cudf.core.column import (
    ColumnBase,
    access_columns,
    as_column,
    column_empty,
)
from cudf.core.dataframe import DataFrame
from cudf.core.dtypes import (
    CategoricalDtype,
    DecimalDtype,
    ListDtype,
    StructDtype,
)
from cudf.core.index import Index, RangeIndex
from cudf.core.multiindex import MultiIndex
from cudf.core.reshape import concat
from cudf.options import get_option
from cudf.utils import ioutils
from cudf.utils.performance_tracking import _performance_tracking

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Sequence
    from typing import Self

    from cudf._typing import DtypeObj
    from cudf.core.series import Series


BYTE_SIZES = {
    "kb": 1000,
    "mb": 1000000,
    "gb": 1000000000,
    "tb": 1000000000000,
    "pb": 1000000000000000,
    "kib": 1024,
    "mib": 1048576,
    "gib": 1073741824,
    "tib": 1099511627776,
    "pib": 1125899906842624,
    "b": 1,
    "": 1,
    "k": 1000,
    "m": 1000000,
    "g": 1000000000,
    "t": 1000000000000,
    "p": 1000000000000000,
    "ki": 1024,
    "mi": 1048576,
    "gi": 1073741824,
    "ti": 1099511627776,
    "pi": 1125899906842624,
}


def _plc_write_parquet(
    table,
    filepaths_or_buffers,
    index: bool | None = None,
    compression: Literal["snappy", "ZSTD", "ZLIB", "LZ4", None] = "snappy",
    statistics: Literal["ROWGROUP", "PAGE", "COLUMN", "NONE"] = "ROWGROUP",
    metadata_file_path: str | None = None,
    int96_timestamps: bool = False,
    row_group_size_bytes: int | None = None,
    row_group_size_rows: int | None = None,
    max_page_size_bytes: int | None = None,
    max_page_size_rows: int | None = None,
    max_dictionary_size: int | None = None,
    partitions_info=None,
    force_nullable_schema: bool = False,
    header_version: Literal["1.0", "2.0"] = "1.0",
    use_dictionary: bool = True,
    skip_compression: set[Hashable] | None = None,
    column_encoding: dict[
        Hashable,
        Literal[
            "PLAIN",
            "DICTIONARY",
            "DELTA_BINARY_PACKED",
            "DELTA_LENGTH_BYTE_ARRAY",
            "DELTA_BYTE_ARRAY",
            "BYTE_STREAM_SPLIT",
            "USE_DEFAULT",
        ],
    ]
    | None = None,
    column_type_length: dict | None = None,
    output_as_binary: set[Hashable] | None = None,
    write_arrow_schema: bool = False,
    page_level_compression: bool = False,
) -> np.ndarray | None:
    """
    Cython function to call into libcudf API, see `write_parquet`.

    See Also
    --------
    cudf.io.parquet.write_parquet
    """
    if index is True or (
        index is None and not isinstance(table.index, RangeIndex)
    ):
        iter_columns = itertools.chain(table.index._columns, table._columns)
    else:
        iter_columns = table._columns

    with access_columns(*iter_columns, mode="read", scope="internal"):
        if index is True or (
            index is None and not isinstance(table.index, RangeIndex)
        ):
            columns = itertools.chain(table.index._columns, table._columns)
            plc_table = plc.Table([col.plc_column for col in columns])
            tbl_meta = plc.io.types.TableInputMetadata(plc_table)
            for level, idx_name in enumerate(table.index.names):
                idx_name_str = ioutils._index_level_name(
                    idx_name, level, table._column_names
                )
                if not isinstance(idx_name_str, str):
                    raise ValueError(
                        f"Index name must be a string, got {type(idx_name_str)}"
                    )
                tbl_meta.column_metadata[level].set_name(idx_name_str)
            num_index_cols_meta = table.index.nlevels
        else:
            plc_table = plc.Table([col.plc_column for col in table._columns])
            tbl_meta = plc.io.types.TableInputMetadata(plc_table)
            num_index_cols_meta = 0

        for i, name in enumerate(table._column_names, num_index_cols_meta):
            if not isinstance(name, str):
                if get_option("mode.pandas_compatible"):
                    tbl_meta.column_metadata[i].set_name(str(name))
                else:
                    raise ValueError(
                        "Writing a Parquet file requires string column names"
                    )
            else:
                tbl_meta.column_metadata[i].set_name(name)

            _set_col_metadata(
                table[name]._column.dtype,
                tbl_meta.column_metadata[i],
                force_nullable_schema,
                None,
                skip_compression,
                column_encoding,
                column_type_length,
                output_as_binary,
            )
        if partitions_info is not None:
            user_data = [
                {
                    "pandas": ioutils.generate_pandas_metadata(
                        table.iloc[start_row : start_row + num_row].copy(
                            deep=False
                        ),
                        index,
                    )
                }
                for start_row, num_row in partitions_info
            ]
        else:
            user_data = [
                {"pandas": ioutils.generate_pandas_metadata(table, index)}
            ]

        if header_version not in ("1.0", "2.0"):
            raise ValueError(
                f"Invalid parquet header version: {header_version}. "
                "Valid values are '1.0' and '2.0'"
            )

        dict_policy = (
            plc.io.types.DictionaryPolicy.ADAPTIVE
            if use_dictionary
            else plc.io.types.DictionaryPolicy.NEVER
        )

        comp_type = _get_comp_type(compression)
        stat_freq = _get_stat_freq(statistics)
        options = (
            plc.io.parquet.ParquetWriterOptions.builder(
                plc.io.SinkInfo(filepaths_or_buffers), plc_table
            )
            .metadata(tbl_meta)
            .key_value_metadata(user_data)
            .compression(comp_type)
            .stats_level(stat_freq)
            .int96_timestamps(int96_timestamps)
            .write_v2_headers(header_version == "2.0")
            .page_level_compression(page_level_compression)
            .dictionary_policy(dict_policy)
            .utc_timestamps(False)
            .write_arrow_schema(write_arrow_schema)
            .build()
        )
        if partitions_info is not None:
            options.set_partitions(
                [
                    plc.io.types.PartitionInfo(part[0], part[1])
                    for part in partitions_info
                ]
            )
        if metadata_file_path is not None:
            if is_list_like(metadata_file_path):
                options.set_column_chunks_file_paths(metadata_file_path)
            else:
                options.set_column_chunks_file_paths([metadata_file_path])
        if row_group_size_bytes is not None:
            options.set_row_group_size_bytes(row_group_size_bytes)
        if row_group_size_rows is not None:
            options.set_row_group_size_rows(row_group_size_rows)
        if max_page_size_bytes is not None:
            options.set_max_page_size_bytes(max_page_size_bytes)
        if max_page_size_rows is not None:
            options.set_max_page_size_rows(max_page_size_rows)
        if max_dictionary_size is not None:
            options.set_max_dictionary_size(max_dictionary_size)
        blob = plc.io.parquet.write_parquet(options)
        if metadata_file_path is not None:
            return np.asarray(blob.obj)
        else:
            return None


@_performance_tracking
def _write_parquet(
    df,
    paths,
    compression: Literal["snappy", "ZSTD", "ZLIB", "LZ4", None] = "snappy",
    index: bool | None = None,
    statistics: Literal["ROWGROUP", "PAGE", "COLUMN", "NONE"] = "ROWGROUP",
    metadata_file_path: str | None = None,
    int96_timestamps: bool = False,
    row_group_size_bytes: int | None = None,
    row_group_size_rows: int | None = None,
    max_page_size_bytes: int | None = None,
    max_page_size_rows: int | None = None,
    max_dictionary_size: int | None = None,
    partitions_info=None,
    storage_options=None,
    force_nullable_schema: bool = False,
    header_version: Literal["1.0", "2.0"] = "1.0",
    use_dictionary: bool = True,
    skip_compression: set[Hashable] | None = None,
    column_encoding: dict[
        Hashable,
        Literal[
            "PLAIN",
            "DICTIONARY",
            "DELTA_BINARY_PACKED",
            "DELTA_LENGTH_BYTE_ARRAY",
            "DELTA_BYTE_ARRAY",
            "BYTE_STREAM_SPLIT",
            "USE_DEFAULT",
        ],
    ]
    | None = None,
    column_type_length: dict | None = None,
    output_as_binary: set[Hashable] | None = None,
    write_arrow_schema: bool = True,
    page_level_compression: bool = False,
) -> np.ndarray | None:
    if is_list_like(paths) and len(paths) > 1:
        if partitions_info is None:
            raise ValueError("partition info is required for multiple paths")
        elif not is_list_like(partitions_info):
            raise ValueError(
                "partition info must be list-like for multiple paths"
            )
        elif not len(paths) == len(partitions_info):
            raise ValueError("partitions_info and paths must be of same size")
    if is_list_like(partitions_info) and len(partitions_info) > 1:
        if not is_list_like(paths):
            raise ValueError(
                "paths must be list-like when partitions_info provided"
            )

    paths_or_bufs = [
        ioutils.get_writer_filepath_or_buffer(
            path_or_data=path, mode="wb", storage_options=storage_options
        )
        for path in paths
    ]
    common_args = {
        "index": index,
        "compression": compression,
        "statistics": statistics,
        "metadata_file_path": metadata_file_path,
        "int96_timestamps": int96_timestamps,
        "row_group_size_bytes": row_group_size_bytes,
        "row_group_size_rows": row_group_size_rows,
        "max_page_size_bytes": max_page_size_bytes,
        "max_page_size_rows": max_page_size_rows,
        "max_dictionary_size": max_dictionary_size,
        "partitions_info": partitions_info,
        "force_nullable_schema": force_nullable_schema,
        "header_version": header_version,
        "use_dictionary": use_dictionary,
        "skip_compression": skip_compression,
        "column_encoding": column_encoding,
        "column_type_length": column_type_length,
        "output_as_binary": output_as_binary,
        "write_arrow_schema": write_arrow_schema,
        "page_level_compression": page_level_compression,
    }
    if all(ioutils.is_fsspec_open_file(buf) for buf in paths_or_bufs):
        with ExitStack() as stack:
            fsspec_objs = [stack.enter_context(file) for file in paths_or_bufs]
            file_objs = [
                ioutils.get_IOBase_writer(file_obj) for file_obj in fsspec_objs
            ]
            write_parquet_res = _plc_write_parquet(
                df, filepaths_or_buffers=file_objs, **common_args
            )
    else:
        write_parquet_res = _plc_write_parquet(
            df, filepaths_or_buffers=paths_or_bufs, **common_args
        )

    return write_parquet_res


# Logic chosen to match: https://arrow.apache.org/
# docs/_modules/pyarrow/parquet.html#write_to_dataset
@_performance_tracking
def write_to_dataset(
    df,
    root_path,
    compression: Literal["snappy", "ZSTD", "ZLIB", "LZ4", None] = "snappy",
    filename=None,
    partition_cols=None,
    fs=None,
    preserve_index: bool = False,
    return_metadata: bool = False,
    statistics: Literal["ROWGROUP", "PAGE", "COLUMN", "NONE"] = "ROWGROUP",
    int96_timestamps: bool = False,
    row_group_size_bytes: int | None = None,
    row_group_size_rows: int | None = None,
    max_page_size_bytes: int | None = None,
    max_page_size_rows: int | None = None,
    storage_options=None,
    force_nullable_schema: bool = False,
    header_version: Literal["1.0", "2.0"] = "1.0",
    use_dictionary: bool = True,
    skip_compression: set[Hashable] | None = None,
    column_encoding: dict[
        Hashable,
        Literal[
            "PLAIN",
            "DICTIONARY",
            "DELTA_BINARY_PACKED",
            "DELTA_LENGTH_BYTE_ARRAY",
            "DELTA_BYTE_ARRAY",
            "BYTE_STREAM_SPLIT",
            "USE_DEFAULT",
        ],
    ]
    | None = None,
    column_type_length: dict | None = None,
    output_as_binary: set[Hashable] | None = None,
    store_schema=False,
    page_level_compression: bool = False,
):
    """Wraps `to_parquet` to write partitioned Parquet datasets.
    For each combination of partition group and value,
    subdirectories are created as follows:

    .. code-block:: bash

        root_dir/
            group=value1
                <filename>.parquet
            ...
            group=valueN
                <filename>.parquet

    Parameters
    ----------
    df : cudf.DataFrame
    root_path : string,
        The root directory of the dataset
    compression : {'snappy', 'ZSTD', None}, default 'snappy'
        Name of the compression to use. Use ``None`` for no compression.
    filename : string, default None
        The file name to use (within each partition directory). If None,
        a random uuid4 hex string will be used for each file name.
    partition_cols : list,
        Column names by which to partition the dataset.
        Columns are partitioned in the order they are given.
    fs : FileSystem, default None
        If nothing passed, paths assumed to be found in the local on-disk
        filesystem
    preserve_index : bool, default False
        Preserve index values in each parquet file.
    return_metadata : bool, default False
        Return parquet metadata for written data. Returned metadata will
        include the file-path metadata (relative to `root_path`).
    int96_timestamps : bool, default False
        If ``True``, write timestamps in int96 format. This will convert
        timestamps from timestamp[ns], timestamp[ms], timestamp[s], and
        timestamp[us] to the int96 format, which is the number of Julian
        days and the number of nanoseconds since midnight of 1970-01-01.
        If ``False``, timestamps will not be altered.
    row_group_size_bytes: integer or None, default None
        Maximum size of each stripe of the output.
        If None, no limit on row group stripe size will be used.
    row_group_size_rows: integer or None, default None
        Maximum number of rows of each stripe of the output.
        If None, 1000000 will be used.
    max_page_size_bytes: integer or None, default None
        Maximum uncompressed size of each page of the output.
        If None, 524288 (512KB) will be used.
    max_page_size_rows: integer or None, default None
        Maximum number of rows of each page of the output.
        If None, 20000 will be used.
    storage_options : dict, optional, default None
        Extra options that make sense for a particular storage connection,
        e.g. host, port, username, password, etc. For HTTP(S) URLs the
        key-value pairs are forwarded to ``urllib.request.Request`` as
        header options. For other URLs (e.g. starting with "s3://", and
        "gcs://") the key-value pairs are forwarded to ``fsspec.open``.
        Please see ``fsspec`` and ``urllib`` for more details.
    force_nullable_schema : bool, default False.
        If True, writes all columns as `null` in schema.
        If False, columns are written as `null` if they contain null values,
        otherwise as `not null`.
    header_version : {{'1.0', '2.0'}}, default "1.0"
        Controls whether to use version 1.0 or version 2.0 page headers when
        encoding. Version 1.0 is more portable, but version 2.0 enables the
        use of newer encoding schemes.
    force_nullable_schema : bool, default False.
        If True, writes all columns as `null` in schema.
        If False, columns are written as `null` if they contain null values,
        otherwise as `not null`.
    skip_compression : set, optional, default None
        If a column name is present in the set, that column will not be compressed,
        regardless of the ``compression`` setting.
    column_encoding : dict, optional, default None
        Sets the page encoding to use on a per-column basis. The key is a column
        name, and the value is one of: 'PLAIN', 'DICTIONARY', 'DELTA_BINARY_PACKED',
        'DELTA_LENGTH_BYTE_ARRAY', 'DELTA_BYTE_ARRAY', 'BYTE_STREAM_SPLIT', or
        'USE_DEFAULT'.
    column_type_length : dict, optional, default None
        Specifies the width in bytes of ``FIXED_LEN_BYTE_ARRAY`` column elements.
        The key is a column name and the value is an integer. The named column
        will be output as unannotated binary (i.e. the column will behave as if
        ``output_as_binary`` was set).
    output_as_binary : set, optional, default None
        If a column name is present in the set, that column will be output as
        unannotated binary, rather than the default 'UTF-8'.
    store_schema : bool, default False
        If ``True``, enable computing and writing arrow schema to Parquet
        file footer's key-value metadata section for faithful round-tripping.
    page_level_compression : bool, default False
        If ``True``, when writing version 2.0 pages (``header_version="2.0"``),
        allow each page to independently decide whether to compress based on
        compression ratio. Pages that do not benefit from compression will
        be written uncompressed.
    """

    fs = ioutils._ensure_filesystem(fs, root_path, storage_options)
    fs.mkdirs(root_path, exist_ok=True)

    if partition_cols is not None and len(partition_cols) > 0:
        (
            full_paths,
            metadata_file_paths,
            grouped_df,
            part_offsets,
            _,
        ) = _get_partitioned(
            df=df,
            root_path=root_path,
            partition_cols=partition_cols,
            filename=filename,
            fs=fs,
            preserve_index=preserve_index,
            storage_options=storage_options,
        )
        metadata_file_path = metadata_file_paths if return_metadata else None
        metadata = to_parquet(
            df=grouped_df,
            path=full_paths,
            compression=compression,
            index=preserve_index,
            partition_offsets=part_offsets,
            storage_options=storage_options,
            metadata_file_path=metadata_file_path,
            statistics=statistics,
            int96_timestamps=int96_timestamps,
            row_group_size_bytes=row_group_size_bytes,
            row_group_size_rows=row_group_size_rows,
            max_page_size_bytes=max_page_size_bytes,
            max_page_size_rows=max_page_size_rows,
            force_nullable_schema=force_nullable_schema,
            header_version=header_version,
            use_dictionary=use_dictionary,
            skip_compression=skip_compression,
            column_encoding=column_encoding,
            column_type_length=column_type_length,
            output_as_binary=output_as_binary,
            store_schema=store_schema,
            page_level_compression=page_level_compression,
        )

    else:
        filename = filename or _generate_filename()
        full_path = fs.sep.join([root_path, filename])

        metadata_file_path = filename if return_metadata else None

        metadata = df.to_parquet(
            path=full_path,
            compression=compression,
            index=preserve_index,
            storage_options=storage_options,
            metadata_file_path=metadata_file_path,
            statistics=statistics,
            int96_timestamps=int96_timestamps,
            row_group_size_bytes=row_group_size_bytes,
            row_group_size_rows=row_group_size_rows,
            max_page_size_bytes=max_page_size_bytes,
            max_page_size_rows=max_page_size_rows,
            force_nullable_schema=force_nullable_schema,
            header_version=header_version,
            use_dictionary=use_dictionary,
            skip_compression=skip_compression,
            column_encoding=column_encoding,
            column_type_length=column_type_length,
            output_as_binary=output_as_binary,
            store_schema=store_schema,
            page_level_compression=page_level_compression,
        )

    return metadata


_BIN_OPS = {
    "==": plc_expr.ASTOperator.EQUAL,
    "!=": plc_expr.ASTOperator.NOT_EQUAL,
    "<": plc_expr.ASTOperator.LESS,
    "<=": plc_expr.ASTOperator.LESS_EQUAL,
    ">": plc_expr.ASTOperator.GREATER,
    ">=": plc_expr.ASTOperator.GREATER_EQUAL,
}


def _raise_for_unsupported_scalar_types(val: Any):
    if not (
        isinstance(val, (int, float))
        or isinstance(val, (datetime.date, datetime.timedelta))
        or isinstance(val, str)
    ):
        return val
    raise NotImplementedError(
        "Only numeric, string, or timestamp/duration scalars are accepted"
    )


def make_literal(v: Any) -> plc_expr.Literal:
    return plc_expr.Literal(
        plc.Scalar.from_py(_raise_for_unsupported_scalar_types(v))
    )


def make_expr(col: str, op: str, val: Any) -> plc_expr.Expression:
    col_ref = plc_expr.ColumnNameReference(col)

    match op:
        case op if op in _BIN_OPS:
            return plc_expr.Operation(
                _BIN_OPS[op],
                col_ref,
                make_literal(val),
            )

        case "in":
            equal_ops = [
                plc_expr.Operation(
                    plc_expr.ASTOperator.EQUAL, col_ref, make_literal(v)
                )
                for v in val
            ]
            return reduce(
                partial(plc_expr.Operation, plc_expr.ASTOperator.LOGICAL_OR),
                equal_ops,
            )

        case "not in":
            not_equal_ops = [
                plc_expr.Operation(
                    plc_expr.ASTOperator.NOT_EQUAL, col_ref, make_literal(v)
                )
                for v in val
            ]
            return reduce(
                partial(plc_expr.Operation, plc_expr.ASTOperator.LOGICAL_AND),
                not_equal_ops,
            )

        case "is":
            if val is None:
                return plc_expr.Operation(
                    plc_expr.ASTOperator.IS_NULL, col_ref
                )
            raise NotImplementedError("Only `is None` supported")

        case "is not":
            if val is None:
                return plc_expr.Operation(
                    plc_expr.ASTOperator.NOT,
                    plc_expr.Operation(plc_expr.ASTOperator.IS_NULL, col_ref),
                )
            raise NotImplementedError("Only `is not None` supported")

        case _:
            raise NotImplementedError(f"Unsupported op: {op}")


def translate_filters_to_ast(
    filters: list[list[tuple[str, str, Any]]],
) -> plc_expr.Expression:
    """
    Convert a filter expression in Disjunctive Normal Form (DNF) to a pylibcudf AST.

    Parameters
    ----------
    filters : list[list[tuple[str, str, Any]]]
        A filter expression in DNF.
        DNF is represented as a list of OR-ed clauses, where each clause is a list of
        AND-ed predicates. Each predicate is a tuple of the form:
            (column_name, operator, value)

    Returns
    -------
    plc.expressions.Expression
        A pylibcudf AST expression representing the filter.

    Notes
    -----
    This function supports the following operators:
      - Binary comparisons: ==, !=, <, <=, >, >=
      - Membership tests: "in", "not in"
      - Null checks: "is None", "is not None"

    The translation is performed by recursively reducing:
      - Each clause's predicates into a conjunction (AND)
      - All clauses into a disjunction (OR) of those conjunctions
    """

    clauses = [
        reduce(
            partial(plc_expr.Operation, plc_expr.ASTOperator.LOGICAL_AND),
            (make_expr(*pred) for pred in clause),
        )
        for clause in filters
    ]

    if not clauses:
        raise ValueError("Empty filter expression")

    return (
        reduce(
            partial(plc_expr.Operation, plc_expr.ASTOperator.LOGICAL_OR),
            clauses,
        )
        if len(clauses) > 1
        else clauses[0]
    )


def _parse_metadata(meta) -> tuple[bool, Any, None | np.dtype]:
    file_is_range_index = False
    file_index_cols = None
    file_column_dtype = None

    if "index_columns" in meta and len(meta["index_columns"]) > 0:
        file_index_cols = meta["index_columns"]

        if (
            isinstance(file_index_cols[0], dict)
            and file_index_cols[0]["kind"] == "range"
        ):
            file_is_range_index = True
    if "column_indexes" in meta and len(meta["column_indexes"]) == 1:
        file_column_dtype = np.dtype(meta["column_indexes"][0]["numpy_type"])
    return file_is_range_index, file_index_cols, file_column_dtype


@ioutils.doc_read_parquet_metadata()
@_performance_tracking
def read_parquet_metadata(
    filepath_or_buffer,
) -> tuple[int, int, Sequence[Hashable], int, Sequence[dict[str, int]]]:
    """{docstring}"""

    # List of filepaths or buffers
    filepaths_or_buffers = ioutils.get_reader_filepath_or_buffer(
        path_or_data=filepath_or_buffer,
        bytes_per_thread=None,
    )

    parquet_metadata = plc.io.parquet_metadata.read_parquet_metadata(
        plc.io.SourceInfo(filepaths_or_buffers)
    )

    # read all column names including index column, if any
    col_names = [
        info.name() for info in parquet_metadata.schema().root().children()
    ]

    index_col_names = set()
    json_str = parquet_metadata.metadata()["pandas"]
    if json_str != "":
        meta = json.loads(json_str)
        file_is_range_index, index_col, _ = _parse_metadata(meta)
        if not file_is_range_index and index_col is not None:
            columns = meta["columns"]
            for idx_col in index_col:
                for c in columns:
                    if c["field_name"] == idx_col:
                        index_col_names.add(idx_col)

    # remove the index column from the list of column names
    # only if index_col_names is not None
    if len(index_col_names) >= 0:
        col_names = [name for name in col_names if name not in index_col_names]

    return (
        parquet_metadata.num_rows(),
        parquet_metadata.num_rowgroups(),
        col_names,
        len(col_names),
        parquet_metadata.rowgroup_metadata(),
    )


@_performance_tracking
def _process_dataset(
    paths,
    fs,
    filters=None,
    row_groups=None,
    categorical_partitions=True,
    dataset_kwargs=None,
):
    # Returns:
    #     file_list - Expanded/filtered list of paths
    #     row_groups - Filtered list of row-group selections
    #     partition_keys - list of partition keys for each file
    #     partition_categories - Categories for each partition

    # The general purpose of this function is to (1) expand
    # directory input into a list of paths (using the pyarrow
    # dataset API), (2) to apply row-group filters, and (3)
    # to discover directory-partitioning information

    # Lazily import pyarrow.dataset due to import time
    import pyarrow.dataset as ds

    # Deal with case that the user passed in a directory name
    file_list = paths
    if len(paths) == 1 and ioutils.is_directory(paths[0]):
        paths = ioutils.stringify_pathlike(paths[0])
    elif (
        filters is None
        and isinstance(dataset_kwargs, dict)
        and dataset_kwargs.get("partitioning") is None
    ):
        # Skip dataset processing if we have no filters
        # or hive/directory partitioning to deal with.
        return paths, row_groups, [], {}

    # Convert filters to ds.Expression
    if filters is not None:
        from pyarrow.parquet import filters_to_expression

        filters = filters_to_expression(filters)

    # Initialize ds.FilesystemDataset
    # TODO: Remove the if len(paths) workaround after following bug is fixed:
    # https://issues.apache.org/jira/browse/ARROW-16438
    dataset = ds.dataset(
        source=paths[0] if len(paths) == 1 else paths,
        filesystem=fs,
        **(
            dataset_kwargs
            or {
                "format": "parquet",
                "partitioning": "hive",
            }
        ),
    )

    file_list = dataset.files
    if len(file_list) == 0:
        raise FileNotFoundError(f"{paths} could not be resolved to any files")

    # Deal with directory partitioning
    # Get all partition keys (without filters)
    partition_categories = defaultdict(list)
    file_fragment = None
    for file_fragment in dataset.get_fragments():
        keys = ds._get_partition_keys(file_fragment.partition_expression)
        if not (keys or partition_categories):
            # Bail - This is not a directory-partitioned dataset
            break
        for k, v in keys.items():
            if v not in partition_categories[k]:
                partition_categories[k].append(v)
        if not categorical_partitions:
            # Bail - We don't need to discover all categories.
            # We only need to save the partition keys from this
            # first `file_fragment`
            break

    if partition_categories and file_fragment is not None:
        # Check/correct order of `categories` using last file_frag,
        # because `_get_partition_keys` does NOT preserve the
        # partition-hierarchy order of the keys.
        cat_keys = [
            part.split("=")[0]
            for part in file_fragment.path.split(fs.sep)
            if "=" in part
        ]
        if set(partition_categories) == set(cat_keys):
            partition_categories = {
                k: partition_categories[k]
                for k in cat_keys
                if k in partition_categories
            }

    # If we do not have partitioned data and
    # are not filtering, we can return here
    if filters is None and not partition_categories:
        return file_list, row_groups, [], {}

    # Record initial row_groups input
    row_groups_map = {}
    if row_groups is not None:
        # Make sure paths and row_groups map 1:1
        # and save the initial mapping
        if len(paths) != len(file_list):
            raise ValueError(
                "Cannot specify a row_group selection for a directory path."
            )
        row_groups_map = {
            path: rgs for path, rgs in zip(paths, row_groups, strict=True)
        }

    # Apply filters and discover partition columns
    partition_keys = []
    if partition_categories or filters is not None:
        file_list = []
        if filters is not None:
            row_groups = []
        for file_fragment in dataset.get_fragments(filter=filters):
            path = file_fragment.path

            # Extract hive-partition keys, and make sure they
            # are ordered the same as they are in `partition_categories`
            if partition_categories:
                raw_keys = ds._get_partition_keys(
                    file_fragment.partition_expression
                )
                partition_keys.append(
                    [
                        (name, raw_keys[name])
                        for name in partition_categories.keys()
                    ]
                )

            # Apply row-group filtering
            selection = row_groups_map.get(path, None)
            if selection is not None or filters is not None:
                filtered_row_groups = [
                    rg_info.id
                    for rg_fragment in file_fragment.split_by_row_group(
                        filters,
                        schema=dataset.schema,
                    )
                    for rg_info in rg_fragment.row_groups
                ]
            file_list.append(path)
            if filters is not None:
                if selection is None:
                    row_groups.append(filtered_row_groups)
                else:
                    row_groups.append(
                        [
                            rg_id
                            for rg_id in filtered_row_groups
                            if rg_id in selection
                        ]
                    )

    return (
        file_list,
        row_groups,
        partition_keys,
        partition_categories if categorical_partitions else {},
    )


@ioutils.doc_read_parquet()
@_performance_tracking
def read_parquet(
    filepath_or_buffer,
    engine="cudf",
    columns=None,
    storage_options=None,
    filesystem=None,
    filters=None,
    row_groups=None,
    use_pandas_metadata=True,
    categorical_partitions=True,
    bytes_per_thread=None,
    dataset_kwargs=None,
    nrows=None,
    skip_rows=None,
    allow_mismatched_pq_schemas=False,
    ignore_missing_columns=True,
    case_sensitive_names=True,
    *args,
    **kwargs,
):
    """{docstring}"""
    if engine not in {"cudf", "pyarrow"}:
        raise ValueError(
            f"Only supported engines are {{'cudf', 'pyarrow'}}, got {engine=}"
        )
    if bytes_per_thread is None:
        bytes_per_thread = ioutils._BYTES_PER_THREAD_DEFAULT

    # Multiple sources are passed as a list. If a single source is passed,
    # wrap it in a list for unified processing downstream.
    if not is_list_like(filepath_or_buffer):
        filepath_or_buffer = [filepath_or_buffer]

    # a list of row groups per source should be passed. make the list of
    # lists that is expected for multiple sources
    if row_groups is not None:
        if not is_list_like(row_groups):
            row_groups = [[row_groups]]
        elif not is_list_like(row_groups[0]):
            row_groups = [row_groups]

    # Check columns input
    if columns is not None:
        if not is_list_like(columns):
            raise ValueError("Expected list like for columns")

    # Start by trying construct a filesystem object, so we
    # can apply filters on remote file-systems
    fs, paths = ioutils._get_filesystem_and_paths(
        path_or_data=filepath_or_buffer,
        storage_options=storage_options,
        filesystem=filesystem,
    )

    # Normalize and validate filters
    filters = _normalize_filters(filters)

    # Attempt to translate filters to a libcudf AST
    ast_filter = None
    if (
        engine == "cudf"
        and filters is not None
        and categorical_partitions is None
        and dataset_kwargs is None
    ):
        try:
            ast_filter = translate_filters_to_ast(filters)
        except NotImplementedError:
            pass

    # Use pyarrow dataset to detect/process directory-partitioned
    # data and apply filters. Note that we can only support partitioned
    # data and filtering if the input is a single directory or list of
    # paths.
    partition_keys = []
    partition_categories = {}
    if ast_filter is None and fs and paths:
        (
            paths,
            row_groups,
            partition_keys,
            partition_categories,
        ) = _process_dataset(
            paths=paths,
            fs=fs,
            filters=filters,
            row_groups=row_groups,
            categorical_partitions=categorical_partitions,
            dataset_kwargs=dataset_kwargs,
        )
    filepath_or_buffer = paths if paths else filepath_or_buffer

    # Prepare remote-IO options
    prefetch_options = kwargs.pop("prefetch_options", {})
    if ast_filter is None and not ioutils._is_local_filesystem(fs):
        # The default prefetch method depends on the
        # `row_groups` argument. In most cases we will use
        # method="all" by default, because it is fastest
        # when we need to read most of the file(s).
        # If a (simple) `row_groups` selection is made, we
        # use method="parquet" to avoid transferring the
        # entire file over the network
        method = prefetch_options.get("method")
        _row_groups = None
        if method in (None, "parquet"):
            if row_groups is None:
                # If the user didn't specify a method, don't use
                # 'parquet' prefetcher for column projection alone.
                method = method or "all"
            elif all(r == row_groups[0] for r in row_groups):
                # Row group selection means we are probably
                # reading half the file or less. We should
                # avoid a full file transfer by default.
                method = "parquet"
                _row_groups = row_groups[0]
            elif (method := method or "all") == "parquet":
                raise ValueError(
                    "The 'parquet' prefetcher requires a uniform "
                    "row-group selection for all paths within the "
                    "same `read_parquet` call. "
                    "Got: {row_groups}"
                )
        if method == "parquet":
            prefetch_options = prefetch_options.update(
                {
                    "method": method,
                    "columns": columns,
                    "row_groups": _row_groups,
                }
            )

    filepaths_or_buffers = ioutils.get_reader_filepath_or_buffer(
        path_or_data=filepath_or_buffer,
        fs=fs,
        storage_options=storage_options,
        bytes_per_thread=bytes_per_thread,
        prefetch_options=prefetch_options,
    )

    # Warn user if they are not using cudf for IO
    # (There is a good chance this was not the intention)
    if engine != "cudf":
        warnings.warn(
            "Using CPU via PyArrow to read Parquet dataset. "
            "This option is both inefficient and unstable!"
        )
        if filters is not None:
            warnings.warn(
                "Parquet row-group filtering is only supported with "
                "'engine=cudf'. Use pandas or pyarrow API directly "
                "for full CPU-based filtering functionality."
            )

    # Make sure we read in the columns needed for row-wise
    # filtering after IO. This means that one or more columns
    # will be dropped almost immediately after IO. However,
    # we do NEED these columns for accurate filtering.
    projected_columns = None
    if ast_filter is None and columns and filters:
        projected_columns = columns
        columns = sorted(
            set(v[0] for v in itertools.chain.from_iterable(filters))
            | set(columns)
        )

    # Convert parquet data to a cudf.DataFrame
    df = _parquet_to_frame(
        filepaths_or_buffers,
        engine,
        *args,
        columns=columns,
        row_groups=row_groups,
        use_pandas_metadata=use_pandas_metadata,
        partition_keys=partition_keys,
        partition_categories=partition_categories,
        dataset_kwargs=dataset_kwargs,
        nrows=nrows,
        skip_rows=skip_rows,
        allow_mismatched_pq_schemas=allow_mismatched_pq_schemas,
        ignore_missing_columns=ignore_missing_columns,
        case_sensitive_names=case_sensitive_names,
        filters=ast_filter,
        **kwargs,
    )
    # Build a lookup from (possibly lowered) name -> actual DataFrame column name
    _key = str.lower if not case_sensitive_names else str
    col_names = {_key(name): name for name in df._column_names}

    # Apply filters row-wise (if any are defined), and return
    if ast_filter is None:
        if not case_sensitive_names and filters:
            filters = [
                [
                    (col_names[_key(col)], op, val)
                    for col, op, val in conjunction
                ]
                for conjunction in filters
            ]
        df = _apply_post_filters(df, filters)

    if projected_columns:
        # Elements of `projected_columns` may now be in the index.
        # We must filter these names from our projection. For structs,
        # only the top-level name needs remapping via col_names
        projected_columns = [
            col_names[_key(col.split(".")[0])] for col in projected_columns
        ]
        return df[projected_columns]
    return df


def _normalize_filters(filters: list | None) -> list[list[tuple]] | None:
    # Utility to normalize and validate the `filters`
    # argument to `read_parquet`
    if not filters:
        return None

    msg = (
        f"filters must be None, or non-empty List[Tuple] "
        f"or List[List[Tuple]]. Got {filters}"
    )
    if not isinstance(filters, list):
        raise TypeError(msg)

    def _validate_predicate(item):
        if not isinstance(item, tuple) or len(item) != 3:
            raise TypeError(
                f"Predicate must be Tuple[str, str, Any], got {predicate}."
            )

    filters = filters if isinstance(filters[0], list) else [filters]
    for conjunction in filters:
        if not conjunction or not isinstance(conjunction, list):
            raise TypeError(msg)
        for predicate in conjunction:
            _validate_predicate(predicate)

    return filters


def _apply_post_filters(
    df: DataFrame, filters: list[list[tuple]] | None
) -> DataFrame:
    """Apply DNF filters to an in-memory DataFrame

    Disjunctive normal form (DNF) means that the inner-most
    tuple describes a single column predicate. These inner
    predicates are combined with an AND conjunction into a
    larger predicate. The outer-most list then combines all
    of the combined filters with an OR disjunction.
    """

    if not filters:
        # No filters to apply
        return df

    def _handle_in(column: Series, value, *, negate) -> Series:
        if not isinstance(value, (list, set, tuple)):
            raise TypeError(
                "Value of 'in'/'not in' filter must be a list, set, or tuple."
            )
        return ~column.isin(value) if negate else column.isin(value)

    def _handle_is(column: Series, value, *, negate) -> Series:
        if value not in {np.nan, None}:
            raise TypeError(
                "Value of 'is'/'is not' filter must be np.nan or None."
            )
        return ~column.isna() if negate else column.isna()

    handlers: dict[str, Callable] = {
        "==": operator.eq,
        "!=": operator.ne,
        "<": operator.lt,
        "<=": operator.le,
        ">": operator.gt,
        ">=": operator.ge,
        "in": partial(_handle_in, negate=False),
        "not in": partial(_handle_in, negate=True),
        "is": partial(_handle_is, negate=False),
        "is not": partial(_handle_is, negate=True),
    }

    # Can re-set the index before returning if we filter
    # out rows from a DataFrame with a default RangeIndex
    # (to reduce memory usage)
    reset_index = (
        isinstance(df.index, RangeIndex)
        and df.index.name is None
        and df.index.start == 0
        and df.index.step == 1
    )

    try:
        selection: Series = reduce(
            operator.or_,
            (
                reduce(
                    operator.and_,
                    (
                        handlers[op](df[column], value)
                        for (column, op, value) in expr
                    ),
                )
                for expr in filters
            ),
        )
        if reset_index:
            return df[selection].reset_index(drop=True)
        return df[selection]
    except (KeyError, TypeError):
        warnings.warn(
            f"Row-wise filtering failed in read_parquet for {filters}"
        )
        return df


@_performance_tracking
def _parquet_to_frame(
    paths_or_buffers,
    *args,
    row_groups=None,
    partition_keys=None,
    partition_categories=None,
    dataset_kwargs=None,
    nrows=None,
    skip_rows=None,
    case_sensitive_names=True,
    **kwargs,
):
    # If this is not a partitioned read, only need
    # one call to `_read_parquet`
    if not partition_keys:
        return _read_parquet(
            paths_or_buffers,
            nrows=nrows,
            skip_rows=skip_rows,
            case_sensitive_names=case_sensitive_names,
            *args,
            row_groups=row_groups,
            **kwargs,
        )

    if nrows is not None or skip_rows is not None:
        raise NotImplementedError(
            "nrows/skip_rows is not supported when reading a partitioned parquet dataset"
        )

    partition_meta = None
    partitioning = (dataset_kwargs or {}).get("partitioning", None)
    if hasattr(partitioning, "schema"):
        partition_meta = DataFrame.from_arrow(
            partitioning.schema.empty_table()
        )

    # For partitioned data, we need a distinct read for each
    # unique set of partition keys. Therefore, we start by
    # aggregating all paths with matching keys using a dict
    plan = {}
    for i, (keys, path) in enumerate(
        zip(partition_keys, paths_or_buffers, strict=True)
    ):
        rgs = row_groups[i] if row_groups else None
        tkeys = tuple(keys)
        if tkeys in plan:
            plan[tkeys][0].append(path)
            if rgs is not None:
                plan[tkeys][1].append(rgs)
        else:
            plan[tkeys] = ([path], None if rgs is None else [rgs])

    dfs = []
    for part_key, (key_paths, key_row_groups) in plan.items():
        # Add new DataFrame to our list
        dfs.append(
            _read_parquet(
                key_paths,
                *args,
                row_groups=key_row_groups,
                case_sensitive_names=case_sensitive_names,
                **kwargs,
            )
        )
        # Add partition columns to the last DataFrame
        for name, value in part_key:
            _len = len(dfs[-1])
            if partition_categories and name in partition_categories:
                # Build the categorical column from `codes`
                cat_dtype = CategoricalDtype(
                    categories=partition_categories[name],
                    ordered=False,
                )
                codes = as_column(
                    partition_categories[name].index(value),
                    length=_len,
                    dtype=cat_dtype._codes_dtype,
                )
                col = ColumnBase.create(
                    codes.plc_column,
                    cat_dtype,
                )
            else:
                # Not building categorical columns, so
                # `value` is already what we want
                if pd.isna(value):
                    col = column_empty(row_count=_len)
                else:
                    col = as_column(value, length=_len)
                if partition_meta is not None:
                    col = col.astype(partition_meta[name].dtype)
            dfs[-1][name] = col

    if len(dfs) > 1:
        # Concatenate dfs and return.
        # Assume we can ignore the index if it has no name.
        return concat(dfs, ignore_index=dfs[-1].index.name is None)
    else:
        return dfs[0]


@_performance_tracking
def _read_parquet(
    filepaths_or_buffers,
    engine,
    columns=None,
    row_groups=None,
    use_pandas_metadata=None,
    nrows: int | None = None,
    skip_rows: int | None = None,
    allow_mismatched_pq_schemas: bool = False,
    ignore_missing_columns: bool = True,
    case_sensitive_names: bool = True,
    filters: plc_expr.Expression | None = None,
    *args,
    **kwargs,
) -> DataFrame:
    # Simple helper function to dispatch between
    # cudf and pyarrow to read parquet data
    if engine == "cudf":
        if set(kwargs.keys()).difference(
            set(("_chunk_read_limit", "_pass_read_limit", "filters"))
        ):
            raise ValueError(
                "cudf engine doesn't support the "
                f"following keyword arguments: {list(kwargs.keys())}"
            )
        if args:
            raise ValueError(
                "cudf engine doesn't support the "
                f"following positional arguments: {list(args)}"
            )
        if nrows is None:
            nrows = -1
        if skip_rows is None:
            skip_rows = 0
        if get_option("io.parquet.low_memory"):
            # If filters are used, we can't rely on a RangeIndex
            # (row count may be reduced)
            allow_range_index = (
                filters is None and columns is not None and len(columns) != 0
            )

            options = (
                plc.io.parquet.ParquetReaderOptions.builder(
                    plc.io.SourceInfo(filepaths_or_buffers)
                )
                .use_pandas_metadata(use_pandas_metadata)
                .allow_mismatched_pq_schemas(allow_mismatched_pq_schemas)
                .ignore_missing_columns(ignore_missing_columns)
                .case_sensitive_names(case_sensitive_names)
                .build()
            )
            if row_groups is not None:
                options.set_row_groups(row_groups)
            if nrows > -1:
                options.set_num_rows(nrows)
            if skip_rows != 0:
                options.set_skip_rows(skip_rows)
            if columns is not None:
                options.set_column_names(columns)
            if filters is not None:
                options.set_filter(filters)

            reader = plc.io.parquet.ChunkedParquetReader(
                options,
                chunk_read_limit=kwargs.get("_chunk_read_limit", 0),
                pass_read_limit=kwargs.get("_pass_read_limit", 1024000000),
            )

            tbl_w_meta = reader.read_chunk()
            column_names = tbl_w_meta.column_names(include_children=False)
            child_names = tbl_w_meta.child_names
            per_file_user_data = tbl_w_meta.per_file_user_data
            concatenated_columns = tbl_w_meta.tbl.release()

            # save memory
            del tbl_w_meta

            while reader.has_next():
                columns = reader.read_chunk().tbl.release()
                # Iterate in reverse to avoid O(n²) cost from popping
                for i in range(len(concatenated_columns) - 1, -1, -1):
                    concatenated_columns[i] = plc.concatenate.concatenate(
                        [concatenated_columns[i], columns.pop()]
                    )

            plc_table = plc.Table(concatenated_columns)
            df = DataFrame.from_pylibcudf(
                plc_table,
                metadata={
                    "columns": column_names,
                    "child_names": child_names,
                },
            )
            df = _process_metadata(
                df,
                column_names,
                per_file_user_data,
                row_groups,
                filepaths_or_buffers,
                allow_range_index,
                use_pandas_metadata,
                nrows=nrows,
                skip_rows=skip_rows,
            )
            return df
        else:
            allow_range_index = True
            if columns is not None and len(columns) == 0 or filters:
                allow_range_index = False

            options = (
                plc.io.parquet.ParquetReaderOptions.builder(
                    plc.io.SourceInfo(filepaths_or_buffers)
                )
                .use_pandas_metadata(use_pandas_metadata)
                .allow_mismatched_pq_schemas(allow_mismatched_pq_schemas)
                .ignore_missing_columns(ignore_missing_columns)
                .case_sensitive_names(case_sensitive_names)
                .build()
            )
            if row_groups is not None:
                options.set_row_groups(row_groups)
            if nrows > -1:
                options.set_num_rows(nrows)
            if skip_rows != 0:
                options.set_skip_rows(skip_rows)
            if columns is not None:
                options.set_column_names(columns)
            if filters is not None:
                options.set_filter(filters)

            tbl_w_meta = plc.io.parquet.read_parquet(options)
            df = DataFrame.from_pylibcudf(tbl_w_meta)
            df = _process_metadata(
                df,
                tbl_w_meta.column_names(include_children=False),
                tbl_w_meta.per_file_user_data,
                row_groups,
                filepaths_or_buffers,
                allow_range_index,
                use_pandas_metadata,
                nrows=nrows,
                skip_rows=skip_rows,
            )
            return df
    else:
        if (
            isinstance(filepaths_or_buffers, list)
            and len(filepaths_or_buffers) == 1
        ):
            filepaths_or_buffers = filepaths_or_buffers[0]

        return DataFrame(
            pd.read_parquet(
                filepaths_or_buffers,
                columns=columns,
                engine=engine,
                *args,
                **kwargs,
            )
        )


@ioutils.doc_to_parquet()
@_performance_tracking
def to_parquet(
    df,
    path,
    engine="cudf",
    compression: Literal["snappy", "ZSTD", "ZLIB", "LZ4", None] = "snappy",
    index: bool | None = None,
    partition_cols=None,
    partition_file_name=None,
    partition_offsets=None,
    statistics: Literal["ROWGROUP", "PAGE", "COLUMN", "NONE"] = "ROWGROUP",
    metadata_file_path: str | None = None,
    int96_timestamps: bool = False,
    row_group_size_bytes: int | None = None,
    row_group_size_rows: int | None = None,
    max_page_size_bytes: int | None = None,
    max_page_size_rows: int | None = None,
    max_dictionary_size: int | None = None,
    storage_options=None,
    return_metadata: bool = False,
    force_nullable_schema: bool = False,
    header_version: Literal["1.0", "2.0"] = "1.0",
    use_dictionary: bool = True,
    skip_compression: set[Hashable] | None = None,
    column_encoding: dict[
        Hashable,
        Literal[
            "PLAIN",
            "DICTIONARY",
            "DELTA_BINARY_PACKED",
            "DELTA_LENGTH_BYTE_ARRAY",
            "DELTA_BYTE_ARRAY",
            "BYTE_STREAM_SPLIT",
            "USE_DEFAULT",
        ],
    ]
    | None = None,
    column_type_length: dict | None = None,
    output_as_binary: set[Hashable] | None = None,
    store_schema=False,
    page_level_compression: bool = False,
    *args,
    **kwargs,
):
    """{docstring}"""

    if engine == "cudf":
        if kwargs:
            raise ValueError(
                "cudf engine doesn't support the "
                f"following keyword arguments: {list(kwargs.keys())}"
            )
        if args:
            raise ValueError(
                "cudf engine doesn't support the "
                f"following positional arguments: {list(args)}"
            )
        # Ensure that no columns dtype is 'category'
        for col in df._column_names:
            if partition_cols is None or col not in partition_cols:
                if df[col].dtype.name == "category":
                    raise ValueError(
                        "'category' column dtypes are currently not "
                        + "supported by the gpu accelerated parquet writer"
                    )

        if partition_cols:
            if metadata_file_path is not None:
                warnings.warn(
                    "metadata_file_path will be ignored/overwritten when "
                    "partition_cols are provided. To request returning the "
                    "metadata binary blob, pass `return_metadata=True`"
                )

            return write_to_dataset(
                df,
                filename=partition_file_name,
                partition_cols=partition_cols,
                root_path=path,
                preserve_index=index,
                compression=compression,
                statistics=statistics,
                int96_timestamps=int96_timestamps,
                row_group_size_bytes=row_group_size_bytes,
                row_group_size_rows=row_group_size_rows,
                max_page_size_bytes=max_page_size_bytes,
                max_page_size_rows=max_page_size_rows,
                return_metadata=return_metadata,
                storage_options=storage_options,
                force_nullable_schema=force_nullable_schema,
                header_version=header_version,
                use_dictionary=use_dictionary,
                skip_compression=skip_compression,
                column_encoding=column_encoding,
                column_type_length=column_type_length,
                output_as_binary=output_as_binary,
                store_schema=store_schema,
                page_level_compression=page_level_compression,
            )

        partition_info = (
            [(i, j - i) for i, j in itertools.pairwise(partition_offsets)]
            if partition_offsets is not None
            else None
        )
        return _write_parquet(
            df,
            paths=path if is_list_like(path) else [path],
            compression=compression,
            index=index,
            statistics=statistics,
            metadata_file_path=metadata_file_path,
            int96_timestamps=int96_timestamps,
            row_group_size_bytes=row_group_size_bytes,
            row_group_size_rows=row_group_size_rows,
            max_page_size_bytes=max_page_size_bytes,
            max_page_size_rows=max_page_size_rows,
            max_dictionary_size=max_dictionary_size,
            partitions_info=partition_info,
            storage_options=storage_options,
            force_nullable_schema=force_nullable_schema,
            header_version=header_version,
            use_dictionary=use_dictionary,
            skip_compression=skip_compression,
            column_encoding=column_encoding,
            column_type_length=column_type_length,
            output_as_binary=output_as_binary,
            write_arrow_schema=store_schema,
            page_level_compression=page_level_compression,
        )

    else:
        import pyarrow.parquet as pq

        if partition_offsets is not None:
            warnings.warn(
                "partition_offsets will be ignored when engine is not cudf"
            )

        # If index is empty set it to the expected default value of True
        if index is None:
            index = True

        pa_table = df.to_arrow(preserve_index=index)
        # Check for conflicting arguments in kwargs
        if "root_path" in kwargs:
            raise ValueError(
                "'root_path' should be passed as 'path' argument to to_parquet(), not in kwargs"
            )
        if "partition_cols" in kwargs:
            raise ValueError(
                "'partition_cols' should be passed directly to to_parquet(), not in kwargs"
            )
        # Type ignore: mypy complains about potential duplicate arguments from *args
        # but our API design allows passing additional args/kwargs to pyarrow
        return pq.write_to_dataset(  # type: ignore[misc]
            pa_table,
            root_path=path,
            partition_cols=partition_cols,
            *args,
            **kwargs,
        )


@ioutils.doc_merge_parquet_filemetadata()
def merge_parquet_filemetadata(filemetadata_list: list) -> np.ndarray:
    """{docstring}"""
    return np.asarray(
        plc.io.parquet.merge_row_group_metadata(filemetadata_list).obj
    )


def _generate_filename():
    return uuid4().hex + ".parquet"


def _get_estimated_file_size(df):
    # NOTE: This is purely a guesstimation method
    # and the y = mx+c has been arrived
    # after extensive experimentation of parquet file size
    # vs dataframe sizes.
    df_mem_usage = df.memory_usage().sum()
    # Parquet file size of a dataframe with all unique values
    # seems to be 1/1.5 times as that of on GPU for >10000 rows
    # and 0.6 times else-wise.
    # Y(file_size) = M(0.6) * X(df_mem_usage) + C(705)
    file_size = int((df_mem_usage * 0.6) + 705)
    # 1000 Bytes accounted for row-group metadata.
    # A parquet file takes roughly ~810 Bytes of metadata per column.
    file_size = file_size + 1000 + (810 * df.shape[1])
    return file_size


@_performance_tracking
def _get_partitioned(
    df,
    root_path,
    partition_cols,
    filename=None,
    fs=None,
    preserve_index=False,
    storage_options=None,
):
    fs = ioutils._ensure_filesystem(
        fs, root_path, storage_options=storage_options
    )
    fs.mkdirs(root_path, exist_ok=True)

    part_names, grouped_df, part_offsets = _get_groups_and_offsets(
        df, partition_cols, preserve_index
    )

    full_paths = []
    metadata_file_paths = []
    for keys in part_names.itertuples(index=False):
        subdir = fs.sep.join(
            [
                _hive_dirname(name, val)
                for name, val in zip(partition_cols, keys, strict=True)
            ]
        )
        prefix = fs.sep.join([root_path, subdir])
        fs.mkdirs(prefix, exist_ok=True)
        filename = filename or _generate_filename()
        full_path = fs.sep.join([prefix, filename])
        full_paths.append(full_path)
        metadata_file_paths.append(fs.sep.join([subdir, filename]))

    return full_paths, metadata_file_paths, grouped_df, part_offsets, filename


@_performance_tracking
def _get_groups_and_offsets(
    df,
    partition_cols,
    preserve_index=False,
    **kwargs,
):
    if not (set(df._data) - set(partition_cols)):
        warnings.warn("No data left to save outside partition columns")

    _, part_offsets, part_keys, grouped_df = df.groupby(
        partition_cols,
        dropna=False,
    )._grouped()
    if not preserve_index:
        grouped_df.reset_index(drop=True, inplace=True)
    grouped_df.drop(columns=partition_cols, inplace=True)
    # Copy the entire keys df in one operation rather than using iloc
    part_names = (
        part_keys.take(part_offsets[:-1])
        .to_pandas(nullable=True)
        .to_frame(index=False)
    )
    return part_names, grouped_df, part_offsets


class ParquetWriter:
    """
    ParquetWriter lets you incrementally write out a Parquet file from a series
    of cudf tables

    Parameters
    ----------
    filepath_or_buffer : str, io.IOBase, os.PathLike, or list
        File path or buffer to write to. The argument may also correspond
        to a list of file paths or buffers.
    index : bool or None, default None
        If ``True``, include a dataframe's index(es) in the file output.
        If ``False``, they will not be written to the file. If ``None``,
        index(es) other than RangeIndex will be saved as columns.
    compression : {'snappy', None}, default 'snappy'
        Name of the compression to use. Use ``None`` for no compression.
    statistics : {'ROWGROUP', 'PAGE', 'COLUMN', 'NONE'}, default 'ROWGROUP'
        Level at which column statistics should be included in file.
    row_group_size_bytes: int, default ``uint64 max``
        Maximum size of each stripe of the output.
        By default, a virtually infinite size equal to ``uint64 max`` will be used.
    row_group_size_rows: int, default 1000000
        Maximum number of rows of each stripe of the output.
        By default, 1000000 (10^6 rows) will be used.
    max_page_size_bytes: int, default 524288
        Maximum uncompressed size of each page of the output.
        By default, 524288 (512KB) will be used.
    max_page_size_rows: int, default 20000
        Maximum number of rows of each page of the output.
        By default, 20000 will be used.
    max_dictionary_size: int, default 1048576
        Maximum size of the dictionary page for each output column chunk. Dictionary
        encoding for column chunks that exceeds this limit will be disabled.
        By default, 1048576 (1MB) will be used.
    use_dictionary : bool, default True
        If ``True``, enable dictionary encoding for Parquet page data
        subject to ``max_dictionary_size`` constraints.
        If ``False``, disable dictionary encoding for Parquet page data.
    store_schema : bool, default False
        If ``True``, enable computing and writing arrow schema to Parquet
        file footer's key-value metadata section for faithful round-tripping.

    See Also
    --------
    cudf.io.parquet.write_parquet
    """

    def __init__(
        self,
        filepath_or_buffer,
        index: bool | None = None,
        compression: Literal["snappy", "ZSTD", "ZLIB", "LZ4", None] = "snappy",
        statistics: Literal["ROWGROUP", "PAGE", "COLUMN", "NONE"] = "ROWGROUP",
        row_group_size_bytes: int = int(np.iinfo(np.uint64).max),
        row_group_size_rows: int = 1000000,
        max_page_size_bytes: int = 524288,
        max_page_size_rows: int = 20000,
        max_dictionary_size: int = 1048576,
        use_dictionary: bool = True,
        store_schema: bool = False,
    ):
        filepaths_or_buffers = (
            list(filepath_or_buffer)
            if is_list_like(filepath_or_buffer)
            else [filepath_or_buffer]
        )
        self.sink = plc.io.SinkInfo(filepaths_or_buffers)
        self.statistics = statistics
        self.compression = compression
        self.index = index
        self.initialized = False
        self.row_group_size_bytes = row_group_size_bytes
        self.row_group_size_rows = row_group_size_rows
        self.max_page_size_bytes = max_page_size_bytes
        self.max_page_size_rows = max_page_size_rows
        self.max_dictionary_size = max_dictionary_size
        self.use_dictionary = use_dictionary
        self.write_arrow_schema = store_schema

    def write_table(self, table, partitions_info=None) -> None:
        """Writes a single table to the file"""
        if not self.initialized:
            self._initialize_chunked_state(
                table,
                num_partitions=len(partitions_info) if partitions_info else 1,
            )
        if self.index is not False and (
            table.index.name is not None or isinstance(table.index, MultiIndex)
        ):
            columns = itertools.chain(table.index._columns, table._columns)
            plc_table = plc.Table([col.plc_column for col in columns])
        else:
            plc_table = plc.Table([col.plc_column for col in table._columns])
        self.writer.write(plc_table, partitions_info)

    def close(self, metadata_file_path=None) -> np.ndarray | None:
        if not self.initialized:
            return None
        column_chunks_file_paths = []
        if metadata_file_path is not None:
            if is_list_like(metadata_file_path):
                column_chunks_file_paths = list(metadata_file_path)
            else:
                column_chunks_file_paths = [metadata_file_path]
        blob = self.writer.close(column_chunks_file_paths)
        if metadata_file_path is not None:
            return np.asarray(blob.obj)
        return None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def _initialize_chunked_state(
        self, table, num_partitions: int = 1
    ) -> None:
        """Prepares all the values required to build the
        chunked_parquet_writer_options and creates a writer
        """

        # Set the table_metadata
        num_index_cols_meta = 0
        plc_table = plc.Table([col.plc_column for col in table._columns])
        self.tbl_meta = plc.io.types.TableInputMetadata(plc_table)
        if self.index is not False:
            if isinstance(table.index, MultiIndex):
                plc_table = plc.Table(
                    [
                        col.plc_column
                        for col in itertools.chain(
                            table.index._columns, table._columns
                        )
                    ]
                )
                self.tbl_meta = plc.io.types.TableInputMetadata(plc_table)
                for level, idx_name in enumerate(table.index.names):
                    self.tbl_meta.column_metadata[level].set_name(idx_name)
                num_index_cols_meta = len(table.index.names)
            else:
                if table.index.name is not None:
                    plc_table = plc.Table(
                        [
                            col.plc_column
                            for col in itertools.chain(
                                table.index._columns, table._columns
                            )
                        ]
                    )
                    self.tbl_meta = plc.io.types.TableInputMetadata(plc_table)
                    self.tbl_meta.column_metadata[0].set_name(table.index.name)
                    num_index_cols_meta = 1

        for i, name in enumerate(table._column_names, num_index_cols_meta):
            self.tbl_meta.column_metadata[i].set_name(name)
            _set_col_metadata(
                table[name]._column.dtype,
                self.tbl_meta.column_metadata[i],
            )

        index = False if isinstance(table.index, RangeIndex) else self.index
        user_data = [
            {"pandas": ioutils.generate_pandas_metadata(table, index)}
        ] * num_partitions
        comp_type = _get_comp_type(self.compression)
        stat_freq = _get_stat_freq(self.statistics)
        dict_policy = (
            plc.io.types.DictionaryPolicy.ADAPTIVE
            if self.use_dictionary
            else plc.io.types.DictionaryPolicy.NEVER
        )
        options = (
            plc.io.parquet.ChunkedParquetWriterOptions.builder(self.sink)
            .metadata(self.tbl_meta)
            .key_value_metadata(user_data)
            .compression(comp_type)
            .stats_level(stat_freq)
            .row_group_size_bytes(self.row_group_size_bytes)
            .row_group_size_rows(self.row_group_size_rows)
            .max_page_size_bytes(self.max_page_size_bytes)
            .max_page_size_rows(self.max_page_size_rows)
            .max_dictionary_size(self.max_dictionary_size)
            .write_arrow_schema(self.write_arrow_schema)
            .build()
        )
        options.set_dictionary_policy(dict_policy)
        self.writer = plc.io.parquet.ChunkedParquetWriter.from_options(options)
        self.initialized = True


def _parse_bytes(s: str) -> int:
    """Parse byte string to numbers

    Utility function vendored from Dask.

    >>> _parse_bytes('100')
    100
    >>> _parse_bytes('100 MB')
    100000000
    >>> _parse_bytes('100M')
    100000000
    >>> _parse_bytes('5kB')
    5000
    >>> _parse_bytes('5.4 kB')
    5400
    >>> _parse_bytes('1kiB')
    1024
    >>> _parse_bytes('1e6')
    1000000
    >>> _parse_bytes('1e6 kB')
    1000000000
    >>> _parse_bytes('MB')
    1000000
    >>> _parse_bytes(123)
    123
    >>> _parse_bytes('5 foos')
    Traceback (most recent call last):
        ...
    ValueError: Could not interpret 'foos' as a byte unit
    """
    if isinstance(s, (int, float)):
        return int(s)
    s = s.replace(" ", "")
    if not any(char.isdigit() for char in s):
        s = "1" + s

    for i in range(len(s) - 1, -1, -1):
        if not s[i].isalpha():
            break
    index = i + 1

    prefix = s[:index]
    suffix = s[index:]

    try:
        n = float(prefix)
    except ValueError as e:
        raise ValueError(
            "Could not interpret '%s' as a number" % prefix
        ) from e

    try:
        multiplier = BYTE_SIZES[suffix.lower()]
    except KeyError as e:
        raise ValueError(
            "Could not interpret '%s' as a byte unit" % suffix
        ) from e

    result = n * multiplier
    return int(result)


class ParquetDatasetWriter:
    """
    Write a parquet file or dataset incrementally

    Parameters
    ----------
    path : str
        A local directory path or S3 URL. Will be used as root directory
        path while writing a partitioned dataset.
    partition_cols : list
        Column names by which to partition the dataset
        Columns are partitioned in the order they are given
    index : bool, default None
        If ``True``, include the dataframe's index(es) in the file output.
        If ``False``, they will not be written to the file. If ``None``,
        index(es) other than RangeIndex will be saved as columns.
    compression : {'snappy', None}, default 'snappy'
        Name of the compression to use. Use ``None`` for no compression.
    statistics : {'ROWGROUP', 'PAGE', 'COLUMN', 'NONE'}, default 'ROWGROUP'
        Level at which column statistics should be included in file.
    max_file_size : int or str, default None
        A file size that cannot be exceeded by the writer.
        It is in bytes, if the input is int.
        Size can also be a str in form or "10 MB", "1 GB", etc.
        If this parameter is used, it is mandatory to pass
        `file_name_prefix`.
    file_name_prefix : str
        This is a prefix to file names generated only when
        `max_file_size` is specified.
    storage_options : dict, optional, default None
        Extra options that make sense for a particular storage connection,
        e.g. host, port, username, password, etc. For HTTP(S) URLs the
        key-value pairs are forwarded to ``urllib.request.Request`` as
        header options. For other URLs (e.g. starting with "s3://", and
        "gcs://") the key-value pairs are forwarded to ``fsspec.open``.
        Please see ``fsspec`` and ``urllib`` for more details.


    Examples
    --------
    Using a context

    >>> df1 = cudf.DataFrame({"a": [1, 1, 2, 2, 1], "b": [9, 8, 7, 6, 5]})
    >>> df2 = cudf.DataFrame({"a": [1, 3, 3, 1, 3], "b": [4, 3, 2, 1, 0]})
    >>> with ParquetDatasetWriter("./dataset", partition_cols=["a"]) as cw:
    ...     cw.write_table(df1)
    ...     cw.write_table(df2)

    By manually calling ``close()``

    >>> cw = ParquetDatasetWriter("./dataset", partition_cols=["a"])
    >>> cw.write_table(df1)
    >>> cw.write_table(df2)
    >>> cw.close()

    Both the methods will generate the same directory structure

    .. code-block:: none

        dataset/
            a=1
                <filename>.parquet
            a=2
                <filename>.parquet
            a=3
                <filename>.parquet

    """

    @_performance_tracking
    def __init__(
        self,
        path,
        partition_cols,
        index=None,
        compression: Literal["snappy", "ZSTD", "ZLIB", "LZ4", None] = "snappy",
        statistics: Literal["ROWGROUP", "PAGE", "COLUMN", "NONE"] = "ROWGROUP",
        max_file_size=None,
        file_name_prefix=None,
        storage_options=None,
    ) -> None:
        if isinstance(path, str) and path.startswith("s3://"):
            self.fs_meta = {"is_s3": True, "actual_path": path}
            self.dir_: tempfile.TemporaryDirectory | None = (
                tempfile.TemporaryDirectory()
            )
            self.path = self.dir_.name
        else:
            self.fs_meta = {}
            self.dir_ = None
            self.path = path

        self.common_args = {
            "index": index,
            "compression": compression,
            "statistics": statistics,
        }
        self.partition_cols = partition_cols
        # Collection of `ParquetWriter`s, and the corresponding
        # partition_col values they're responsible for
        self._chunked_writers: list[tuple[ParquetWriter, list[str], str]] = []
        # Map of partition_col values to their ParquetWriter's index
        # in self._chunked_writers for reverse lookup
        self.path_cw_map: dict[str, int] = {}
        self.storage_options = storage_options
        self.filename = file_name_prefix
        self.max_file_size = max_file_size
        if max_file_size is not None:
            if file_name_prefix is None:
                raise ValueError(
                    "file_name_prefix cannot be None if max_file_size is "
                    "passed"
                )
            self.max_file_size = _parse_bytes(max_file_size)

        self._file_sizes: dict[str, int] = {}

    @_performance_tracking
    def write_table(self, df):
        """
        Write a dataframe to the file/dataset
        """
        part_names, grouped_df, part_offsets = _get_groups_and_offsets(
            df=df,
            partition_cols=self.partition_cols,
            preserve_index=self.common_args["index"],
        )
        fs = ioutils._ensure_filesystem(None, self.path, None)
        fs.mkdirs(self.path, exist_ok=True)

        full_paths = []
        metadata_file_paths = []
        full_offsets = [0]

        for idx, keys in enumerate(part_names.itertuples(index=False)):
            subdir = fs.sep.join(
                [
                    f"{name}={val}"
                    for name, val in zip(
                        self.partition_cols, keys, strict=True
                    )
                ]
            )
            prefix = fs.sep.join([self.path, subdir])
            fs.mkdirs(prefix, exist_ok=True)
            current_offset = (part_offsets[idx], part_offsets[idx + 1])
            num_chunks = 1
            parts = 1

            if self.max_file_size is not None:
                # get the current partition
                start, end = current_offset
                sliced_df = grouped_df[start:end]

                current_file_size = _get_estimated_file_size(sliced_df)
                if current_file_size > self.max_file_size:
                    # if the file is too large, compute metadata for
                    # smaller chunks
                    parts = math.ceil(current_file_size / self.max_file_size)
                    new_offsets = list(
                        range(start, end, int((end - start) / parts))
                    )[1:]
                    new_offsets.append(end)
                    num_chunks = len(new_offsets)
                    parts = len(new_offsets)
                    full_offsets.extend(new_offsets)
                else:
                    full_offsets.append(end)

                curr_file_num = 0
                num_chunks = 0
                while num_chunks < parts:
                    new_file_name = f"{self.filename}_{curr_file_num}.parquet"
                    new_full_path = fs.sep.join([prefix, new_file_name])

                    # Check if the same `new_file_name` exists and
                    # generate a `new_file_name`
                    while new_full_path in self._file_sizes and (
                        self._file_sizes[new_full_path]
                        + (current_file_size / parts)
                    ) > (self.max_file_size):
                        curr_file_num += 1
                        new_file_name = (
                            f"{self.filename}_{curr_file_num}.parquet"
                        )
                        new_full_path = fs.sep.join([prefix, new_file_name])

                    self._file_sizes[new_full_path] = self._file_sizes.get(
                        new_full_path, 0
                    ) + (current_file_size / parts)
                    full_paths.append(new_full_path)
                    metadata_file_paths.append(
                        fs.sep.join([subdir, new_file_name])
                    )
                    num_chunks += 1
                    curr_file_num += 1
            else:
                self.filename = self.filename or _generate_filename()
                full_path = fs.sep.join([prefix, self.filename])
                full_paths.append(full_path)
                metadata_file_paths.append(
                    fs.sep.join([subdir, self.filename])
                )
                full_offsets.append(current_offset[1])

        paths, metadata_file_paths, offsets = (
            full_paths,
            metadata_file_paths,
            full_offsets,
        )
        existing_cw_batch = defaultdict(dict)
        new_cw_paths = []
        partition_info = [(i, j - i) for i, j in itertools.pairwise(offsets)]

        for path, part_info, meta_path in zip(
            paths,
            partition_info,
            metadata_file_paths,
            strict=True,
        ):
            if path in self.path_cw_map:  # path is a currently open file
                cw_idx = self.path_cw_map[path]
                existing_cw_batch[cw_idx][path] = part_info
            else:  # path not currently handled by any chunked writer
                new_cw_paths.append((path, part_info, meta_path))

        # Write out the parts of grouped_df currently handled by existing cw's
        for cw_idx, path_to_part_info_map in existing_cw_batch.items():
            cw = self._chunked_writers[cw_idx][0]
            # match found paths with this cw's paths and nullify partition info
            # for partition_col values not in this batch
            this_cw_part_info = [
                path_to_part_info_map.get(path, (0, 0))
                for path in self._chunked_writers[cw_idx][1]
            ]
            cw.write_table(grouped_df, this_cw_part_info)

        if new_cw_paths:
            # Create new cw for unhandled paths encountered in this write_table
            new_paths, part_info, meta_paths = zip(*new_cw_paths, strict=True)
            self._chunked_writers.append(
                (
                    ParquetWriter(new_paths, **self.common_args),
                    new_paths,
                    meta_paths,
                )
            )
            new_cw_idx = len(self._chunked_writers) - 1
            self.path_cw_map.update({k: new_cw_idx for k in new_paths})
            self._chunked_writers[-1][0].write_table(grouped_df, part_info)

    @_performance_tracking
    def close(self, return_metadata=False):
        """
        Close all open files and optionally return footer metadata as a binary
        blob
        """

        metadata = [
            cw.close(metadata_file_path=meta_path if return_metadata else None)
            for cw, _, meta_path in self._chunked_writers
        ]

        if self.fs_meta.get("is_s3", False):
            local_path = self.path
            s3_path = self.fs_meta["actual_path"]
            s3_file, _ = ioutils._get_filesystem_and_paths(
                s3_path, storage_options=self.storage_options
            )
            s3_file.put(local_path, s3_path, recursive=True)
            shutil.rmtree(self.path)

        if self.dir_ is not None:
            self.dir_.cleanup()

        if return_metadata:
            return (
                merge_parquet_filemetadata(metadata)
                if len(metadata) > 1
                else metadata[0]
            )

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def _hive_dirname(name, val):
    # Simple utility to produce hive directory name
    if pd.isna(val):
        val = "__HIVE_DEFAULT_PARTITION__"
    return f"{name}={val}"


def _set_col_metadata(
    dtype: DtypeObj,
    col_meta: plc.io.types.ColumnInMetadata,
    force_nullable_schema: bool = False,
    path: str | None = None,
    skip_compression: set[Hashable] | None = None,
    column_encoding: dict[
        Hashable,
        Literal[
            "PLAIN",
            "DICTIONARY",
            "DELTA_BINARY_PACKED",
            "DELTA_LENGTH_BYTE_ARRAY",
            "DELTA_BYTE_ARRAY",
            "BYTE_STREAM_SPLIT",
            "USE_DEFAULT",
        ],
    ]
    | None = None,
    column_type_length: dict | None = None,
    output_as_binary: set[Hashable] | None = None,
) -> None:
    need_path = (
        skip_compression is not None
        or column_encoding is not None
        or column_type_length is not None
        or output_as_binary is not None
    )
    name = col_meta.get_name() if need_path else None
    full_path = (
        path + "." + name if (path is not None and name is not None) else name
    )

    if force_nullable_schema:
        # Only set nullability if `force_nullable_schema`
        # is true.
        col_meta.set_nullability(True)

    if skip_compression is not None and full_path in skip_compression:
        col_meta.set_skip_compression(True)

    if column_encoding is not None and full_path in column_encoding:
        encoding = column_encoding[full_path]
        if encoding is None:
            c_encoding = plc.io.types.ColumnEncoding.USE_DEFAULT
        else:
            enc = str(encoding).upper()
            c_encoding = getattr(plc.io.types.ColumnEncoding, enc, None)
            if c_encoding is None:
                raise ValueError("Unsupported `column_encoding` type")
        col_meta.set_encoding(c_encoding)

    if column_type_length is not None and full_path in column_type_length:
        col_meta.set_output_as_binary(True)
        col_meta.set_type_length(column_type_length[full_path])

    if output_as_binary is not None and full_path in output_as_binary:
        col_meta.set_output_as_binary(True)

    if isinstance(dtype, StructDtype):
        for i, (name, field_dtype) in enumerate(dtype.fields.items()):
            col_meta.child(i).set_name(name)
            _set_col_metadata(
                field_dtype,
                col_meta.child(i),
                force_nullable_schema,
                full_path,
                skip_compression,
                column_encoding,
                column_type_length,
                output_as_binary,
            )
    elif isinstance(dtype, ListDtype):
        if full_path is not None:
            full_path = full_path + ".list"
            col_meta.child(1).set_name("element")
        _set_col_metadata(
            dtype.element_type,
            col_meta.child(1),
            force_nullable_schema,
            full_path,
            skip_compression,
            column_encoding,
            column_type_length,
            output_as_binary,
        )
    elif isinstance(dtype, DecimalDtype):
        col_meta.set_decimal_precision(dtype.precision)


def _get_comp_type(
    compression: Literal[
        "snappy",
        "SNAPPY",
        "GZIP",
        "BROTLI",
        "LZ4",
        "ZSTD",
        "ZLIB",
        "NONE",
        None,
    ],
) -> plc.io.types.CompressionType:
    if compression is None or compression == "NONE":
        return plc.io.types.CompressionType.NONE
    result = getattr(plc.io.types.CompressionType, compression.upper(), None)
    if result is None:
        raise ValueError("Unsupported `compression` type")
    return result


def _get_stat_freq(
    statistics: Literal["ROWGROUP", "PAGE", "COLUMN", "NONE"],
) -> plc.io.types.StatisticsFreq:
    result = getattr(
        plc.io.types.StatisticsFreq, f"STATISTICS_{statistics.upper()}", None
    )
    if result is None:
        raise ValueError("Unsupported `statistics_freq` type")
    return result


def _process_metadata(
    df: DataFrame,
    names: Sequence[Hashable],
    per_file_user_data: list,
    row_groups,
    filepaths_or_buffers,
    allow_range_index: bool,
    use_pandas_metadata: bool,
    nrows: int = -1,
    skip_rows: int = 0,
) -> DataFrame:
    index_col = None
    is_range_index = True
    column_index_type = None
    index_col_names = None
    meta = None
    for single_file in per_file_user_data:
        if b"pandas" not in single_file:
            continue
        json_str = single_file[b"pandas"].decode("utf-8")
        meta = json.loads(json_str)
        file_is_range_index, index_col, column_index_type = _parse_metadata(
            meta
        )
        is_range_index &= file_is_range_index

        if (
            not file_is_range_index
            and index_col is not None
            and index_col_names is None
        ):
            index_col_names = {}
            for idx_col in index_col:
                for c in meta["columns"]:
                    if c["field_name"] == idx_col:
                        index_col_names[idx_col] = c["name"]

    if meta is not None:
        # Book keep each column metadata as the order
        # of `meta["columns"]` and `column_names` are not
        # guaranteed to be deterministic and same always.
        meta_data_per_column = {
            col_meta["name"]: col_meta for col_meta in meta["columns"]
        }

        # update the decimal precision of each column
        for col in names:
            if isinstance(df._data[col].dtype, DecimalDtype):
                df._data[col].dtype.precision = meta_data_per_column[col][  # type: ignore[union-attr]
                    "metadata"
                ]["precision"]
            elif (
                isinstance(df._data[col].dtype, pd.StringDtype)
                and col in meta_data_per_column
                and meta_data_per_column[col]["pandas_type"] == "empty"
                and meta_data_per_column[col]["numpy_type"] == "object"
            ):
                df._data[col] = df._data[col].astype(np.dtype("object"))

    # Set the index column
    if index_col is not None and len(index_col) > 0:
        if is_range_index:
            if not allow_range_index:
                return df

            if len(per_file_user_data) > 1:
                range_index_meta = {
                    "kind": "range",
                    "name": None,
                    "start": 0,
                    "stop": len(df),
                    "step": 1,
                }
            else:
                range_index_meta = index_col[0]

            if row_groups is not None:
                import pyarrow.parquet as pq

                per_file_metadata = [
                    pq.read_metadata(
                        # Pyarrow cannot read directly from bytes
                        io.BytesIO(s) if isinstance(s, bytes) else s
                    )
                    for s in filepaths_or_buffers
                ]

                filtered_idx = []
                for i, file_meta in enumerate(per_file_metadata):
                    row_groups_i = []
                    start = 0
                    for row_group in range(file_meta.num_row_groups):
                        stop = start + file_meta.row_group(row_group).num_rows
                        row_groups_i.append((start, stop))
                        start = stop

                    for rg in row_groups[i]:
                        filtered_idx.append(
                            RangeIndex(
                                start=row_groups_i[rg][0],
                                stop=row_groups_i[rg][1],
                                step=range_index_meta["step"],
                            )
                        )

                if len(filtered_idx) > 0:
                    idx = concat(filtered_idx)
                else:
                    idx = Index._from_column(column_empty(0))
            else:
                start = range_index_meta["start"] + skip_rows  # type: ignore[operator]
                stop = int(range_index_meta["stop"])  # type: ignore[arg-type]
                if nrows > -1:
                    stop = start + nrows
                idx = RangeIndex(
                    start=start,
                    stop=stop,
                    step=range_index_meta["step"],
                    name=range_index_meta["name"],
                )

            df.index = idx
        elif set(index_col).issubset(names):
            index_data = df[index_col]
            actual_index_names = iter(index_col_names.values())
            if index_data._num_columns == 1:
                idx = Index._from_column(
                    index_data._columns[0], name=next(actual_index_names)
                )
            else:
                idx = MultiIndex.from_frame(
                    index_data, names=list(actual_index_names)
                )
            df.drop(columns=index_col, inplace=True)
            df.index = idx
        else:
            if use_pandas_metadata:
                df.index.names = index_col

    if df._num_columns == 0 and column_index_type is not None:
        df._data.label_dtype = column_index_type

    return df


def is_supported_read_parquet(
    compression: Literal["SNAPPY", "GZIP", "BROTLI", "LZ4", "ZSTD", "NONE"],
) -> bool:
    """Check if the compression type is supported for reading Parquet files.

    Parameters
    ----------
    compression : str
        The compression type to check (e.g., "SNAPPY", "GZIP", "LZ4")

    Returns
    -------
    bool
        True if the compression type is supported for reading Parquet files

    Examples
    --------
    >>> import cudf
    >>> cudf.io.parquet.is_supported_read_parquet("LZ4")  # doctest: +SKIP
    True
    """
    return plc.io.parquet.is_supported_read_parquet(
        _get_comp_type(compression)
    )


def is_supported_write_parquet(
    compression: Literal["SNAPPY", "GZIP", "BROTLI", "LZ4", "ZSTD", "NONE"],
) -> bool:
    """Check if the compression type is supported for writing Parquet files.

    Parameters
    ----------
    compression : str
        The compression type to check (e.g., "SNAPPY", "GZIP", "LZ4")

    Returns
    -------
    bool
        True if the compression type is supported for writing Parquet files

    Examples
    --------
    >>> import cudf
    >>> cudf.io.parquet.is_supported_write_parquet("SNAPPY")  # doctest: +SKIP
    True
    """
    return plc.io.parquet.is_supported_write_parquet(
        _get_comp_type(compression)
    )
