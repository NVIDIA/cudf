# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import itertools
import json
import warnings
from typing import TYPE_CHECKING, Literal

import pyarrow as pa

import pylibcudf as plc

from cudf.api.types import is_list_like
from cudf.core.column import access_columns, column_empty
from cudf.core.dataframe import DataFrame
from cudf.core.dtypes import (
    CategoricalDtype,
    ListDtype,
    StructDtype,
    dtype as cudf_dtype,
)
from cudf.core.index import CategoricalIndex, RangeIndex
from cudf.core.multiindex import MultiIndex
from cudf.utils import ioutils
from cudf.utils.dtypes import cudf_dtype_from_pa_type, dtype_to_pylibcudf_type

if TYPE_CHECKING:
    from cudf._typing import DtypeObj


@ioutils.doc_read_orc_metadata()
def read_orc_metadata(path):
    """{docstring}"""
    from pyarrow import orc

    orc_file = orc.ORCFile(path)

    num_rows = orc_file.nrows
    num_stripes = orc_file.nstripes
    col_names = orc_file.schema.names

    return num_rows, num_stripes, col_names


@ioutils.doc_read_orc_statistics()
def read_orc_statistics(
    filepaths_or_buffers,
    columns=None,
    **kwargs,
):
    """{docstring}"""

    files_statistics = []
    stripes_statistics = []
    for source in filepaths_or_buffers:
        path_or_buf = ioutils.get_reader_filepath_or_buffer(
            path_or_data=source, **kwargs
        )
        path_or_buf = ioutils._select_single_source(
            path_or_buf, "read_orc_statistics"
        )
        parsed = plc.io.orc.read_parsed_orc_statistics(
            plc.io.SourceInfo([path_or_buf])
        )
        column_names = parsed.column_names
        parsed_file_statistics = parsed.file_stats
        parsed_stripes_statistics = parsed.stripes_stats

        # Parse file statistics
        file_statistics = {
            column_name: column_stats
            for column_name, column_stats in zip(
                column_names, parsed_file_statistics, strict=True
            )
            if columns is None or column_name in columns
        }
        files_statistics.append(file_statistics)

        # Parse stripe statistics
        for parsed_stripe_statistics in parsed_stripes_statistics:
            stripe_statistics = {
                column_name: column_stats
                for column_name, column_stats in zip(
                    column_names, parsed_stripe_statistics, strict=True
                )
                if columns is None or column_name in columns
            }
            if any(
                not parsed_statistics
                for parsed_statistics in stripe_statistics.values()
            ):
                continue
            else:
                stripes_statistics.append(stripe_statistics)

    return files_statistics, stripes_statistics


def _filter_stripes(
    filters, filepath_or_buffer, stripes=None, skip_rows=None, num_rows=None
):
    # Multiple sources are passed as a list. If a single source is passed,
    # wrap it in a list for unified processing downstream.
    if not is_list_like(filepath_or_buffer):
        filepath_or_buffer = [filepath_or_buffer]

    # Prepare filters
    filters = ioutils._prepare_filters(filters)

    # Get columns relevant to filtering
    columns_in_predicate = [
        col for conjunction in filters for (col, op, val) in conjunction
    ]

    # Read and parse file-level and stripe-level statistics
    file_statistics, stripes_statistics = read_orc_statistics(
        filepath_or_buffer, columns_in_predicate
    )

    file_stripe_map = []
    for file_stat in file_statistics:
        # Filter using file-level statistics
        if not ioutils._apply_filters(filters, file_stat):
            continue

        # Filter using stripe-level statistics
        selected_stripes = []
        num_rows_scanned = 0
        for i, stripe_statistics in enumerate(stripes_statistics):
            num_rows_before_stripe = num_rows_scanned
            num_rows_scanned += next(
                iter(stripe_statistics.values())
            ).number_of_values
            if stripes is not None and i not in stripes:
                continue
            if skip_rows is not None and num_rows_scanned <= skip_rows:
                continue
            else:
                skip_rows = 0
            if (
                skip_rows is not None
                and num_rows is not None
                and num_rows_before_stripe >= skip_rows + num_rows
            ):
                continue
            if ioutils._apply_filters(filters, stripe_statistics):
                selected_stripes.append(i)

        file_stripe_map.append(selected_stripes)

    return file_stripe_map


@ioutils.doc_read_orc()
def read_orc(
    filepath_or_buffer,
    engine="cudf",
    columns=None,
    filters=None,
    stripes=None,
    skiprows: int | None = None,
    num_rows: int | None = None,
    use_index: bool = True,
    timestamp_type=None,
    storage_options=None,
    bytes_per_thread=None,
    filesystem=None,
):
    """{docstring}"""
    if skiprows is not None:
        # Do not remove until cuIO team approves its removal.
        warnings.warn(
            "skiprows is deprecated and will be removed.",
            FutureWarning,
        )

    if num_rows is not None:
        # Do not remove until cuIO team approves its removal.
        warnings.warn(
            "num_rows is deprecated and will be removed.",
            FutureWarning,
        )

    # Multiple sources are passed as a list. If a single source is passed,
    # wrap it in a list for unified processing downstream.
    if not is_list_like(filepath_or_buffer):
        filepath_or_buffer = [filepath_or_buffer]

    # Each source must have a correlating stripe list. If a single stripe list
    # is provided rather than a list of list of stripes then extrapolate that
    # stripe list across all input sources
    if stripes is not None:
        if any(not isinstance(stripe, list) for stripe in stripes):
            stripes = [stripes]

        # Must ensure a stripe for each source is specified, unless None
        if not len(stripes) == len(filepath_or_buffer):
            raise ValueError(
                "A list of stripes must be provided for each input source"
            )

    filepaths_or_buffers = ioutils.get_reader_filepath_or_buffer(
        path_or_data=filepath_or_buffer,
        storage_options=storage_options,
        filesystem=filesystem,
        bytes_per_thread=bytes_per_thread,
        expand_dir_pattern="*.orc",
    )

    if filters is not None:
        selected_stripes = _filter_stripes(
            filters, filepaths_or_buffers, stripes, skiprows, num_rows
        )

        # Return empty if everything was filtered
        if len(selected_stripes) == 0:
            from pyarrow import orc

            orc_file = orc.ORCFile(filepaths_or_buffers[0])
            schema = orc_file.schema
            col_names = schema.names if columns is None else columns
            return DataFrame._from_data(
                data={
                    col_name: column_empty(
                        row_count=0,
                        dtype=cudf_dtype_from_pa_type(
                            schema.field(col_name).type
                        ),
                    )
                    for col_name in col_names
                }
            )
        else:
            stripes = selected_stripes

    if engine == "cudf":
        if columns is not None:
            columns = [str(col) for col in columns]

        if skiprows is None:
            skiprows = 0
        elif not isinstance(skiprows, int) or skiprows < 0:
            raise TypeError("skiprows must be an int >= 0")

        if num_rows is None:
            num_rows = -1
        elif not isinstance(num_rows, int) or num_rows < -1:
            raise TypeError("num_rows must be an int >= -1")

        options = (
            plc.io.orc.OrcReaderOptions.builder(
                plc.io.types.SourceInfo(filepaths_or_buffers)
            )
            .use_index(use_index)
            .build()
        )
        if num_rows >= 0:
            options.set_num_rows(num_rows)
        if skiprows >= 0:
            options.set_skip_rows(skiprows)
        if stripes is not None and len(stripes) > 0:
            options.set_stripes(stripes)
        if timestamp_type is not None:
            options.set_timestamp_type(
                dtype_to_pylibcudf_type(cudf_dtype(timestamp_type))
            )
        if columns is not None and len(columns) > 0:
            options.set_columns(columns)

        tbl_w_meta = plc.io.orc.read_orc(options)
        df = DataFrame.from_pylibcudf(tbl_w_meta)

        if isinstance(columns, list) and len(columns) == 0:
            # Index to deselect all columns
            df = df.loc[:, columns]
        else:
            index_col = None
            is_range_index = False
            reset_index_name = False
            range_idx = None

            if len(tbl_w_meta.per_file_user_data) > 0:
                json_str = (
                    tbl_w_meta.per_file_user_data[0]
                    .get(b"pandas", b"")
                    .decode("utf-8")
                )
                if json_str != "":
                    meta = json.loads(json_str)
                    if (
                        "index_columns" in meta
                        and len(meta["index_columns"]) > 0
                    ):
                        index_col = meta["index_columns"]
                        if (
                            isinstance(index_col[0], dict)
                            and index_col[0]["kind"] == "range"
                        ):
                            is_range_index = True
                        else:
                            index_col_names = {}
                            for idx_col in index_col:
                                for c in meta["columns"]:
                                    if c["field_name"] == idx_col:
                                        index_col_names[idx_col] = (
                                            c["name"] or c["field_name"]
                                        )
                                        if c["name"] is None:
                                            reset_index_name = True

            if index_col is not None and len(index_col) > 0:
                if is_range_index:
                    range_index_meta = index_col[0]
                    range_idx = RangeIndex(
                        start=range_index_meta["start"],
                        stop=range_index_meta["stop"],
                        step=range_index_meta["step"],
                        name=range_index_meta["name"],
                    )
                    if skiprows != 0:
                        range_idx = range_idx[skiprows:]
                    if num_rows != -1:
                        range_idx = range_idx[:num_rows]
                    df.index = range_idx
                else:
                    df = df.set_index(list(index_col_names.values()))

            if reset_index_name:
                df.index.names = [None] * df.index.nlevels
        return df
    else:
        from pyarrow import orc

        warnings.warn("Using CPU via PyArrow to read ORC dataset.")
        if len(filepath_or_buffer) > 1:
            raise NotImplementedError(
                "Using CPU via PyArrow only supports a single a "
                "single input source"
            )

        orc_file = orc.ORCFile(filepath_or_buffer[0])
        if stripes is not None and len(stripes) > 0:
            for stripe_source_file in stripes:
                pa_tables = (
                    orc_file.read_stripe(i, columns)
                    for i in stripe_source_file
                )
                pa_table = pa.concat_tables(
                    [
                        pa.Table.from_batches([table])
                        if isinstance(table, pa.RecordBatch)
                        else table
                        for table in pa_tables
                    ]
                )
        else:
            pa_table = orc_file.read(columns=columns)
        df = DataFrame.from_arrow(pa_table)

    return df


@ioutils.doc_to_orc()
def to_orc(
    df: DataFrame,
    fname,
    compression: Literal[
        False, None, "SNAPPY", "ZLIB", "ZSTD", "LZ4"
    ] = "SNAPPY",
    statistics: Literal["NONE", "STRIPE", "ROWGROUP"] = "ROWGROUP",
    stripe_size_bytes: int | None = None,
    stripe_size_rows: int | None = None,
    row_index_stride: int | None = None,
    cols_as_map_type=None,
    storage_options=None,
    index: bool | None = None,
    filesystem=None,
):
    """{docstring}"""

    for _, dtype in df._dtypes:
        if isinstance(dtype, CategoricalDtype):
            raise NotImplementedError(
                "Writing to ORC format is not yet supported with "
                "Categorical columns."
            )

    if isinstance(df.index, CategoricalIndex):
        raise NotImplementedError(
            "Writing to ORC format is not yet supported with "
            "Categorical columns."
        )

    if cols_as_map_type is not None and not isinstance(cols_as_map_type, list):
        raise TypeError("cols_as_map_type must be a list of column names.")

    path_or_buf = ioutils.get_writer_filepath_or_buffer(
        path_or_data=fname,
        mode="wb",
        storage_options=storage_options,
        filesystem=filesystem,
    )
    if ioutils.is_fsspec_open_file(path_or_buf):
        with path_or_buf as file_obj:
            file_obj = ioutils.get_IOBase_writer(file_obj)
            _plc_write_orc(
                df,
                file_obj,
                compression,
                statistics,
                stripe_size_bytes,
                stripe_size_rows,
                row_index_stride,
                cols_as_map_type,
                index,
            )
    else:
        _plc_write_orc(
            df,
            path_or_buf,
            compression,
            statistics,
            stripe_size_bytes,
            stripe_size_rows,
            row_index_stride,
            cols_as_map_type,
            index,
        )


def _plc_write_orc(
    table: DataFrame,
    path_or_buf,
    compression: Literal[
        False, None, "SNAPPY", "ZLIB", "ZSTD", "LZ4"
    ] = "SNAPPY",
    statistics: Literal["NONE", "STRIPE", "ROWGROUP"] = "ROWGROUP",
    stripe_size_bytes: int | None = None,
    stripe_size_rows: int | None = None,
    row_index_stride: int | None = None,
    cols_as_map_type=None,
    index: bool | None = None,
) -> None:
    """
    See `cudf::io::write_orc`.

    See Also
    --------
    cudf.read_orc
    """
    user_data = {"pandas": ioutils.generate_pandas_metadata(table, index)}
    if index is True or (
        index is None and not isinstance(table.index, RangeIndex)
    ):
        iter_columns = (
            table._columns
            if table.index is None
            else itertools.chain(table.index._columns, table._columns)
        )
    else:
        iter_columns = table._columns

    with access_columns(*iter_columns, mode="read", scope="internal"):
        if index is True or (
            index is None and not isinstance(table.index, RangeIndex)
        ):
            columns = (
                table._columns
                if table.index is None
                else itertools.chain(table.index._columns, table._columns)
            )
            plc_table = plc.Table([col.plc_column for col in columns])
            tbl_meta = plc.io.types.TableInputMetadata(plc_table)
            for level, idx_name in enumerate(table._index.names):
                tbl_meta.column_metadata[level].set_name(
                    ioutils._index_level_name(
                        idx_name,
                        level,
                        table._column_names,  # type: ignore[arg-type]
                    )
                )
            num_index_cols_meta = table.index.nlevels
        else:
            plc_table = plc.Table([col.plc_column for col in table._columns])
            tbl_meta = plc.io.types.TableInputMetadata(plc_table)
            num_index_cols_meta = 0

        has_map_type = False
        if cols_as_map_type is not None:
            cols_as_map_type = set(cols_as_map_type)
            has_map_type = True

        for i, (name, col) in enumerate(
            table._column_labels_and_values, start=num_index_cols_meta
        ):
            # generate_pandas_metadata will reject tables with non-string column names
            tbl_meta.column_metadata[i].set_name(name)  # type: ignore[arg-type]
            _set_col_children_metadata(
                col.dtype,
                tbl_meta.column_metadata[i],
                has_map_type and name in cols_as_map_type,
            )

        options = (
            plc.io.orc.OrcWriterOptions.builder(
                plc.io.SinkInfo([path_or_buf]), plc_table
            )
            .metadata(tbl_meta)
            .key_value_metadata(user_data)
            .compression(_get_comp_type(compression))
            .enable_statistics(_get_orc_stat_freq(statistics))
            .build()
        )
        if stripe_size_bytes is not None:
            options.set_stripe_size_bytes(stripe_size_bytes)
        if stripe_size_rows is not None:
            options.set_stripe_size_rows(stripe_size_rows)
        if row_index_stride is not None:
            options.set_row_index_stride(row_index_stride)

        plc.io.orc.write_orc(options)


class ORCWriter:
    """
    ORCWriter lets you you incrementally write out a ORC file from a series
    of cudf tables

    See Also
    --------
    cudf.io.orc.to_orc
    """

    def __init__(
        self,
        path,
        index: bool | None = None,
        compression: Literal[
            False, None, "SNAPPY", "ZLIB", "ZSTD", "LZ4"
        ] = "SNAPPY",
        statistics: Literal["NONE", "STRIPE", "ROWGROUP"] = "ROWGROUP",
        cols_as_map_type=None,
        stripe_size_bytes: int | None = None,
        stripe_size_rows: int | None = None,
        row_index_stride: int | None = None,
    ):
        self.sink = plc.io.SinkInfo([path])
        self.statistics = statistics
        self.compression = compression
        self.index = index
        self.cols_as_map_type = (
            cols_as_map_type
            if cols_as_map_type is None
            else set(cols_as_map_type)
        )
        self.stripe_size_bytes = stripe_size_bytes
        self.stripe_size_rows = stripe_size_rows
        self.row_index_stride = row_index_stride
        self.initialized = False
        self.writer = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def write_table(self, table):
        """Writes a single table to the file"""
        if not self.initialized:
            self._initialize_chunked_state(table)

        keep_index = self.index is not False and (
            table.index.name is not None or isinstance(table.index, MultiIndex)
        )
        if keep_index:
            cols_to_write = itertools.chain(
                table.index._columns, table._columns
            )
        else:
            cols_to_write = table._columns

        self.writer.write(plc.Table([col.plc_column for col in cols_to_write]))

    def close(self):
        if not self.initialized:
            return
        self.writer.close()

    def _initialize_chunked_state(self, table):
        """
        Prepare all the values required to build the
        chunked_orc_writer_options anb creates a writer
        """

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
                num_index_cols_meta = table.index.nlevels
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

        has_map_type = self.cols_as_map_type is not None
        for i, (name, col) in enumerate(
            table._column_labels_and_values, start=num_index_cols_meta
        ):
            self.tbl_meta.column_metadata[i].set_name(name)
            _set_col_children_metadata(
                col.dtype,
                self.tbl_meta.column_metadata[i],
                has_map_type and name in self.cols_as_map_type,
            )

        user_data = {
            "pandas": ioutils.generate_pandas_metadata(table, self.index)
        }

        options = (
            plc.io.orc.ChunkedOrcWriterOptions.builder(self.sink)
            .metadata(self.tbl_meta)
            .key_value_metadata(user_data)
            .compression(_get_comp_type(self.compression))
            .enable_statistics(_get_orc_stat_freq(self.statistics))
            .build()
        )
        if self.stripe_size_bytes is not None:
            options.set_stripe_size_bytes(self.stripe_size_bytes)
        if self.stripe_size_rows is not None:
            options.set_stripe_size_rows(self.stripe_size_rows)
        if self.row_index_stride is not None:
            options.set_row_index_stride(self.row_index_stride)

        self.writer = plc.io.orc.OrcChunkedWriter.from_options(options)

        self.initialized = True


def _get_comp_type(
    compression: Literal[False, None, "SNAPPY", "ZLIB", "ZSTD", "LZ4", "NONE"],
) -> plc.io.types.CompressionType:
    if compression is None or compression is False or compression == "NONE":
        return plc.io.types.CompressionType.NONE

    normed_compression = compression.upper()
    if normed_compression == "SNAPPY":
        return plc.io.types.CompressionType.SNAPPY
    elif normed_compression == "ZLIB":
        return plc.io.types.CompressionType.ZLIB
    elif normed_compression == "ZSTD":
        return plc.io.types.CompressionType.ZSTD
    elif normed_compression == "LZ4":
        return plc.io.types.CompressionType.LZ4
    else:
        raise ValueError(f"Unsupported `compression` type {compression}")


def _get_orc_stat_freq(
    statistics: Literal["NONE", "STRIPE", "ROWGROUP"],
) -> plc.io.types.StatisticsFreq:
    """
    Convert ORC statistics terms to CUDF convention:
      - ORC "STRIPE"   == CUDF "ROWGROUP"
      - ORC "ROWGROUP" == CUDF "PAGE"
    """
    normed_statistics = statistics.upper()
    if normed_statistics == "NONE":
        return plc.io.types.StatisticsFreq.STATISTICS_NONE
    elif normed_statistics == "STRIPE":
        return plc.io.types.StatisticsFreq.STATISTICS_ROWGROUP
    elif normed_statistics == "ROWGROUP":
        return plc.io.types.StatisticsFreq.STATISTICS_PAGE
    else:
        raise ValueError(f"Unsupported `statistics_freq` type {statistics}")


def _set_col_children_metadata(
    dtype: DtypeObj,
    col_meta: plc.io.types.ColumnInMetadata,
    list_column_as_map: bool = False,
) -> None:
    if isinstance(dtype, StructDtype):
        for i, (name, field_dtype) in enumerate(dtype.fields.items()):
            col_meta.child(i).set_name(name)
            _set_col_children_metadata(
                field_dtype, col_meta.child(i), list_column_as_map
            )
    elif isinstance(dtype, ListDtype):
        if list_column_as_map:
            col_meta.set_list_column_as_map()
        _set_col_children_metadata(
            dtype.element_type, col_meta.child(1), list_column_as_map
        )
    else:
        return


def is_supported_read_orc(
    compression: Literal["SNAPPY", "ZLIB", "ZSTD", "LZ4", "NONE"],
) -> bool:
    """Check if the compression type is supported for reading ORC files.

    Parameters
    ----------
    compression : str
        The compression type to check (e.g., "SNAPPY", "ZLIB", "LZ4")

    Returns
    -------
    bool
        True if the compression type is supported for reading ORC files

    Examples
    --------
    >>> import cudf
    >>> cudf.io.orc.is_supported_read_orc("LZ4")  # doctest: +SKIP
    True
    """
    return plc.io.orc.is_supported_read_orc(_get_comp_type(compression))


def is_supported_write_orc(
    compression: Literal["SNAPPY", "ZLIB", "ZSTD", "LZ4", "NONE"],
) -> bool:
    """Check if the compression type is supported for writing ORC files.

    Parameters
    ----------
    compression : str
        The compression type to check (e.g., "SNAPPY", "ZLIB", "LZ4")

    Returns
    -------
    bool
        True if the compression type is supported for writing ORC files

    Examples
    --------
    >>> import cudf
    >>> cudf.io.orc.is_supported_write_orc("SNAPPY")  # doctest: +SKIP
    True
    """
    return plc.io.orc.is_supported_write_orc(_get_comp_type(compression))
