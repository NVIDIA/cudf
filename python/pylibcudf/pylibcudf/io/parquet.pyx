# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from cython.operator cimport dereference
import warnings

from libc.stdint cimport int32_t, int64_t, uint8_t

from libcpp cimport bool
from libcpp.memory cimport unique_ptr, make_unique
from libcpp.string cimport string
from libcpp.utility cimport move
from libcpp.vector cimport vector

from rmm.pylibrmm.stream cimport Stream
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource

from pylibcudf.contiguous_split cimport HostBuffer
from pylibcudf.expressions cimport Expression
from pylibcudf.io.types cimport (
    SinkInfo,
    SourceInfo,
    PartitionInfo,
    TableInputMetadata,
    TableWithMetadata,
)
from pylibcudf.io.parquet_metadata cimport FileMetaData
from pylibcudf.libcudf.expressions cimport expression
from pylibcudf.libcudf.io.datasource cimport datasource, make_datasources
from pylibcudf.libcudf.io.parquet cimport (
    chunked_parquet_reader as cpp_chunked_parquet_reader,
    parquet_reader_options,
    read_parquet as cpp_read_parquet,
    write_parquet as cpp_write_parquet,
    is_supported_read_parquet as cpp_is_supported_read_parquet,
    is_supported_write_parquet as cpp_is_supported_write_parquet,
    parquet_writer_options,
    chunked_parquet_writer as cpp_chunked_parquet_writer,
    chunked_parquet_writer_options,
    merge_row_group_metadata as cpp_merge_row_group_metadata,
)
from pylibcudf.libcudf.io.parquet_schema cimport FileMetaData as cpp_FileMetaData
from pylibcudf.libcudf.io.types cimport (
    compression_type,
    dictionary_policy as dictionary_policy_t,
    partition_info,
    statistics_freq,
    table_with_metadata,
)
from pylibcudf.libcudf.table.table_view cimport table_view
from pylibcudf.libcudf.types cimport size_type, type_id
from pylibcudf.table cimport Table
from pylibcudf.utils cimport _get_stream, _get_memory_resource
from cuda.bindings.cyruntime cimport cudaStream_t

__all__ = [
    "ChunkedParquetReader",
    "ChunkedParquetWriterOptions",
    "ChunkedParquetWriterOptionsBuilder",
    "ParquetChunkedWriter",
    "ParquetReaderOptions",
    "ParquetReaderOptionsBuilder",
    "ParquetWriterOptions",
    "ParquetWriterOptionsBuilder",
    "is_supported_read_parquet",
    "is_supported_write_parquet",
    "merge_row_group_metadata",
    "read_parquet",
    "write_parquet",
]


def _warn_deprecated(api_name, new_api):
    warnings.warn(
        f"{api_name} is deprecated and will be removed in a "
        f"future version of cudf. Use {new_api} instead.",
        FutureWarning
    )


cdef vector[cpp_FileMetaData] _build_parquet_metadatas(
    object parquet_metadatas,
    size_t num_sources,
) except *:
    cdef vector[cpp_FileMetaData] c_metadatas
    cdef vector[cpp_FileMetaData*] metadata_ptrs
    cdef object metadata
    cdef size_t i
    if parquet_metadatas is None:
        return c_metadatas

    for metadata in parquet_metadatas:
        if not isinstance(metadata, FileMetaData):
            raise TypeError(
                "parquet_metadatas must contain only FileMetaData objects"
            )
        metadata_ptrs.push_back(&(<FileMetaData>metadata).c_obj)

    if metadata_ptrs.size() != num_sources:
        raise ValueError(
            f"Length of 'parquet_metadatas' ({metadata_ptrs.size()}) "
            f"must match the number of input sources "
            f"({num_sources})"
        )

    c_metadatas.reserve(metadata_ptrs.size())
    with nogil:
        # This copies the (potentially large) metadata object. We don't
        # want to hold the GIL for that.
        for i in range(metadata_ptrs.size()):
            c_metadatas.push_back(dereference(metadata_ptrs[i]))

    return c_metadatas


cdef class ParquetReaderOptions:
    """The settings to use for ``read_parquet``
    For details, see :cpp:class:`cudf::io::parquet_reader_options`
    """
    @staticmethod
    def builder(SourceInfo source):
        """
        Create a ParquetReaderOptionsBuilder object

        For details, see :cpp:func:`cudf::io::parquet_reader_options::builder`

        Parameters
        ----------
        sink : SourceInfo
            The source to read the Parquet file from.

        Returns
        -------
        ParquetReaderOptionsBuilder
            Builder to build ParquetReaderOptions
        """
        cdef ParquetReaderOptionsBuilder parquet_builder = (
            ParquetReaderOptionsBuilder.__new__(ParquetReaderOptionsBuilder)
        )
        parquet_builder.c_obj = parquet_reader_options.builder(source.c_obj)
        parquet_builder.source = source
        return parquet_builder

    cpdef void set_row_groups(self, list row_groups):
        """
        Sets list of individual row groups to read.

        Parameters
        ----------
        row_groups : list[list[int]]
            Row groups to read, one inner list per input source.

        Notes
        -----
        Rows are emitted in input-source order; all rows selected from
        source 0 are emitted before rows selected from source 1, and so on.
        Within each source, row groups are read in the order provided;
        indices are not sorted or deduplicated, and repeated indices are
        emitted multiple times. Empty inner lists contribute no rows.
        When unset, all row groups are read in source order, then in
        on-disk order within each source. Predicate pushdown drops row
        groups in place; remaining row groups keep their relative order.

        Returns
        -------
        None
        """
        cdef vector[vector[size_type]] outer
        cdef vector[size_type] inner
        for row_group in row_groups:
            for x in row_group:
                inner.push_back(x)
            outer.push_back(inner)
            inner.clear()

        self.c_obj.set_row_groups(outer)

    cpdef void set_num_rows(self, int64_t nrows):
        """
        Sets number of rows to read.

        Parameters
        ----------
        nrows : int64_t
            Number of rows to read after skip

        Notes
        -----
        Although this allows one to request more than `size_type::max()`
        rows, if any single read would produce a table larger than this row
        limit, an error is thrown.

        Returns
        -------
        None
        """
        self.c_obj.set_num_rows(nrows)

    cpdef void set_skip_rows(self, int64_t skip_rows):
        """
        Sets number of rows to skip.

        Parameters
        ----------
        skip_rows : int64_t
            Number of rows to skip from start

        Returns
        -------
        None
        """
        self.c_obj.set_skip_rows(skip_rows)

    cpdef void set_columns(self, list col_names):
        """
        Sets names of the columns to be read. Deprecated and will be
        removed in a future version. Use set_column_names instead.

        Parameters
        ----------
        col_names : list
            List of column names

        Returns
        -------
        None
        """
        _warn_deprecated("set_columns", "set_column_names")
        self.set_column_names(col_names)

    cpdef void set_column_names(self, list col_names):
        """
        Sets names of the columns to be read.

        Parameters
        ----------
        col_names : list
            List of column names

        Returns
        -------
        None
        """
        cdef vector[string] vec
        for name in col_names:
            vec.push_back(<string>str(name).encode())
        self.c_obj.set_column_names(vec)

    cpdef void set_column_indices(self, list col_indices):
        """
        Sets indices of the top-level columns to be read.

        Parameters
        ----------
        col_indices : list
            List of top-level column indices

        Returns
        -------
        None
        """
        cdef vector[size_type] vec
        for idx in col_indices:
            vec.push_back(idx)
        self.c_obj.set_column_indices(vec)

    cpdef void set_column_field_ids(self, list column_field_ids):
        """
        Sets Parquet field IDs of the columns/fields to be read.

        Parameters
        ----------
        column_field_ids : list
            List of Parquet field IDs

        Returns
        -------
        None
        """
        cdef vector[int32_t] vec
        for field_id in column_field_ids:
            vec.push_back(field_id)
        self.c_obj.set_column_field_ids(vec)

    cpdef void set_filter(self, Expression filter):
        """
        Sets AST based filter for predicate pushdown.

        Parameters
        ----------
        filter : Expression
            AST expression to use as filter

        Returns
        -------
        None
        """
        self.c_obj.set_filter(<expression &>dereference(filter.c_obj))

    cpdef void set_source(self, SourceInfo src):
        """
        Set a new source info location.

        Parameters
        ----------
        src : SourceInfo
            New source information, replacing existing information.

        Returns
        -------
        None
        """
        self.c_obj.set_source(src.c_obj)

    cpdef bool is_enabled_use_jit_filter(self):
        """Returns whether to use JIT compilation for filtering."""
        return self.c_obj.is_enabled_use_jit_filter()

    cpdef void enable_case_sensitive_names(self, bool val):
        """
        Sets whether column names are matched case-sensitively.

        Parameters
        ----------
        val : bool
            Enables case-sensitive matching

        Returns
        -------
        None
        """
        self.c_obj.enable_case_sensitive_names(val)

    cpdef bool is_enabled_case_sensitive_names(self):
        """
        Returns whether column name matching is case sensitive.

        Returns
        -------
        bool
            Whether column names are matched case-sensitively
        """
        return self.c_obj.is_enabled_case_sensitive_names()

cdef class ParquetReaderOptionsBuilder:
    cpdef ParquetReaderOptionsBuilder convert_strings_to_categories(self, bool val):
        """
        Sets enable/disable conversion of strings to categories.

        Parameters
        ----------
        val : bool
            Boolean value to enable/disable conversion of string columns to categories

        Returns
        -------
        ParquetReaderOptionsBuilder
        """
        self.c_obj.convert_strings_to_categories(val)
        return self

    cpdef ParquetReaderOptionsBuilder use_pandas_metadata(self, bool val):
        """
        Sets to enable/disable use of pandas metadata to read.

        Parameters
        ----------
        val : bool
            Boolean value whether to use pandas metadata

        Returns
        -------
        ParquetReaderOptionsBuilder
        """
        self.c_obj.use_pandas_metadata(val)
        return self

    cpdef ParquetReaderOptionsBuilder allow_mismatched_pq_schemas(self, bool val):
        """
        Sets to enable/disable reading of matching projected and filter
        columns from mismatched Parquet sources.

        Parameters
        ----------
        val : bool
            Boolean value whether to read matching projected and filter
            columns from mismatched Parquet sources.

        Returns
        -------
        ParquetReaderOptionsBuilder
        """
        self.c_obj.allow_mismatched_pq_schemas(val)
        return self

    cpdef ParquetReaderOptionsBuilder ignore_missing_columns(self, bool val):
        """
        Sets to enable/disable ignoring of non-existent projected columns while reading.

        Parameters
        ----------
        val : bool
            Boolean indicating whether to ignore non-existent projected columns
            while reading.

        Returns
        -------
        ParquetReaderOptionsBuilder
        """
        self.c_obj.ignore_missing_columns(val)
        return self

    cpdef ParquetReaderOptionsBuilder use_arrow_schema(self, bool val):
        """
        Sets to enable/disable use of arrow schema to read.

        Parameters
        ----------
        val : bool
            Boolean value whether to use arrow schema

        Returns
        -------
        ParquetReaderOptionsBuilder
        """
        self.c_obj.use_arrow_schema(val)
        return self

    cpdef ParquetReaderOptionsBuilder filter(self, Expression filter):
        """
        Sets AST based filter for predicate pushdown.

        Parameters
        ----------
        filter : Expression
            AST expression to use as filter

        Returns
        -------
        ParquetReaderOptionsBuilder
        """
        self.c_obj.filter(<expression &>dereference(filter.c_obj))
        return self

    cpdef ParquetReaderOptionsBuilder columns(self, list col_names):
        """
        Sets names of the columns to be read. Deprecated and will be
        removed in a future version. Use column_names instead.

        Parameters
        ----------
        col_names : list[str]
            List of column names

        Returns
        -------
        ParquetReaderOptionsBuilder
        """
        _warn_deprecated("columns", "column_names")
        return self.column_names(col_names)

    cpdef ParquetReaderOptionsBuilder column_names(self, list col_names):
        """
        Sets names of the columns to be read.

        Parameters
        ----------
        col_names : list[str]
            List of column names

        Returns
        -------
        ParquetReaderOptionsBuilder
        """
        cdef vector[string] vec
        for name in col_names:
            vec.push_back(<string>str(name).encode())
        self.c_obj.column_names(vec)
        return self

    cpdef ParquetReaderOptionsBuilder column_indices(self, list col_indices):
        """
        Sets indices of the top-level columns to be read.

        Parameters
        ----------
        col_indices : list[int]
            List of top-level column indices

        Returns
        -------
        ParquetReaderOptionsBuilder
        """
        cdef vector[size_type] vec
        for idx in col_indices:
            vec.push_back(idx)
        self.c_obj.column_indices(vec)
        return self

    cpdef ParquetReaderOptionsBuilder column_field_ids(self, list column_field_ids):
        """
        Sets Parquet field IDs of the columns/fields to be read.

        Parameters
        ----------
        column_field_ids : list[int]
            List of Parquet field IDs

        Returns
        -------
        ParquetReaderOptionsBuilder
        """
        cdef vector[int32_t] vec
        for field_id in column_field_ids:
            vec.push_back(field_id)
        self.c_obj.column_field_ids(vec)
        return self

    cpdef ParquetReaderOptionsBuilder use_jit_filter(self, bool use_jit_filter):
        """
        Sets whether to use JIT compilation for filtering.

        Parameters
        ----------
        use_jit_filter : bool
            Boolean value whether to use JIT filter

        Returns
        -------
        ParquetReaderOptionsBuilder
        """
        self.c_obj.use_jit_filter(use_jit_filter)
        return self

    cpdef ParquetReaderOptionsBuilder case_sensitive_names(self, bool val):
        """
        Sets whether column name matching is case sensitive.

        Parameters
        ----------
        val : bool
            ``True`` to enable case-sensitive matching (default),
            ``False`` for case-insensitive matching.

        Returns
        -------
        ParquetReaderOptionsBuilder
        """
        self.c_obj.case_sensitive_names(val)
        return self

    cpdef ParquetReaderOptionsBuilder decimal_width(self, type_id width):
        """
        Sets the decimal width used to cast all decimal columns.

        Parameters
        ----------
        width : TypeId
            The decimal type_id (DECIMAL32, DECIMAL64, or DECIMAL128) to which
            all decimal columns should be cast.

        Returns
        -------
        ParquetReaderOptionsBuilder
        """
        self.c_obj.decimal_width(width)
        return self

    cpdef build(self):
        """Create a ParquetReaderOptions object"""
        cdef ParquetReaderOptions parquet_options = ParquetReaderOptions.__new__(
            ParquetReaderOptions
        )
        parquet_options.c_obj = move(self.c_obj.build())
        parquet_options.source = self.source
        return parquet_options


cdef class ChunkedParquetReader:
    """
    Reads chunks of a Parquet file into a :py:class:`~.types.TableWithMetadata`.

    For details, see :cpp:class:`chunked_parquet_reader`.

    Parameters
    ----------
    options : ParquetReaderOptions
        Settings for controlling reading behavior
    stream : Stream | None
        CUDA stream used for device memory operations and kernel launches
    mr : DeviceMemoryResource, optional
        Device memory resource used to allocate the returned table's device memory.
    chunk_read_limit : size_t, default 0
        Limit on total number of bytes to be returned per read,
        or 0 if there is no limit.
    pass_read_limit : size_t, default 1024000000
        Limit on the amount of memory used for reading and decompressing data
        or 0 if there is no limit.
    parquet_metadatas : list[FileMetaData], optional
        Pre-materialized parquet footer metadata, one for each source. If not
        provided, footers are read from the sources internally.
    """
    def __init__(
        self,
        ParquetReaderOptions options,
        object stream = None,
        DeviceMemoryResource mr = None,
        size_t chunk_read_limit=0,
        size_t pass_read_limit=1024000000,
        object parquet_metadatas=None,
    ):
        self._stream = _get_stream(stream)
        self.mr = _get_memory_resource(mr)
        cdef vector[unique_ptr[datasource]] sources
        cdef vector[cpp_FileMetaData] c_metadatas
        cdef cudaStream_t stream_view = self._stream.view().value()
        if parquet_metadatas is None:
            with nogil:
                self.reader.reset(
                    new cpp_chunked_parquet_reader(
                        chunk_read_limit,
                        pass_read_limit,
                        options.c_obj,
                        stream_view,
                        self.mr.get_mr()
                    )
                )
        else:
            with nogil:
                sources = make_datasources(options.c_obj.get_source())
            c_metadatas = _build_parquet_metadatas(
                parquet_metadatas, sources.size()
            )
            with nogil:
                self.reader.reset(
                    new cpp_chunked_parquet_reader(
                        chunk_read_limit,
                        pass_read_limit,
                        move(sources),
                        move(c_metadatas),
                        options.c_obj,
                        stream_view,
                        self.mr.get_mr()
                    )
                )

    __hash__ = None

    cpdef bool has_next(self):
        """
        Returns True if there is another chunk in the Parquet file
        to be read.

        Returns
        -------
        True if we have not finished reading the file.
        """
        with nogil:
            return self.reader.get()[0].has_next()

    cpdef TableWithMetadata read_chunk(self, DeviceMemoryResource mr=None):
        """
        Read the next chunk into a :py:class:`~.types.TableWithMetadata`

        Parameters
        ----------
        mr : DeviceMemoryResource, optional
            Device memory resource used to allocate the returned table's device memory.

        Returns
        -------
        TableWithMetadata
            The Table and its corresponding metadata (column names) that were read in.
        """
        # Read Parquet
        cdef table_with_metadata c_result
        mr = _get_memory_resource(mr)

        with nogil:
            c_result = move(self.reader.get()[0].read_chunk())

        return TableWithMetadata.from_libcudf(c_result, self._stream, mr)


cpdef read_parquet(
    ParquetReaderOptions options,
    object stream = None,
    DeviceMemoryResource mr=None,
    object parquet_metadatas=None,
):
    """
    Read from Parquet format.

    The source to read from and options are encapsulated
    by the `options` object.

    For details, see :cpp:func:`read_parquet`.

    Parameters
    ----------
    options: ParquetReaderOptions
        Settings for controlling reading behavior
    stream : Stream | None
        CUDA stream used for device memory operations and kernel launches
    mr : DeviceMemoryResource, optional
        Device memory resource used to allocate the returned table's device memory.
    parquet_metadatas : list[FileMetaData], optional
        Pre-materialized parquet footer metadata, one for each source. If not
        provided, footers are read from the sources internally.
    """
    cdef Stream s = _get_stream(stream)
    cdef cudaStream_t _cs = s.view().value()
    cdef vector[unique_ptr[datasource]] sources
    cdef vector[cpp_FileMetaData] c_metadatas
    cdef table_with_metadata c_result
    mr = _get_memory_resource(mr)
    if parquet_metadatas is None:
        with nogil:
            c_result = move(cpp_read_parquet(options.c_obj, _cs, mr.get_mr()))
    else:
        with nogil:
            sources = make_datasources(options.c_obj.get_source())
        c_metadatas = _build_parquet_metadatas(parquet_metadatas, sources.size())
        with nogil:
            c_result = move(
                cpp_read_parquet(
                    move(sources),
                    move(c_metadatas),
                    options.c_obj,
                    _cs,
                    mr.get_mr(),
                )
            )

    return TableWithMetadata.from_libcudf(c_result, s, mr)


cdef class ChunkedParquetWriter:
    cpdef memoryview close(self, list metadata_file_path):
        """
        Closes the chunked Parquet writer.

        Parameters
        ----------
        metadata_file_path: list
            Column chunks file path to be set in the raw output metadata

        Returns
        -------
        None
        """
        cdef vector[string] column_chunks_file_paths
        cdef unique_ptr[vector[uint8_t]] out_metadata_c
        if metadata_file_path:
            for path in metadata_file_path:
                column_chunks_file_paths.push_back(path.encode())
        with nogil:
            out_metadata_c = move(self.c_obj.get()[0].close(column_chunks_file_paths))
        return memoryview(HostBuffer.from_unique_ptr(move(out_metadata_c)))

    cpdef void write(self, Table table, object partitions_info=None):
        """
        Writes table to output.

        Parameters
        ----------
        table: Table
            Table that needs to be written
        partitions_info: object, default None
            Optional partitions to divide the table into.
            If specified, must be same size as number of sinks.

        Returns
        -------
        None
        """
        cdef vector[partition_info] partitions
        if partitions_info is not None:
            for part in partitions_info:
                partitions.push_back(
                    partition_info(part[0], part[1])
                )
        cdef table_view c_table = table.view()
        with nogil:
            self.c_obj.get()[0].write(c_table, partitions)

    @staticmethod
    def from_options(ChunkedParquetWriterOptions options, object stream = None):
        """
        Creates a chunked Parquet writer from options

        Parameters
        ----------
        options: ChunkedParquetWriterOptions
            Settings for controlling writing behavior
        stream : Stream | None
            CUDA stream used for device memory operations and kernel launches

        Returns
        -------
        ChunkedParquetWriter
        """
        cdef ChunkedParquetWriter parquet_writer = ChunkedParquetWriter.__new__(
            ChunkedParquetWriter
        )
        cdef Stream s = _get_stream(stream)
        cdef cudaStream_t _cs = s.view().value()
        parquet_writer.c_obj.reset(
            new cpp_chunked_parquet_writer(options.c_obj, _cs)
        )
        return parquet_writer


cdef class ChunkedParquetWriterOptions:
    @staticmethod
    def builder(SinkInfo sink):
        """
        Create builder to create ChunkedParquetWriterOptions.

        Parameters
        ----------
        sink: SinkInfo
            The sink used for writer output

        Returns
        -------
        ChunkedParquetWriterOptionsBuilder
        """
        cdef ChunkedParquetWriterOptionsBuilder parquet_builder = (
            ChunkedParquetWriterOptionsBuilder.__new__(
                ChunkedParquetWriterOptionsBuilder
            )
        )
        parquet_builder.c_obj = chunked_parquet_writer_options.builder(sink.c_obj)
        parquet_builder.sink = sink
        return parquet_builder

    cpdef void set_dictionary_policy(self, dictionary_policy_t policy):
        """
        Sets the policy for dictionary use.

        Parameters
        ----------
        policy : DictionaryPolicy
            Policy for dictionary use

        Returns
        -------
        None
        """
        self.c_obj.set_dictionary_policy(policy)


cdef class ChunkedParquetWriterOptionsBuilder:
    cpdef ChunkedParquetWriterOptionsBuilder metadata(
        self,
        TableInputMetadata metadata
    ):
        self.c_obj.metadata(metadata.c_obj)
        return self

    cpdef ChunkedParquetWriterOptionsBuilder key_value_metadata(self, metadata):
        """
        Sets Key-Value footer metadata.

        Parameters
        ----------
        metadata : list[dict[str, str]]
            Key-Value footer metadata

        Returns
        -------
        Self
        """
        self.c_obj.key_value_metadata(
            [
                {key.encode(): value.encode() for key, value in mapping.items()}
                for mapping in metadata
            ]
        )
        return self

    cpdef ChunkedParquetWriterOptionsBuilder compression(
        self,
        compression_type compression
    ):
        """
        Sets compression type.

        Parameters
        ----------
        compression : CompressionType
            The compression type to use

        Returns
        -------
        Self
        """
        self.c_obj.compression(compression)
        return self

    cpdef ChunkedParquetWriterOptionsBuilder stats_level(self, statistics_freq sf):
        """
        Sets the level of statistics.

        Parameters
        ----------
        sf : StatisticsFreq
            Level of statistics requested in the output file

        Returns
        -------
        Self
        """
        self.c_obj.stats_level(sf)
        return self

    cpdef ChunkedParquetWriterOptionsBuilder row_group_size_bytes(self, size_t val):
        """
        Sets the maximum row group size, in bytes.

        Parameters
        ----------
        val : size_t
            Maximum row group size, in bytes to set

        Returns
        -------
        Self
        """
        self.c_obj.row_group_size_bytes(val)
        return self

    cpdef ChunkedParquetWriterOptionsBuilder row_group_size_rows(self, size_type val):
        """
        Sets the maximum row group size, in rows.

        Parameters
        ----------
        val : size_type
            Maximum row group size, in rows to set

        Returns
        -------
        Self
        """
        self.c_obj.row_group_size_rows(val)
        return self

    cpdef ChunkedParquetWriterOptionsBuilder max_page_size_bytes(self, size_t val):
        """
        Sets the maximum uncompressed page size, in bytes.

        Parameters
        ----------
        val : size_t
            Maximum uncompressed page size, in bytes to set

        Returns
        -------
        Self
        """
        self.c_obj.max_page_size_bytes(val)
        return self

    cpdef ChunkedParquetWriterOptionsBuilder max_page_size_rows(self, size_type val):
        """
        Sets the maximum page size, in rows.

        Parameters
        ----------
        val : size_type
            Maximum page size, in rows to set.

        Returns
        -------
        Self
        """
        self.c_obj.max_page_size_rows(val)
        return self

    cpdef ChunkedParquetWriterOptionsBuilder max_dictionary_size(self, size_t val):
        """
        Sets the maximum dictionary size, in bytes.

        Parameters
        ----------
        val : size_t
            Sets the maximum dictionary size, in bytes.

        Returns
        -------
        Self
        """
        self.c_obj.max_dictionary_size(val)
        return self

    cpdef ChunkedParquetWriterOptionsBuilder write_arrow_schema(self, bool enabled):
        """
        Set to true if arrow schema is to be written.

        Parameters
        ----------
        enabled : bool
            Boolean value to enable/disable writing of arrow schema.

        Returns
        -------
        Self
        """
        self.c_obj.write_arrow_schema(enabled)
        return self

    cpdef ChunkedParquetWriterOptions build(self):
        """Create a ChunkedParquetWriterOptions object"""
        cdef ChunkedParquetWriterOptions parquet_options = (
            ChunkedParquetWriterOptions.__new__(ChunkedParquetWriterOptions)
        )
        parquet_options.c_obj = move(self.c_obj.build())
        parquet_options.sink = self.sink
        return parquet_options


cdef class ParquetWriterOptions:

    @staticmethod
    def builder(SinkInfo sink, Table table):
        """
        Create builder to create ParquetWriterOptionsBuilder.

        Parameters
        ----------
        sink : SinkInfo
            The sink used for writer output

        table : Table
            Table to be written to output

        Returns
        -------
        ParquetWriterOptionsBuilder
        """
        cdef ParquetWriterOptionsBuilder bldr = ParquetWriterOptionsBuilder.__new__(
            ParquetWriterOptionsBuilder
        )
        bldr.c_obj = parquet_writer_options.builder(sink.c_obj, table.view())
        bldr.table_ref = table
        bldr.sink_ref = sink
        return bldr

    cpdef void set_partitions(self, list partitions):
        """
        Sets partitions.

        Parameters
        ----------
        partitions : list[Partitions]
            Partitions of input table in {start_row, num_rows} pairs.

        Returns
        -------
        None
        """
        cdef vector[partition_info] c_partions
        cdef PartitionInfo partition

        c_partions.reserve(len(partitions))
        for partition in partitions:
            c_partions.push_back(partition.c_obj)

        self.c_obj.set_partitions(c_partions)

    cpdef void set_column_chunks_file_paths(self, file_paths):
        """
        Sets column chunks file path to be set in the raw output metadata.

        Parameters
        ----------
        file_paths : list[str]
            Vector of strings which indicate file paths.

        Returns
        -------
        None
        """
        self.c_obj.set_column_chunks_file_paths([fp.encode() for fp in file_paths])

    cpdef void set_row_group_size_bytes(self, size_t size_bytes):
        """
        Sets the maximum row group size, in bytes.

        Parameters
        ----------
        size_bytes : int
            Maximum row group size, in bytes to set

        Returns
        -------
        None
        """
        self.c_obj.set_row_group_size_bytes(size_bytes)

    cpdef void set_row_group_size_rows(self, size_type size_rows):
        """
        Sets the maximum row group size, in rows.

        Parameters
        ----------
        size_rows : int
            Maximum row group size, in rows to set

        Returns
        -------
        None
        """
        self.c_obj.set_row_group_size_rows(size_rows)

    cpdef void set_max_page_size_bytes(self, size_t size_bytes):
        """
        Sets the maximum uncompressed page size, in bytes.

        Parameters
        ----------
        size_bytes : int
            Maximum uncompressed page size, in bytes to set

        Returns
        -------
        None
        """
        self.c_obj.set_max_page_size_bytes(size_bytes)

    cpdef void set_max_page_size_rows(self, size_type size_rows):
        """
        Sets the maximum page size, in rows.

        Parameters
        ----------
        size_rows : int
            Maximum page size, in rows to set.

        Returns
        -------
        None
        """
        self.c_obj.set_max_page_size_rows(size_rows)

    cpdef void set_max_dictionary_size(self, size_t size_bytes):
        """
        Sets the maximum dictionary size, in bytes.

        Parameters
        ----------
        size_bytes : int
            Sets the maximum dictionary size, in bytes.

        Returns
        -------
        None
        """
        self.c_obj.set_max_dictionary_size(size_bytes)


cdef class ParquetWriterOptionsBuilder:

    cpdef ParquetWriterOptionsBuilder metadata(self, TableInputMetadata metadata):
        """
        Sets metadata.

        Parameters
        ----------
        metadata : TableInputMetadata
            Associated metadata

        Returns
        -------
        Self
        """
        self.c_obj.metadata(metadata.c_obj)
        return self

    cpdef ParquetWriterOptionsBuilder key_value_metadata(self, metadata):
        """
        Sets Key-Value footer metadata.

        Parameters
        ----------
        metadata : list[dict[str, str]]
            Key-Value footer metadata

        Returns
        -------
        Self
        """
        self.c_obj.key_value_metadata(
            [
                {key.encode(): value.encode() for key, value in mapping.items()}
                for mapping in metadata
            ]
        )
        return self

    cpdef ParquetWriterOptionsBuilder compression(self, compression_type compression):
        """
        Sets compression type.

        Parameters
        ----------
        compression : CompressionType
            The compression type to use

        Returns
        -------
        Self
        """
        self.c_obj.compression(compression)
        return self

    cpdef ParquetWriterOptionsBuilder stats_level(self, statistics_freq sf):
        """
        Sets the level of statistics.

        Parameters
        ----------
        sf : StatisticsFreq
            Level of statistics requested in the output file

        Returns
        -------
        Self
        """
        self.c_obj.stats_level(sf)
        return self

    cpdef ParquetWriterOptionsBuilder int96_timestamps(self, bool enabled):
        """
        Sets whether timestamps are written as int96 or timestamp micros.

        Parameters
        ----------
        enabled : bool
            Boolean value to enable/disable int96 timestamps

        Returns
        -------
        Self
        """
        self.c_obj.int96_timestamps(enabled)
        return self

    cpdef ParquetWriterOptionsBuilder write_v2_headers(self, bool enabled):
        """
        Set to true to write V2 page headers, otherwise false to write V1 page headers.

        Parameters
        ----------
        enabled : bool
            Boolean value to enable/disable writing of V2 page headers.

        Returns
        -------
        Self
        """
        self.c_obj.write_v2_headers(enabled)
        return self

    cpdef ParquetWriterOptionsBuilder page_level_compression(self, bool enabled):
        """
        Set to true to enable per-page compression decisions for V2 data pages.

        When enabled, each V2 data page independently decides whether to compress
        based on compression ratio. When disabled (default), all V2 data pages
        in a chunk follow the same compression decision as dictionary pages.

        Parameters
        ----------
        enabled : bool
            Boolean value to enable/disable per-page compression decisions.

        Returns
        -------
        Self
        """
        self.c_obj.page_level_compression(enabled)
        return self

    cpdef ParquetWriterOptionsBuilder dictionary_policy(self, dictionary_policy_t val):
        """
        Sets the policy for dictionary use.

        Parameters
        ----------
        val : DictionaryPolicy
            Policy for dictionary use.

        Returns
        -------
        Self
        """
        self.c_obj.dictionary_policy(val)
        return self

    cpdef ParquetWriterOptionsBuilder utc_timestamps(self, bool enabled):
        """
        Set to true if timestamps are to be written as UTC.

        Parameters
        ----------
        enabled : bool
            Boolean value to enable/disable writing of timestamps as UTC.

        Returns
        -------
        Self
        """
        self.c_obj.utc_timestamps(enabled)
        return self

    cpdef ParquetWriterOptionsBuilder write_arrow_schema(self, bool enabled):
        """
        Set to true if arrow schema is to be written.

        Parameters
        ----------
        enabled : bool
            Boolean value to enable/disable writing of arrow schema.

        Returns
        -------
        Self
        """
        self.c_obj.write_arrow_schema(enabled)
        return self

    cpdef ParquetWriterOptionsBuilder row_group_size_rows(self, size_type val):
        """
        Sets the maximum row group size, in rows.

        Parameters
        ----------
        val : size_type
            Maximum row group size, in rows to set

        Returns
        -------
        Self
        """
        self.c_obj.row_group_size_rows(val)
        return self

    cpdef ParquetWriterOptionsBuilder max_page_size_bytes(self, size_t val):
        """
        Sets the maximum uncompressed page size, in bytes.

        Parameters
        ----------
        val : size_t
            Maximum uncompressed page size, in bytes to set

        Returns
        -------
        Self
        """
        self.c_obj.max_page_size_bytes(val)
        return self

    cpdef ParquetWriterOptions build(self):
        """
        Create a ParquetWriterOptions from the set options.

        Returns
        -------
        ParquetWriterOptions
        """
        cdef ParquetWriterOptions parquet_options = ParquetWriterOptions.__new__(
            ParquetWriterOptions
        )
        parquet_options.c_obj = move(self.c_obj.build())
        parquet_options.table_ref = self.table_ref
        parquet_options.sink_ref = self.sink_ref
        return parquet_options


cpdef memoryview write_parquet(ParquetWriterOptions options, object stream = None):
    """
    Writes a set of columns to parquet format.

    Parameters
    ----------
    options : ParquetWriterOptions
        Settings for controlling writing behavior
    stream : Stream | None
        CUDA stream used for device memory operations and kernel launches

    Returns
    -------
    memoryview
        A blob that contains the file metadata
        (parquet FileMetadata thrift message) if requested in
        parquet_writer_options (empty blob otherwise).
    """
    cdef unique_ptr[vector[uint8_t]] c_result
    cdef Stream s = _get_stream(stream)
    cdef cudaStream_t _cs = s.view().value()
    with nogil:
        c_result = cpp_write_parquet(move(options.c_obj), _cs)

    return memoryview(HostBuffer.from_unique_ptr(move(c_result)))


cpdef bool is_supported_read_parquet(compression_type compression):
    """Check if the compression type is supported for reading Parquet files.

    For details, see :cpp:func:`is_supported_read_parquet`.

    Parameters
    ----------
    compression : CompressionType
        The compression type to check

    Returns
    -------
    bool
        True if the compression type is supported for reading Parquet files
    """
    return cpp_is_supported_read_parquet(compression)


cpdef bool is_supported_write_parquet(compression_type compression):
    """Check if the compression type is supported for writing Parquet files.

    For details, see :cpp:func:`is_supported_write_parquet`.

    Parameters
    ----------
    compression : CompressionType
        The compression type to check

    Returns
    -------
    bool
        True if the compression type is supported for writing Parquet files
    """
    return cpp_is_supported_write_parquet(compression)


cpdef memoryview merge_row_group_metadata(list metdata_list):
    """
    Merges multiple raw metadata blobs that were previously
    created by write_parquet into a single metadata blob.

    For details, see :cpp:func:`merge_row_group_metadata`.

    Parameters
    ----------
    metdata_list : list
        List of input file metadata

    Returns
    -------
    memoryview
        A parquet-compatible blob that contains the data for all row groups in the list
    """
    cdef vector[unique_ptr[vector[uint8_t]]] list_c
    cdef unique_ptr[vector[uint8_t]] output_c

    for blob in metdata_list:
        list_c.push_back(move(make_unique[vector[uint8_t]](<vector[uint8_t]> blob)))

    with nogil:
        output_c = move(cpp_merge_row_group_metadata(list_c))

    return memoryview(HostBuffer.from_unique_ptr(move(output_c)))
