# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from libcpp cimport bool
from libcpp.string cimport string
from libcpp.utility cimport move
from libcpp.vector cimport vector

import datetime

from rmm.pylibrmm.stream cimport Stream
from cuda.bindings.cyruntime cimport cudaStream_t
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource

from pylibcudf.io.types cimport SourceInfo, TableWithMetadata, SinkInfo

from pylibcudf.libcudf.io.orc cimport (
    orc_reader_options,
    read_orc as cpp_read_orc,
    write_orc as cpp_write_orc,
    is_supported_read_orc as cpp_is_supported_read_orc,
    is_supported_write_orc as cpp_is_supported_write_orc,
)

from pylibcudf.libcudf.io.orc_metadata cimport (
    binary_statistics,
    bucket_statistics,
    column_statistics,
    date_statistics,
    decimal_statistics,
    double_statistics,
    integer_statistics,
    no_statistics,
    read_parsed_orc_statistics as cpp_read_parsed_orc_statistics,
    statistics_type,
    string_statistics,
    timestamp_statistics,
)

from pylibcudf.libcudf.io.types cimport (
    table_with_metadata,
    compression_type,
    statistics_freq,
)

from pylibcudf.libcudf.io.orc cimport (
    orc_chunked_writer,
    orc_writer_options,
    chunked_orc_writer_options,
)

from pylibcudf.libcudf.types cimport size_type
from pylibcudf.libcudf.table.table_view cimport table_view

from pylibcudf.types cimport DataType

from pylibcudf.variant cimport get_if, holds_alternative

from pylibcudf.utils cimport _get_stream, _get_memory_resource


__all__ = [
    "OrcColumnStatistics",
    "ParsedOrcStatistics",
    "read_orc",
    "read_parsed_orc_statistics",
    "write_orc",
    "is_supported_read_orc",
    "is_supported_write_orc",
    "OrcReaderOptions",
    "OrcReaderOptionsBuilder",
    "OrcWriterOptions",
    "OrcWriterOptionsBuilder",
    "OrcChunkedWriter",
    "ChunkedOrcWriterOptions",
    "ChunkedOrcWriterOptionsBuilder",
]

cdef class OrcColumnStatistics:
    def __init__(self):
        raise TypeError(
            "OrcColumnStatistics should not be instantiated by users. If it is "
            "being constructed in Cython from a preexisting libcudf object, "
            "use `OrcColumnStatistics.from_libcudf` instead."
        )

    __hash__ = None

    @property
    def number_of_values(self):
        if self.number_of_values_c.has_value():
            return self.number_of_values_c.value()
        return None

    @property
    def has_null(self):
        if self.has_null_c.has_value():
            return self.has_null_c.value()
        return None

    cdef void _init_stats_dict(self):
        # Initialize stats to return and parse stats blob
        self.column_stats = {}

        cdef statistics_type type_specific_stats = self.type_specific_stats_c

        cdef integer_statistics* int_stats
        cdef double_statistics* dbl_stats
        cdef string_statistics* str_stats
        cdef bucket_statistics* bucket_stats
        cdef decimal_statistics* dec_stats
        cdef date_statistics* date_stats
        cdef binary_statistics* bin_stats
        cdef timestamp_statistics* ts_stats

        if holds_alternative[no_statistics](type_specific_stats):
            pass
        elif int_stats := get_if[integer_statistics](&type_specific_stats):
            if int_stats.minimum.has_value():
                self.column_stats["minimum"] = int_stats.minimum.value()
            else:
                self.column_stats["minimum"] = None
            if int_stats.maximum.has_value():
                self.column_stats["maximum"] = int_stats.maximum.value()
            else:
                self.column_stats["maximum"] = None
            if int_stats.sum.has_value():
                self.column_stats["sum"] = int_stats.sum.value()
            else:
                self.column_stats["sum"] = None
        elif dbl_stats := get_if[double_statistics](&type_specific_stats):
            if dbl_stats.minimum.has_value():
                self.column_stats["minimum"] = dbl_stats.minimum.value()
            else:
                self.column_stats["minimum"] = None
            if dbl_stats.maximum.has_value():
                self.column_stats["maximum"] = dbl_stats.maximum.value()
            else:
                self.column_stats["maximum"] = None
            if dbl_stats.sum.has_value():
                self.column_stats["sum"] = dbl_stats.sum.value()
            else:
                self.column_stats["sum"] = None
        elif str_stats := get_if[string_statistics](&type_specific_stats):
            if str_stats.minimum.has_value():
                self.column_stats["minimum"] = str_stats.minimum.value().decode("utf-8")
            else:
                self.column_stats["minimum"] = None
            if str_stats.maximum.has_value():
                self.column_stats["maximum"] = str_stats.maximum.value().decode("utf-8")
            else:
                self.column_stats["maximum"] = None
            if str_stats.sum.has_value():
                self.column_stats["sum"] = str_stats.sum.value()
            else:
                self.column_stats["sum"] = None
        elif bucket_stats := get_if[bucket_statistics](&type_specific_stats):
            self.column_stats["true_count"] = bucket_stats.count[0]
            self.column_stats["false_count"] = (
                self.number_of_values
                - self.column_stats["true_count"]
            )
        elif dec_stats := get_if[decimal_statistics](&type_specific_stats):
            if dec_stats.minimum.has_value():
                self.column_stats["minimum"] = dec_stats.minimum.value().decode("utf-8")
            else:
                self.column_stats["minimum"] = None
            if dec_stats.maximum.has_value():
                self.column_stats["maximum"] = dec_stats.maximum.value().decode("utf-8")
            else:
                self.column_stats["maximum"] = None
            if dec_stats.sum.has_value():
                self.column_stats["sum"] = dec_stats.sum.value().decode("utf-8")
            else:
                self.column_stats["sum"] = None
        elif date_stats := get_if[date_statistics](&type_specific_stats):
            if date_stats.minimum.has_value():
                self.column_stats["minimum"] = datetime.datetime.fromtimestamp(
                    datetime.timedelta(date_stats.minimum.value()).total_seconds(),
                    datetime.timezone.utc,
                )
            else:
                self.column_stats["minimum"] = None
            if date_stats.maximum.has_value():
                self.column_stats["maximum"] = datetime.datetime.fromtimestamp(
                    datetime.timedelta(date_stats.maximum.value()).total_seconds(),
                    datetime.timezone.utc,
                )
            else:
                self.column_stats["maximum"] = None
        elif bin_stats := get_if[binary_statistics](&type_specific_stats):
            if bin_stats.sum.has_value():
                self.column_stats["sum"] = bin_stats.sum.value()
            else:
                self.column_stats["sum"] = None
        elif ts_stats := get_if[timestamp_statistics](&type_specific_stats):
            # Before ORC-135, the local timezone offset was included and they were
            # stored as minimum and maximum. After ORC-135, the timestamp is
            # adjusted to UTC before being converted to milliseconds and stored
            # in minimumUtc and maximumUtc.
            # TODO: Support minimum and maximum by reading writer's local timezone
            if ts_stats.minimum_utc.has_value() and ts_stats.maximum_utc.has_value():
                self.column_stats["minimum"] = datetime.datetime.fromtimestamp(
                    ts_stats.minimum_utc.value() / 1000, datetime.timezone.utc
                )
                self.column_stats["maximum"] = datetime.datetime.fromtimestamp(
                    ts_stats.maximum_utc.value() / 1000, datetime.timezone.utc
                )
        else:
            raise ValueError("Unsupported statistics type")

    def __getitem__(self, item):
        return self.column_stats[item]

    def __contains__(self, item):
        return item in self.column_stats

    def get(self, item, default=None):
        return self.column_stats.get(item, default)

    @staticmethod
    cdef OrcColumnStatistics from_libcudf(column_statistics& col_stats):
        cdef OrcColumnStatistics out = OrcColumnStatistics.__new__(OrcColumnStatistics)
        out.number_of_values_c = col_stats.number_of_values
        out.has_null_c = col_stats.has_null
        out.type_specific_stats_c = col_stats.type_specific_stats
        out._init_stats_dict()
        return out


cdef class ParsedOrcStatistics:
    """
    Holds column names and parsed file-level and stripe-level statistics.

    For details, see :cpp:class:`cudf::io::parsed_orc_statistics`
    """

    __hash__ = None

    @property
    def column_names(self):
        return [name.decode() for name in self.c_obj.column_names]

    @property
    def file_stats(self):
        return [
            OrcColumnStatistics.from_libcudf(self.c_obj.file_stats[i])
            for i in range(self.c_obj.file_stats.size())
        ]

    @property
    def stripes_stats(self):
        return [
            [
                OrcColumnStatistics.from_libcudf(stripe_stats_c[i])
                for i in range(stripe_stats_c.size())
            ]
            for stripe_stats_c in self.c_obj.stripes_stats
        ]

    @staticmethod
    cdef ParsedOrcStatistics from_libcudf(parsed_orc_statistics& orc_stats):
        cdef ParsedOrcStatistics out = ParsedOrcStatistics.__new__(ParsedOrcStatistics)
        out.c_obj = move(orc_stats)
        return out


cdef class OrcReaderOptions:
    """
    The settings to use for ``read_orc``

    For details, see :cpp:class:`cudf::io::orc_reader_options`
    """
    @staticmethod
    def builder(SourceInfo source):
        """
        Create a OrcReaderOptionsBuilder object

        For details, see :cpp:func:`cudf::io::orc_reader_options::builder`

        Parameters
        ----------
        sink : SourceInfo
            The source to read the ORC file from.

        Returns
        -------
        OrcReaderOptionsBuilder
            Builder to build OrcReaderOptions
        """
        cdef OrcReaderOptionsBuilder orc_builder = (
            OrcReaderOptionsBuilder.__new__(OrcReaderOptionsBuilder)
        )
        orc_builder.c_obj = orc_reader_options.builder(source.c_obj)
        orc_builder.source = source
        return orc_builder

    cpdef void set_num_rows(self, int64_t nrows):
        """
        Sets number of row to read.

        Parameters
        ----------
        nrows: int64_t
            Number of rows

        Returns
        -------
        None
        """
        self.c_obj.set_num_rows(nrows)

    cpdef void set_skip_rows(self, int64_t skip_rows):
        """
        Sets number of rows to skip from the start.

        Parameters
        ----------
        skip_rows: int64_t
            Number of rows

        Returns
        -------
        None
        """
        self.c_obj.set_skip_rows(skip_rows)

    cpdef void set_stripes(self, list stripes):
        """
        Sets list of stripes to read for each input source.

        Parameters
        ----------
        stripes: list[list[size_type]]
            List of lists, mapping stripes to read to input sources

        Returns
        -------
        None
        """
        cdef vector[vector[size_type]] c_stripes
        cdef vector[size_type] vec
        for sub_list in stripes:
            for x in sub_list:
                vec.push_back(x)
            c_stripes.push_back(vec)
            vec.clear()
        self.c_obj.set_stripes(c_stripes)

    cpdef void set_decimal128_columns(self, list val):
        """
        Set columns that should be read as 128-bit Decimal.

        Parameters
        ----------
        val: list[str]
            List of fully qualified column names

        Returns
        -------
        None
        """
        cdef vector[string] c_decimal128_columns
        c_decimal128_columns.reserve(len(val))
        for col in val:
            if not isinstance(col, str):
                raise TypeError("Decimal 128 column names must be strings!")
            c_decimal128_columns.push_back(col.encode())
        self.c_obj.set_decimal128_columns(c_decimal128_columns)

    cpdef void set_timestamp_type(self, DataType type_):
        """
        Sets timestamp type to which timestamp column will be cast.

        Parameters
        ----------
        type_: DataType
            Type of timestamp

        Returns
        -------
        None
        """
        self.c_obj.set_timestamp_type(type_.c_obj)

    cpdef void set_columns(self, list col_names):
        """
        Sets names of the column to read.

        Parameters
        ----------
        col_names: list[str]
            List of column names

        Returns
        -------
        None
        """
        cdef vector[string] c_column_names
        c_column_names.reserve(len(col_names))
        for col in col_names:
            if not isinstance(col, str):
                raise TypeError("Column names must be strings!")
            c_column_names.push_back(col.encode())
        self.c_obj.set_columns(c_column_names)

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

cdef class OrcReaderOptionsBuilder:
    cpdef OrcReaderOptionsBuilder use_index(self, bool use):
        """
        Enable/Disable use of row index to speed-up reading.

        Parameters
        ----------
        use : bool
            Boolean value to enable/disable row index use

        Returns
        -------
        OrcReaderOptionsBuilder
        """
        self.c_obj.use_index(use)
        return self

    cpdef OrcReaderOptions build(self):
        """Create a OrcReaderOptions object"""
        cdef OrcReaderOptions orc_options = OrcReaderOptions.__new__(
            OrcReaderOptions
        )
        orc_options.c_obj = move(self.c_obj.build())
        orc_options.source = self.source
        return orc_options


cpdef TableWithMetadata read_orc(
    OrcReaderOptions options, object stream = None, DeviceMemoryResource mr=None
):
    """
    Read from ORC format.

    The source to read from and options are encapsulated
    by the `options` object.

    For details, see :cpp:func:`read_orc`.

    Parameters
    ----------
    options: OrcReaderOptions
        Settings for controlling reading behavior
    stream : Stream | None
        CUDA stream used for device memory operations and kernel launches
    mr : DeviceMemoryResource, optional
        Device memory resource used to allocate the returned table's device memory.
    """
    cdef table_with_metadata c_result
    cdef Stream s = _get_stream(stream)
    cdef cudaStream_t _cs = s.view().value()
    mr = _get_memory_resource(mr)
    with nogil:
        c_result = move(cpp_read_orc(options.c_obj, _cs, mr.get_mr()))

    return TableWithMetadata.from_libcudf(c_result, s, mr)


cpdef ParsedOrcStatistics read_parsed_orc_statistics(
    SourceInfo source_info,
    object stream=None
):
    """
    Read ORC statistics from a source.

    Parameters
    ----------
    source_info : SourceInfo
        The source to read statistics from.
    stream : Stream | None
        CUDA stream used for device memory operations and kernel launches.

    Returns
    -------
    ParsedOrcStatistics
        The parsed ORC statistics.
    """
    cdef Stream s = _get_stream(stream)
    cdef parsed_orc_statistics parsed
    cdef cudaStream_t _cs = s.view().value()
    with nogil:
        parsed = cpp_read_parsed_orc_statistics(source_info.c_obj, _cs)
    return ParsedOrcStatistics.from_libcudf(parsed)


cdef class OrcWriterOptions:
    cpdef void set_stripe_size_bytes(self, size_t size_bytes):
        """
        Sets the maximum stripe size, in bytes.

        For details, see :cpp:func:`cudf::io::orc_writer_options::set_stripe_size_bytes`

        Parameters
        ----------
        size_bytes: size_t
            Sets the maximum stripe size, in bytes.

        Returns
        -------
        None
        """
        self.c_obj.set_stripe_size_bytes(size_bytes)

    cpdef void set_stripe_size_rows(self, size_type size_rows):
        """
        Sets the maximum stripe size, in rows.

        If the stripe size is smaller that the row group size,
        row group size will be reduced to math the stripe size.

        For details, see :cpp:func:`cudf::io::orc_writer_options::set_stripe_size_rows`

        Parameters
        ----------
        size_bytes: size_type
            Maximum stripe size, in rows to be set

        Returns
        -------
        None
        """
        self.c_obj.set_stripe_size_rows(size_rows)

    cpdef void set_row_index_stride(self, size_type stride):
        """
        Sets the row index stride.

        Rounded down to a multiple of 8.

        For details, see :cpp:func:`cudf::io::orc_writer_options::set_row_index_stride`

        Parameters
        ----------
        size_bytes: size_type
            Maximum stripe size, in rows to be set

        Returns
        -------
        None
        """
        self.c_obj.set_row_index_stride(stride)

    @staticmethod
    def builder(SinkInfo sink, Table table):
        """
        Create builder to create OrcWriterOptions.

        For details, see :cpp:func:`cudf::io::orc_writer_options::builder`

        Parameters
        ----------
        sink: SinkInfo
            The sink used for writer output
        table: Table
            Table to be written to output

        Returns
        -------
        OrcWriterOptionsBuilder
        """
        cdef OrcWriterOptionsBuilder orc_builder = OrcWriterOptionsBuilder.__new__(
            OrcWriterOptionsBuilder
        )
        orc_builder.c_obj = orc_writer_options.builder(sink.c_obj, table.view())
        orc_builder.table = table
        orc_builder.sink = sink
        return orc_builder


cdef class OrcWriterOptionsBuilder:
    cpdef OrcWriterOptionsBuilder compression(self, compression_type comp):
        """
        Sets compression type.

        For details, see :cpp:func:`cudf::io::orc_writer_options_builder::compression`

        Parameters
        ----------
        comp: CompressionType
            The compression type to use

        Returns
        -------
        OrcWriterOptionsBuilder
        """
        self.c_obj.compression(comp)
        return self

    cpdef OrcWriterOptionsBuilder enable_statistics(self, statistics_freq val):
        """
        Choose granularity of column statistics to be written.

        For details, see :cpp:func:`enable_statistics`

        Parameters
        ----------
        val: StatisticsFreq
            Level of statistics collection

        Returns
        -------
        OrcWriterOptionsBuilder
        """
        self.c_obj.enable_statistics(val)
        return self

    cpdef OrcWriterOptionsBuilder key_value_metadata(self, dict kvm):
        """
        Sets Key-Value footer metadata.

        Parameters
        ----------
        kvm: dict
            Key-Value footer metadata

        Returns
        -------
        OrcWriterOptionsBuilder
        """
        self.c_obj.key_value_metadata(
            {key.encode(): value.encode() for key, value in kvm.items()}
        )
        return self

    cpdef OrcWriterOptionsBuilder metadata(self, TableInputMetadata meta):
        """
        Sets associated metadata.

        For details, see :cpp:func:`cudf::io::orc_writer_options_builder::metadata`

        Parameters
        ----------
        meta: TableInputMetadata
            Associated metadata

        Returns
        -------
        OrcWriterOptionsBuilder
        """
        self.c_obj.metadata(meta.c_obj)
        return self

    cpdef OrcWriterOptions build(self):
        """Moves the ORC writer options builder"""
        cdef OrcWriterOptions orc_options = OrcWriterOptions.__new__(
            OrcWriterOptions
        )
        orc_options.c_obj = move(self.c_obj.build())
        orc_options.table = self.table
        orc_options.sink = self.sink
        return orc_options


cpdef void write_orc(OrcWriterOptions options, object stream = None):
    """
    Write to ORC format.

    The table to write, output paths, and options are encapsulated
    by the `options` object.

    For details, see :cpp:func:`write_orc`.

    Parameters
    ----------
    options: OrcWriterOptions
        Settings for controlling writing behavior
    stream : Stream | None
        CUDA stream used for device memory operations and kernel launches

    Returns
    -------
    None
    """
    cdef Stream s = _get_stream(stream)
    cdef cudaStream_t _cs = s.view().value()
    with nogil:
        cpp_write_orc(move(options.c_obj), _cs)


cdef class OrcChunkedWriter:
    cpdef void close(self):
        """
        Closes the chunked ORC writer.

        Returns
        -------
        None
        """
        with nogil:
            self.c_obj.get()[0].close()

    cpdef void write(self, Table table):
        """
        Writes table to output.

        Parameters
        ----------
        table: Table
            able that needs to be written

        Returns
        -------
        None
        """
        cdef table_view c_table = table.view()
        with nogil:
            self.c_obj.get()[0].write(c_table)

    @staticmethod
    def from_options(ChunkedOrcWriterOptions options, object stream = None):
        """
        Creates a chunked ORC writer from options

        Parameters
        ----------
        options: ChunkedOrcWriterOptions
            Settings for controlling writing behavior
        stream : Stream | None
            CUDA stream used for device memory operations and kernel launches

        Returns
        -------
        OrcChunkedWriter
        """
        cdef OrcChunkedWriter orc_writer = OrcChunkedWriter.__new__(
            OrcChunkedWriter
        )
        cdef Stream s = _get_stream(stream)
        cdef cudaStream_t _cs = s.view().value()
        orc_writer.c_obj.reset(new orc_chunked_writer(options.c_obj, _cs))
        return orc_writer


cdef class ChunkedOrcWriterOptions:
    cpdef void set_stripe_size_bytes(self, size_t size_bytes):
        """
        Sets the maximum stripe size, in bytes.

        Parameters
        ----------
        size_bytes: size_t
            Sets the maximum stripe size, in bytes.

        Returns
        -------
        None
        """
        self.c_obj.set_stripe_size_bytes(size_bytes)

    cpdef void set_stripe_size_rows(self, size_type size_rows):
        """
        Sets the maximum stripe size, in rows.

        If the stripe size is smaller that the row group size,
        row group size will be reduced to math the stripe size.

        Parameters
        ----------
        size_bytes: size_type
            Maximum stripe size, in rows to be set

        Returns
        -------
        None
        """
        self.c_obj.set_stripe_size_rows(size_rows)

    cpdef void set_row_index_stride(self, size_type stride):
        """
        Sets the row index stride.

        Rounded down to a multiple of 8.

        Parameters
        ----------
        size_bytes: size_type
            Maximum stripe size, in rows to be set

        Returns
        -------
        None
        """
        self.c_obj.set_row_index_stride(stride)

    @staticmethod
    def builder(SinkInfo sink):
        """
        Create builder to create ChunkedOrcWriterOptions.

        Parameters
        ----------
        sink: SinkInfo
            The sink used for writer output
        table: Table
            Table to be written to output

        Returns
        -------
        ChunkedOrcWriterOptionsBuilder
        """
        cdef ChunkedOrcWriterOptionsBuilder orc_builder = \
            ChunkedOrcWriterOptionsBuilder.__new__(
                ChunkedOrcWriterOptionsBuilder
            )
        orc_builder.c_obj = chunked_orc_writer_options.builder(sink.c_obj)
        orc_builder.sink = sink
        return orc_builder


cdef class ChunkedOrcWriterOptionsBuilder:
    cpdef ChunkedOrcWriterOptionsBuilder compression(self, compression_type comp):
        """
        Sets compression type.

        Parameters
        ----------
        comp: CompressionType
            The compression type to use

        Returns
        -------
        ChunkedOrcWriterOptionsBuilder
        """
        self.c_obj.compression(comp)
        return self

    cpdef ChunkedOrcWriterOptionsBuilder enable_statistics(self, statistics_freq val):
        """
        Choose granularity of column statistics to be written.

        Parameters
        ----------
        val: StatisticsFreq
            Level of statistics collection

        Returns
        -------
        ChunkedOrcWriterOptionsBuilder
        """
        self.c_obj.enable_statistics(val)
        return self

    cpdef ChunkedOrcWriterOptionsBuilder key_value_metadata(
        self,
        dict kvm
    ):
        """
        Sets Key-Value footer metadata.

        Parameters
        ----------
        kvm: dict
            Key-Value footer metadata

        Returns
        -------
        ChunkedOrcWriterOptionsBuilder
        """
        self.c_obj.key_value_metadata(
            {key.encode(): value.encode() for key, value in kvm.items()}
        )
        return self

    cpdef ChunkedOrcWriterOptionsBuilder metadata(self, TableInputMetadata meta):
        """
        Sets associated metadata.

        Parameters
        ----------
        meta: TableInputMetadata
            Associated metadata

        Returns
        -------
        ChunkedOrcWriterOptionsBuilder
        """
        self.c_obj.metadata(meta.c_obj)
        return self

    cpdef ChunkedOrcWriterOptions build(self):
        """Create a OrcWriterOptions object"""
        cdef ChunkedOrcWriterOptions orc_options = ChunkedOrcWriterOptions.__new__(
            ChunkedOrcWriterOptions
        )
        orc_options.c_obj = move(self.c_obj.build())
        orc_options.sink = self.sink
        return orc_options


cpdef bool is_supported_read_orc(compression_type compression):
    """Check if the compression type is supported for reading ORC files.

    For details, see :cpp:func:`is_supported_read_orc`.

    Parameters
    ----------
    compression : CompressionType
        The compression type to check

    Returns
    -------
    bool
        True if the compression type is supported for reading ORC files
    """
    return cpp_is_supported_read_orc(compression)


cpdef bool is_supported_write_orc(compression_type compression):
    """Check if the compression type is supported for writing ORC files.

    For details, see :cpp:func:`is_supported_write_orc`.

    Parameters
    ----------
    compression : CompressionType
        The compression type to check

    Returns
    -------
    bool
        True if the compression type is supported for writing ORC files
    """
    return cpp_is_supported_write_orc(compression)
