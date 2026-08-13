# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libcpp cimport bool
from libcpp.map cimport map
from libcpp.string cimport string
from libcpp.utility cimport move
from libcpp.vector cimport vector

from rmm.pylibrmm.stream cimport Stream
from cuda.bindings.cyruntime cimport cudaStream_t
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource

from pylibcudf.io.types cimport SourceInfo, SinkInfo, TableWithMetadata

from pylibcudf.libcudf.io.csv cimport (
    csv_reader_options,
    csv_writer_options,
    read_csv as cpp_read_csv,
    write_csv as cpp_write_csv,
    is_supported_write_csv as cpp_is_supported_write_csv,
)

from pylibcudf.libcudf.io.types cimport (
    compression_type,
    quote_style,
    table_with_metadata,
)

from pylibcudf.libcudf.types cimport data_type, size_type

from pylibcudf.table cimport Table

from pylibcudf.types cimport DataType

from pylibcudf.utils cimport _get_stream, _get_memory_resource

__all__ = [
    "read_csv",
    "write_csv",
    "CsvWriterOptions",
    "CsvWriterOptionsBuilder",
    "CsvReaderOptions",
    "CsvReaderOptionsBuilder",
]

cdef class CsvReaderOptions:
    """The settings to use for ``read_csv``
    For details, see :cpp:class:`cudf::io::csv_reader_options`
    """
    @staticmethod
    def builder(SourceInfo source):
        """
        Create a CsvWriterOptionsBuilder object

        For details, see :cpp:func:`cudf::io::csv_reader_options::builder`

        Parameters
        ----------
        sink : SourceInfo
            The source to read the CSV file from.

        Returns
        -------
        CsvReaderOptionsBuilder
            Builder to build CsvReaderOptions
        """
        cdef CsvReaderOptionsBuilder csv_builder = CsvReaderOptionsBuilder.__new__(
            CsvReaderOptionsBuilder
        )
        csv_builder.c_obj = csv_reader_options.builder(source.c_obj)
        csv_builder.source = source
        return csv_builder

    cpdef void set_header(self, size_type header):
        """
        Sets header row index.

        Parameters
        ----------
        header : size_type
            Index where header row is located

        Returns
        -------
        None
        """
        self.c_obj.set_header(header)

    cpdef void set_names(self, list col_names):
        """
        Sets names of the column.

        Parameters
        ----------
        col_names : list[str]
            List of column names

        Returns
        -------
        None
        """
        cdef vector[string] vec
        for name in col_names:
            vec.push_back(name.encode())
        self.c_obj.set_names(vec)

    cpdef void set_prefix(self, str prefix):
        """
        Sets prefix to be used for column ID.

        Parameters
        ----------
        prefix : str
            String used as prefix in for each column name

        Returns
        -------
        None
        """
        self.c_obj.set_prefix(prefix.encode())

    cpdef void set_use_cols_indexes(self, list col_indices):
        """
        Sets indexes of columns to read.

        Parameters
        ----------
        col_indices : list[int]
            List of column indices that are needed

        Returns
        -------
        None
        """
        cdef vector[int] vec
        for i in col_indices:
            vec.push_back(i)
        self.c_obj.set_use_cols_indexes(vec)

    cpdef void set_use_cols_names(self, list col_names):
        """
        Sets names of the columns to be read.

        Parameters
        ----------
        col_names : list[str]
            List of column indices that are needed

        Returns
        -------
        None
        """
        cdef vector[string] vec
        for name in col_names:
            vec.push_back(name.encode())
        self.c_obj.set_use_cols_names(vec)

    cpdef void set_delimiter(self, str delimiter):
        """
        Sets field delimiter.

        Parameters
        ----------
        delimiter : str
            A character to indicate delimiter

        Returns
        -------
        None
        """
        self.c_obj.set_delimiter(ord(delimiter))

    cpdef void set_thousands(self, str thousands):
        """
        Sets numeric data thousands separator.

        Parameters
        ----------
        thousands : str
            A character that separates thousands

        Returns
        -------
        None
        """
        self.c_obj.set_thousands(ord(thousands))

    cpdef void set_comment(self, str comment):
        """
        Sets comment line start character.

        Parameters
        ----------
        comment : str
            A character that indicates comment

        Returns
        -------
        None
        """
        self.c_obj.set_comment(ord(comment))

    cpdef void set_parse_dates(self, list val):
        """
        Sets indexes or names of columns to read as datetime.

        Parameters
        ----------
        val : list[int | str]
            List column indices or names to infer as datetime.

        Returns
        -------
        None
        """
        cdef vector[string] vec_str
        cdef vector[int] vec_int
        if not all([isinstance(col, (str, int)) for col in val]):
            raise TypeError("Must be a list of int or str")
        else:
            for date in val:
                if isinstance(date, str):
                    vec_str.push_back(date.encode())
                else:
                    vec_int.push_back(date)
            self.c_obj.set_parse_dates(vec_str)
            self.c_obj.set_parse_dates(vec_int)

    cpdef void set_parse_hex(self, list val):
        """
        Sets indexes or names of columns to parse as hexadecimal.

        Parameters
        ----------
        val : list[int | str]
            List of column indices or names to parse as hexadecimal

        Returns
        -------
        None
        """
        cdef vector[string] vec_str
        cdef vector[int] vec_int
        if not all([isinstance(col, (str, int)) for col in val]):
            raise TypeError("Must be a list of int or str")
        else:
            for hx in val:
                if isinstance(hx, str):
                    vec_str.push_back(hx.encode())
                else:
                    vec_int.push_back(hx)

            self.c_obj.set_parse_hex(vec_str)
            self.c_obj.set_parse_hex(vec_int)

    cpdef void set_dtypes(self, object types):
        """
        Sets per-column types.

        Parameters
        ----------
        types : dict[str, data_type] | list[data_type]
            Column name to data type map specifying the columns' target data types.
            Or a list specifying the columns' target data types.

        Returns
        -------
        None
        """
        cdef map[string, data_type] dtype_map
        cdef vector[data_type] dtype_list
        if isinstance(types, dict):
            for name, dtype in types.items():
                dtype_map[str(name).encode()] = (<DataType>dtype).c_obj
            self.c_obj.set_dtypes(dtype_map)
        elif isinstance(types, list):
            for dtype in types:
                dtype_list.push_back((<DataType>dtype).c_obj)
            self.c_obj.set_dtypes(dtype_list)
        else:
            raise TypeError("Must pass an dict or list")

    cpdef void set_true_values(self, list true_values):
        """
        Sets additional values to recognize as boolean true values.

        Parameters
        ----------
        true_values : list[str]
            List of values to be considered to be true

        Returns
        -------
        None
        """
        cdef vector[string] vec
        for val in true_values:
            vec.push_back(val.encode())
        self.c_obj.set_true_values(vec)

    cpdef void set_false_values(self, list false_values):
        """
        Sets additional values to recognize as boolean false values.

        Parameters
        ----------
        false_values : list[str]
            List of values to be considered to be false

        Returns
        -------
        None
        """
        cdef vector[string] vec
        for val in false_values:
            vec.push_back(val.encode())
        self.c_obj.set_false_values(vec)

    cpdef void set_na_values(self, list na_values):
        """
        Sets additional values to recognize as null values.

        Parameters
        ----------
        na_values : list[str]
            List of values to be considered to be null

        Returns
        -------
        None
        """
        cdef vector[string] vec
        for val in na_values:
            vec.push_back(val.encode())
        self.c_obj.set_na_values(vec)

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


cdef class CsvReaderOptionsBuilder:
    """
    Builder to build options for ``read_csv``

    For details, see :cpp:class:`cudf::io::csv_reader_options_builder`
    """
    cpdef CsvReaderOptionsBuilder compression(self, compression_type compression):
        """
        Sets compression format of the source.

        Parameters
        ----------
        compression : compression_type
            Compression type

        Returns
        -------
        CsvReaderOptionsBuilder
        """
        self.c_obj.compression(compression)
        return self

    cpdef CsvReaderOptionsBuilder mangle_dupe_cols(self, bool mangle_dupe_cols):
        """
        Sets whether to rename duplicate column names.

        Parameters
        ----------
        mangle_dupe_cols : bool
            Boolean value to enable/disable

        Returns
        -------
        CsvReaderOptionsBuilder
        """
        self.c_obj.mangle_dupe_cols(mangle_dupe_cols)
        return self

    cpdef CsvReaderOptionsBuilder byte_range_offset(self, size_t byte_range_offset):
        """
        Sets number of bytes to skip from source start.

        Parameters
        ----------
        byte_range_offset : size_t
            Number of bytes of offset

        Returns
        -------
        CsvReaderOptionsBuilder
        """
        self.c_obj.byte_range_offset(byte_range_offset)
        return self

    cpdef CsvReaderOptionsBuilder byte_range_size(self, size_t byte_range_size):
        """
        Sets number of bytes to read.

        Parameters
        ----------
        byte_range_offset : size_t
            Number of bytes to read

        Returns
        -------
        CsvReaderOptionsBuilder
        """
        self.c_obj.byte_range_size(byte_range_size)
        return self

    cpdef CsvReaderOptionsBuilder nrows(self, size_type nrows):
        """
        Sets number of rows to read.

        Parameters
        ----------
        nrows : size_type
            Number of rows to read

        Returns
        -------
        CsvReaderOptionsBuilder
        """
        self.c_obj.nrows(nrows)
        return self

    cpdef CsvReaderOptionsBuilder skiprows(self, size_type skiprows):
        """
        Sets number of rows to skip from start.

        Parameters
        ----------
        skiprows : size_type
            Number of rows to skip

        Returns
        -------
        CsvReaderOptionsBuilder
        """
        self.c_obj.skiprows(skiprows)
        return self

    cpdef CsvReaderOptionsBuilder skipfooter(self, size_type skipfooter):
        """
        Sets number of rows to skip from end.

        Parameters
        ----------
        skipfooter : size_type
            Number of rows to skip

        Returns
        -------
        CsvReaderOptionsBuilder
        """
        self.c_obj.skipfooter(skipfooter)
        return self

    cpdef CsvReaderOptionsBuilder quoting(self, quote_style quoting):
        """
        Sets quoting style.

        Parameters
        ----------
        quoting : quote_style
            Quoting style used

        Returns
        -------
        CsvReaderOptionsBuilder
        """
        self.c_obj.quoting(quoting)
        return self

    cpdef CsvReaderOptionsBuilder lineterminator(self, str lineterminator):
        """
        Sets line terminator.

        Parameters
        ----------
        quoting : str
            A character to indicate line termination

        Returns
        -------
        CsvReaderOptionsBuilder
        """
        self.c_obj.lineterminator(ord(lineterminator))
        return self

    cpdef CsvReaderOptionsBuilder quotechar(self, str quotechar):
        """
        Sets quoting character.

        Parameters
        ----------
        quotechar : str
            A character to indicate quoting

        Returns
        -------
        CsvReaderOptionsBuilder
        """
        self.c_obj.quotechar(ord(quotechar))
        return self

    cpdef CsvReaderOptionsBuilder decimal(self, str decimal):
        """
        Sets decimal point character.

        Parameters
        ----------
        quotechar : str
            A character that indicates decimal values

        Returns
        -------
        CsvReaderOptionsBuilder
        """
        self.c_obj.decimal(ord(decimal))
        return self

    cpdef CsvReaderOptionsBuilder delim_whitespace(self, bool delim_whitespace):
        """
        Sets whether to treat whitespace as field delimiter.

        Parameters
        ----------
        delim_whitespace : bool
            Boolean value to enable/disable

        Returns
        -------
        CsvReaderOptionsBuilder
        """
        self.c_obj.delim_whitespace(delim_whitespace)
        return self

    cpdef CsvReaderOptionsBuilder skipinitialspace(self, bool skipinitialspace):
        """
        Sets whether to skip whitespace after the delimiter.

        Parameters
        ----------
        skipinitialspace : bool
            Boolean value to enable/disable

        Returns
        -------
        CsvReaderOptionsBuilder
        """
        self.c_obj.skipinitialspace(skipinitialspace)
        return self

    cpdef CsvReaderOptionsBuilder skip_blank_lines(self, bool skip_blank_lines):
        """
        Sets whether to ignore empty lines or parse line values as invalid.

        Parameters
        ----------
        skip_blank_lines : bool
            Boolean value to enable/disable

        Returns
        -------
        CsvReaderOptionsBuilder
        """
        self.c_obj.skip_blank_lines(skip_blank_lines)
        return self

    cpdef CsvReaderOptionsBuilder doublequote(self, bool doublequote):
        """
        Sets a quote inside a value is double-quoted.

        Parameters
        ----------
        doublequote : bool
            Boolean value to enable/disable

        Returns
        -------
        CsvReaderOptionsBuilder
        """
        self.c_obj.doublequote(doublequote)
        return self

    cpdef CsvReaderOptionsBuilder keep_default_na(self, bool keep_default_na):
        """
        Sets whether to keep the built-in default NA values.

        Parameters
        ----------
        keep_default_na : bool
            Boolean value to enable/disable

        Returns
        -------
        CsvReaderOptionsBuilder
        """
        self.c_obj.keep_default_na(keep_default_na)
        return self

    cpdef CsvReaderOptionsBuilder na_filter(self, bool na_filter):
        """
        Sets whether to disable null filter.

        Parameters
        ----------
        na_filter : bool
            Boolean value to enable/disable

        Returns
        -------
        CsvReaderOptionsBuilder
        """
        self.c_obj.na_filter(na_filter)
        return self

    cpdef CsvReaderOptionsBuilder dayfirst(self, bool dayfirst):
        """
        Sets whether to parse dates as DD/MM versus MM/DD.

        Parameters
        ----------
        dayfirst : bool
            Boolean value to enable/disable

        Returns
        -------
        CsvReaderOptionsBuilder
        """
        self.c_obj.dayfirst(dayfirst)
        return self

    cpdef CsvReaderOptionsBuilder delimiter(self, str delimiter):
        """
        Sets field delimiter.

        Parameters
        ----------
        delimiter : str
            A character to indicate delimiter

        Returns
        -------
        CsvReaderOptionsBuilder
        """
        self.c_obj.delimiter(ord(delimiter))
        return self

    cpdef CsvReaderOptions build(self):
        """Create a CsvReaderOptions object"""
        cdef CsvReaderOptions csv_options = CsvReaderOptions.__new__(
            CsvReaderOptions
        )
        csv_options.c_obj = move(self.c_obj.build())
        csv_options.source = self.source
        return csv_options


cpdef TableWithMetadata read_csv(
    CsvReaderOptions options,
    object stream = None,
    DeviceMemoryResource mr=None,
):
    """
    Read from CSV format.

    The source to read from and options are encapsulated
    by the `options` object.

    For details, see :cpp:func:`read_csv`.

    Parameters
    ----------
    options: CsvReaderOptions
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
        c_result = move(cpp_read_csv(options.c_obj, _cs, mr.get_mr()))

    cdef TableWithMetadata tbl_meta = TableWithMetadata.from_libcudf(c_result, s, mr)
    return tbl_meta


# TODO: Implement the remaining methods
cdef class CsvWriterOptions:
    """The settings to use for ``write_csv``

    For details, see :cpp:class:`cudf::io::csv_writer_options`
    """
    @staticmethod
    def builder(SinkInfo sink, Table table):
        """Create a CsvWriterOptionsBuilder object

        For details, see :cpp:func:`cudf::io::csv_writer_options::builder`

        Parameters
        ----------
        sink : SinkInfo
            The sink used for writer output
        table : Table
            Table to be written to output

        Returns
        -------
        CsvWriterOptionsBuilder
            Builder to build CsvWriterOptions
        """
        cdef CsvWriterOptionsBuilder csv_builder = CsvWriterOptionsBuilder.__new__(
            CsvWriterOptionsBuilder
        )
        csv_builder.c_obj = csv_writer_options.builder(sink.c_obj, table.view())
        csv_builder.table = table
        csv_builder.sink = sink
        return csv_builder


# TODO: Implement the remaining methods
cdef class CsvWriterOptionsBuilder:
    """Builder to build options for ``write_csv``

    For details, see :cpp:class:`cudf::io::csv_writer_options_builder`
    """
    cpdef CsvWriterOptionsBuilder names(self, list names):
        """Sets optional column names.

        Parameters
        ----------
        names : list[str]
            Column names

        Returns
        -------
        CsvWriterOptionsBuilder
            Builder to build CsvWriterOptions
        """
        self.c_obj.names([name.encode() for name in names])
        return self

    cpdef CsvWriterOptionsBuilder na_rep(self, str val):
        """Sets string to used for null entries.

        Parameters
        ----------
        val : str
            String to represent null value

        Returns
        -------
        CsvWriterOptionsBuilder
            Builder to build CsvWriterOptions
        """
        self.c_obj.na_rep(val.encode())
        return self

    cpdef CsvWriterOptionsBuilder include_header(self, bool val):
        """Enables/Disables headers being written to csv.

        Parameters
        ----------
        val : bool
            Boolean value to enable/disable

        Returns
        -------
        CsvWriterOptionsBuilder
            Builder to build CsvWriterOptions
        """
        self.c_obj.include_header(val)
        return self

    cpdef CsvWriterOptionsBuilder rows_per_chunk(self, int val):
        """Sets maximum number of rows to process for each file write.

        Parameters
        ----------
        val : int
            Number of rows per chunk

        Returns
        -------
        CsvWriterOptionsBuilder
            Builder to build CsvWriterOptions
        """
        self.c_obj.rows_per_chunk(val)
        return self

    cpdef CsvWriterOptionsBuilder line_terminator(self, str term):
        """Sets character used for separating lines.

        Parameters
        ----------
        term : str
            Character to represent line termination

        Returns
        -------
        CsvWriterOptionsBuilder
            Builder to build CsvWriterOptions
        """
        self.c_obj.line_terminator(term.encode())
        return self

    cpdef CsvWriterOptionsBuilder inter_column_delimiter(self, str delim):
        """Sets character used for separating column values.

        Parameters
        ----------
        delim : str
            Character to delimit column values

        Returns
        -------
        CsvWriterOptionsBuilder
            Builder to build CsvWriterOptions
        """
        self.c_obj.inter_column_delimiter(ord(delim))
        return self

    cpdef CsvWriterOptionsBuilder true_value(self, str val):
        """Sets string used for values != 0

        Parameters
        ----------
        val : str
            String to represent values != 0

        Returns
        -------
        CsvWriterOptionsBuilder
            Builder to build CsvWriterOptions
        """
        self.c_obj.true_value(val.encode())
        return self

    cpdef CsvWriterOptionsBuilder false_value(self, str val):
        """Sets string used for values == 0

        Parameters
        ----------
        val : str
            String to represent values == 0

        Returns
        -------
        CsvWriterOptionsBuilder
            Builder to build CsvWriterOptions
        """
        self.c_obj.false_value(val.encode())
        return self

    cpdef CsvWriterOptionsBuilder quoting(self, quote_style style):
        """Sets quoting style for string columns.

        Parameters
        ----------
        style : quote_style
            Quoting style (MINIMAL or NONE).
            MINIMAL: Quote only fields containing special characters.
            NONE: Never quote fields.

        Returns
        -------
        CsvWriterOptionsBuilder
            Builder to build CsvWriterOptions
        """
        self.c_obj.quoting(style)
        return self

    cpdef CsvWriterOptions build(self):
        """Create a CsvWriterOptions object"""
        cdef CsvWriterOptions csv_options = CsvWriterOptions.__new__(
            CsvWriterOptions
        )
        csv_options.c_obj = move(self.c_obj.build())
        csv_options.table = self.table
        csv_options.sink = self.sink
        return csv_options


cpdef void write_csv(
    CsvWriterOptions options,
    object stream = None,
):
    """
    Write to CSV format.

    The table to write, output paths, and options are encapsulated
    by the `options` object.

    For details, see :cpp:func:`write_csv`.

    Parameters
    ----------
    options: CsvWriterOptions
        Settings for controlling writing behavior
    stream : Stream | None
        CUDA stream used for device memory operations and kernel launches
    """
    cdef Stream s = _get_stream(stream)
    cdef cudaStream_t _cs = s.view().value()
    with nogil:
        cpp_write_csv(move(options.c_obj), _cs)


cpdef bool is_supported_write_csv(DataType type):
    """Check if the dtype is supported for CSV writing

    For details, see :cpp:func:`is_supported_write_type`.
    """
    return cpp_is_supported_write_csv(type.c_obj)
