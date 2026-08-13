# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from libcpp cimport bool
from libcpp.map cimport map
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.utility cimport move
from libcpp.vector cimport vector

from rmm.pylibrmm.stream cimport Stream

from pylibcudf.concatenate cimport concatenate
from pylibcudf.column cimport Column
from pylibcudf.scalar cimport Scalar
from pylibcudf.utils cimport _get_memory_resource
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource

from pylibcudf.io.types cimport SinkInfo, SourceInfo, TableWithMetadata

from pylibcudf.libcudf.io.json cimport (
    json_reader_options,
    json_recovery_mode_t,
    json_writer_options,
    read_json as cpp_read_json,
    schema_element,
    write_json as cpp_write_json,
    is_supported_write_json as cpp_is_supported_write_json,

)

from pylibcudf.libcudf.strings cimport combine as cpp_combine

from pylibcudf.libcudf.scalar.scalar cimport string_scalar

from pylibcudf.libcudf.io.types cimport (
    compression_type,
    table_with_metadata,
)

from pylibcudf.libcudf.io.json import json_recovery_mode_t as JsonRecoveryModeType  # no-cython-lint

from pylibcudf.libcudf.types cimport data_type, size_type
from pylibcudf.libcudf.column.column cimport column, column_contents
from pylibcudf.libcudf.column.column_view cimport column_view

from pylibcudf.types cimport DataType

from pylibcudf.utils cimport _get_stream

from cython.operator import dereference

from rmm.pylibrmm.device_buffer cimport DeviceBuffer
from cuda.bindings.cyruntime cimport cudaStream_t

__all__ = [
    "chunked_read_json",
    "read_json",
    "read_json_from_string_column",
    "write_json",
    "JsonReaderOptions",
    "JsonReaderOptionsBuilder",
    "JsonWriterOptions",
    "JsonWriterOptionsBuilder"
]

cdef map[string, schema_element] _generate_schema_map(list dtypes):
    cdef map[string, schema_element] schema_map
    cdef schema_element s_elem
    cdef string c_name

    for name, dtype, child_dtypes in dtypes:
        if not (isinstance(name, str) and
                isinstance(dtype, DataType) and
                isinstance(child_dtypes, list)):

            raise ValueError("Must pass a list of a tuple containing "
                             "(column_name, column_dtype, list of child_dtypes)")

        c_name = <str>name.encode()

        s_elem.type = (<DataType>dtype).c_obj
        s_elem.child_types = _generate_schema_map(child_dtypes)

        schema_map[c_name] = s_elem
    return schema_map


cpdef JsonReaderOptions _setup_json_reader_options(
        SourceInfo source_info,
        list dtypes,
        compression_type compression = compression_type.AUTO,
        bool lines = False,
        size_t byte_range_offset = 0,
        size_t byte_range_size = 0,
        bool keep_quotes = False,
        bool mixed_types_as_string = False,
        bool prune_columns = False,
        json_recovery_mode_t recovery_mode = json_recovery_mode_t.FAIL,
        dict extra_parameters=None,
):
    options = (
        JsonReaderOptions.builder(source_info)
        .compression(compression)
        .lines(lines)
        .byte_range_offset(byte_range_offset)
        .byte_range_size(byte_range_size)
        .recovery_mode(recovery_mode)
        .build()
    )

    if dtypes is not None:
        options.set_dtypes(dtypes)

    options.enable_keep_quotes(keep_quotes)
    options.enable_mixed_types_as_string(mixed_types_as_string)
    options.enable_prune_columns(prune_columns)

    # These hidden options are subjected to change without deprecation cycle.
    # These are used to test libcudf JSON reader features, not used in cuDF.
    if extra_parameters is not None:
        for key, value in extra_parameters.items():
            if key == 'delimiter':
                options.set_delimiter(value)
            elif key == 'dayfirst':
                options.enable_dayfirst(value)
            elif key == 'experimental':
                options.enable_experimental(value)
            elif key == 'normalize_single_quotes':
                options.enable_normalize_single_quotes(value)
            elif key == 'normalize_whitespace':
                options.enable_normalize_whitespace(value)
            elif key == 'strict_validation':
                options.set_strict_validation(value)
            elif key == 'allow_unquoted_control_chars':
                options.allow_unquoted_control_chars(value)
            elif key == 'allow_numeric_leading_zeros':
                options.allow_numeric_leading_zeros(value)
            elif key == 'allow_nonnumeric_numbers':
                options.allow_nonnumeric_numbers(value)
            elif key == 'na_values':
                options.set_na_values(value)
            else:
                raise ValueError(
                    "cudf engine doesn't support the "
                    f"'{key}' keyword argument for read_json"
                )
    return options


cdef class JsonReaderOptions:
    """
    The settings to use for ``read_json``

    For details, see `:cpp:class:`cudf::io::json_reader_options`
    """
    @staticmethod
    def builder(SourceInfo source):
        """
        Create a JsonReaderOptionsBuilder object

        For details, see :cpp:func:`cudf::io::json_reader_options::builder`

        Parameters
        ----------
        sink : SourceInfo
            The source to read the JSON file from.

        Returns
        -------
        JsonReaderOptionsBuilder
            Builder to build JsonReaderOptions
        """
        cdef JsonReaderOptionsBuilder json_builder = (
            JsonReaderOptionsBuilder.__new__(JsonReaderOptionsBuilder)
        )
        json_builder.c_obj = json_reader_options.builder(source.c_obj)
        json_builder.source = source
        return json_builder

    cpdef void set_dtypes(self, list types):
        """
        Set data types for columns to be read.

        Parameters
        ----------
        types : list
            List of dtypes or a list of tuples of
            column names, dtypes, and list of tuples
            (to support nested column hierarchy)

        Returns
        -------
        None
        """
        cdef vector[data_type] types_vec
        if isinstance(types[0], tuple):
            self.c_obj.set_dtypes(_generate_schema_map(types))
        else:
            types_vec.reserve(len(types))
            for dtype in types:
                types_vec.push_back((<DataType>dtype).c_obj)
            self.c_obj.set_dtypes(types_vec)

    cpdef void enable_keep_quotes(self, bool keep_quotes):
        """
        Set whether the reader should keep quotes of string values.

        Parameters
        ----------
        keep_quotes : bool
           Boolean value to indicate whether the reader should
           keep quotes of string values

        Returns
        -------
        None
        """
        self.c_obj.enable_keep_quotes(keep_quotes)

    cpdef void enable_mixed_types_as_string(self, bool mixed_types_as_string):
        """
        Set whether to parse mixed types as a string column.
        Also enables forcing to read a struct as string column using schema.

        Parameters
        ----------
        mixed_types_as_string : bool
           Boolean value to enable/disable parsing mixed types
           as a string column

        Returns
        -------
        None
        """
        self.c_obj.enable_mixed_types_as_string(mixed_types_as_string)

    cpdef void enable_prune_columns(self, bool prune_columns):
        """
        Set whether to prune columns on read, selected
        based on the ``set_dtypes`` option.

        Parameters
        ----------
        prune_columns : bool
           When set as true, if the reader options include
           ``set_dtypes``, then the reader will only return those
           columns which are mentioned in ``set_dtypes``. If false,
           then all columns are returned, independent of the
           ``set_dtypes`` setting.

        Returns
        -------
        None
        """
        self.c_obj.enable_prune_columns(prune_columns)

    cpdef void set_byte_range_offset(self, size_t offset):
        """
        Set number of bytes to skip from source start.

        Parameters
        ----------
        offset : size_t
            Number of bytes of offset

        Returns
        -------
        None
        """
        self.c_obj.set_byte_range_offset(offset)

    cpdef void set_byte_range_size(self, size_t size):
        """
        Set number of bytes to read.

        Parameters
        ----------
        size : size_t
            Number of bytes to read

        Returns
        -------
        None
        """
        self.c_obj.set_byte_range_size(size)

    cpdef void enable_lines(self, bool val):
        """
        Set whether to read the file as a json object per line.

        Parameters
        ----------
        val : bool
            Boolean value to enable/disable the option
            to read each line as a json object

        Returns
        -------
        None
        """
        self.c_obj.enable_lines(val)

    # These hidden options are subjected to change without deprecation cycle.
    # These are used to test libcudf JSON reader features, not used in cuDF.

    cpdef void set_delimiter(self, str val):
        self.c_obj.set_delimiter(val.encode())

    cpdef void enable_dayfirst(self, bool val):
        self.c_obj.enable_dayfirst(val)

    cpdef void enable_experimental(self, bool val):
        self.c_obj.enable_experimental(val)

    cpdef void enable_normalize_single_quotes(self, bool val):
        self.c_obj.enable_normalize_single_quotes(val)

    cpdef void enable_normalize_whitespace(self, bool val):
        self.c_obj.enable_normalize_whitespace(val)

    cpdef void set_strict_validation(self, bool val):
        self.c_obj.set_strict_validation(val)

    cpdef void allow_unquoted_control_chars(self, bool val):
        self.c_obj.allow_unquoted_control_chars(val)

    cpdef void allow_numeric_leading_zeros(self, bool val):
        self.c_obj.allow_numeric_leading_zeros(val)

    cpdef void allow_nonnumeric_numbers(self, bool val):
        self.c_obj.allow_nonnumeric_numbers(val)

    cpdef void set_na_values(self, list vals):
        cdef vector[string] vec
        for val in vals:
            if isinstance(val, str):
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


cdef class JsonReaderOptionsBuilder:
    cpdef JsonReaderOptionsBuilder byte_range_offset(self, size_t byte_range_offset):
        """
        Set number of bytes to skip from source start.

        Parameters
        ----------
        byte_range_offset : size_t
            Number of bytes of offset

        Returns
        -------
        Self
        """
        self.c_obj.byte_range_offset(byte_range_offset)
        return self

    cpdef JsonReaderOptionsBuilder byte_range_size(self, size_t byte_range_size):
        """
        Set number of bytes to read.

        Parameters
        ----------
        byte_range_size : size_t
            Number of bytes to read

        Returns
        -------
        Self
        """
        self.c_obj.byte_range_size(byte_range_size)
        return self

    cpdef JsonReaderOptionsBuilder compression(self, compression_type compression):
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

    cpdef JsonReaderOptionsBuilder dayfirst(self, bool val):
        """
        Set whether the reader should parse dates as DD/MM versus MM/DD.

        Parameters
        ----------
        val : bool
            Boolean value to indicate whether the
            reader should enable/disable DD/MM parsing

        Returns
        -------
        Self
        """
        self.c_obj.dayfirst(val)
        return self

    cpdef JsonReaderOptionsBuilder delimiter(self, str delimiter):
        """
        Set delimiter character separating records in JSON lines inputs

        Parameters
        ----------
        delimiter : str
            Character to be used as delimiter separating records

        Returns
        -------
        Self
        """
        self.c_obj.delimiter(delimiter)
        return self

    cpdef JsonReaderOptionsBuilder dtypes(self, list types):
        """
        Set data type for columns to be read

        Parameters
        ----------
        types : list
            List of dtypes or a list of tuples of
            column names, dtypes, and list of tuples
            (to support nested column hierarchy)

        Returns
        -------
        Self
        """
        cdef vector[data_type] types_vec
        if isinstance(types[0], tuple):
            self.c_obj.dtypes(_generate_schema_map(types))
            return self
        else:
            types_vec.reserve(len(types))
            for dtype in types:
                types_vec.push_back((<DataType>dtype).c_obj)
            self.c_obj.dtypes(types_vec)
            return self

    cpdef JsonReaderOptionsBuilder experimental(self, bool val):
        """
        Set whether to enable experimental features.
        When set to true, experimental features, such as the new column tree
        construction, utf-8 matching of field names will be enabled.

        Parameters
        ----------
        val : bool
            Boolean value to enable/disable experimental features

        Returns
        -------
        Self
        """
        self.c_obj.experimental(val)
        return self

    cpdef JsonReaderOptionsBuilder keep_quotes(self, bool val):
        """
        Set whether the reader should keep quotes of string values.

        Parameters
        ----------
        val : bool
            Boolean value to indicate whether the
            reader should keep quotes of string values

        Returns
        -------
        Self
        """
        self.c_obj.keep_quotes(val)
        return self

    cpdef JsonReaderOptionsBuilder lines(self, bool val):
        """
        Set whether to read the file as a json object per line.

        Parameters
        ----------
        val : bool
            Boolean value to enable/disable the option
            to read each line as a json object

        Returns
        -------
        Self
        """
        self.c_obj.lines(val)
        return self

    cpdef JsonReaderOptionsBuilder mixed_types_as_string(self, bool val):
        """
        Set whether to parse mixed types as a string column.
        Also enables forcing to read a struct as string column using schema.

        Parameters
        ----------
        val : bool
            Boolean value to enable/disable parsing mixed types as a string column

        Returns
        -------
        Self
        """
        self.c_obj.mixed_types_as_string(val)
        return self

    cpdef JsonReaderOptionsBuilder na_values(self, list vals):
        """
        Sets additional values to recognize as null values.

        Parameters
        ----------
        vals : list
            Vector of values to be considered to be null

        Returns
        -------
        Self
        """
        cdef vector[string] vec
        for val in vals:
            if isinstance(val, str):
                vec.push_back(val.encode())
        self.c_obj.na_values(vec)
        return self

    cpdef JsonReaderOptionsBuilder nonnumeric_numbers(self, bool val):
        """
        Set whether unquoted number values should be allowed NaN, +INF, -INF, +Infinity,
        Infinity, and -Infinity. Strict validation must be enabled for this to work.

        Parameters
        ----------
        val : bool
            Boolean value to indicate whether leading zeros are allowed in numeric
            values

        Returns
        -------
        Self
        """
        self.c_obj.nonnumeric_numbers(val)
        return self

    cpdef JsonReaderOptionsBuilder normalize_single_quotes(self, bool val):
        """
        Sets whether to normalize single quotes around strings.

        Parameters
        ----------
        val : bool
            Boolean value to enable/disable the option to normalize single quotes
            around strings

        Returns
        -------
        Self
        """
        self.c_obj.normalize_single_quotes(val)
        return self

    cpdef JsonReaderOptionsBuilder normalize_whitespace(self, bool val):
        """
        Sets whether to normalize unquoted whitespace characters

        Parameters
        ----------
        val : bool
            Boolean value to enable/disable the option to normalize unquoted
            whitespace characters

        Returns
        -------
        Self
        """
        self.c_obj.normalize_whitespace(val)
        return self

    cpdef JsonReaderOptionsBuilder numeric_leading_zeros(self, bool val):
        """
        Set whether leading zeros are allowed in numeric values. Strict validation
        must be enabled for this to work.

        Parameters
        ----------
        val : bool
            Boolean value to indicate whether leading zeros are allowed in numeric
            values

        Returns
        -------
        Self
        """
        self.c_obj.numeric_leading_zeros(val)
        return self

    cpdef JsonReaderOptionsBuilder prune_columns(self, bool val):
        """
        Set whether to prune columns on read, selected based on the @ref dtypes option.
        When set as true, if the reader options include @ref dtypes, then
        the reader will only return those columns which are mentioned in @ref dtypes.
        If false, then all columns are returned, independent of the @ref dtypes setting.

        Parameters
        ----------
        val : bool
            Boolean value to enable/disable column pruning

        Returns
        -------
        Self
        """
        self.c_obj.prune_columns(val)
        return self

    cpdef JsonReaderOptionsBuilder recovery_mode(
        self,
        json_recovery_mode_t recovery_mode
    ):
        """
        Specifies the JSON reader's behavior on invalid JSON lines.

        Parameters
        ----------
        recovery_mode : json_recovery_mode_t
            An enum value to indicate the JSON reader's
            behavior on invalid JSON lines.

        Returns
        -------
        Self
        """
        self.c_obj.recovery_mode(recovery_mode)
        return self

    cpdef JsonReaderOptionsBuilder strict_validation(self, bool val):
        """
        Set whether strict validation is enabled or not

        Parameters
        ----------
        val : bool
            Boolean value to indicate whether strict validation is to be enabled

        Returns
        -------
        Self
        """
        self.c_obj.strict_validation(val)
        return self

    cpdef JsonReaderOptionsBuilder unquoted_control_chars(self, bool val):
        """
        Set whether in a quoted string should characters greater than or equal to 0
        and less than 32 be allowed without some form of escaping. Strict validation
        must be enabled for this to work.

        Parameters
        ----------
        val : bool
            Boolean value to indicate whether unquoted control chars are allowed

        Returns
        -------
        Self
        """
        self.c_obj.unquoted_control_chars(val)
        return self

    cpdef build(self):
        """Create a JsonReaderOptions object"""
        cdef JsonReaderOptions json_options = JsonReaderOptions.__new__(
            JsonReaderOptions
        )
        json_options.c_obj = move(self.c_obj.build())
        json_options.source = self.source
        return json_options


cpdef tuple chunked_read_json(
    JsonReaderOptions options,
    int chunk_size=100_000_000,
    object stream = None,
    DeviceMemoryResource mr = None,
):
    """
    Reads chunks of a JSON file into a :py:class:`~.types.TableWithMetadata`.

    Parameters
    ----------
    options : JsonReaderOptions
        Settings for controlling reading behavior
    chunk_size : int, default 100_000_000 bytes.
        The number of bytes to be read in chunks.
        The chunk_size should be set to at least row_size.
    stream : Stream | None
        CUDA stream used for device memory operations and kernel launches

    Returns
    -------
    tuple
        A tuple of (columns, column_name, child_names)
    """
    cdef size_type c_range_size = (
        chunk_size if chunk_size is not None else 0
    )
    cdef table_with_metadata c_result

    final_columns = []
    meta_names = None
    child_names = None
    i = 0
    cdef Stream s = _get_stream(stream)
    cdef cudaStream_t _cs = s.view().value()
    mr = _get_memory_resource(mr)
    while True:
        options.enable_lines(True)
        options.set_byte_range_offset(c_range_size * i)
        options.set_byte_range_size(c_range_size)

        try:
            with nogil:
                c_result = move(cpp_read_json(options.c_obj, _cs, mr.get_mr()))
        except (ValueError, OverflowError):
            break
        if meta_names is None:
            meta_names = [info.name.decode() for info in c_result.metadata.schema_info]
        if child_names is None:
            child_names = TableWithMetadata._parse_col_names(
                c_result.metadata.schema_info
            )
        new_chunk = [
            col for col in TableWithMetadata.from_libcudf(
                c_result, s, mr).columns
        ]

        if len(final_columns) == 0:
            final_columns = new_chunk
        else:
            for col_idx in range(len(meta_names)):
                final_columns[col_idx] = concatenate(
                    [final_columns[col_idx], new_chunk[col_idx]]
                )
                # Must drop any residual GPU columns to save memory
                new_chunk[col_idx] = None
        i += 1
    return (final_columns, meta_names, child_names)


cpdef TableWithMetadata read_json(
    JsonReaderOptions options,
    object stream = None,
    DeviceMemoryResource mr = None
):
    """
    Read from JSON format.

    The source to read from and options are encapsulated
    by the `options` object.

    For details, see :cpp:func:`read_json`.

    Parameters
    ----------
    options: JsonReaderOptions
        Settings for controlling reading behavior
    stream : Stream | None
        CUDA stream used for device memory operations and kernel launches

    Returns
    -------
    TableWithMetadata
        The Table and its corresponding metadata (column names) that were read in.
    """
    cdef table_with_metadata c_result
    cdef Stream s = _get_stream(stream)
    cdef cudaStream_t _cs = s.view().value()
    mr = _get_memory_resource(mr)
    with nogil:
        c_result = move(cpp_read_json(options.c_obj, _cs, mr.get_mr()))

    return TableWithMetadata.from_libcudf(c_result, s, mr)

cpdef TableWithMetadata read_json_from_string_column(
    Column input,
    Scalar separator,
    Scalar narep,
    list dtypes = None,
    compression_type compression = compression_type.NONE,
    json_recovery_mode_t recovery_mode = json_recovery_mode_t.RECOVER_WITH_NULL,
    object stream = None,
    DeviceMemoryResource mr = None
):
    """
    Joins a column of JSON strings into a device buffer and reads it into
    a table using the JSON reader.

    The source to read from is a string column of JSON records.

    For details, see :cpp:func:`join_strings` and :cpp:func:`read_json`.

    Parameters
    ----------
    input: Column
        String column with json-like strings as rows
    separator: Scalar
        String scalar used to join the input strings
    narep: Scalar
        String scalar used to replace null values during join
    dtypes: List
        Set data types for columns to be read.
    compression: CompressionType
        Set compression type of the string column contents
    recovery_mode: JSONRecoveryMode
        Set recovery option for corrupted JSON input in string column
    stream : Stream | None
        CUDA stream used for device memory operations and kernel launches

    Returns
    -------
    TableWithMetadata
        The Table and its corresponding metadata (column names)
    """
    cdef const string_scalar* c_separator = <const string_scalar*>(
        separator.c_obj.get()
    )
    cdef const string_scalar* c_narep = <const string_scalar*>(
        narep.c_obj.get()
    )
    cdef unique_ptr[column] c_join_string_column
    cdef column_contents c_contents
    cdef table_with_metadata c_result
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    # Join the string column into a single string
    cdef column_view c_input = input.view()
    with nogil:
        c_join_string_column = move(
            cpp_combine.join_strings(
                c_input,
                dereference(c_separator),
                dereference(c_narep),
                _cs,
                mr.get_mr()
            )
        )
        c_contents = c_join_string_column.get().release()

    # Create a new source from the joined string data
    cdef SourceInfo joined_source = SourceInfo(
            [DeviceBuffer.c_from_unique_ptr(move(c_contents.data), _stream, mr)])

    # Create new options using the joined string as source
    cdef JsonReaderOptions options = (
        JsonReaderOptions.builder(joined_source)
        .lines(True)
        .compression(compression)
        .recovery_mode(recovery_mode)
        .build()
    )

    if dtypes is not None and len(dtypes) > 0:
        options.set_dtypes(dtypes)

    # Read JSON from the joined string
    with nogil:
        c_result = move(cpp_read_json(options.c_obj, _cs, mr.get_mr()))

    return TableWithMetadata.from_libcudf(c_result, _stream, mr)

cdef class JsonWriterOptions:
    """
    The settings to use for ``write_json``

    For details, see :cpp:class:`cudf::io::json_writer_options`
    """
    @staticmethod
    def builder(SinkInfo sink, Table table):
        """
        Create a JsonWriterOptionsBuilder object

        Parameters
        ----------
        sink : SinkInfo
            The sink used for writer output
        table : Table
            Table to be written to output

        Returns
        -------
        JsonWriterOptionsBuilder
            Builder to build JsonWriterOptions
        """
        cdef JsonWriterOptionsBuilder json_builder = (
            JsonWriterOptionsBuilder.__new__(JsonWriterOptionsBuilder)
        )
        json_builder.c_obj = json_writer_options.builder(sink.c_obj, table.view())
        json_builder.sink = sink
        json_builder.table = table
        return json_builder

    cpdef void set_rows_per_chunk(self, size_type val):
        """
        Sets string to used for null entries.

        Parameters
        ----------
        val : size_type
            String to represent null value

        Returns
        -------
        None
        """
        self.c_obj.set_rows_per_chunk(val)

    cpdef void set_true_value(self, str val):
        """
        Sets string used for values != 0

        Parameters
        ----------
        val : str
            String to represent values != 0

        Returns
        -------
        None
        """
        self.c_obj.set_true_value(val.encode())

    cpdef void set_false_value(self, str val):
        """
        Sets string used for values == 0

        Parameters
        ----------
        val : str
            String to represent values == 0

        Returns
        -------
        None
        """
        self.c_obj.set_false_value(val.encode())

    cpdef void set_compression(self, compression_type comptype):
        """
        Sets compression type to be used

        Parameters
        ----------
        comptype : CompressionType
            Compression type for sink

        Returns
        -------
        None
        """
        self.c_obj.set_compression(comptype)

cdef class JsonWriterOptionsBuilder:
    cpdef JsonWriterOptionsBuilder metadata(self, TableWithMetadata tbl_w_meta):
        """
        Sets optional metadata (with column names).

        Parameters
        ----------
        tbl_w_meta : TableWithMetadata
            Associated metadata

        Returns
        -------
        Self
        """
        self.c_obj.metadata(tbl_w_meta.metadata)
        return self

    cpdef JsonWriterOptionsBuilder na_rep(self, str val):
        """
        Sets string to used for null entries.

        Parameters
        ----------
        val : str
            String to represent null value

        Returns
        -------
        Self
        """
        self.c_obj.na_rep(val.encode())
        return self

    cpdef JsonWriterOptionsBuilder include_nulls(self, bool val):
        """
        Enables/Disables output of nulls as 'null'.

        Parameters
        ----------
        val : bool
            Boolean value to enable/disable

        Returns
        -------
        Self
        """
        self.c_obj.include_nulls(val)
        return self

    cpdef JsonWriterOptionsBuilder lines(self, bool val):
        """
        Enables/Disables JSON lines for records format.

        Parameters
        ----------
        val : bool
            Boolean value to enable/disable

        Returns
        -------
        Self
        """
        self.c_obj.lines(val)
        return self

    cpdef JsonWriterOptionsBuilder compression(self, compression_type comptype):
        """
        Sets compression type of output sink.

        Parameters
        ----------
        comptype : CompressionType
            Compression type used

        Returns
        -------
        Self
        """
        self.c_obj.compression(comptype)
        return self

    cpdef JsonWriterOptionsBuilder utf8_escaped(self, bool val):
        """
        Sets whether to write UTF-8 characters in string fields
        without escaping them.

        Parameters
        ----------
        val : bool
            If False, disables escaping of UTF-8 characters in output

        Returns
        -------
        Self
        """
        self.c_obj.utf8_escaped(val)
        return self

    cpdef JsonWriterOptions build(self):
        """Create a JsonWriterOptions object"""
        cdef JsonWriterOptions json_options = JsonWriterOptions.__new__(
            JsonWriterOptions
        )
        json_options.c_obj = move(self.c_obj.build())
        json_options.sink = self.sink
        json_options.table = self.table
        return json_options


cpdef void write_json(JsonWriterOptions options, object stream = None):
    """
    Writes a set of columns to JSON format.

    Parameters
    ----------
    options : JsonWriterOptions
        Settings for controlling writing behavior
    stream: Stream
        CUDA stream used for device memory operations and kernel launches

    Returns
    -------
    None
    """
    cdef Stream s = _get_stream(stream)
    cdef cudaStream_t _cs = s.view().value()
    with nogil:
        cpp_write_json(options.c_obj, _cs)

cpdef bool is_supported_write_json(DataType type):
    """Check if the dtype is supported for JSON writing

    For details, see :cpp:func:`is_supported_write_json`.
    """
    return cpp_is_supported_write_json(type.c_obj)

JsonRecoveryModeType.__str__ = JsonRecoveryModeType.__repr__
