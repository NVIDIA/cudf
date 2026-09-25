# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport int32_t, int64_t, uint8_t
from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from libcpp.vector cimport vector

from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream
from cuda.bindings.cyruntime cimport cudaStream_t

from pylibcudf.column cimport Column
from pylibcudf.utils cimport _get_memory_resource, _get_stream

from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.io.protobuf cimport (
    decode_protobuf as cpp_decode_protobuf,
    decode_protobuf_options,
    nested_field_descriptor,
    proto_encoding,
    proto_wire_type,
)
from pylibcudf.libcudf.types cimport type_id

__all__ = [
    "decode_protobuf",
]


cdef vector[uint8_t] _bytes_to_vector(object value):
    cdef vector[uint8_t] out
    cdef object item

    if value is None:
        return out

    out.reserve(len(value))
    for item in value:
        out.push_back(<uint8_t>item)
    return out


cdef vector[int32_t] _ints_to_vector(object values):
    cdef vector[int32_t] out
    cdef object value

    if values is None:
        return out

    out.reserve(len(values))
    for value in values:
        out.push_back(<int32_t>value)
    return out


cdef vector[vector[uint8_t]] _bytes_list_to_vectors(object values):
    cdef vector[vector[uint8_t]] out
    cdef vector[uint8_t] value
    cdef object item

    if values is None:
        return out

    out.reserve(len(values))
    for item in values:
        value = _bytes_to_vector(item)
        out.push_back(move(value))
    return out


cpdef Column decode_protobuf(
    Column binary_input,
    list schema,
    list default_ints,
    list default_floats,
    list default_bools,
    list default_strings,
    list enum_valid_values,
    list enum_names,
    bint fail_on_errors,
    object stream = None,
    DeviceMemoryResource mr = None,
):
    """
    Decode serialized protobuf messages from a LIST<INT8/UINT8> column
    into a STRUCT column.

    Parameters
    ----------
    binary_input : Column
        LIST<INT8/UINT8> column of serialized protobuf messages.
    schema : list of tuples
        Each tuple is (field_number, parent_idx, depth, wire_type, output_type_id,
        encoding, is_repeated, is_required, has_default_value).
    default_ints : list of int
        Default integer values per field.
    default_floats : list of float
        Default float values per field.
    default_bools : list of bool
        Default boolean values per field.
    default_strings : list of bytes
        Default string values per field (as raw bytes).
    enum_valid_values : list of list of int
        Valid enum numbers per field.
    enum_names : list of list of bytes
        UTF-8 enum names per field.
    fail_on_errors : bool
        If True, raise on malformed messages. If False, return nulls.
    stream : Stream, optional
        CUDA stream for device operations.
    mr : DeviceMemoryResource, optional
        Device memory resource.

    Returns
    -------
    Column
        A STRUCT column containing decoded protobuf fields.
    """
    cdef decode_protobuf_options options
    cdef int n = len(schema)

    # Build schema vector
    cdef vector[nested_field_descriptor] c_schema
    c_schema.reserve(n)
    cdef nested_field_descriptor desc
    for field in schema:
        desc.field_number = <int>field[0]
        desc.parent_idx = <int>field[1]
        desc.depth = <int>field[2]
        desc.wire_type = <proto_wire_type>(<int>field[3])
        desc.output_type = <type_id>(<int>field[4])
        desc.encoding = <proto_encoding>(<int>field[5])
        desc.is_repeated = <bool>field[6]
        desc.is_required = <bool>field[7]
        desc.has_default_value = <bool>field[8]
        c_schema.push_back(desc)

    options.schema = move(c_schema)

    # Default values
    cdef vector[int64_t] c_default_ints
    c_default_ints.reserve(n)
    for v in default_ints:
        c_default_ints.push_back(<int64_t>v)
    options.default_ints = move(c_default_ints)

    cdef vector[double] c_default_floats
    c_default_floats.reserve(n)
    for v in default_floats:
        c_default_floats.push_back(<double>v)
    options.default_floats = move(c_default_floats)

    cdef vector[uint8_t] c_default_bools
    c_default_bools.reserve(n)
    for v in default_bools:
        if v:
            c_default_bools.push_back(<uint8_t>1)
        else:
            c_default_bools.push_back(<uint8_t>0)
    options.default_bools = move(c_default_bools)

    cdef vector[vector[uint8_t]] c_default_strings
    cdef vector[uint8_t] c_bytes
    c_default_strings.reserve(n)
    for v in default_strings:
        c_bytes = _bytes_to_vector(v)
        c_default_strings.push_back(move(c_bytes))
    options.default_strings = move(c_default_strings)

    cdef vector[vector[int32_t]] c_enum_valid_values
    cdef vector[int32_t] c_ints
    c_enum_valid_values.reserve(n)
    for v in enum_valid_values:
        c_ints = _ints_to_vector(v)
        c_enum_valid_values.push_back(move(c_ints))
    options.enum_valid_values = move(c_enum_valid_values)

    cdef vector[vector[vector[uint8_t]]] c_enum_names
    cdef vector[vector[uint8_t]] c_names
    c_enum_names.reserve(n)
    for v in enum_names:
        c_names = _bytes_list_to_vectors(v)
        c_enum_names.push_back(move(c_names))
    options.enum_names = move(c_enum_names)

    options.fail_on_errors = fail_on_errors

    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    cdef unique_ptr[column] c_result
    with nogil:
        c_result = move(
            cpp_decode_protobuf(
                binary_input.view(),
                options,
                _cs,
                mr.get_mr(),
            )
        )

    return Column.from_libcudf(move(c_result), _stream, mr)
