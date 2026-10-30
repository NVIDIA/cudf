# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport int32_t, int64_t, uint8_t, uint32_t
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
from pylibcudf.libcudf.column.column_view cimport column_view
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
    cdef object item
    if values is None:
        return out
    out.reserve(len(values))
    for item in values:
        out.push_back(_bytes_to_vector(item))
    return out


cdef vector[bool] _bools_to_vector(object values):
    cdef vector[bool] out
    cdef object value
    if values is None:
        return out
    out.reserve(len(values))
    for value in values:
        out.push_back(<bool>value)
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
    list output_fields = None,
    object stream = None,
    DeviceMemoryResource mr = None,
):
    """
    Decode serialized protobuf messages from a LIST<INT8/UINT8> column
    into a STRUCT column.

    For details, see :cpp:func:`decode_protobuf`.

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
        Sorted valid enum numbers per field.
    enum_names : list of list of bytes
        UTF-8 enum names per field.
    fail_on_errors : bool
        If True, raise on malformed messages. If False, return nulls.
    output_fields : list of bool, optional
        Per-field flags selecting which fields appear in the result. Hidden
        fields are still validated. A nested field must use the same flag as
        its parent. ``None`` returns every field.
    stream : Stream, optional
        CUDA stream for device operations.
    mr : DeviceMemoryResource, optional
        Device memory resource.

    Returns
    -------
    Column
        A STRUCT column containing the decoded output fields.
    """
    cdef decode_protobuf_options options
    cdef nested_field_descriptor desc
    options.schema.reserve(len(schema))
    for field in schema:
        desc.field_number = <int>field[0]
        desc.parent_idx = <int>field[1]
        desc.depth = <int>field[2]
        desc.wire_type = <proto_wire_type>(<uint32_t>field[3])
        desc.output_type = <type_id>(<int>field[4])
        desc.encoding = <proto_encoding>(<int>field[5])
        desc.is_repeated = <bool>field[6]
        desc.is_required = <bool>field[7]
        desc.has_default_value = <bool>field[8]
        options.schema.push_back(desc)

    for v in default_ints:
        options.default_ints.push_back(<int64_t>v)
    for v in default_floats:
        options.default_floats.push_back(<double>v)
    options.default_bools = _bools_to_vector(default_bools)
    options.default_strings = _bytes_list_to_vectors(default_strings)
    for v in enum_valid_values:
        options.enum_valid_values.push_back(_ints_to_vector(v))
    for v in enum_names:
        options.enum_names.push_back(_bytes_list_to_vectors(v))
    options.fail_on_errors = fail_on_errors
    options.output_fields = _bools_to_vector(output_fields)

    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().get()
    mr = _get_memory_resource(mr)

    cdef column_view c_input = binary_input.view()
    cdef unique_ptr[column] c_result
    with nogil:
        c_result = cpp_decode_protobuf(
            c_input,
            options,
            _cs,
            mr.get_mr(),
        )

    return Column.from_libcudf(move(c_result), _stream, mr)
