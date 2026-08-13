# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from cython.operator cimport dereference
from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.column cimport Column
from pylibcudf.libcudf cimport json as cpp_json
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.scalar.scalar cimport string_scalar
from pylibcudf.scalar cimport Scalar

from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream

from .utils cimport _get_stream, _get_memory_resource
from cuda.bindings.cyruntime cimport cudaStream_t

__all__ = ["GetJsonObjectOptions", "get_json_object"]

cdef class GetJsonObjectOptions:
    """Settings for `get_json_object()`"""
    def __init__(
        self,
        *,
        allow_single_quotes=False,
        strip_quotes_from_single_strings=True,
        missing_fields_as_nulls=False
    ):
        self.set_allow_single_quotes(allow_single_quotes)
        self.set_strip_quotes_from_single_strings(
            strip_quotes_from_single_strings
        )
        self.set_missing_fields_as_nulls(missing_fields_as_nulls)

    __hash__ = None

    def get_allow_single_quotes(self):
        """
        Returns true/false depending on whether single-quotes for representing strings
        are allowed.

        Returns
        -------
        bool
            true if single-quotes are allowed, false otherwise.
        """
        return self.options.get_allow_single_quotes()

    def get_strip_quotes_from_single_strings(self):
        """
        Returns true/false depending on whether individually returned string values have
        their quotes stripped.

        Returns
        -------
        bool
            true if individually returned string values have their quotes stripped.
        """
        return self.options.get_strip_quotes_from_single_strings()

    def get_missing_fields_as_nulls(self):
        """
        Whether a field not contained by an object is to be interpreted as null.

        Returns
        -------
        bool
            true if missing fields are interpreted as null.
        """
        return self.options.get_missing_fields_as_nulls()

    def set_allow_single_quotes(self, bool val):
        """
        Set whether single-quotes for strings are allowed.

        Parameters
        ----------
        val : bool
            Whether to allow single quotes

        Returns
        -------
        None
        """
        self.options.set_allow_single_quotes(val)

    def set_strip_quotes_from_single_strings(self, bool val):
        """
        Set whether individually returned string values have their quotes stripped.

        Parameters
        ----------
        val : bool
            Whether to strip quotes from single strings.

        Returns
        -------
        None
        """
        self.options.set_strip_quotes_from_single_strings(val)

    def set_missing_fields_as_nulls(self, bool val):
        """
        Set whether missing fields are interpreted as null.

        Parameters
        ----------
        val : bool
            Whether to treat missing fields as nulls.

        Returns
        -------
        None
        """
        self.options.set_missing_fields_as_nulls(val)


cpdef Column get_json_object(
    Column col,
    Scalar json_path,
    GetJsonObjectOptions options=None,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """
    Apply a JSONPath string to all rows in an input strings column.

    For details, see :cpp:func:`get_json_object`

    Parameters
    ----------
    col : Column
        The input strings column. Each row must contain a valid json string.

    json_path : Scalar
        The JSONPath string to be applied to each row.

    options : GetJsonObjectOptions
        Options for controlling the behavior of the function.

    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Column
        New strings column containing the retrieved json object strings.
    """
    cdef unique_ptr[column] c_result
    cdef column_view c_col

    cdef string_scalar* c_json_path = <string_scalar*>(
        json_path.c_obj.get()
    )
    if options is None:
        options = GetJsonObjectOptions()

    cdef cpp_json.get_json_object_options c_options = options.options
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    c_col = col.view()
    with nogil:
        c_result = cpp_json.get_json_object(
            c_col,
            dereference(c_json_path),
            c_options,
            _cs,
            mr.get_mr()
        )

    return Column.from_libcudf(move(c_result), _stream, mr)
