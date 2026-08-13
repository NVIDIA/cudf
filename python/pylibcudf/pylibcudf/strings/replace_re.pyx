# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from cython.operator cimport dereference
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.utility cimport move
from pylibcudf.column cimport Column
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.scalar.scalar cimport string_scalar
from pylibcudf.libcudf.scalar.scalar_factories cimport (
    make_string_scalar as cpp_make_string_scalar,
)
from pylibcudf.libcudf.strings cimport replace_re as cpp_replace_re
from pylibcudf.libcudf.types cimport size_type
from pylibcudf.scalar cimport Scalar
from pylibcudf.strings.regex_program cimport RegexProgram
from pylibcudf.utils cimport _get_stream, _get_memory_resource
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream
from cuda.bindings.cyruntime cimport cudaStream_t

__all__ = ["replace_re", "replace_with_backrefs"]

cpdef Column replace_re(
    Column input,
    RegexProgram pattern,
    Scalar replacement=None,
    size_type max_replace_count=-1,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """
    For each string, replaces any character sequence matching the given patterns
    with the provided replacement.

    For details, see :cpp:func:`replace_re`

    Parameters
    ----------
    input : Column
        Strings instance for this operation.
    pattern: RegexProgram
        The regex to match to each string and replace.
    replacement : Scalar
        The string used to replace the matched sequence in each string.
    max_replace_count : int
        The maximum number of times to replace the matched pattern
        within each string.
        Default replaces every substring that is matched.

    Returns
    -------
    Column
        New strings column
    """
    cdef unique_ptr[column] c_result
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)
    cdef column_view c_input

    if replacement is None:
        replacement = Scalar.from_libcudf(
            cpp_make_string_scalar("".encode(), _stream.view().value(), mr.get_mr())
        )
    c_input = input.view()
    with nogil:
        c_result = move(
            cpp_replace_re.replace_re(
                c_input,
                pattern.c_obj.get()[0],
                dereference(<string_scalar*>(replacement.get())),
                max_replace_count,
                _cs,
                mr.get_mr()
            )
        )
    return Column.from_libcudf(move(c_result), _stream, mr)


cpdef Column replace_with_backrefs(
    Column input,
    RegexProgram prog,
    str replacement,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """
    For each string, replaces any character sequence matching the given regex
    using the replacement template for back-references.

    For details, see :cpp:func:`replace_with_backrefs`

    Parameters
    ----------
    input : Column
        Strings instance for this operation.

    prog: RegexProgram
        Regex program instance.

    replacement : str
         The replacement template for creating the output string.

    Returns
    -------
    Column
        New strings column.
    """
    cdef unique_ptr[column] c_result
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)
    cdef string c_replacement = replacement.encode()

    cdef column_view c_input = input.view()
    with nogil:
        c_result = cpp_replace_re.replace_with_backrefs(
            c_input,
            prog.c_obj.get()[0],
            c_replacement,
            _cs,
            mr.get_mr()
        )

    return Column.from_libcudf(move(c_result), _stream, mr)
