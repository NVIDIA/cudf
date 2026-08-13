# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from cython.operator import dereference

from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.libcudf cimport replace as cpp_replace
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view, mutable_column_view
from rmm.pylibrmm.stream cimport Stream
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource

from pylibcudf.libcudf.replace import \
    replace_policy as ReplacePolicy  # no-cython-lint

from .column cimport Column
from .scalar cimport Scalar
from .utils cimport _get_stream, _get_memory_resource
from cuda.bindings.cyruntime cimport cudaStream_t

__all__ = [
    "ReplacePolicy",
    "clamp",
    "find_and_replace_all",
    "normalize_nans_and_zeros",
    "replace_nulls",
]


cpdef Column replace_nulls(
    Column source_column,
    ReplacementType replacement,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """Replace nulls in source_column.

    The values used to replace nulls depends on the type of replacement:
        - If replacement is a Column, the corresponding value from replacement
          is used.
        - If replacement is a Scalar, the same value is used for all nulls.
        - If replacement is a replace_policy, the policy is used to determine
          the replacement value:

            - PRECEDING: The first non-null value that precedes the null is used.
            - FOLLOWING: The first non-null value that follows the null is used.

    For more details, see :cpp:func:`replace_nulls`.

    Parameters
    ----------
    source_column : Column
        The column in which to replace nulls.
    replacement_column : Union[Column, Scalar, replace_policy]
        If a Column, the values to use as replacements. If a Scalar, the value
        to use as a replacement. If a replace_policy, the policy to use to
        determine the replacement value.
    stream : Stream | None
        CUDA stream on which to perform the operation.
    mr : DeviceMemoryResource | None
        Device memory resource used to allocate the returned column's device memory.

    Returns
    -------
    Column
        A copy of source_column with nulls replaced by values from
        replacement_column.
    """
    cdef unique_ptr[column] c_result
    cdef replace_policy policy
    cdef column_view c_replacement

    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    cdef column_view c_source_column = source_column.view()
    # Due to https://github.com/cython/cython/issues/5984, if this function is
    # called as a Python function (i.e. without typed inputs, which is always
    # true in pure Python files), the type of `replacement` will be `object`
    # instead of `replace_policy`. This is a workaround to handle that case.
    if ReplacementType is object:
        if isinstance(replacement, ReplacePolicy):
            policy = replacement
            with nogil:
                c_result = cpp_replace.replace_nulls(
                    c_source_column,
                    policy,
                    _cs,
                    mr.get_mr()
                )
            return Column.from_libcudf(move(c_result), _stream, mr)
        else:
            raise TypeError("replacement must be a Column, Scalar, or replace_policy")

    if ReplacementType is Column:
        c_replacement = replacement.view()

    with nogil:
        if ReplacementType is Column:
            c_result = cpp_replace.replace_nulls(
                c_source_column,
                c_replacement,
                _cs,
                mr.get_mr()
            )
        elif ReplacementType is Scalar:
            c_result = cpp_replace.replace_nulls(
                c_source_column,
                dereference(replacement.c_obj),
                _cs,
                mr.get_mr()
            )
        elif ReplacementType is replace_policy:
            c_result = cpp_replace.replace_nulls(
                c_source_column,
                replacement,
                _cs,
                mr.get_mr()
            )
        else:
            assert False, "Internal error. Please contact pylibcudf developers"
    return Column.from_libcudf(move(c_result), _stream, mr)


cpdef Column find_and_replace_all(
    Column source_column,
    Column values_to_replace,
    Column replacement_values,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """Replace all occurrences of values_to_replace with replacement_values.

    For details, see :cpp:func:`find_and_replace_all`.

    Parameters
    ----------
    source_column : Column
        The column in which to replace values.
    values_to_replace : Column
        The column containing values to replace.
    replacement_values : Column
        The column containing replacement values.
    stream : Stream | None
        CUDA stream on which to perform the operation.
    mr : DeviceMemoryResource | None
        Device memory resource used to allocate the returned column's device memory.

    Returns
    -------
    Column
        A copy of source_column with all occurrences of values_to_replace
        replaced by replacement_values.
    """
    cdef unique_ptr[column] c_result

    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    cdef column_view c_source_column = source_column.view()
    cdef column_view c_values_to_replace = values_to_replace.view()
    cdef column_view c_replacement_values = replacement_values.view()
    with nogil:
        c_result = cpp_replace.find_and_replace_all(
            c_source_column,
            c_values_to_replace,
            c_replacement_values,
            _cs,
            mr.get_mr()
        )
    return Column.from_libcudf(move(c_result), _stream, mr)


cpdef Column clamp(
    Column source_column,
    Scalar lo,
    Scalar hi,
    Scalar lo_replace=None,
    Scalar hi_replace=None,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """Clamp the values in source_column to the range [lo, hi].

    For details, see :cpp:func:`clamp`.

    Parameters
    ----------
    source_column : Column
        The column to clamp.
    lo : Scalar
        The lower bound of the clamp range.
    hi : Scalar
        The upper bound of the clamp range.
    lo_replace : Scalar, optional
        The value to use for elements that are less than lo. If not specified,
        the value of lo is used.
    hi_replace : Scalar, optional
        The value to use for elements that are greater than hi. If not
        specified, the value of hi is used.
    stream : Stream | None
        CUDA stream on which to perform the operation.
    mr : DeviceMemoryResource | None
        Device memory resource used to allocate the returned column's device memory.

    Returns
    -------
    Column
        A copy of source_column with values clamped to the range [lo, hi].
    """
    if (lo_replace is None) != (hi_replace is None):
        raise ValueError("lo_replace and hi_replace must be specified together")

    cdef unique_ptr[column] c_result

    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    cdef column_view c_source_column = source_column.view()
    with nogil:
        if lo_replace is None:
            c_result = cpp_replace.clamp(
                c_source_column,
                dereference(lo.c_obj),
                dereference(hi.c_obj),
                _cs,
                mr.get_mr()
            )
        else:
            c_result = cpp_replace.clamp(
                c_source_column,
                dereference(lo.c_obj),
                dereference(lo_replace.c_obj),
                dereference(hi.c_obj),
                dereference(hi_replace.c_obj),
                _cs,
                mr.get_mr()
            )
    return Column.from_libcudf(move(c_result), _stream, mr)


cpdef Column normalize_nans_and_zeros(
    Column source_column,
    bool inplace=False,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """Normalize NaNs and zeros in source_column.

    For details, see :cpp:func:`normalize_nans_and_zeros`.

    Parameters
    ----------
    source_column : Column
        The column to normalize.
    inplace : bool, optional
        If True, normalize source_column in place. If False, return a new
        column with the normalized values.
    stream : Stream | None
        CUDA stream on which to perform the operation.
    mr : DeviceMemoryResource | None
        Device memory resource used to allocate the returned column's device memory.

    Returns
    -------
    Column
        A copy of source_column with NaNs and zeros normalized.
    """
    cdef unique_ptr[column] c_result

    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    cdef column_view c_source_column = source_column.view()
    cdef mutable_column_view c_mutable_source_column
    if inplace:
        c_mutable_source_column = source_column.mutable_view()
    with nogil:
        if inplace:
            cpp_replace.normalize_nans_and_zeros(
                c_mutable_source_column,
                _cs,
                mr.get_mr()
            )
        else:
            c_result = cpp_replace.normalize_nans_and_zeros(
                c_source_column,
                _cs,
                mr.get_mr()
            )

    if not inplace:
        return Column.from_libcudf(move(c_result), _stream, mr)

ReplacePolicy.__str__ = ReplacePolicy.__repr__
