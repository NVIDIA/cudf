# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libcpp cimport bool
from pylibcudf.libcudf.table cimport equality as cpp_table_equality
from pylibcudf.libcudf.table.table_view cimport table_view
from pylibcudf.libcudf.types cimport null_equality

from rmm.pylibrmm.stream cimport Stream

from .table cimport Table
from .utils cimport _get_stream
from cuda.bindings.cyruntime cimport cudaStream_t

__all__ = ["tables_equal"]


cpdef bool tables_equal(
    Table left,
    Table right,
    null_equality nulls_equal=null_equality.EQUAL,
    object stream=None,
):
    """Check if two tables are equal.

    Returns true if the input tables have the same number of rows, the same
    number of columns, matching column types, and every row in ``left``
    compares equal to the row at the same index in ``right``. Null equality
    is controlled by ``nulls_equal``. Floating point NaN values compare equal.

    For details, see :cpp:func:`tables_equal`.

    Parameters
    ----------
    left : Table
        The first table to compare.
    right : Table
        The second table to compare.
    nulls_equal : NullEquality, default NullEquality.EQUAL
        Flag to denote if null elements should be considered equal.
    stream : Stream, default None
        CUDA stream on which to perform the operation.

    Returns
    -------
    bool
        True if the tables are equal, False otherwise.
    """
    cdef bool c_result
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    cdef table_view c_left = left.view()
    cdef table_view c_right = right.view()

    with nogil:
        c_result = cpp_table_equality.tables_equal(
            c_left, c_right, nulls_equal, _cs
        )
    return c_result
