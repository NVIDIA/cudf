# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from libcpp.vector cimport vector
from pylibcudf.libcudf cimport concatenate as cpp_concatenate
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.table.table cimport table
from pylibcudf.libcudf.table.table_view cimport table_view

from rmm.pylibrmm.stream cimport Stream
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource

from .column cimport Column
from .table cimport Table
from .utils cimport _get_stream, _get_memory_resource
from cuda.bindings.cyruntime cimport cudaStream_t

__all__ = ["concatenate"]

cpdef concatenate(objects, object stream=None, DeviceMemoryResource mr=None):
    """Concatenate columns or tables.

    Parameters
    ----------
    objects : Sequence[Column] | Sequence[Table]
        The Columns or Tables to concatenate.
    stream : Stream | None
        CUDA stream on which to perform the operation.
    mr : DeviceMemoryResource | None
        Device memory resource used to allocate the returned object's device memory.

    Returns
    -------
    Union[Column, Table]
        The concatenated Column or Table.
    """
    if len(objects) == 0:
        raise ValueError("input list may not be empty")

    cdef vector[column_view] c_columns
    cdef vector[table_view] c_tables
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    cdef unique_ptr[column] c_col_result
    cdef unique_ptr[table] c_tbl_result

    if isinstance(objects[0], Table):
        for tbl in objects:
            c_tables.push_back((<Table?>tbl).view())

        with nogil:
            c_tbl_result = cpp_concatenate.concatenate(
                c_tables, _cs, mr.get_mr()
            )
        return Table.from_libcudf(move(c_tbl_result), _stream, mr)
    elif isinstance(objects[0], Column):
        for column in objects:
            c_columns.push_back((<Column?>column).view())

        with nogil:
            c_col_result = cpp_concatenate.concatenate(
                c_columns, _cs, mr.get_mr()
            )
        return Column.from_libcudf(move(c_col_result), _stream, mr)
    else:
        raise ValueError("input must be a list of Columns or Tables")
