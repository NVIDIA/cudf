# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from libcpp.vector cimport vector
from pylibcudf.libcudf cimport merge as cpp_merge
from pylibcudf.libcudf.table.table cimport table
from pylibcudf.libcudf.table.table_view cimport table_view
from pylibcudf.libcudf.types cimport null_order, order, size_type

from rmm.pylibrmm.stream cimport Stream
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource

from .table cimport Table
from .utils cimport _get_stream, _get_memory_resource
from cuda.bindings.cyruntime cimport cudaStream_t

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pylibcudf.types import NullOrder, Order
    from pylibcudf.typing import CudaStreamLike

__all__ = ["merge"]

cpdef Table merge (
    list tables_to_merge: list[Table],
    list key_cols: list[int],
    list column_order: list[Order],
    list null_precedence: list[NullOrder],
    object stream: CudaStreamLike | None = None,
    DeviceMemoryResource mr=None
):
    """Merge a set of sorted tables.

    For details see :cpp:func:`merge`.

    Parameters
    ----------
    tables_to_merge : list
        List of tables to merge.
    key_cols : list
        List of column indexes to merge on.
    column_order : List[ColumnOrder]
        Whether each column should be sorted in ascending or descending order.
    null_precedence : List[NullOrder]
        Whether nulls should come before or after non-nulls.
    stream : Stream | None
        CUDA stream on which to perform the operation.
    mr : DeviceMemoryResource | None
        Device memory resource used to allocate the returned table's device memory.

    Returns
    -------
    Table
        The merged table.
    """
    cdef vector[size_type] c_key_cols = key_cols
    cdef vector[order] c_column_order = column_order
    cdef vector[null_order] c_null_precedence = null_precedence
    cdef vector[table_view] c_tables_to_merge

    for tbl in tables_to_merge:
        c_tables_to_merge.push_back((<Table?> tbl).view())

    cdef unique_ptr[table] c_result
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().get()
    mr = _get_memory_resource(mr)

    with nogil:
        c_result = cpp_merge.merge(
            c_tables_to_merge,
            c_key_cols,
            c_column_order,
            c_null_precedence,
            _cs,
            mr.get_mr()
        )
    return Table.from_libcudf(move(c_result), _stream, mr)
