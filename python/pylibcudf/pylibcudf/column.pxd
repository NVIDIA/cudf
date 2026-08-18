# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector
from libc.stdint cimport uint64_t

from rmm.librmm.device_buffer cimport device_buffer
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport (
    column_view,
    mutable_column_view,
)
from pylibcudf.libcudf.lists.lists_column_view cimport lists_column_view
from pylibcudf.libcudf.types cimport bitmask_type, size_type
from pylibcudf.libcudf.structs.structs_column_view cimport structs_column_view

from .gpumemoryview cimport gpumemoryview
from .types cimport DataType
from .scalar cimport Scalar


cdef class OwnerWithCAI:
    cdef object owner
    cdef dict cai

    @staticmethod
    cdef create(column_view cv, object owner, object stream)


cdef class OwnerMaskWithCAI:
    cdef object owner
    cdef dict cai

    @staticmethod
    cdef create(column_view cv, object owner)


cdef gpumemoryview _copy_array_to_device(object buf, object stream=*)


cdef class Column:
    # TODO: Should we document these attributes? Should we mark them readonly?
    cdef:
        # Core data
        DataType _data_type
        size_type _size
        object _data
        object _mask
        size_type _null_count
        size_type _offset
        # _children: List[Column]
        list _children
        size_type _num_children
        object __weakref__

    cdef column_view view(self)
    cdef mutable_column_view mutable_view(self)

    @staticmethod
    cdef Column from_libcudf(
        unique_ptr[column] libcudf_col,
        object stream,
        DeviceMemoryResource mr
    )

    @staticmethod
    cdef Column from_column_view(const column_view& cv, Column owner)

    @staticmethod
    cdef Column from_column_view_of_arbitrary(
        const column_view& cv,
        object owner,
        object stream,
    )

    @staticmethod
    cdef Column _wrap_nested_list_column(
        object data,
        tuple shape,
        DataType dtype,
        Column base=*,
        object stream=*,
    )

    cpdef list to_pylist(self)
    cpdef Scalar to_scalar(self, object stream=*, DeviceMemoryResource mr=*)
    cpdef DataType type(self)
    cpdef Column child(self, size_type index)
    cpdef size_type num_children(self)
    cpdef size_type size(self)
    cpdef size_type null_count(self)
    cpdef void set_null_count(self, size_type null_count)
    cpdef size_type offset(self)
    cpdef object data(self)
    cpdef object null_mask(self)
    cpdef list children(self)
    cpdef Column copy(self, object stream=*, DeviceMemoryResource mr=*)
    cpdef uint64_t device_buffer_size(self)
    cpdef Column with_mask(self, object, size_type, bint validate=*)

    cpdef ListsColumnView list_view(self)
    cpdef StructsColumnView struct_view(self)


cdef class ListsColumnView:
    cdef Column _column
    cpdef child(self)
    cpdef offsets(self)
    cdef lists_column_view view(self)
    cpdef Column get_sliced_child(self, object stream=*)


cdef class StructsColumnView:
    cdef Column _column
    cdef structs_column_view view(self)
    cpdef Column get_sliced_child(self, int index, object stream=*)
