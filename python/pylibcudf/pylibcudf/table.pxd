# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libcpp.memory cimport unique_ptr
from pylibcudf.libcudf.table.table cimport table
from pylibcudf.libcudf.table.table_view cimport table_view
from pylibcudf.libcudf.types cimport size_type
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource

cdef class Table:
    # Tuple[pylibcudf.Column]
    cdef tuple _columns
    cdef size_type _num_rows

    cdef table_view view(self)

    cpdef int num_columns(self)
    cpdef int num_rows(self)
    cpdef tuple shape(self)

    @staticmethod
    cdef Table from_libcudf(
        unique_ptr[table] libcudf_tbl,
        object stream,
        DeviceMemoryResource mr
    )

    @staticmethod
    cdef Table from_table_view(const table_view& tv, Table owner)

    @staticmethod
    cdef Table from_table_view_of_arbitrary(
        const table_view& tv,
        object owner,
        object stream,
    )

    cpdef tuple columns(self)
    cpdef list release(self)
    cpdef Table copy(self, object stream = *, DeviceMemoryResource mr=*)
