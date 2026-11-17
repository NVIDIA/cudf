# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from libcpp cimport bool
from libcpp.vector cimport vector
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.types cimport bitmask_type, data_type, size_type


cdef extern from "cudf/column/column_view.hpp" namespace "cudf" nogil:
    cdef cppclass column_view_base:
        const T* data[T]() except +libcudf_exception_handler
        const T* head[T]() except +libcudf_exception_handler
        const bitmask_type* null_mask() except +libcudf_exception_handler
        size_type size() except +libcudf_exception_handler
        data_type type() except +libcudf_exception_handler
        bool nullable() except +libcudf_exception_handler
        size_type null_count() except +libcudf_exception_handler
        bool has_nulls() except +libcudf_exception_handler
        size_type offset() except +libcudf_exception_handler
        size_type num_children() except +libcudf_exception_handler

    cdef cppclass column_view(column_view_base):
        column_view() except +libcudf_exception_handler
        column_view(const column_view& other) except +libcudf_exception_handler

        column_view& operator=(const column_view&) except +libcudf_exception_handler

        column_view(
            data_type type,
            size_type size,
            const void* data,
            const bitmask_type* null_mask,
            size_type null_count,
            size_type offset,
            vector[column_view] children
        ) except +libcudf_exception_handler

        column_view child(size_type) except +libcudf_exception_handler

    cdef cppclass mutable_column_view(column_view_base):
        mutable_column_view() except +libcudf_exception_handler
        mutable_column_view(
            const mutable_column_view&
        ) except +libcudf_exception_handler

        mutable_column_view& operator=(
            const mutable_column_view&
        ) except +libcudf_exception_handler

        mutable_column_view(
            data_type type, size_type size, const void* data,
            const bitmask_type* null_mask, size_type null_count,
            size_type offset, vector[mutable_column_view] children
        ) except +libcudf_exception_handler

        T* data[T]() except +libcudf_exception_handler
        T* head[T]() except +libcudf_exception_handler
        bitmask_type* null_mask() except +libcudf_exception_handler
        mutable_column_view& child(size_type) except +libcudf_exception_handler

    cdef column_view bit_cast(
        const column_view& input,
        data_type type
    ) except +libcudf_exception_handler
