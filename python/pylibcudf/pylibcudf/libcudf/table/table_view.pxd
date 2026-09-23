# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from libcpp.vector cimport vector
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column_view cimport (
    column_view,
    mutable_column_view,
)
from pylibcudf.libcudf.types cimport size_type


cdef extern from "cudf/table/table_view.hpp" namespace "cudf" nogil:
    cdef cppclass table_view:
        table_view() except +libcudf_exception_handler
        table_view(const vector[column_view]) except +libcudf_exception_handler
        table_view(
            const vector[column_view], size_type num_rows
        ) except +libcudf_exception_handler
        column_view column(size_type column_index) except +libcudf_exception_handler
        size_type num_columns() except +libcudf_exception_handler
        size_type num_rows() except +libcudf_exception_handler
        table_view select(
            vector[size_type] column_indices
        ) except +libcudf_exception_handler

    cdef cppclass mutable_table_view:
        mutable_table_view() except +libcudf_exception_handler
        mutable_table_view(
            const vector[mutable_column_view]
        ) except +libcudf_exception_handler
        mutable_column_view column(
            size_type column_index
        ) except +libcudf_exception_handler
        size_type num_columns() except +libcudf_exception_handler
        size_type num_rows() except +libcudf_exception_handler
