# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from libcpp cimport int
from libcpp.memory cimport unique_ptr
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.scalar.scalar cimport string_scalar
from pylibcudf.libcudf.table.table_view cimport table_view
from cuda.bindings.cyruntime cimport cudaStream_t
from rmm.librmm.memory_resource cimport device_async_resource_ref


cdef extern from "cudf/strings/combine.hpp" namespace "cudf::strings" nogil:

    cpdef enum class separator_on_nulls(int):
        YES
        NO

    cpdef enum class output_if_empty_list(int):
        EMPTY_STRING
        NULL_ELEMENT

    cdef unique_ptr[column] concatenate(
        table_view strings_columns,
        string_scalar separator,
        string_scalar narep,
        separator_on_nulls separate_nulls,
        cudaStream_t stream,
        device_async_resource_ref mr) except +libcudf_exception_handler

    cdef unique_ptr[column] concatenate(
        table_view strings_columns,
        column_view separators,
        string_scalar separator_narep,
        string_scalar col_narep,
        separator_on_nulls separate_nulls,
        cudaStream_t stream,
        device_async_resource_ref mr) except +libcudf_exception_handler

    cdef unique_ptr[column] join_strings(
        column_view input,
        string_scalar separator,
        string_scalar narep,
        cudaStream_t stream,
        device_async_resource_ref mr) except +libcudf_exception_handler

    cdef unique_ptr[column] join_list_elements(
        column_view lists_strings_column,
        column_view separators,
        string_scalar separator_narep,
        string_scalar string_narep,
        separator_on_nulls separate_nulls,
        output_if_empty_list empty_list_policy,
        cudaStream_t stream,
        device_async_resource_ref mr) except +libcudf_exception_handler

    cdef unique_ptr[column] join_list_elements(
        column_view lists_strings_column,
        string_scalar separator,
        string_scalar narep,
        separator_on_nulls separate_nulls,
        output_if_empty_list empty_list_policy,
        cudaStream_t stream,
        device_async_resource_ref mr) except +libcudf_exception_handler
