# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.table.table_view cimport table_view
from pylibcudf.libcudf.types cimport (
    nan_policy,
    null_equality,
    null_policy,
    size_type,
)
from cuda.bindings.cyruntime cimport cudaStream_t


cdef extern from "cudf/reduction/distinct_count.hpp" namespace "cudf" nogil:
    cdef size_type distinct_count(
        column_view column,
        null_policy null_handling,
        nan_policy nan_handling,
        cudaStream_t stream) except +libcudf_exception_handler

    cdef size_type distinct_count(
        table_view source_table,
        null_equality nulls_equal,
        cudaStream_t stream) except +libcudf_exception_handler
