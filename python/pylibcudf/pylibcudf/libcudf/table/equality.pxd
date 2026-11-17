# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from libcpp cimport bool
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.table.table_view cimport table_view
from pylibcudf.libcudf.types cimport null_equality
from cuda.bindings.cyruntime cimport cudaStream_t


cdef extern from "cudf/table/equality.hpp" namespace "cudf" nogil:
    cdef bool tables_equal(
        const table_view& left,
        const table_view& right,
        null_equality nulls_equal,
        cudaStream_t stream,
    ) except +libcudf_exception_handler
