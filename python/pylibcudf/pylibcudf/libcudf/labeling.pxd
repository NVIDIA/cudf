# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from libcpp cimport int
from libcpp.memory cimport unique_ptr
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view

from cuda.bindings.cyruntime cimport cudaStream_t
from rmm.librmm.memory_resource cimport device_async_resource_ref


cdef extern from "cudf/labeling/label_bins.hpp" namespace "cudf" nogil:
    cpdef enum class inclusive(int):
        YES
        NO

    cdef unique_ptr[column] label_bins (
        const column_view &input,
        const column_view &left_edges,
        inclusive left_inclusive,
        const column_view &right_edges,
        inclusive right_inclusive,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler
