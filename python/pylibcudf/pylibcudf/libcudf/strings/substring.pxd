# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from libcpp.memory cimport unique_ptr
from libcpp.optional cimport optional
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.types cimport size_type

from cuda.bindings.cyruntime cimport cudaStream_t
from rmm.librmm.memory_resource cimport device_async_resource_ref


cdef extern from "cudf/strings/slice.hpp" namespace "cudf::strings" nogil:
    cdef unique_ptr[column] slice_strings(
        column_view source_strings,
        optional[size_type] start,
        optional[size_type] stop,
        optional[size_type] step,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] slice_strings(
        column_view source_strings,
        column_view starts,
        column_view stops,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler
