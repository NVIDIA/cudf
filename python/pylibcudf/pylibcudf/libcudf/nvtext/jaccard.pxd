# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from libcpp.memory cimport unique_ptr
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.types cimport size_type
from cuda.bindings.cyruntime cimport cudaStream_t
from rmm.librmm.memory_resource cimport device_async_resource_ref


cdef extern from "nvtext/jaccard.hpp" namespace "nvtext" nogil:

    cdef unique_ptr[column] jaccard_index(
        const column_view &input1,
        const column_view &input2,
        size_type width,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler
