# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view

from cuda.bindings.cyruntime cimport cudaStream_t
from rmm.librmm.memory_resource cimport device_async_resource_ref


cdef extern from "nvtext/edit_distance.hpp" namespace "nvtext" nogil:

    cdef unique_ptr[column] edit_distance(
        const column_view & strings,
        const column_view & targets,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler
