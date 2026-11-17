# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from libcpp.memory cimport unique_ptr
from libcpp.pair cimport pair
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.table.table_view cimport table_view

from cuda.bindings.cyruntime cimport cudaStream_t
from rmm.librmm.memory_resource cimport device_async_resource_ref


cdef extern from "cudf/transpose.hpp" namespace "cudf" nogil:
    cdef pair[
        unique_ptr[column],
        table_view
    ] transpose(
        table_view input_table,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler
