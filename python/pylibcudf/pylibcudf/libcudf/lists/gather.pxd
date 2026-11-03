# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from libcpp.memory cimport unique_ptr
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.copying cimport out_of_bounds_policy
from pylibcudf.libcudf.lists.lists_column_view cimport lists_column_view
from cuda.bindings.cyruntime cimport cudaStream_t
from rmm.librmm.memory_resource cimport device_async_resource_ref

cdef extern from "cudf/lists/gather.hpp" namespace "cudf::lists" nogil:
    cdef unique_ptr[column] segmented_gather(
        const lists_column_view& source_column,
        const lists_column_view& gather_map_list,
        out_of_bounds_policy bounds_policy,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler
