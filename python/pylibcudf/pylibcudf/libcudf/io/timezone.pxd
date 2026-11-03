# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.optional cimport optional
from libcpp.string cimport string
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.table.table cimport table
from cuda.bindings.cyruntime cimport cudaStream_t
from rmm.librmm.memory_resource cimport device_async_resource_ref


cdef extern from "cudf/timezone.hpp" namespace "cudf" nogil:
    unique_ptr[table] make_timezone_transition_table(
        optional[string] tzif_dir,
        string timezone_name,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler
