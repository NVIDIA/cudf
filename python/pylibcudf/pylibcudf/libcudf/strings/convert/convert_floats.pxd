# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from libcpp.memory cimport unique_ptr
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.types cimport data_type

from cuda.bindings.cyruntime cimport cudaStream_t
from rmm.librmm.memory_resource cimport device_async_resource_ref


cdef extern from "cudf/strings/convert/convert_floats.hpp" namespace \
        "cudf::strings" nogil:
    cdef unique_ptr[column] to_floats(
        column_view strings,
        data_type output_type,
        cudaStream_t stream,
        device_async_resource_ref mr) except +libcudf_exception_handler

    cdef unique_ptr[column] from_floats(
        column_view floats,
        cudaStream_t stream,
        device_async_resource_ref mr) except +libcudf_exception_handler

    cdef unique_ptr[column] is_float(
        column_view input,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler
