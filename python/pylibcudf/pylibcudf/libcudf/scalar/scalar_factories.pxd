# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.types cimport data_type
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.scalar.scalar cimport scalar
from pylibcudf.libcudf.fixed_point.fixed_point cimport scale_type
from pylibcudf.libcudf.types cimport int128 as int128_t

from cuda.bindings.cyruntime cimport cudaStream_t
from rmm.librmm.memory_resource cimport device_async_resource_ref


cdef extern from "cudf/scalar/scalar_factories.hpp" namespace "cudf" nogil:
    cdef unique_ptr[scalar] make_string_scalar(
        const string & _string,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler
    cdef unique_ptr[scalar] make_fixed_width_scalar[T](
        T value,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler
    cdef unique_ptr[scalar] make_fixed_point_scalar[T](
        int128_t value,
        scale_type scale,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler
    cdef unique_ptr[scalar] make_numeric_scalar(
        data_type type_,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler
    cdef unique_ptr[scalar] make_timestamp_scalar(
        data_type type_,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler
    cdef unique_ptr[scalar] make_empty_scalar_like(
        const column_view &,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler
    cdef unique_ptr[scalar] make_duration_scalar(
        data_type type_,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler
    cdef unique_ptr[scalar] make_default_constructed_scalar(
        data_type type_,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler
