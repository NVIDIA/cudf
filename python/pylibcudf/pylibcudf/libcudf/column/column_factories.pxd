# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from libcpp.memory cimport unique_ptr
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.scalar.scalar cimport scalar
from pylibcudf.libcudf.types cimport (
    bitmask_type,
    data_type,
    mask_state,
    size_type,
    type_id,
)

from pylibcudf.libcudf.utilities.device_buffer cimport byte, device_buffer
from cuda.bindings.cyruntime cimport cudaStream_t
from rmm.librmm.memory_resource cimport device_async_resource_ref


cdef extern from "cudf/column/column_factories.hpp" namespace "cudf" nogil:
    cdef unique_ptr[column] make_numeric_column(
        data_type type,
        size_type size,
        mask_state state,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] make_numeric_column(
        data_type type,
        size_type size,
        device_buffer[byte] mask,
        size_type null_count,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] make_fixed_point_column(
        data_type type,
        size_type size,
        mask_state state,
        cudaStream_t stream,
        device_async_resource_ref mr) except +libcudf_exception_handler

    cdef unique_ptr[column] make_fixed_point_column(
        data_type type,
        size_type size,
        device_buffer[byte] mask,
        size_type null_count,
        cudaStream_t stream,
        device_async_resource_ref mr) except +libcudf_exception_handler

    cdef unique_ptr[column] make_timestamp_column(
        data_type type,
        size_type size,
        mask_state state,
        cudaStream_t stream,
        device_async_resource_ref mr) except +libcudf_exception_handler

    cdef unique_ptr[column] make_timestamp_column(
        data_type type,
        size_type size,
        device_buffer[byte] mask,
        size_type null_count,
        cudaStream_t stream,
        device_async_resource_ref mr) except +libcudf_exception_handler

    cdef unique_ptr[column] make_duration_column(
        data_type type,
        size_type size,
        mask_state state,
        cudaStream_t stream,
        device_async_resource_ref mr) except +libcudf_exception_handler

    cdef unique_ptr[column] make_duration_column(
        data_type type,
        size_type size,
        device_buffer[byte] mask,
        size_type null_count,
        cudaStream_t stream,
        device_async_resource_ref mr) except +libcudf_exception_handler

    cdef unique_ptr[column] make_fixed_width_column(
        data_type type,
        size_type size,
        mask_state state,
        cudaStream_t stream,
        device_async_resource_ref mr) except +libcudf_exception_handler

    cdef unique_ptr[column] make_fixed_width_column(
        data_type type,
        size_type size,
        device_buffer[byte] mask,
        size_type null_count,
        cudaStream_t stream,
        device_async_resource_ref mr) except +libcudf_exception_handler

    cdef unique_ptr[column] make_column_from_scalar(
        const scalar& s,
        size_type size,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] make_dictionary_from_scalar(
        const scalar& s,
        size_type size,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] make_dictionary_column(
        unique_ptr[column] keys_column,
        unique_ptr[column] indices_column,
        cudaStream_t stream,
        device_async_resource_ref mr) except +libcudf_exception_handler

    cdef unique_ptr[column] make_empty_column(
        type_id id
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] make_empty_column(
        data_type type_
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] make_empty_lists_column(
        data_type child_type
    ) except +libcudf_exception_handler
