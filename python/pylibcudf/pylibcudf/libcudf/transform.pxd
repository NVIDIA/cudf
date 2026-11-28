# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.optional cimport optional
from libcpp.pair cimport pair
from libcpp.span cimport span
from libcpp.string_view cimport string_view
from libc.stdint cimport uint8_t
from libcpp.vector cimport vector
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.expressions cimport expression
from pylibcudf.libcudf.table.table cimport table
from pylibcudf.libcudf.table.table_view cimport table_view
from pylibcudf.libcudf.types cimport (
    bitmask_type,
    data_type,
    null_aware,
    output_nullability,
    size_type,
)

from pylibcudf.libcudf.utilities.device_buffer cimport device_buffer
from cuda.bindings.cyruntime cimport cudaStream_t
from rmm.librmm.memory_resource cimport device_async_resource_ref


ctypedef const uint8_t const_uint8_t


cdef extern from "cudf/udf.hpp" namespace "cudf" nogil:
    cdef enum class lto_binary_type(uint8_t):
        LTO_IR
        FATBIN
        PTX

    cdef cppclass lto_fragment:
        lto_binary_type type
        span[const_uint8_t] data

    cdef cppclass transform_udf:
        string_view symbol
        span[const lto_fragment] fragments
        null_aware is_null_aware
        bool indexed

ctypedef const lto_fragment const_lto_fragment


cdef extern from "cudf/transform.hpp" namespace "cudf" nogil:
    cdef cppclass scalar_column_view:
        scalar_column_view(
            const column_view& input
        ) except +libcudf_exception_handler

    cdef cppclass user_data:
        user_data() noexcept
        void* data

    cdef cppclass transform_input:
        transform_input(
            const column_view& input
        ) except +libcudf_exception_handler
        transform_input(
            const scalar_column_view& input
        ) except +libcudf_exception_handler
        transform_input(
            const user_data& input
        ) except +libcudf_exception_handler

    ctypedef const transform_input const_transform_input

    cdef cppclass transform_output:
        data_type type
        output_nullability nullability

    ctypedef const transform_output const_transform_output

    cdef pair[unique_ptr[device_buffer[bitmask_type]], size_type] bools_to_mask (
        const column_view& input,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] mask_to_bools (
        const bitmask_type* bitmask,
        size_type begin_bit,
        size_type end_bit,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] column_nans_to_nulls(
        const column_view& input,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[table] transform(
        const transform_udf& udf,
        span[const_transform_input] inputs,
        span[const_transform_output] outputs,
        vector[unique_ptr[column]]&& string_offsets,
        optional[size_type] row_size,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler

    cdef pair[unique_ptr[table], unique_ptr[column]] encode(
        table_view input,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler

    cdef pair[unique_ptr[column], table_view] one_hot_encode(
        column_view input_column,
        column_view categories,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] compute_column(
        const table_view table,
        const expression& expr,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] compute_column_jit(
        const table_view table,
        const expression& expr,
        span[const_lto_fragment] fragments,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler
