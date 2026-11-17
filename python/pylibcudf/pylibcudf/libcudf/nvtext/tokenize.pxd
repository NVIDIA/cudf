# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from libcpp.memory cimport unique_ptr
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.scalar.scalar cimport string_scalar
from pylibcudf.libcudf.types cimport size_type
from cuda.bindings.cyruntime cimport cudaStream_t
from rmm.librmm.memory_resource cimport device_async_resource_ref


cdef extern from "nvtext/tokenize.hpp" namespace "nvtext" nogil:

    cdef unique_ptr[column] tokenize(
        const column_view & strings,
        const string_scalar & delimiter,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] tokenize(
        const column_view & strings,
        const column_view & delimiters,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] count_tokens(
        const column_view & strings,
        const string_scalar & delimiter,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] count_tokens(
        const column_view & strings,
        const column_view & delimiters,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] character_tokenize(
        const column_view & strings,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] detokenize(
        const column_view & strings,
        const column_view & row_indices,
        const string_scalar & separator,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler

    cdef struct tokenize_vocabulary:
        pass

    cdef unique_ptr[tokenize_vocabulary] load_vocabulary(
        const column_view & strings,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] tokenize_with_vocabulary(
        const column_view & strings,
        const tokenize_vocabulary & vocabulary,
        const string_scalar & delimiter,
        size_type default_id,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler
