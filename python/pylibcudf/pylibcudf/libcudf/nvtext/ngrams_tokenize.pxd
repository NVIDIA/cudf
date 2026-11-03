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


cdef extern from "nvtext/ngrams_tokenize.hpp" namespace "nvtext" nogil:

    cdef unique_ptr[column] ngrams_tokenize(
        const column_view & strings,
        size_type ngrams,
        const string_scalar & delimiter,
        const string_scalar & separator,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler
