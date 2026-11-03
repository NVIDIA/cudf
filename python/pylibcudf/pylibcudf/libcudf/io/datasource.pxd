# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from libc.stddef cimport size_t
from libc.stdint cimport uint8_t
from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector
from pylibcudf.libcudf.io.types cimport source_info
from pylibcudf.exception_handler cimport libcudf_exception_handler


cdef extern from "cudf/io/datasource.hpp" \
        namespace "cudf::io" nogil:

    cdef cppclass datasource:
        cppclass buffer:
            const uint8_t* data() except +libcudf_exception_handler const
            size_t size() except +libcudf_exception_handler const

    cdef vector[unique_ptr[datasource]] make_datasources(
        source_info info
    ) except +libcudf_exception_handler
