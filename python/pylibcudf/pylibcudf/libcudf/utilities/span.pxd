# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from libc.stddef cimport size_t
from libcpp.vector cimport vector
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.types cimport size_type


cdef extern from "cudf/utilities/span.hpp" namespace "cudf" nogil:
    cdef cppclass host_span[T]:
        host_span() except +libcudf_exception_handler
        host_span(vector[T]) except +libcudf_exception_handler
        host_span(T* data, size_type size) noexcept

    cdef cppclass device_span[T]:
        device_span() noexcept
        device_span(T *data, size_t size) noexcept
        T *data() noexcept const
        size_t size() noexcept const
