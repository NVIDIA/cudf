# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from libc.stdint cimport int32_t, int64_t
from libcpp cimport bool
from libcpp.string cimport string
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.fixed_point.fixed_point cimport scale_type
from pylibcudf.libcudf.table.table_view cimport table_view
from pylibcudf.libcudf.types cimport data_type
from cuda.bindings.cyruntime cimport cudaStream_t
from rmm.librmm.memory_resource cimport device_async_resource_ref


cdef extern from "cudf/scalar/scalar.hpp" namespace "cudf" nogil:
    cdef cppclass scalar:
        scalar() except +libcudf_exception_handler
        scalar(scalar other) except +libcudf_exception_handler
        data_type type() except +libcudf_exception_handler
        void set_valid_async(
            bool is_valid, cudaStream_t stream
        ) except +libcudf_exception_handler
        bool is_valid(cudaStream_t stream) except +libcudf_exception_handler

    cdef cppclass numeric_scalar[T](scalar):
        void set_value(
            T value,
            cudaStream_t stream
        ) except +libcudf_exception_handler
        T value(cudaStream_t stream) except +libcudf_exception_handler

    cdef cppclass timestamp_scalar[T](scalar):
        void set_value(
            T value,
            cudaStream_t stream
        ) except +libcudf_exception_handler

    cdef cppclass duration_scalar[T](scalar):
        void set_value(
            T value,
            cudaStream_t stream
        ) except +libcudf_exception_handler

    cdef cppclass string_scalar(scalar):
        string to_string(cudaStream_t stream) except +libcudf_exception_handler

    cdef cppclass list_scalar(scalar):
        pass

    cdef cppclass struct_scalar(scalar):
        pass

    cdef cppclass fixed_point_scalar[T](scalar):
        fixed_point_scalar() except +libcudf_exception_handler
        fixed_point_scalar(
            T value,
            scale_type scale,
            bool is_valid
        ) except +libcudf_exception_handler
        T value(cudaStream_t stream) except +libcudf_exception_handler
