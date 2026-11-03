# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.scalar.scalar cimport scalar, string_scalar

from cuda.bindings.cyruntime cimport cudaStream_t
from rmm.librmm.memory_resource cimport device_async_resource_ref


cdef extern from "cudf/json/json.hpp" namespace "cudf" nogil:
    cdef cppclass get_json_object_options:
        get_json_object_options() except +libcudf_exception_handler
        # getters
        bool get_allow_single_quotes() except +libcudf_exception_handler
        bool get_strip_quotes_from_single_strings() except +libcudf_exception_handler
        bool get_missing_fields_as_nulls() except +libcudf_exception_handler
        # setters
        void set_allow_single_quotes(bool val) except +libcudf_exception_handler
        void set_strip_quotes_from_single_strings(
            bool val
        ) except +libcudf_exception_handler
        void set_missing_fields_as_nulls(bool val) except +libcudf_exception_handler

    cdef unique_ptr[column] get_json_object(
        column_view col,
        string_scalar json_path,
        get_json_object_options options,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler
