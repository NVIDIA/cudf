# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from libcpp.memory cimport unique_ptr
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view

from cuda.bindings.cyruntime cimport cudaStream_t
from pylibcudf.libcudf.utilities.memory_resource cimport memory_resources


cdef extern from "cudf/strings/convert/convert_urls.hpp" namespace \
        "cudf::strings" nogil:
    cdef unique_ptr[column] url_encode(
        column_view input,
        cudaStream_t stream,
        memory_resources mr) except +libcudf_exception_handler

    cdef unique_ptr[column] url_decode(
        column_view input,
        cudaStream_t stream,
        memory_resources mr) except +libcudf_exception_handler
