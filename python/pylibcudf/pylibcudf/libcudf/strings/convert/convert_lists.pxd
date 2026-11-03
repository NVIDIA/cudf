# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from libcpp.memory cimport unique_ptr
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.scalar.scalar cimport string_scalar

from cuda.bindings.cyruntime cimport cudaStream_t
from rmm.librmm.memory_resource cimport device_async_resource_ref


cdef extern from "cudf/strings/convert/convert_lists.hpp" namespace \
        "cudf::strings" nogil:

    cdef unique_ptr[column] format_list_column(
        column_view input,
        string_scalar na_rep,
        column_view separators,
        cudaStream_t stream,
        device_async_resource_ref mr) except +libcudf_exception_handler
