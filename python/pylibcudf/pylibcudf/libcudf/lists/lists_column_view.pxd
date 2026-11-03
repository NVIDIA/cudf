# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from cuda.bindings.cyruntime cimport cudaStream_t

from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column_view cimport (
    column_view,
    mutable_column_view,
)
from pylibcudf.libcudf.types cimport size_type


cdef extern from "cudf/lists/lists_column_view.hpp" namespace "cudf" nogil:
    cdef cppclass lists_column_view(column_view):
        lists_column_view() except +libcudf_exception_handler
        lists_column_view(
            const lists_column_view& lists_column
        ) except +libcudf_exception_handler
        lists_column_view(
            const column_view& lists_column
        ) except +libcudf_exception_handler
        lists_column_view& operator=(
            const lists_column_view&
        ) except +libcudf_exception_handler
        column_view parent() except +libcudf_exception_handler
        column_view offsets() except +libcudf_exception_handler
        column_view child() except +libcudf_exception_handler
        column_view get_sliced_child(
            cudaStream_t stream
        ) except +libcudf_exception_handler

    cdef enum:
        offsets_column_index "cudf::lists_column_view::offsets_column_index"
        child_column_index "cudf::lists_column_view::child_column_index"
