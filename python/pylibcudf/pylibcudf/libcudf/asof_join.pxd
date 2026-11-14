# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from rmm.librmm.cuda_stream_ref cimport stream_ref
from rmm.librmm.memory_resource cimport device_async_resource_ref
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.table.table_view cimport table_view

cdef extern from "cudf/join/asof_join.hpp" namespace "cudf::experimental" nogil:
    cpdef enum class asof_join_strategy:
        BACKWARD

    cpdef enum class asof_join_exact_matches:
        YES

    cdef cppclass asof_join:
        asof_join(
            const table_view& right_by,
            const column_view& right_on,
            stream_ref stream,
            device_async_resource_ref mr,
        ) except +libcudf_exception_handler
