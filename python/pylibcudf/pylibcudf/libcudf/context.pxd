# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from libcpp cimport bool
from pylibcudf.exception_handler cimport libcudf_exception_handler


cdef extern from "cudf/context.hpp" namespace "cudf" nogil:
    void enable_jit_cache(bool enable) noexcept
    void clear_jit_cache() except +libcudf_exception_handler
