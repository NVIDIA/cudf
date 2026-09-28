# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport uint32_t
from libcpp.optional cimport optional
from pylibcudf.exception_handler cimport libcudf_exception_handler

cdef extern from "cudf/io/config_utils.hpp" \
        namespace "cudf::io::kvikio_integration" nogil:

    void set_up_kvikio(optional[uint32_t] nthreads) except +libcudf_exception_handler
