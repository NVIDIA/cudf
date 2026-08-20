# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pylibcudf.exception_handler cimport libcudf_exception_handler

cdef extern from "cudf/io/config_utils.hpp" \
        namespace "cudf::io::kvikio_integration" nogil:

    void set_up_kvikio() except +libcudf_exception_handler
