# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cdef extern from "cudf/version_config.hpp" nogil:
    enum:
        CUDF_VERSION_MAJOR
        CUDF_VERSION_MINOR
        CUDF_VERSION_PATCH
