# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from cuda.bindings.cyruntime cimport cudaStream_t
from libcpp cimport bool


cdef extern from "cudf/utilities/default_stream.hpp" namespace "cudf" nogil:
    cdef bool is_ptds_enabled()
    cdef cudaStream_t get_default_stream()
