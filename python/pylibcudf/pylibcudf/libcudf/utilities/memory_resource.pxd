# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from rmm.librmm.memory_resource cimport device_async_resource_ref


cdef extern from "cudf/utilities/memory_resource.hpp" namespace "cudf" nogil:
    cdef cppclass memory_resources:
        memory_resources(
            device_async_resource_ref output_mr,
            device_async_resource_ref temporary_mr,
        ) noexcept
