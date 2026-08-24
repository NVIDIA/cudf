# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from .table cimport Table

from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource


cpdef Table merge (
    object tables_to_merge,
    object key_cols,
    object column_order,
    object null_precedence,
    object stream = *,
    DeviceMemoryResource mr=*
)
