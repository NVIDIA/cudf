# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from .table cimport Table

from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource


cpdef Table merge (
    list tables_to_merge,
    list key_cols,
    list column_order,
    list null_precedence,
    object stream = *,
    DeviceMemoryResource mr=*
)
