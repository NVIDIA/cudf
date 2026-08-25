# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource

from .column cimport Column
from .table cimport Table


cpdef Column lower_bound(
    Table haystack,
    Table needles,
    list column_order,
    list null_precedence,
    object stream = *,
    DeviceMemoryResource mr = *,
)

cpdef Column upper_bound(
    Table haystack,
    Table needles,
    list column_order,
    list null_precedence,
    object stream = *,
    DeviceMemoryResource mr = *,
)

cpdef Column contains(
    Column haystack, Column needles, object stream = *, DeviceMemoryResource mr = *
)
