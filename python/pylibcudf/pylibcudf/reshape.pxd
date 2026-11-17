# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from libc.stddef cimport size_t
from libc.stdint cimport uintptr_t

from pylibcudf.libcudf.types cimport size_type

from rmm.pylibrmm.device_buffer cimport DeviceBuffer
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource

from .column cimport Column
from .scalar cimport Scalar
from .table cimport Table
from .types cimport DataType


cpdef Column interleave_columns(
    Table source_table, object stream = *, DeviceMemoryResource mr=*
)
cpdef Table tile(
    Table source_table,
    size_type count,
    object stream = *,
    DeviceMemoryResource mr=*
)
cpdef void table_to_array(
    Table input_table,
    uintptr_t ptr,
    size_t size,
    object stream = *
)
