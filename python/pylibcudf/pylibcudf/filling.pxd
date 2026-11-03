# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from pylibcudf.libcudf.types cimport size_type
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource

from .column cimport Column
from .scalar cimport Scalar
from .table cimport Table

ctypedef fused ColumnOrSize:
    Column
    size_type

cpdef Column fill(
    Column destination,
    size_type begin,
    size_type end,
    Scalar value,
    object stream = *,
    DeviceMemoryResource mr = *,
)

cpdef void fill_in_place(
    Column destination,
    size_type c_begin,
    size_type c_end,
    Scalar value,
    object stream = *,
)

cpdef Column sequence(
    size_type size,
    Scalar init,
    Scalar step,
    object stream = *,
    DeviceMemoryResource mr = *,
)

cpdef Table repeat(
    Table input_table,
    ColumnOrSize count,
    object stream = *,
    DeviceMemoryResource mr = *,
)

cpdef Column calendrical_month_sequence(
    size_type n,
    Scalar init,
    size_type months,
    object stream = *,
    DeviceMemoryResource mr = *,
)
