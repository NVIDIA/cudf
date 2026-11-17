# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from pylibcudf.column cimport Column
from pylibcudf.types cimport DataType
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource


cpdef Column to_fixed_point(
    Column input,
    DataType output_type,
    object stream = *,
    DeviceMemoryResource mr=*
)

cpdef Column from_fixed_point(
    Column input, object stream = *, DeviceMemoryResource mr=*
)

cpdef Column is_fixed_point(
    Column input,
    DataType decimal_type=*,
    object stream = *,
    DeviceMemoryResource mr=*
)
