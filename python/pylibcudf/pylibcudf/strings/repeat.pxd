# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from pylibcudf.column cimport Column
from pylibcudf.libcudf.types cimport size_type
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource

ctypedef fused ColumnorSizeType:
    Column
    size_type

cpdef Column repeat_strings(
    Column input,
    ColumnorSizeType repeat_times,
    object stream = *,
    DeviceMemoryResource mr=*,
)
