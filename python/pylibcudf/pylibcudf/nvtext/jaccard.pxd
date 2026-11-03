# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from pylibcudf.column cimport Column
from pylibcudf.libcudf.types cimport size_type
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource


cpdef Column jaccard_index(
    Column input1,
    Column input2,
    size_type width,
    object stream = *,
    DeviceMemoryResource mr=*,
)
