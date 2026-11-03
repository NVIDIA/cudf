# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from pylibcudf.column cimport Column
from pylibcudf.table cimport Table
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource


cpdef Column find_multiple(
    Column input,
    Column targets,
    object stream = *,
    DeviceMemoryResource mr=*,
)
cpdef Table contains_multiple(
    Column input,
    Column targets,
    object stream = *,
    DeviceMemoryResource mr=*,
)
