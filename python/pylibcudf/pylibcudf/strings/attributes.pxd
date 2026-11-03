# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from pylibcudf.column cimport Column
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource


cpdef Column count_characters(
    Column source_strings, object stream = *, DeviceMemoryResource mr=*
)

cpdef Column count_bytes(
    Column source_strings, object stream = *, DeviceMemoryResource mr=*
)

cpdef Column code_points(
    Column source_strings, object stream = *, DeviceMemoryResource mr=*
)
