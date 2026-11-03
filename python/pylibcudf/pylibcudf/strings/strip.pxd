# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from pylibcudf.column cimport Column
from pylibcudf.scalar cimport Scalar
from pylibcudf.strings.side_type cimport side_type
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource


cpdef Column strip(
    Column input,
    side_type side=*,
    Scalar to_strip=*,
    object stream = *,
    DeviceMemoryResource mr=*
)
