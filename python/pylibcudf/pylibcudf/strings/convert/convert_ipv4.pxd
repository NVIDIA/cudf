# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from pylibcudf.column cimport Column
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource


cpdef Column ipv4_to_integers(
    Column input, object stream = *, DeviceMemoryResource mr=*
)

cpdef Column integers_to_ipv4(
    Column integers, object stream = *, DeviceMemoryResource mr=*
)

cpdef Column is_ipv4(
    Column input, object stream = *, DeviceMemoryResource mr=*
)
