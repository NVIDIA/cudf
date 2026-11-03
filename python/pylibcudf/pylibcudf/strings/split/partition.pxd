# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from pylibcudf.column cimport Column
from pylibcudf.scalar cimport Scalar
from pylibcudf.table cimport Table
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource


cpdef Table partition(
    Column input, Scalar delimiter=*, object stream = *, DeviceMemoryResource mr=*
)

cpdef Table rpartition(
    Column input, Scalar delimiter=*, object stream = *, DeviceMemoryResource mr=*
)
