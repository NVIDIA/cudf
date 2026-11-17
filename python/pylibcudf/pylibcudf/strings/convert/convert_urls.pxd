# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from pylibcudf.column cimport Column
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource


cpdef Column url_encode(
    Column Input, object stream = *, DeviceMemoryResource mr=*
)

cpdef Column url_decode(
    Column Input, object stream = *, DeviceMemoryResource mr=*
)
