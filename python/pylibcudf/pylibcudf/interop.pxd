# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from pylibcudf.table cimport Table
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource

cpdef Table from_dlpack(
    object managed_tensor, object stream = *, DeviceMemoryResource mr=*
)

cpdef object to_dlpack(Table input, object stream = *, DeviceMemoryResource mr=*)
