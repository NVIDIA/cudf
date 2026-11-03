# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from pylibcudf.column cimport Column
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource


cpdef Column to_lower(Column input, object stream = *, DeviceMemoryResource mr=*)
cpdef Column to_upper(Column input, object stream = *, DeviceMemoryResource mr=*)
cpdef Column swapcase(Column input, object stream = *, DeviceMemoryResource mr=*)
