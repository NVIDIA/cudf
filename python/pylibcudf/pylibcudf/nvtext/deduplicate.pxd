# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from pylibcudf.column cimport Column
from pylibcudf.libcudf.types cimport size_type
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource

cpdef Column build_suffix_array(
    Column input,
    size_type min_width,
    object stream = *,
    DeviceMemoryResource mr=*
)
cpdef Column resolve_duplicates(
    Column input,
    Column indices,
    size_type min_width,
    object stream = *,
    DeviceMemoryResource mr=*
)
cpdef Column resolve_duplicates_pair(
    Column input1,
    Column indices1,
    Column input2,
    Column indices2,
    size_type min_width,
    object stream = *,
    DeviceMemoryResource mr=*
)
