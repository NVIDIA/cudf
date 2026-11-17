# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from pylibcudf.column cimport Column
from pylibcudf.libcudf.types cimport size_type
from pylibcudf.scalar cimport Scalar
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource


cpdef Column replace_tokens(
    Column input,
    Column targets,
    Column replacements,
    Scalar delimiter=*,
    object stream = *,
    DeviceMemoryResource mr=*,
)

cpdef Column filter_tokens(
    Column input,
    size_type min_token_length,
    Scalar replacement=*,
    Scalar delimiter=*,
    object stream = *,
    DeviceMemoryResource mr=*,
)
