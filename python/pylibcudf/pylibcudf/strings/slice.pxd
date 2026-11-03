# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pylibcudf.column cimport Column
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource

ctypedef fused ColumnOrScalar:
    Column
    object

cpdef Column slice_strings(
    Column input,
    ColumnOrScalar start=*,
    ColumnOrScalar stop=*,
    object step=*,
    object stream = *,
    DeviceMemoryResource mr=*
)
