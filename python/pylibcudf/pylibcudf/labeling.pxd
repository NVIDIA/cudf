# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from libcpp cimport bool
from pylibcudf.libcudf.labeling cimport inclusive

from .column cimport Column

from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource


cpdef Column label_bins(
    Column input,
    Column left_edges,
    inclusive left_inclusive,
    Column right_edges,
    inclusive right_inclusive,
    object stream = *,
    DeviceMemoryResource mr=*
)
