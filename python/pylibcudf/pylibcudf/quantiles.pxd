# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from libcpp.vector cimport vector
from pylibcudf.libcudf.types cimport interpolation, sorted
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource

from .column cimport Column
from .table cimport Table


cpdef Column quantile(
    Column input,
    vector[double] q,
    interpolation interp = *,
    Column ordered_indices = *,
    bint exact = *,
    object stream = *,
    DeviceMemoryResource mr = *,
)

cpdef Table quantiles(
    Table input,
    vector[double] q,
    interpolation interp = *,
    sorted is_input_sorted = *,
    list column_order = *,
    list null_precedence = *,
    object stream = *,
    DeviceMemoryResource mr = *,
)
