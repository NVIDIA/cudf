# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from libc.stdint cimport int32_t
from pylibcudf.libcudf.round cimport rounding_method

from .column cimport Column

from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource


cpdef Column round(
    Column source,
    int32_t decimal_places = *,
    rounding_method round_method = *,
    object stream = *,
    DeviceMemoryResource mr = *
)

cpdef Column round_decimal(
    Column source,
    int32_t decimal_places = *,
    rounding_method round_method = *,
    object stream = *,
    DeviceMemoryResource mr = *
)
