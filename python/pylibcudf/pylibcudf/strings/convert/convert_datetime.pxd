# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from libcpp.string cimport string
from pylibcudf.column cimport Column
from pylibcudf.types cimport DataType
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource


cpdef Column to_timestamps(
    Column input,
    DataType timestamp_type,
    str format,
    object stream = *,
    DeviceMemoryResource mr=*
)

cpdef Column from_timestamps(
    Column timestamps,
    str format,
    Column input_strings_names,
    object stream = *,
    DeviceMemoryResource mr=*
)

cpdef Column is_timestamp(
    Column input,
    str format,
    object stream = *,
    DeviceMemoryResource mr=*
)
