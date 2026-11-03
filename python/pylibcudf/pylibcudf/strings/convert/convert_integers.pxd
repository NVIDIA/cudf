# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from pylibcudf.column cimport Column
from pylibcudf.types cimport DataType
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource


cpdef Column to_integers(
    Column input, DataType output_type, object stream = *, DeviceMemoryResource mr=*
)

cpdef Column from_integers(
    Column integers, object stream = *, DeviceMemoryResource mr=*
)

cpdef Column is_integer(
    Column input, DataType int_type=*, object stream = *, DeviceMemoryResource mr=*
)

cpdef Column hex_to_integers(
    Column input, DataType output_type, object stream = *, DeviceMemoryResource mr=*
)

cpdef Column is_hex(
    Column input, object stream = *, DeviceMemoryResource mr=*
)

cpdef Column integers_to_hex(
    Column input, object stream = *, DeviceMemoryResource mr=*
)
