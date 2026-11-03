# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from pylibcudf.column cimport Column
from pylibcudf.scalar cimport Scalar
from pylibcudf.libcudf.strings.char_types cimport string_character_types
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource


cpdef Column capitalize(
    Column input, Scalar delimiters=*, object stream = *, DeviceMemoryResource mr=*
)
cpdef Column title(
    Column input,
    string_character_types sequence_type=*,
    object stream = *,
    DeviceMemoryResource mr=*
)
cpdef Column is_title(Column input, object stream = *, DeviceMemoryResource mr=*)
