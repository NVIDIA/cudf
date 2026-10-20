# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pylibcudf.column cimport Column
from pylibcudf.libcudf.strings.char_types cimport string_character_types
from pylibcudf.scalar cimport Scalar
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource


cpdef Column all_characters_of_type(
    Column source_strings,
    string_character_types types,
    string_character_types verify_types,
    object stream = *,
    DeviceMemoryResource mr=*
)

cpdef Column filter_characters_of_type(
    Column source_strings,
    string_character_types types_to_remove,
    Scalar replacement,
    string_character_types types_to_keep,
    object stream = *,
    DeviceMemoryResource mr=*
)

cpdef get_character_flags_table_ptr(object stream = *)
