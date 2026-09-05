# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from pylibcudf.column cimport Column
from pylibcudf.libcudf.nvtext.normalize cimport (
    character_normalizer,
    underlying_type_t_normalize_flags,
)
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource

cdef class CharacterNormalizer:
    cdef unique_ptr[character_normalizer] c_obj

cpdef Column normalize_spaces(
    Column input, object stream = *, DeviceMemoryResource mr=*
)

cpdef Column normalize_characters(
    Column input,
    CharacterNormalizer normalizer,
    underlying_type_t_normalize_flags flags = *,
    object stream = *,
    DeviceMemoryResource mr=*
)
