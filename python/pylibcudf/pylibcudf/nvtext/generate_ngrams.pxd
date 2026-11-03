# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport uint32_t
from pylibcudf.column cimport Column
from pylibcudf.libcudf.types cimport size_type
from pylibcudf.scalar cimport Scalar
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource


cpdef Column generate_ngrams(
    Column input,
    size_type ngrams,
    Scalar separator,
    object stream = *,
    DeviceMemoryResource mr=*,
)

cpdef Column generate_character_ngrams(
    Column input,
    size_type ngrams=*,
    object stream = *,
    DeviceMemoryResource mr=*,
)

cpdef Column hash_character_ngrams(
    Column input,
    size_type ngrams,
    uint32_t seed,
    object stream = *,
    DeviceMemoryResource mr=*,
)
