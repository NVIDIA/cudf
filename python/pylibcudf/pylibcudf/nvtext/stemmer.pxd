# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from libcpp cimport bool
from pylibcudf.column cimport Column
from pylibcudf.libcudf.nvtext.stemmer cimport letter_type
from pylibcudf.libcudf.types cimport size_type
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource

ctypedef fused ColumnOrSize:
    Column
    size_type

cpdef Column is_letter(
    Column input,
    bool check_vowels,
    ColumnOrSize indices,
    object stream = *,
    DeviceMemoryResource mr=*,
)

cpdef Column porter_stemmer_measure(
    Column input, object stream = *, DeviceMemoryResource mr=*
)
