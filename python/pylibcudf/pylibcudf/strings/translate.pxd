# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from pylibcudf.column cimport Column
from pylibcudf.libcudf.strings.translate cimport filter_type
from pylibcudf.scalar cimport Scalar
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource


cpdef Column translate(
    Column input, dict chars_table, object stream = *, DeviceMemoryResource mr=*
)

cpdef Column filter_characters(
    Column input,
    dict characters_to_filter,
    filter_type keep_characters,
    Scalar replacement,
    object stream = *,
    DeviceMemoryResource mr=*
)
