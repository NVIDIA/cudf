# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from pylibcudf.column cimport Column
from pylibcudf.libcudf.types cimport size_type
from pylibcudf.scalar cimport Scalar
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource


cpdef Column ngrams_tokenize(
    Column input,
    size_type ngrams,
    Scalar delimiter,
    Scalar separator,
    object stream = *,
    DeviceMemoryResource mr=*
)
