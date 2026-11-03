# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from pylibcudf.column cimport Column
from pylibcudf.strings.regex_program cimport RegexProgram
from pylibcudf.table cimport Table
from pylibcudf.libcudf.types cimport size_type
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource


cpdef Table extract(
    Column input, RegexProgram prog, object stream = *, DeviceMemoryResource mr=*
)

cpdef Column extract_all_record(
    Column input, RegexProgram prog, object stream = *, DeviceMemoryResource mr=*
)

cpdef Column extract_single(
    Column input,
    RegexProgram prog,
    size_type group,
    object stream = *,
    DeviceMemoryResource mr=*,
)
