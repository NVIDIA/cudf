# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from pylibcudf.column cimport Column
from pylibcudf.strings.regex_program cimport RegexProgram
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource


cpdef Column contains_re(
    Column input, RegexProgram prog, object stream = *, DeviceMemoryResource mr=*
)

cpdef Column count_re(
    Column input, RegexProgram prog, object stream = *, DeviceMemoryResource mr=*
)

cpdef Column matches_re(
    Column input, RegexProgram prog, object stream = *, DeviceMemoryResource mr=*
)

cpdef Column like(
    Column input,
    str pattern,
    str escape_character=*,
    object stream = *,
    DeviceMemoryResource mr=*,
)
