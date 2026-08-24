# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from pylibcudf.column cimport Column
from pylibcudf.libcudf.types cimport size_type
from pylibcudf.scalar cimport Scalar
from pylibcudf.strings.regex_flags cimport regex_flags
from pylibcudf.strings.regex_program cimport RegexProgram
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource


cpdef Column replace_re(
    Column input,
    RegexProgram patterns,
    Scalar replacement=*,
    size_type max_replace_count=*,
    object stream = *,
    DeviceMemoryResource mr=*
)

cpdef Column replace_with_backrefs(
    Column input,
    RegexProgram prog,
    str replacement,
    object stream = *,
    DeviceMemoryResource mr=*
)
