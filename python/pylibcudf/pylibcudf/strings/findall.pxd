# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from pylibcudf.column cimport Column
from pylibcudf.strings.regex_program cimport RegexProgram
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource


cpdef Column find_re(
    Column input, RegexProgram pattern, object stream = *, DeviceMemoryResource mr=*
)
cpdef Column findall(
    Column input, RegexProgram pattern, object stream = *, DeviceMemoryResource mr=*
)
