# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from pylibcudf.column cimport Column
from pylibcudf.libcudf.types cimport size_type
from pylibcudf.scalar cimport Scalar
from pylibcudf.strings.regex_program cimport RegexProgram
from pylibcudf.table cimport Table
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource


cpdef Table split(
    Column strings_column, Scalar delimiter, size_type maxsplit, object stream = *,
    DeviceMemoryResource mr=*,
)

cpdef Table rsplit(
    Column strings_column, Scalar delimiter, size_type maxsplit, object stream = *,
    DeviceMemoryResource mr=*,
)

cpdef Column split_record(
    Column strings, Scalar delimiter, size_type maxsplit, object stream = *,
    DeviceMemoryResource mr=*,
)

cpdef Column rsplit_record(
    Column strings, Scalar delimiter, size_type maxsplit, object stream = *,
    DeviceMemoryResource mr=*,
)

cpdef Table split_re(
    Column input, RegexProgram prog, size_type maxsplit, object stream = *,
    DeviceMemoryResource mr=*,
)

cpdef Table rsplit_re(
    Column input, RegexProgram prog, size_type maxsplit, object stream = *,
    DeviceMemoryResource mr=*,
)

cpdef Column split_record_re(
    Column input, RegexProgram prog, size_type maxsplit, object stream = *,
    DeviceMemoryResource mr=*,
)

cpdef Column rsplit_record_re(
    Column input, RegexProgram prog, size_type maxsplit, object stream = *,
    DeviceMemoryResource mr=*,
)

cpdef Column split_part(
    Column input, Scalar delimiter, size_type index, object stream = *,
    DeviceMemoryResource mr=*,
)
