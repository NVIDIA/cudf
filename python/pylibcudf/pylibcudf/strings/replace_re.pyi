# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from rmm.pylibrmm.memory_resource import DeviceMemoryResource

from pylibcudf.column import Column
from pylibcudf.scalar import Scalar
from pylibcudf.strings.regex_program import RegexProgram
from pylibcudf.utils import CudaStreamLike

def replace_re(
    input: Column,
    pattern: RegexProgram,
    replacement: Scalar,
    max_replace_count: int = -1,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def replace_with_backrefs(
    input: Column,
    prog: RegexProgram,
    replacement: str,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
