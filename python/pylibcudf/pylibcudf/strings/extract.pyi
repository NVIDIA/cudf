# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from rmm.pylibrmm.memory_resource import DeviceMemoryResource

from pylibcudf.column import Column
from pylibcudf.strings.regex_program import RegexProgram
from pylibcudf.table import Table
from pylibcudf.utils import CudaStreamLike

def extract(
    input: Column,
    prog: RegexProgram,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Table: ...
def extract_all_record(
    input: Column,
    prog: RegexProgram,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def extract_single(
    input: Column,
    prog: RegexProgram,
    group: int,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
