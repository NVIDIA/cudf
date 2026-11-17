# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from rmm.pylibrmm.memory_resource import DeviceMemoryResource

from pylibcudf.column import Column
from pylibcudf.scalar import Scalar
from pylibcudf.strings.regex_program import RegexProgram
from pylibcudf.table import Table
from pylibcudf.utils import CudaStreamLike

def split(
    strings_column: Column,
    delimiter: Scalar,
    maxsplit: int,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Table: ...
def rsplit(
    strings_column: Column,
    delimiter: Scalar,
    maxsplit: int,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Table: ...
def split_record(
    strings: Column,
    delimiter: Scalar,
    maxsplit: int,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def rsplit_record(
    strings: Column,
    delimiter: Scalar,
    maxsplit: int,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def split_re(
    input: Column,
    prog: RegexProgram,
    maxsplit: int,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Table: ...
def rsplit_re(
    input: Column,
    prog: RegexProgram,
    maxsplit: int,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Table: ...
def split_record_re(
    input: Column,
    prog: RegexProgram,
    maxsplit: int,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def rsplit_record_re(
    input: Column,
    prog: RegexProgram,
    maxsplit: int,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def split_part(
    input: Column,
    delimiter: Scalar,
    index: int,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
