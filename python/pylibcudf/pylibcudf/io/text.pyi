# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from rmm.pylibrmm.memory_resource import DeviceMemoryResource

from pylibcudf.column import Column
from pylibcudf.utils import CudaStreamLike

class ByteRangeInfo:
    def __init__(self, offset: int, size: int) -> None: ...
    @property
    def offset(self) -> int: ...
    @property
    def size(self) -> int: ...

class ParseOptions:
    def __init__(
        self,
        *,
        byte_range: tuple[int, int] | None = None,
        strip_delimiters: bool = False,
    ) -> None: ...

class DataChunkSource:
    def __init__(self, data: str) -> None: ...

def make_source(data: str) -> DataChunkSource: ...
def make_source_from_file(filename: str) -> DataChunkSource: ...
def make_source_from_bgzip_file(
    filename: str,
    virtual_begin: int = -1,
    virtual_end: int = -1,
) -> DataChunkSource: ...
def multibyte_split(
    source: DataChunkSource,
    delimiter: str,
    options: ParseOptions | None = None,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
