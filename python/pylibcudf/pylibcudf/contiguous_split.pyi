# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from rmm.mr import DeviceMemoryResource

from pylibcudf.gpumemoryview import gpumemoryview
from pylibcudf.span import Span
from pylibcudf.table import Table
from pylibcudf.utils import CudaStreamLike

class PackedColumns:
    def __init__(self): ...
    def release(self) -> tuple[memoryview[bytes], gpumemoryview]: ...

def pack(
    input: Table,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> PackedColumns: ...
def unpack(
    input: PackedColumns, stream: CudaStreamLike | None = None
) -> Table: ...
def unpack_from_memoryviews(
    metadata: memoryview[bytes],
    gpu_data: Span,
    stream: CudaStreamLike | None = None,
) -> Table: ...

class ChunkedPack:
    def __init__(self): ...
    @staticmethod
    def create(
        input: Table,
        user_buffer_size: int,
        stream: CudaStreamLike | None = None,
        temp_mr: DeviceMemoryResource | None = None,
    ) -> ChunkedPack: ...
    def has_next(self) -> bool: ...
    def get_total_contiguous_size(self) -> int: ...
    def next(self, buf: Span) -> int: ...
    def build_metadata(self) -> memoryview[bytes]: ...
    def pack_to_host(
        self, buf: Span
    ) -> tuple[memoryview[bytes], memoryview[bytes]]: ...
