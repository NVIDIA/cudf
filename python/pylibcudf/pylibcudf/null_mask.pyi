# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence

from pylibcudf.gpumemoryview import gpumemoryview
from rmm.pylibrmm.memory_resource import DeviceMemoryResource

from pylibcudf.column import Column
from pylibcudf.span import Span
from pylibcudf.types import MaskState
from pylibcudf.utils import CudaStreamLike

def copy_bitmask(
    col: Column,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> gpumemoryview: ...
def copy_bitmask_from_bitmask(
    bitmask: Span,
    begin_bit: int,
    end_bit: int,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> gpumemoryview: ...
def bitmask_allocation_size_bytes(number_of_bits: int) -> int: ...
def create_null_mask(
    size: int,
    state: MaskState = MaskState.UNINITIALIZED,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> gpumemoryview: ...
def bitmask_and(
    columns: Sequence[Column],
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> tuple[gpumemoryview, int]: ...
def bitmask_or(
    columns: Sequence[Column],
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> tuple[gpumemoryview, int]: ...
def null_count(
    bitmask: Span, start: int, stop: int, stream: CudaStreamLike | None = None
) -> int: ...
def index_of_first_set_bit(
    bitmask: Span, start: int, stop: int, stream: CudaStreamLike | None = None
) -> int: ...
