# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from rmm.pylibrmm.memory_resource import DeviceMemoryResource

from pylibcudf.column import Column
from pylibcudf.utils import CudaStreamLike

def to_lower(
    input: Column,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def to_upper(
    input: Column,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def swapcase(
    input: Column,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
