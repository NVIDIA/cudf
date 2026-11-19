# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from rmm.pylibrmm.memory_resource import DeviceMemoryResource

from pylibcudf.column import Column
from pylibcudf.scalar import Scalar
from pylibcudf.utils import CudaStreamLike

def find(
    input: Column,
    target: Column | Scalar,
    start: int = 0,
    stop: int = -1,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def rfind(
    input: Column,
    target: Scalar,
    start: int = 0,
    stop: int = -1,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def contains(
    input: Column,
    target: Column | Scalar,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def starts_with(
    input: Column,
    target: Column | Scalar,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def ends_with(
    input: Column,
    target: Column | Scalar,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
