# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from rmm.pylibrmm.memory_resource import DeviceMemoryResource

from pylibcudf.column import Column
from pylibcudf.scalar import Scalar
from pylibcudf.utils import CudaStreamLike

def replace(
    input: Column,
    target: Scalar,
    repl: Scalar,
    maxrepl: int = -1,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def replace_multiple(
    input: Column,
    target: Column,
    repl: Column,
    maxrepl: int = -1,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def replace_slice(
    input: Column,
    repl: Scalar | None = None,
    start: int = 0,
    stop: int = -1,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
