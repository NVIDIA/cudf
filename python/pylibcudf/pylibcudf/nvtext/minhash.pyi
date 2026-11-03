# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import numpy as np

from rmm.pylibrmm.memory_resource import DeviceMemoryResource

from pylibcudf.column import Column
from pylibcudf.utils import CudaStreamLike

def minhash(
    input: Column,
    seed: int | np.unsignedinteger[Any],
    a: Column,
    b: Column,
    width: int,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def minhash64(
    input: Column,
    seed: int | np.unsignedinteger[Any],
    a: Column,
    b: Column,
    width: int,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def minhash_ngrams(
    input: Column,
    ngrams: int,
    seed: int | np.unsignedinteger[Any],
    a: Column,
    b: Column,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def minhash64_ngrams(
    input: Column,
    ngrams: int,
    seed: int | np.unsignedinteger[Any],
    a: Column,
    b: Column,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
