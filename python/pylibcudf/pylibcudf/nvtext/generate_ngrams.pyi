# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import numpy as np

from rmm.pylibrmm.memory_resource import DeviceMemoryResource

from pylibcudf.column import Column
from pylibcudf.scalar import Scalar
from pylibcudf.utils import CudaStreamLike

def generate_ngrams(
    input: Column,
    ngrams: int,
    separator: Scalar,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def generate_character_ngrams(
    input: Column,
    ngrams: int = 2,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def hash_character_ngrams(
    input: Column,
    ngrams: int,
    seed: int | np.unsignedinteger[Any],
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
