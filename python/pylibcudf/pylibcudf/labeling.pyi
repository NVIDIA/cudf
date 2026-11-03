# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from enum import IntEnum

from rmm.pylibrmm.memory_resource import DeviceMemoryResource

from pylibcudf.column import Column
from pylibcudf.utils import CudaStreamLike

class Inclusive(IntEnum):
    YES = ...
    NO = ...

def label_bins(
    input: Column,
    left_edges: Column,
    left_inclusive: Inclusive,
    right_edges: Column,
    right_inclusive: Inclusive,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
