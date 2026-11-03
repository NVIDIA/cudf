# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from enum import IntEnum

from rmm.pylibrmm.memory_resource import DeviceMemoryResource

from pylibcudf.column import Column
from pylibcudf.utils import CudaStreamLike

class RoundingMethod(IntEnum):
    HALF_UP = ...
    HALF_EVEN = ...

def round(
    source: Column,
    decimal_places: int = 0,
    round_method: RoundingMethod = RoundingMethod.HALF_UP,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def round_decimal(
    source: Column,
    decimal_places: int = 0,
    round_method: RoundingMethod = RoundingMethod.HALF_UP,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
