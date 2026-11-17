# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Mapping
from enum import IntEnum

from rmm.pylibrmm.memory_resource import DeviceMemoryResource

from pylibcudf.column import Column
from pylibcudf.scalar import Scalar
from pylibcudf.utils import CudaStreamLike

class FilterType(IntEnum):
    KEEP = ...
    REMOVE = ...

def translate(
    input: Column,
    chars_table: Mapping[int | str, int | str],
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def filter_characters(
    input: Column,
    characters_to_filter: Mapping[int | str, int | str],
    keep_characters: FilterType,
    replacement: Scalar,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
