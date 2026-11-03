# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from enum import IntEnum

from rmm.pylibrmm.memory_resource import DeviceMemoryResource

from pylibcudf.column import Column
from pylibcudf.scalar import Scalar
from pylibcudf.utils import CudaStreamLike

class ReplacePolicy(IntEnum):
    PRECEDING = ...
    FOLLOWING = ...

def replace_nulls(
    source_column: Column,
    replacement: Column | Scalar | ReplacePolicy,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def find_and_replace_all(
    source_column: Column,
    values_to_replace: Column,
    replacement_values: Column,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def clamp(
    source_column: Column,
    lo: Scalar,
    hi: Scalar,
    lo_replace: Scalar | None = None,
    hi_replace: Scalar | None = None,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def normalize_nans_and_zeros(
    source_column: Column,
    inplace: bool = False,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
