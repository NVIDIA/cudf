# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from rmm.pylibrmm.memory_resource import DeviceMemoryResource

from pylibcudf.column import Column
from pylibcudf.types import DataType, MaskState, TypeId
from pylibcudf.utils import CudaStreamLike

def make_numeric_column(
    type_: DataType,
    size: int,
    mstate: MaskState,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def make_fixed_point_column(
    type_: DataType,
    size: int,
    mstate: MaskState,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def make_timestamp_column(
    type_: DataType,
    size: int,
    mstate: MaskState,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def make_duration_column(
    type_: DataType,
    size: int,
    mstate: MaskState,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def make_fixed_width_column(
    type_: DataType,
    size: int,
    mstate: MaskState,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def make_empty_column(
    type_or_id: DataType | TypeId,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def make_empty_lists_column(
    child_type: DataType,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
