# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from enum import IntEnum

from rmm.pylibrmm.memory_resource import DeviceMemoryResource

from pylibcudf import Table
from pylibcudf.aggregation import Aggregation
from pylibcudf.column import Column
from pylibcudf.scalar import Scalar
from pylibcudf.types import DataType, NanPolicy, NullEquality, NullPolicy
from pylibcudf.utils import CudaStreamLike

class ScanType(IntEnum):
    INCLUSIVE = ...
    EXCLUSIVE = ...

def reduce(
    col: Column,
    agg: Aggregation,
    data_type: DataType,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Scalar: ...
def scan(
    col: Column,
    agg: Aggregation,
    inclusive: ScanType,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def minmax(
    col: Column,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> tuple[Scalar, Scalar]: ...
def is_valid_reduce_aggregation(
    source: DataType, agg: Aggregation
) -> bool: ...
def unique_count(
    source: Column,
    null_handling: NullPolicy,
    nan_handling: NanPolicy,
    stream: CudaStreamLike | None = None,
) -> int: ...
def distinct_count(
    source: Column,
    null_handling: NullPolicy,
    nan_handling: NanPolicy,
    stream: CudaStreamLike | None = None,
) -> int: ...
def unique_count_table(
    source: Table,
    nulls_equal: NullEquality,
    stream: CudaStreamLike | None = None,
) -> int: ...
def distinct_count_table(
    source: Table,
    nulls_equal: NullEquality,
    stream: CudaStreamLike | None = None,
) -> int: ...
