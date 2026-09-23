# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import IntEnum
from typing import TypeVar

from rmm.pylibrmm.memory_resource import DeviceMemoryResource

from pylibcudf.column import Column
from pylibcudf.scalar import Scalar
from pylibcudf.table import Table
from pylibcudf.utils import CudaStreamLike

class MaskAllocationPolicy(IntEnum):
    NEVER = ...
    RETAIN = ...
    ALWAYS = ...

class OutOfBoundsPolicy(IntEnum):
    NULLIFY = ...
    DONT_CHECK = ...

ColumnOrTable = TypeVar("ColumnOrTable", Column, Table)

def gather(
    source_table: Table,
    gather_map: Column,
    bounds_policy: OutOfBoundsPolicy,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Table: ...
def scatter(
    source: Table | list[Scalar],
    scatter_map: Column,
    target_table: Table,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Table: ...
def empty_like(
    input: ColumnOrTable,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> ColumnOrTable: ...
def allocate_like(
    input_column: Column,
    policy: MaskAllocationPolicy,
    size: int | None = None,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def copy_range_in_place(
    input_column: Column,
    target_column: Column,
    input_begin: int,
    input_end: int,
    target_begin: int,
    stream: CudaStreamLike | None = None,
) -> Column: ...
def copy_range(
    input_column: Column,
    target_column: Column,
    input_begin: int,
    input_end: int,
    target_begin: int,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def shift(
    input: Column,
    offset: int,
    fill_value: Scalar,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def reverse(
    input: ColumnOrTable,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> ColumnOrTable: ...
def slice(
    input: ColumnOrTable,
    indices: list[int],
    stream: CudaStreamLike | None = None,
) -> list[ColumnOrTable]: ...
def split(
    input: ColumnOrTable,
    splits: list[int],
    stream: CudaStreamLike | None = None,
) -> list[ColumnOrTable]: ...
def copy_if_else(
    lhs: Column | Scalar,
    rhs: Column | Scalar,
    boolean_mask: Column,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def boolean_mask_scatter(
    input: Table | list[Scalar],
    target: Table,
    boolean_mask: Column,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Table: ...
def get_element(
    input_column: Column,
    index: int,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Scalar: ...
