# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Sequence

from rmm.pylibrmm.memory_resource import DeviceMemoryResource

from pylibcudf.column import Column
from pylibcudf.expressions import Expression
from pylibcudf.gpumemoryview import gpumemoryview
from pylibcudf.table import Table
from pylibcudf.types import DataType, NullAware, OutputNullability
from pylibcudf.utils import CudaStreamLike

def nans_to_nulls(
    input: Column,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> tuple[gpumemoryview, int]: ...
def column_nans_to_nulls(
    input: Column,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def compute_column(
    input: Table,
    expr: Expression,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def compute_column_jit(
    input: Table,
    expr: Expression,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def bools_to_mask(
    input: Column,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> tuple[gpumemoryview, int]: ...
def mask_to_bools(
    bitmask: int,
    begin_bit: int,
    end_bit: int,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def transform(
    inputs: Sequence[Column],
    transform_udf: str,
    output_type: DataType,
    is_ptx: bool,
    null_aware: NullAware = NullAware.NO,
    null_policy: OutputNullability = OutputNullability.PRESERVE,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def encode(
    input: Table,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> tuple[Table, Column]: ...
def one_hot_encode(
    input: Column,
    categories: Column,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Table: ...
