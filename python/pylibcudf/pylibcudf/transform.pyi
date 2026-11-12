# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable, Sequence
from pylibcudf.udf import TransformUDF, UserData

from rmm.pylibrmm.memory_resource import DeviceMemoryResource

from pylibcudf.column import Column
from pylibcudf.expressions import Expression
from pylibcudf.gpumemoryview import gpumemoryview
from pylibcudf.table import Table
from pylibcudf.types import DataType, NullAware, OutputNullability
from pylibcudf.utils import CudaStreamLike

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
    inputs: Sequence[Column | UserData],
    transform_udf: Callable | TransformUDF,
    output_type: DataType,
    is_null_aware: NullAware = NullAware.NO,
    null_policy: OutputNullability = OutputNullability.PRESERVE,
    row_size: int | None = None,
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
