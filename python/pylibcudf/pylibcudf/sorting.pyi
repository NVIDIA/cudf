# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from rmm.pylibrmm.memory_resource import DeviceMemoryResource

from pylibcudf.aggregation import RankMethod
from pylibcudf.column import Column
from pylibcudf.table import Table
from pylibcudf.types import NullOrder, NullPolicy, Order
from pylibcudf.utils import CudaStreamLike

def sorted_order(
    source_table: Table,
    column_order: list[Order],
    null_precedence: list[NullOrder],
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def stable_sorted_order(
    source_table: Table,
    column_order: list[Order],
    null_precedence: list[NullOrder],
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def rank(
    input_view: Column,
    method: RankMethod,
    column_order: Order,
    null_handling: NullPolicy,
    null_precedence: NullOrder,
    percentage: bool,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def is_sorted(
    tbl: Table,
    column_order: list[Order],
    null_precedence: list[NullOrder],
    stream: CudaStreamLike | None = None,
) -> bool: ...
def segmented_sort_by_key(
    values: Table,
    keys: Table,
    segment_offsets: Column,
    column_order: list[Order],
    null_precedence: list[NullOrder],
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Table: ...
def stable_segmented_sort_by_key(
    values: Table,
    keys: Table,
    segment_offsets: Column,
    column_order: list[Order],
    null_precedence: list[NullOrder],
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Table: ...
def sort_by_key(
    values: Table,
    keys: Table,
    column_order: list[Order],
    null_precedence: list[NullOrder],
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Table: ...
def stable_sort_by_key(
    values: Table,
    keys: Table,
    column_order: list[Order],
    null_precedence: list[NullOrder],
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Table: ...
def sort(
    source_table: Table,
    column_order: list[Order],
    null_precedence: list[NullOrder],
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Table: ...
def stable_sort(
    source_table: Table,
    column_order: list[Order],
    null_precedence: list[NullOrder],
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Table: ...
def top_k(
    col: Column,
    k: int,
    sort_order: Order = Order.DESCENDING,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def top_k_order(
    col: Column,
    k: int,
    sort_order: Order = Order.DESCENDING,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
