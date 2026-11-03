# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from rmm.pylibrmm.memory_resource import DeviceMemoryResource

from pylibcudf.column import Column
from pylibcudf.table import Table
from pylibcudf.types import NullOrder, Order
from pylibcudf.utils import CudaStreamLike

def lower_bound(
    haystack: Table,
    needles: Table,
    column_order: list[Order],
    null_precedence: list[NullOrder],
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def upper_bound(
    haystack: Table,
    needles: Table,
    column_order: list[Order],
    null_precedence: list[NullOrder],
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def contains(
    haystack: Column,
    needles: Column,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
