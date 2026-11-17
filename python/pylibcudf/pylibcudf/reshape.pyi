# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from rmm.pylibrmm.memory_resource import DeviceMemoryResource

from pylibcudf.column import Column
from pylibcudf.table import Table
from pylibcudf.utils import CudaStreamLike

def interleave_columns(
    source_table: Table,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def tile(
    source_table: Table,
    count: int,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Table: ...
def table_to_array(
    input_table: Table,
    ptr: int,
    size: int,
    stream: CudaStreamLike | None = None,
) -> None: ...
