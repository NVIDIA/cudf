# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence

from rmm.pylibrmm.memory_resource import DeviceMemoryResource

from pylibcudf.column import Column
from pylibcudf.table import Table
from pylibcudf.utils import CudaStreamLike

def concatenate[ColumnOrTable: (Column, Table)](
    objects: Sequence[ColumnOrTable],
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> ColumnOrTable: ...
