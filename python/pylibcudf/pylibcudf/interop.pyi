# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Any

from rmm.pylibrmm.memory_resource import DeviceMemoryResource

from pylibcudf.table import Table
from pylibcudf.utils import CudaStreamLike

@dataclass
class ColumnMetadata:
    name: str = ...
    timezone: str = ...
    precision: int | None = ...
    children_meta: list[ColumnMetadata] = ...

def from_dlpack(
    managed_tensor: Any,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Table: ...
def to_dlpack(
    input: Table,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Any: ...
