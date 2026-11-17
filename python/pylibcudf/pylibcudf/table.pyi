# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from typing import Any

from rmm.pylibrmm.memory_resource import DeviceMemoryResource

from pylibcudf._interop_helpers import ArrowLike, ColumnMetadata
from pylibcudf.column import Column
from pylibcudf.types import DataType
from pylibcudf.utils import CudaStreamLike

class Table:
    def __init__(
        self, columns: Sequence[Column], num_rows: int | None = None
    ): ...
    def num_columns(self) -> int: ...
    def num_rows(self) -> int: ...
    def shape(self) -> tuple[int, int]: ...
    def columns(self) -> tuple[Column, ...]: ...
    def release(self) -> list[Column]: ...
    def copy(
        self,
        stream: CudaStreamLike | None = None,
        mr: DeviceMemoryResource | None = None,
    ) -> Table: ...
    def to_arrow(
        self,
        metadata: list[ColumnMetadata | str] | None = None,
        stream: CudaStreamLike | None = None,
    ) -> ArrowLike: ...
    # Private methods below are included because polars is currently using them,
    # but we want to remove stubs for these private methods eventually
    def _to_schema(self, metadata: Any = None) -> Any: ...
    def _to_host_array(self, stream: CudaStreamLike) -> Any: ...
    @staticmethod
    def from_arrow(
        arrow_like: ArrowLike,
        dtype: DataType | None = None,
        stream: CudaStreamLike | None = None,
        mr: DeviceMemoryResource | None = None,
    ) -> Table: ...
