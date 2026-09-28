# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from pylibcudf._interop_helpers import ColumnMetadata
from pylibcudf.column import Column
from pylibcudf.types import DataType
from pylibcudf.utils import CudaStreamLike

NpGeneric = type[Any]

PaScalar = type[Any]

class Scalar:
    def __init__(self): ...
    def type(self) -> DataType: ...
    def is_valid(self, stream: CudaStreamLike | None = None) -> bool: ...
    def to_arrow(
        self,
        metadata: ColumnMetadata | str | None = None,
        stream: CudaStreamLike | None = None,
    ) -> PaScalar: ...
    @staticmethod
    def from_arrow(
        pa_val: Any,
        dtype: DataType | None = None,
        stream: CudaStreamLike | None = None,
    ) -> Scalar: ...
    @classmethod
    def from_py(
        cls,
        py_val: Any,
        dtype: DataType | None = None,
        stream: CudaStreamLike | None = None,
    ) -> Scalar: ...
    @classmethod
    def from_numpy(
        cls, np_val: NpGeneric, stream: CudaStreamLike | None = None
    ) -> Scalar: ...
    def to_py(
        self, stream: CudaStreamLike | None = None
    ) -> None | int | float | str | bool: ...
