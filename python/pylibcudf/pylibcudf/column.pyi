# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable, Sequence
from typing import Any, Protocol, TypedDict

from rmm.pylibrmm.device_buffer import DeviceBuffer
from rmm.pylibrmm.memory_resource import DeviceMemoryResource

from pylibcudf._interop_helpers import ArrowLike, ColumnMetadata
from pylibcudf.scalar import Scalar
from pylibcudf.span import Span
from pylibcudf.types import DataType
from pylibcudf.utils import CudaStreamLike

class ArrayInterfaceBase(TypedDict):
    shape: tuple[int, ...]
    typestr: str
    data: None | tuple[int, bool]
    version: int
    strides: None | tuple[int, ...]
    descr: None | list[tuple[Any, ...]]

class ArrayInterface(ArrayInterfaceBase):
    mask: None | "SupportsArrayInterface"

class CudaArrayInterface(ArrayInterfaceBase):
    stream: None | int
    mask: None | "SupportsCudaArrayInterface"

# Numpy doesn't use a typed dict for their type stubs, they just annotate
# as dict[str, Any]. So do the same here but with a union type so it's
# clearer.
class SupportsCudaArrayInterface(Protocol):
    @property
    def __cuda_array_interface__(
        self,
    ) -> CudaArrayInterface | dict[str, Any]: ...

class SupportsArrayInterface(Protocol):
    @property
    def __array_interface__(self) -> ArrayInterface | dict[str, Any]: ...

class Column:
    def __init__(
        self,
        data_type: DataType,
        size: int,
        data: Span | None,
        mask: Span | None,
        null_count: int,
        offset: int,
        children: Iterable[Column],
        validate: bool = True,
    ) -> None: ...
    def type(self) -> DataType: ...
    def child(self, index: int) -> Column: ...
    def size(self) -> int: ...
    def null_count(self) -> int: ...
    def offset(self) -> int: ...
    def data(self) -> Span | None: ...
    def null_mask(self) -> Span | None: ...
    def children(self) -> list[Column]: ...
    def num_children(self) -> int: ...
    def copy(
        self,
        stream: CudaStreamLike | None = None,
        mr: DeviceMemoryResource | None = None,
    ) -> Column: ...
    def device_buffer_size(self) -> int: ...
    def with_mask(
        self, mask: Span | None, null_count: int, validate: bool = True
    ) -> Column: ...
    def list_view(self) -> ListsColumnView: ...
    def struct_view(self) -> StructsColumnView: ...
    @staticmethod
    def from_scalar(
        scalar: Scalar,
        size: int,
        stream: CudaStreamLike | None = None,
        mr: DeviceMemoryResource | None = None,
    ) -> Column: ...
    def to_scalar(
        self,
        stream: CudaStreamLike | None = None,
        mr: DeviceMemoryResource | None = None,
    ) -> Scalar: ...
    @staticmethod
    def all_null_like(
        like: Column,
        size: int,
        stream: CudaStreamLike | None = None,
        mr: DeviceMemoryResource | None = None,
    ) -> Column: ...
    @staticmethod
    def from_rmm_buffer(
        buff: DeviceBuffer,
        dtype: DataType,
        size: int,
        children: Iterable[Column],
    ) -> Column: ...
    def to_arrow(
        self,
        metadata: ColumnMetadata | str | None = None,
        stream: CudaStreamLike | None = None,
    ) -> ArrowLike: ...
    # Private methods below are included because polars is currently using them,
    # but we want to remove stubs for these private methods eventually
    def _to_schema(self, metadata: Any = None) -> Any: ...
    def _to_host_array(self, stream: CudaStreamLike) -> Any: ...
    @staticmethod
    def from_arrow(
        obj: ArrowLike,
        dtype: DataType | None = None,
        stream: CudaStreamLike | None = None,
        mr: DeviceMemoryResource | None = None,
    ) -> Column: ...
    @classmethod
    def from_cuda_array_interface(
        cls,
        obj: SupportsCudaArrayInterface,
        stream: CudaStreamLike | None = None,
    ) -> Column: ...
    @classmethod
    def from_array_interface(
        cls, obj: SupportsArrayInterface, stream: CudaStreamLike | None = None
    ) -> Column: ...
    @classmethod
    def from_array(
        cls,
        obj: SupportsCudaArrayInterface | SupportsArrayInterface,
        stream: CudaStreamLike | None = None,
    ) -> Column: ...
    @staticmethod
    def struct_from_children(children: Sequence[Column]) -> Column: ...
    @staticmethod
    def from_iterable_of_py(
        obj: Iterable,
        dtype: DataType | None = None,
        stream: CudaStreamLike | None = None,
    ) -> Column: ...
    def to_pylist(self) -> list: ...

class ListsColumnView:
    def __init__(self, column: Column): ...
    def child(self) -> Column: ...
    def offsets(self) -> Column: ...
    def get_sliced_child(
        self, stream: CudaStreamLike | None = None
    ) -> Column: ...

class StructsColumnView:
    def __init__(self, column: Column): ...
    def child(self) -> Column: ...
    def offsets(self) -> Column: ...
    def get_sliced_child(
        self, index: int, stream: CudaStreamLike | None = None
    ) -> Column: ...

def is_c_contiguous(
    shape: Sequence[int], strides: Sequence[int] | None, itemsize: int
) -> bool: ...
