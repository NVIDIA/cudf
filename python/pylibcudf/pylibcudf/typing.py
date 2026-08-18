# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Protocol, TypeAlias, TypedDict

from rmm.pylibrmm.stream import Stream


class HasCudaStream(Protocol):
    def __cuda_stream__(self) -> tuple[int, int]: ...


CudaStreamLike: TypeAlias = Stream | HasCudaStream


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


class SupportsCudaArrayInterface(Protocol):
    @property
    def __cuda_array_interface__(
        self,
    ) -> CudaArrayInterface | dict[str, Any]: ...


class SupportsArrayInterface(Protocol):
    @property
    def __array_interface__(self) -> ArrayInterface | dict[str, Any]: ...


__all__ = [
    "ArrayInterface",
    "ArrayInterfaceBase",
    "CudaArrayInterface",
    "CudaStreamLike",
    "HasCudaStream",
    "SupportsArrayInterface",
    "SupportsCudaArrayInterface",
]
