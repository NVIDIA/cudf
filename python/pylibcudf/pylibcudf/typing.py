# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Protocol, TypeAlias, TypedDict

from cuda.bindings.runtime import cudaStream_t

from rmm.pylibrmm.stream import Stream


class HasCudaStream(Protocol):
    def __cuda_stream__(self) -> object: ...


CudaStreamLike: TypeAlias = Stream | HasCudaStream | cudaStream_t


class ArrayInterfaceBase(TypedDict):
    data: tuple[int, bool]
    shape: tuple[int, ...]
    typestr: str
    version: int


class SupportsCudaArrayInterface(Protocol):
    @property
    def __cuda_array_interface__(self) -> ArrayInterfaceBase: ...


class SupportsArrayInterface(Protocol):
    @property
    def __array_interface__(self) -> ArrayInterfaceBase: ...


__all__ = [
    "ArrayInterfaceBase",
    "CudaStreamLike",
    "HasCudaStream",
    "SupportsArrayInterface",
    "SupportsCudaArrayInterface",
]
