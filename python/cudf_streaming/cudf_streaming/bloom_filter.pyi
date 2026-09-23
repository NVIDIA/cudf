# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from typing import Self

from cudf_streaming.table_chunk import TableChunk
from rapidsmpf.communicator.communicator import Communicator
from rapidsmpf.streaming.core.channel import Channel
from rapidsmpf.streaming.core.context import Context
from rapidsmpf.streaming.core.message import Message

class BloomFilterChunk:
    # Note: if you go looking for this type in the cython bindings, you
    # won't find it. This is purely to provide for better type-checking of
    # the generic Channel argument to BloomFilter.build/apply below.
    @classmethod
    def from_message(cls: type[Self], message: Message[Self]) -> Self: ...
    def into_message(
        self, sequence_number: int, message: Message[Self]
    ) -> None: ...

class BloomFilter:
    def __init__(
        self,
        ctx: Context,
        comm: Communicator,
        seed: int,
        filter_size: int,
    ) -> None: ...
    @property
    def comm(self) -> Communicator: ...
    @staticmethod
    def aligned_size(size: int) -> int: ...
    async def build(
        self,
        ctx: Context,
        ch_in: Channel[TableChunk],
        ch_out: Channel[BloomFilterChunk],
        tag: int,
    ) -> None: ...
    async def apply(
        self,
        ctx: Context,
        bloom_filter: Channel[BloomFilterChunk],
        ch_in: Channel[TableChunk],
        ch_out: Channel[TableChunk],
        keys: Iterable[int],
    ) -> None: ...
