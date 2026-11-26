# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Iterable
from typing import overload

from pylibcudf.table import Table
from typing_extensions import TypeIs

from rapidsmpf.memory.buffer import MemoryType
from rapidsmpf.memory.buffer_resource import BufferResource
from rapidsmpf.memory.memory_reservation import MemoryReservation
from rapidsmpf.memory.packed_data import PackedData
from rapidsmpf.streaming.core.context import Context
from rapidsmpf.streaming.core.message import Message
from rmm.pylibrmm.stream import Stream

class TableChunk:
    @staticmethod
    def from_pylibcudf_table(
        table: Table,
        stream: Stream,
        *,
        exclusive_view: bool,
        br: BufferResource,
    ) -> AvailableTableChunk: ...
    @staticmethod
    def from_packed_data(pd: PackedData, br: BufferResource) -> TableChunk: ...
    @staticmethod
    def from_message(
        message: Message[TableChunk], br: BufferResource
    ) -> TableChunk: ...
    def into_message(
        self, sequence_number: int, message: Message[TableChunk]
    ) -> None: ...
    @property
    def stream(self) -> Stream: ...
    def data_alloc_size(self, mem_type: MemoryType | None = None) -> int: ...
    @staticmethod
    def is_available(chunk: TableChunk) -> TypeIs[AvailableTableChunk]: ...
    def make_available_cost(self) -> int: ...
    def make_available(
        self, reservation: MemoryReservation
    ) -> AvailableTableChunk: ...
    async def make_available_or_wait(
        self, ctx: Context, *, net_memory_delta: int
    ) -> AvailableTableChunk: ...
    def make_available_and_spill(
        self, br: BufferResource, *, allow_overbooking: bool
    ) -> AvailableTableChunk: ...
    def is_spillable(self) -> bool: ...
    def copy(self, reservation: MemoryReservation) -> TableChunk: ...
    def into_packed_data_cost(self) -> int: ...
    def into_packed_data(
        self, reservation: MemoryReservation
    ) -> PackedData: ...
    @property
    def shape(self) -> tuple[int, int]: ...

class AvailableTableChunk(TableChunk):
    """
    A ``TableChunk`` whose data was available on device when it was returned.

    Exists only for type checking. At runtime every chunk is a ``TableChunk``.
    The type does not change when the chunk is consumed by a call such as
    ``make_available()``, or moved into a ``Message`` or ``ChunkStore``. After
    that, ``table_view()`` raises.
    """

    def table_view(self) -> Table: ...

@overload
async def make_table_chunks_available_or_wait(
    ctx: Context,
    chunks: TableChunk,
    *,
    reserve_extra: int,
    net_memory_delta: int,
    allow_overbooking: bool | None = None,
) -> tuple[AvailableTableChunk, MemoryReservation]: ...
@overload
async def make_table_chunks_available_or_wait(
    ctx: Context,
    chunks: Iterable[TableChunk],
    *,
    reserve_extra: int,
    net_memory_delta: int,
    allow_overbooking: bool | None = None,
) -> tuple[list[AvailableTableChunk], MemoryReservation]: ...
