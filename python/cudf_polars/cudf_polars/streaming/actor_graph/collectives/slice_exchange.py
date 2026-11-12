# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exchange whole chunks or row slices between ranks."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pylibcudf as plc
from cudf_streaming.table_chunk import (
    TableChunk,
    make_table_chunks_available_or_wait,
)
from rapidsmpf.memory.buffer import MemoryType
from rapidsmpf.streaming.coll.sparse_alltoall import SparseAlltoall
from rapidsmpf.streaming.core.message import Message

from cudf_polars.streaming.actor_graph.utils import ChunkStore

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.streaming.core.context import Context


@dataclass(frozen=True)
class SliceRoute:
    """A half-open row slice sent to one remote rank."""

    destination: int
    start: int
    stop: int


class SparseExchange:
    """
    Send remote row slices and whole chunks through a sparse all-to-all.

    Each destination must list the sender as a source. Callers must derive
    both peer sets from the same collective plan.
    """

    def __init__(
        self,
        context: Context,
        comm: Communicator,
        collective_id: int,
        *,
        sources: Sequence[int],
        destinations: Sequence[int],
    ) -> None:
        sources = tuple(sources)
        destinations = tuple(destinations)
        self.context = context
        self.destinations = frozenset(destinations)
        self._pending_sources = set(sources)
        self._exchange = (
            SparseAlltoall(
                context,
                comm,
                collective_id,
                srcs=sources,
                dsts=destinations,
            )
            if sources or destinations
            else None
        )
        self._finished = False
        self._closed = False
        self._finish_task: asyncio.Task[None] | None = None

    async def __aenter__(self) -> SparseExchange:
        """Return this exchange."""
        return self

    async def __aexit__(
        self,
        _exc_type: object,
        exc: object,
        _traceback: object,
    ) -> None:
        """Close this exchange."""
        if exc is None:
            await self.aclose()
        else:
            with suppress(BaseException):
                await self.aclose()

    async def insert_slices(
        self,
        chunk: TableChunk,
        routes: Sequence[SliceRoute],
    ) -> None:
        """Insert remote row slices without consuming an available chunk."""
        if self._finish_task is not None or self._finished or self._closed:
            raise RuntimeError("Cannot insert after finishing a sparse exchange")
        if not routes:
            return
        if not chunk.is_available():
            raise ValueError("insert_slices requires an available TableChunk")

        nrows = chunk.shape[0]
        for route in routes:
            if route.destination not in self.destinations:
                raise ValueError(f"Unexpected destination rank {route.destination}")
            if not 0 <= route.start <= route.stop <= nrows:
                raise ValueError(f"Invalid row slice [{route.start}, {route.stop})")
        routes = tuple(route for route in routes if route.start < route.stop)
        if not routes:
            return

        table = chunk.table_view()
        for route in routes:
            piece = (
                plc.Table(table.columns())
                if route.start == 0 and route.stop == nrows
                else plc.copying.slice(
                    table,
                    [route.start, route.stop],
                    stream=chunk.stream,
                )[0]
            )
            await self._insert_chunk(
                route.destination,
                TableChunk.from_pylibcudf_table(
                    piece,
                    chunk.stream,
                    exclusive_view=False,
                    br=self.context.br(),
                ),
            )

    async def insert_chunk(self, destination: int, chunk: TableChunk) -> None:
        """Consume and send one whole chunk to a remote rank."""
        if self._finish_task is not None or self._finished or self._closed:
            raise RuntimeError("Cannot insert after finishing a sparse exchange")
        if destination not in self.destinations:
            raise ValueError(f"Unexpected destination rank {destination}")
        await self._insert_chunk(destination, chunk)

    async def _insert_chunk(self, destination: int, chunk: TableChunk) -> None:
        """Pack and insert one whole chunk."""
        assert self._exchange is not None
        pack_cost = chunk.into_packed_data_cost()
        if pack_cost:
            chunk, reservation = await make_table_chunks_available_or_wait(
                self.context,
                chunk,
                reserve_extra=pack_cost,
                net_memory_delta=0,
            )
        else:
            reservation = await self.context.memory(MemoryType.DEVICE).reserve_or_wait(
                0, net_memory_delta=0
            )
        self._exchange.insert(destination, chunk.into_packed_data(reservation))

    def start_finish(self) -> None:
        """Signal that no more payloads will be inserted without waiting for peers."""
        if self._closed:
            raise RuntimeError("Sparse exchange is closed")
        if self._finish_task is None and self._exchange is not None:
            self._finish_task = asyncio.create_task(
                self._exchange.insert_finished(self.context)
            )

    async def finish(self) -> None:
        """Finish insertion before extraction, even under cancellation."""
        if self._finished:
            return
        self.start_finish()
        cancelled = False
        if self._exchange is not None:
            assert self._finish_task is not None
            # The detached C++ task borrows the exchange pointer. Keep it alive
            # through cancellation until the task has actually completed.
            while not self._finish_task.done():
                try:
                    await asyncio.shield(self._finish_task)
                except asyncio.CancelledError:
                    cancelled = True
            self._finish_task.result()
        self._finished = True
        if cancelled:
            raise asyncio.CancelledError

    def extract(self, source: int) -> ChunkStore:
        """
        Extract one source's payloads into spillable storage.

        The caller must drain or clear the returned store.
        """
        if not self._finished or self._closed:
            raise RuntimeError("Sparse exchange must be finished before extraction")
        if source not in self._pending_sources:
            raise ValueError(f"Unexpected or repeated source rank {source}")
        assert self._exchange is not None
        result = ChunkStore(self.context)
        try:
            for packed in self._exchange.extract(source):
                result.insert(
                    Message(0, TableChunk.from_packed_data(packed, self.context.br()))
                )
        except BaseException:
            with suppress(BaseException):
                result.clear()
            raise
        self._pending_sources.remove(source)
        return result

    async def aclose(self) -> None:
        """Finish insertion and release all unconsumed remote payloads."""
        if self._closed:
            return
        try:
            await self.finish()
        finally:
            self._closed = True
            self._exchange = None
            self._pending_sources.clear()
