# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Memory reservations for the RapidsMPF streaming runtime."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from rapidsmpf.memory.buffer import MemoryType
from rapidsmpf.streaming.core.memory_reserve_or_wait import reserve_memory

from cudf_polars.quent._types import (
    MemoryReservationPurpose,
    MemoryReservationRequest,
    Task,
)

if TYPE_CHECKING:
    from rapidsmpf.memory.memory_reservation import MemoryReservation
    from rapidsmpf.streaming.core.context import Context

    from cudf_polars.dsl.ir import IRExecutionContext

__all__ = ["MemoryReservationPurpose", "reserve_memory_traced"]


async def reserve_memory_traced(
    context: Context,
    size: int,
    *,
    net_memory_delta: int,
    ir_context: IRExecutionContext | None,
    purpose: MemoryReservationPurpose,
    sequence_number: int | None = None,
    mem_type: MemoryType = MemoryType.DEVICE,
    allow_overbooking: bool | None = None,
) -> MemoryReservation:
    """
    Reserve memory, recording the wait as a Quent event.

    This is a drop-in replacement for
    :func:`rapidsmpf.streaming.core.memory_reserve_or_wait.reserve_memory` for
    the reservations made on behalf of an IR node. The reservation is recorded
    as a Quent Task bound to that node's operator, so a trace shows which
    operators are waiting on memory admission and for how much.

    Parameters
    ----------
    context
        The rapidsmpf context.
    size
        The number of bytes to reserve.
    net_memory_delta
        The expected lasting change in memory usage. This is smaller than
        ``size`` for operations whose peak usage is transient.
    ir_context
        The execution context for the IR node making the reservation, which
        supplies the Quent operator the reservation is attributed to. ``None``
        for a reservation that can't be attributed to an IR node, in which
        case nothing is recorded.
    purpose
        What the memory is reserved for. Distinguishes reservations made by a
        single operator.
    sequence_number
        The sequence number of the chunk this reservation is for, if any.
    mem_type
        The memory tier to reserve from.
    allow_overbooking
        Whether the runtime may hand out a reservation it cannot back, or
        ``None`` to use the rapidsmpf default.

    Returns
    -------
    The satisfied memory reservation.

    Notes
    -----
    Nothing is recorded unless the query is being traced, in which case this
    behaves exactly like ``reserve_memory``.
    """
    quent_ir_execution_context = (
        None if ir_context is None else ir_context.quent_ir_execution_context
    )
    if quent_ir_execution_context is None:
        return await reserve_memory(
            context,
            size,
            net_memory_delta=net_memory_delta,
            mem_type=mem_type,
            allow_overbooking=allow_overbooking,
        )

    granted = False
    requested_at = time.time_ns()
    try:
        reservation = await reserve_memory(
            context,
            size,
            net_memory_delta=net_memory_delta,
            mem_type=mem_type,
            allow_overbooking=allow_overbooking,
        )
        granted = True
    finally:
        # A reservation that failed still spent time waiting, so record it too.
        request = MemoryReservationRequest(
            purpose=purpose,
            size_bytes=size,
            mem_type=mem_type.name,
            net_memory_delta=net_memory_delta,
            allow_overbooking=allow_overbooking,
            sequence_number=sequence_number,
            granted=granted,
        )
        quent_ir_execution_context.context._emit_memory_reservation_events(
            Task.for_memory_reservation(request, quent_ir_execution_context),
            quent_ir_execution_context,
            request,
            requested_at=requested_at,
            satisfied_at=time.time_ns(),
        )
    return reservation
