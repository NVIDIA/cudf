# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Background sampling of memory resource occupancy."""

from __future__ import annotations

import contextlib
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import uuid
    from collections.abc import Iterator

    import cudf_polars_quent as quent_bindings

    from rapidsmpf.rmm_resource_adaptor import RmmResourceAdaptor

    from cudf_polars.quent._runtime import QuentSession

__all__ = ["memory_resource_sampler"]


def device_memory_usage(
    target: uuid.UUID, nbytes: int
) -> quent_bindings.SampledMemoryResourceUsageRefDict:
    return {
        "target": target,
        "data": {"bytes": nbytes},
    }


@contextlib.contextmanager
def memory_resource_sampler(
    mr: RmmResourceAdaptor,
    *,
    interval_s: float,
    quent_session: QuentSession,
    query_id: uuid.UUID,
    sampled_memory_resource_id: uuid.UUID,
) -> Iterator[None]:
    """
    Context manager that samples device memory usage on a background thread.

    Parameters
    ----------
    mr
        Memory resource to sample.
    interval_s
        Seconds between readings. Must be positive.
    quent_session
        Quent session to use for declaring the sampler FSM.
    query_id
        UUID identifier for the query. The lifetime of this context should match
        the lifetime of the query execution.
    sampled_memory_resource_id
        Sampled memory resource ID to use for sampling. Typically comes from a
        ``WorkerResources``.  It's expected that the memory resource has already
        been declared.

    Notes
    -----
    This function handles the declaration, sampling, and finalization of the
    sampler FSM.

    - The declaration happens on the main thread upon entry.
    - The sampling happens on a background thread.
    - The finalization happens on the main thread upon exit.

    Raises
    ------
    ValueError
        If ``interval_s`` is not positive.
    Exception
        If the background thread raises an exception.
    """
    stop_event = threading.Event()

    # Declare the sampler FSM on the main thread
    fsm = (
        quent_session.binding_context.memory_resource_sampler_observer()
        .handle()
        .sampling(
            query=query_id,
            memory=device_memory_usage(
                sampled_memory_resource_id, mr.current_allocated
            ),
        )
    )
    errors: list[BaseException] = []

    def run() -> None:
        # We reassign fsm to handle state transitions in `.sampling()`.
        nonlocal fsm

        try:
            while not stop_event.wait(interval_s):
                fsm = fsm.sampling(
                    query=query_id,
                    memory=device_memory_usage(
                        sampled_memory_resource_id, mr.current_allocated
                    ),
                )
        except BaseException as error:
            errors.append(error)
            stop_event.set()

    # Start the background monitoring thread.
    thread = threading.Thread(
        target=run, name="cudf-polars-memory-sampler", daemon=True
    )
    thread.start()

    def stop() -> None:
        stop_event.set()
        thread.join()
        try:
            fsm.stopped()
        except BaseException as error:
            errors.append(error)

    try:
        yield
    except BaseException as query_error:
        stop()
        if errors:
            raise BaseExceptionGroup(
                "query and memory resource sampler failed", [query_error, *errors]
            ) from None
        raise
    else:
        stop()
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise BaseExceptionGroup("memory resource sampler failed", errors)
