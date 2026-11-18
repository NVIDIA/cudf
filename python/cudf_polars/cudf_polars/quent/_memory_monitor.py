# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Background sampling of memory resource occupancy."""

from __future__ import annotations

import contextlib
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import uuid
    from collections.abc import Callable, Iterator, Mapping

    import cudf_polars_quent as quent_bindings

    from cudf_polars.quent._runtime import QuentSession

__all__ = ["memory_resource_sampler"]


def memory_resource_usage(
    target: uuid.UUID, nbytes: int
) -> quent_bindings.SampledMemoryResourceUsageRefDict:
    return {
        "target": target,
        "data": {"bytes": nbytes},
    }


@contextlib.contextmanager
def memory_resource_sampler(
    resources: Mapping[uuid.UUID, Callable[[], int]],
    *,
    interval_s: float,
    quent_session: QuentSession,
    query_id: uuid.UUID,
) -> Iterator[None]:
    """
    Sample memory-resource occupancy on one background thread.

    Parameters
    ----------
    resources
        Mapping from each declared sampled-memory-resource ID to a callable
        returning its current allocation in bytes.
    interval_s
        Seconds between readings. Must be positive.
    quent_session
        Quent session to use for declaring the sampler FSM.
    query_id
        UUID identifier for the query. The lifetime of this context should match
        the lifetime of the query execution.

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

    # Declare one sampler FSM per resource on the main thread. A single
    # background thread advances all of them.
    fsms = {
        resource_id: (
            quent_session.binding_context.memory_resource_sampler_observer()
            .handle()
            .sampling(
                query=query_id,
                memory=memory_resource_usage(resource_id, read_allocated()),
            )
        )
        for resource_id, read_allocated in resources.items()
    }
    errors: list[BaseException] = []

    def run() -> None:
        try:
            while not stop_event.wait(interval_s):
                for resource_id, read_allocated in resources.items():
                    fsms[resource_id] = fsms[resource_id].sampling(
                        query=query_id,
                        memory=memory_resource_usage(resource_id, read_allocated()),
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
        for fsm in fsms.values():
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
