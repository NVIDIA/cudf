# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from cudf_polars.dsl.ir import IRExecutionContext


# Cancelling an await does not stop a running executor worker. Unwinding the
# caller can therefore release resources that the worker is still using.
# The internal offload API lets us coordinate this situation explicitly,
# making the lifetime check independent of native I/O timing or scheduling.
@pytest.mark.parametrize("cancellations", [1, 3])
@pytest.mark.parametrize("worker_raises", [False, True])
def test_to_thread_cancellation_drains_worker(cancellations, worker_raises):
    async def run(executor):
        context = IRExecutionContext(executor)
        loop = asyncio.get_running_loop()
        started = asyncio.Event()
        release = threading.Event()
        finished = threading.Event()
        scope_alive = True

        def worker():
            loop.call_soon_threadsafe(started.set)
            assert release.wait(timeout=10)
            assert scope_alive
            finished.set()
            if worker_raises:
                raise ValueError("worker failed during cancellation")

        async def caller():
            nonlocal scope_alive
            try:
                await context.to_thread(worker)
            finally:
                scope_alive = False

        task = asyncio.create_task(caller())
        await asyncio.wait_for(started.wait(), timeout=10)
        try:
            for _ in range(cancellations):
                task.cancel()
                await asyncio.sleep(0)
                assert not task.done()
                assert scope_alive
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert finished.is_set()
        assert not scope_alive

    with ThreadPoolExecutor(max_workers=1) as executor:
        asyncio.run(run(executor))
