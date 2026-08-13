# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for :class:`DefaultSingletonEngine`.

Every test body runs inside a worker spawned by the module-scoped
``proc_pool`` fixture. This isolates us from any session-scoped shared
streaming engine that may be live in the parent pytest process and would
otherwise trip the "no other engine alive when default is created" guardrail
in :class:`DefaultSingletonEngine`.
"""

from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable, Generator


# ---------------------------------------------------------------------------
# Subprocess infrastructure
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def proc_pool() -> Generator[ProcessPoolExecutor, None, None]:
    """
    Module-scoped ``ProcessPoolExecutor`` used to run test bodies in isolation.

    Spawn (rather than fork) is used so each worker starts from a clean
    interpreter state — no inherited rapidsmpf Context, no inherited
    pytest fixtures, no inherited GPU resources.
    """
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as pool:
        yield pool


def _reset_singleton_module_state() -> None:
    """Tear down any leftover engine and reset every module-level slot."""
    from cudf_polars.engine import (
        default_singleton_engine as dse,
    )
    from cudf_polars.engine.default_singleton_engine import (
        DefaultSingletonEngine,
    )

    DefaultSingletonEngine.shutdown()  # idempotent; no-op if no live instance
    if dse._state.worker is not None:
        # Construction failed mid-flight (no live instance, but a worker
        # was spawned). Signal it to exit and clear the slot.
        dse._state.worker.shutdown()
        dse._state.worker = None


def _run(body: object, timeout_seconds: int) -> None:
    """
    Subprocess entry point.

    Resets the singleton module state, runs ``body``, and resets again on
    exit so a stuck test doesn't bleed state into the next worker
    invocation. ``body`` must be a top-level callable so it is picklable
    across the spawn boundary.
    """
    _reset_singleton_module_state()
    try:
        body(timeout_seconds)  # type: ignore[operator]
    finally:
        _reset_singleton_module_state()


# ---------------------------------------------------------------------------
# Test bodies
# ---------------------------------------------------------------------------


def _body_lifecycle(timeout_seconds: int) -> None:
    """
    Construction, type, get-or-create, shutdown, context manager, fresh-after-shutdown.
    """
    import pytest

    import polars as pl

    from cudf_polars.engine.default_singleton_engine import (
        DefaultSingletonEngine,
    )
    from cudf_polars.engine.spmd import SPMDEngine

    # Construction succeeds; isinstance check skips its parent's
    # ``check_no_live_default_singleton`` so SPMDEngine.__init__ runs cleanly.
    with DefaultSingletonEngine.get_or_create() as engine:
        assert engine.nranks == 1
        assert engine.rank == 0
        assert isinstance(engine, SPMDEngine)
        assert isinstance(engine, pl.GPUEngine)
        # Get-or-create: a second call returns the same instance.
        assert DefaultSingletonEngine.get_or_create() is engine

    # After context-manager exit, the slot is free.
    e2 = DefaultSingletonEngine.get_or_create()
    assert e2 is not engine
    e2.shutdown()
    e2.shutdown()  # idempotent

    # Comm/context properties raise after shutdown.
    with pytest.raises(RuntimeError, match="shutdown"):
        _ = e2.comm
    with pytest.raises(RuntimeError, match="shutdown"):
        _ = e2.context


def _body_default_path_routing(timeout_seconds: int) -> None:
    """
    Both ``engine="gpu"`` and ``engine=pl.GPUEngine(executor="streaming")``
    route through the singleton, reuse it across queries, and pick up an
    explicit user singleton.
    """
    import polars as pl

    from cudf_polars.engine import (
        default_singleton_engine as dse,
    )
    from cudf_polars.engine.default_singleton_engine import (
        DefaultSingletonEngine,
    )

    # Default path #1: literal ``engine="gpu"`` triggers DefaultSingletonEngine.
    assert dse._state.instance is None
    result = pl.LazyFrame({"a": [1, 2, 3], "b": [4, 5, 6]}).collect(engine="gpu")
    assert result.shape == (3, 2)
    assert result["a"].to_list() == [1, 2, 3]
    first = dse._state.instance
    assert first is not None
    first.shutdown()
    assert dse._state.instance is None

    # Default path #2: vanilla streaming GPUEngine also triggers DefaultSingletonEngine.
    engine = pl.GPUEngine(executor="streaming")
    result = pl.LazyFrame({"a": [1, 2, 3], "b": [4, 5, 6]}).collect(engine=engine)
    assert result.shape == (3, 2)
    assert result["a"].to_list() == [1, 2, 3]

    # Second query reuses the same singleton.
    first = dse._state.instance
    assert first is not None
    pl.LazyFrame({"a": [4, 5, 6]}).collect(engine=engine)
    assert dse._state.instance is first
    first.shutdown()

    # An already-live user singleton is reused by the default path.
    with DefaultSingletonEngine.get_or_create() as user_engine:
        pl.LazyFrame({"a": [1]}).collect(engine=engine)
        assert dse._state.instance is user_engine


def _body_concurrent_warm_path(timeout_seconds: int) -> None:
    """Concurrent ``get_or_create()`` calls return the same instance."""
    import threading

    from cudf_polars.engine.default_singleton_engine import (
        DefaultSingletonEngine,
    )

    main_engine = DefaultSingletonEngine.get_or_create()
    try:
        barrier = threading.Barrier(8, timeout=timeout_seconds)
        results: list[DefaultSingletonEngine] = []
        results_lock = threading.Lock()

        def worker() -> None:
            barrier.wait()
            with results_lock:
                results.append(DefaultSingletonEngine.get_or_create())

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=timeout_seconds)
            assert not t.is_alive(), (
                f"worker thread did not finish within {timeout_seconds}s"
            )

        assert len(results) == 8
        assert all(r is main_engine for r in results)
    finally:
        main_engine.shutdown()


def _body_atexit_no_op(timeout_seconds: int) -> None:
    """
    ``DefaultSingletonEngine.shutdown`` (registered once at import as the
    atexit hook) is a no-op when no engine is live.
    """
    from cudf_polars.engine import (
        default_singleton_engine as dse,
    )
    from cudf_polars.engine.default_singleton_engine import (
        DefaultSingletonEngine,
    )

    assert dse._state.instance is None
    DefaultSingletonEngine.shutdown()  # no-op
    assert dse._state.instance is None

    # Construct, shutdown, then call the atexit hook again — still a no-op.
    DefaultSingletonEngine.get_or_create().shutdown()
    assert dse._state.instance is None
    DefaultSingletonEngine.shutdown()
    assert dse._state.instance is None


def _body_singleton_blocked_when_explicit_alive(timeout_seconds: int) -> None:
    """
    The reverse direction: ``get_or_create()`` refuses if any other
    ``StreamingEngine`` is alive when ``DefaultSingletonEngine.__init__``
    finishes. Plus the ``_active_engine_count`` accessor.
    """
    import pytest

    from cudf_polars.engine.core import StreamingEngine
    from cudf_polars.engine.default_singleton_engine import (
        DefaultSingletonEngine,
    )
    from cudf_polars.engine.spmd import SPMDEngine

    assert StreamingEngine._active_engine_count() == 0

    # A real explicit SPMDEngine blocks the default.
    with (
        SPMDEngine(),
        pytest.raises(RuntimeError, match="explicit streaming engine"),
    ):
        DefaultSingletonEngine.get_or_create()
    assert StreamingEngine._active_engine_count() == 0

    # Active-count tracks DefaultSingletonEngine's own lifecycle too.
    with DefaultSingletonEngine.get_or_create():
        assert StreamingEngine._active_engine_count() == 1
    assert StreamingEngine._active_engine_count() == 0


def _body_worker_thread_isolation(timeout_seconds: int) -> None:
    """
    The dedicated worker thread owns construction and shutdown.

    - Construction runs on the named worker thread, not the caller's.
    - ``shutdown`` from a non-creator thread (i.e. the test main thread)
      doesn't crash, because the teardown is dispatched back to the
      worker.
    - If construction raises on the worker, the caller sees the same
      exception and the slot is reset for retry.
    """
    import threading
    from typing import Any
    from unittest.mock import patch

    import pytest

    from cudf_polars.engine import (
        default_singleton_engine as dse,
    )
    from cudf_polars.engine.default_singleton_engine import (
        DefaultSingletonEngine,
    )
    from cudf_polars.engine.spmd import SPMDEngine

    # 1) Construction runs on the worker thread.
    recorded: dict[str, threading.Thread] = {}
    real_init = SPMDEngine.__init__

    def recording_init(self: SPMDEngine, **kwargs: Any) -> None:
        recorded["thread"] = threading.current_thread()
        real_init(self, **kwargs)

    with patch.object(SPMDEngine, "__init__", recording_init):
        engine = DefaultSingletonEngine.get_or_create()
    try:
        assert recorded["thread"] is not threading.current_thread()
        assert recorded["thread"].name.startswith("default-singleton-engine")
    finally:
        engine.shutdown()

    # 2) Cross-thread shutdown via the worker is safe.
    create_done = threading.Event()

    def creator() -> None:
        DefaultSingletonEngine.get_or_create()
        create_done.set()

    t = threading.Thread(target=creator)
    t.start()
    assert create_done.wait(timeout=timeout_seconds), (
        f"creator did not signal completion within {timeout_seconds}s"
    )
    t.join(timeout=timeout_seconds)
    assert not t.is_alive(), f"creator thread did not finish within {timeout_seconds}s"
    live = dse._state.instance
    assert live is not None
    live.shutdown()  # different thread than the one that constructed
    assert dse._state.instance is None
    assert dse._state.worker is None

    # 3) Construction error propagates and resets state for a retry.
    def broken_init(self: SPMDEngine, **kwargs: object) -> None:
        raise RuntimeError("synthetic boom")

    with (
        patch.object(SPMDEngine, "__init__", broken_init),
        pytest.raises(RuntimeError, match="synthetic boom"),
    ):
        DefaultSingletonEngine.get_or_create()
    assert dse._state.instance is None
    assert dse._state.worker is None
    DefaultSingletonEngine.get_or_create().shutdown()  # retry succeeds


def _body_shutdown_timeout(timeout_seconds: int) -> None:
    """
    A hung ``SPMDEngine.shutdown`` causes the timeout branch to fire:
    a warning is emitted, the singleton slot is cleared, and any new
    construction is refused until the leaked worker eventually returns.
    """
    import threading
    from unittest.mock import patch

    import pytest

    from cudf_polars.engine import (
        default_singleton_engine as dse,
    )
    from cudf_polars.engine.default_singleton_engine import (
        DefaultSingletonEngine,
    )
    from cudf_polars.engine.spmd import SPMDEngine

    release_worker = threading.Event()
    real_done = threading.Event()
    real_shutdown = SPMDEngine.shutdown

    with patch.object(dse, "SHUTDOWN_TIMEOUT_SECONDS", 0.1):
        engine = DefaultSingletonEngine.get_or_create()

        def hanging_super_shutdown(self: SPMDEngine) -> None:
            # Only the original ``engine`` should hang; any other
            # ``SPMDEngine.shutdown`` call (e.g. the self-cleanup path
            # inside ``DefaultSingletonEngine.__init__`` when the
            # leaked instance still occupies ``_active_engines``) must
            # run normally, otherwise the test deadlocks.
            if self is not engine:
                real_shutdown(self)
                return
            assert release_worker.wait(timeout=timeout_seconds), (
                f"release_worker did not signal completion within {timeout_seconds}s"
            )
            try:
                # Run the real teardown so the rapidsmpf Context is
                # destroyed on the construction (worker) thread, otherwise
                # GC of the engine on the wrong thread crashes with
                # "Context::shutdown() called from a different thread...".
                real_shutdown(self)
            finally:
                real_done.set()

        with patch.object(SPMDEngine, "shutdown", hanging_super_shutdown):
            with pytest.warns(UserWarning, match="did not complete within"):
                engine.shutdown()
            # Slot cleared, but the leaked engine is still in the
            # active-engine registry — no new construction allowed.
            assert dse._state.instance is None
            assert dse._state.worker is None
            with pytest.raises(RuntimeError, match="explicit streaming engine"):
                DefaultSingletonEngine.get_or_create()
            # Once the leaked worker returns it removes itself from the
            # registry; a fresh construction is allowed again.
            release_worker.set()
            assert real_done.wait(timeout=timeout_seconds), (
                f"real_done did not signal completion within {timeout_seconds}s"
            )
            DefaultSingletonEngine.get_or_create().shutdown()


# ---------------------------------------------------------------------------
# Single parametrized entry point. Each body runs in a fresh subprocess via
# ``proc_pool``; failure messages identify the body by name (``[lifecycle]``,
# ``[shutdown_timeout]``, …), and ``pytest -k <name>`` matches as expected.
# ---------------------------------------------------------------------------


_ALL_BODIES = [
    _body_lifecycle,
    _body_default_path_routing,
    _body_concurrent_warm_path,
    _body_atexit_no_op,
    _body_singleton_blocked_when_explicit_alive,
    _body_worker_thread_isolation,
    _body_shutdown_timeout,
]


@pytest.mark.parametrize(
    "body", _ALL_BODIES, ids=lambda b: b.__name__.removeprefix("_body_")
)
def test_default_singleton_engine(
    proc_pool: ProcessPoolExecutor, body: Callable[[], None], timeout_seconds: int
) -> None:
    """Run each ``_body_*`` function in an isolated subprocess."""
    proc_pool.submit(_run, body, timeout_seconds).result(timeout=timeout_seconds)
