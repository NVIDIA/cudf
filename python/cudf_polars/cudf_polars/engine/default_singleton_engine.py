# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single-GPU, single-instance specialization of :class:`~cudf_polars.engine.spmd.SPMDEngine`."""

from __future__ import annotations

import atexit
import dataclasses
import threading
import warnings
from concurrent.futures import Future
from typing import TYPE_CHECKING, Any

from rapidsmpf.communicator.single import (
    new_communicator as single_communicator,
)
from rapidsmpf.progress_thread import ProgressThread
from rapidsmpf.statistics import Statistics

from cudf_polars.engine.core import (
    resolve_rapidsmpf_options,
)
from cudf_polars.engine.spmd import SPMDEngine

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["DefaultSingletonEngine", "check_no_live_default_singleton"]


def _set_future(fut: Future[Any], fn: Callable[[], Any]) -> bool:
    """
    Run ``fn`` and pipe its result or exception into ``fut``.

    Parameters
    ----------
    fut
        The future to populate with ``fn``'s outcome.
    fn
        Zero-argument callable to invoke.

    Returns
    -------
    ``True`` if ``fn`` returned normally, ``False`` if it raised or
    the future was already cancelled.
    """
    if not fut.set_running_or_notify_cancel():
        return False
    try:
        fut.set_result(fn())
    except BaseException as exc:
        fut.set_exception(exc)
        return False
    return True


class _DaemonWorker:
    """
    Single-shot daemon thread that owns the live engine lifecycle.

    Builds :class:`DefaultSingletonEngine` once on the worker thread.
    After construction completes, blocks until :meth:`shutdown` is
    called, then tears down the live engine on the same thread and
    exits. Both phases therefore run on the same thread, which rapidsmpf
    requires.

    Notes
    -----
    We deliberately do not use ``ThreadPoolExecutor`` here, even though
    a single-worker pool would otherwise be a natural fit.

    The rapidsmpf ``Context`` must be torn down on the same thread that
    constructed it; otherwise rapidsmpf calls ``std::terminate``. The
    engine lifecycle therefore has to live on a single dedicated worker
    thread.

    ``ThreadPoolExecutor`` registers workers in internal shutdown
    machinery and installs a shutdown hook that runs before regular
    ``atexit`` handlers. By the time our own teardown hook runs,
    ``executor.submit`` already raises::

        RuntimeError: cannot schedule new futures after shutdown

    That prevents us from scheduling teardown work onto the construction
    thread. The rapidsmpf ``Context`` is then later finalized on the
    wrong thread, which triggers ``std::terminate``.

    A daemon :class:`threading.Thread` avoids this problem because it
    is not managed by the executor shutdown machinery. The thread stays
    alive until late interpreter finalization, after all ``atexit``
    hooks have run, which gives our teardown hook a chance to enqueue
    cleanup work onto the correct thread.

    Parameters
    ----------
    name
        Thread name passed to :class:`threading.Thread`.

    Attributes
    ----------
    startup_future
        Resolves with the constructed :class:`DefaultSingletonEngine`,
        or with the exception raised during construction.
    shutdown_future
        Resolves once teardown completes, or with the exception raised
        during teardown.
    """

    def __init__(self, name: str) -> None:
        self._shutdown_signal = threading.Event()
        self.startup_future: Future[DefaultSingletonEngine] = Future()
        self.shutdown_future: Future[None] = Future()
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    def shutdown(self) -> Future[None]:
        """
        Signal the worker to tear down the live engine and exit.

        Returns
        -------
        Resolves once the worker has torn down the engine.
        """
        self._shutdown_signal.set()
        return self.shutdown_future

    def _run(self) -> None:
        """Worker thread entry point: build engine, wait, tear it down."""
        if not _set_future(self.startup_future, _build_engine):
            return
        self._shutdown_signal.wait()
        _set_future(self.shutdown_future, _teardown_engine)


def _build_engine() -> DefaultSingletonEngine:
    """
    Construct the live default engine on the dedicated worker thread.

    Returns
    -------
    The newly constructed instance, also assigned to ``_state.instance``.
    """
    with _state.lock:
        try:
            options = resolve_rapidsmpf_options(None)
            statistics = Statistics.from_options(options)
            comm = single_communicator(
                progress_thread=ProgressThread(statistics),
                options=options,
            )
            instance = DefaultSingletonEngine(comm=comm)
            assert instance.nranks == 1
        except BaseException:
            _state.worker = None
            raise
        _state.instance = instance
        return instance


def _teardown_engine() -> None:
    """Tear down ``_state.instance`` on the dedicated worker thread."""
    with _state.lock:
        instance = _state.instance
        _state.instance = None
        _state.worker = None
    if instance is not None:
        SPMDEngine.shutdown(instance)


@dataclasses.dataclass
class _SingletonState:
    """
    Module-level singleton bookkeeping.

    Attributes
    ----------
    instance
        The live :class:`DefaultSingletonEngine`, if one exists.
    worker
        Worker thread that owns the engine lifecycle.
    lock
        Protects mutations to :attr:`instance` and :attr:`worker`.

        NB: Code must not hold this lock across calls to
        ``SPMDEngine.shutdown``.
    """

    instance: DefaultSingletonEngine | None = None
    worker: _DaemonWorker | None = None
    lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)


_state = _SingletonState()
SHUTDOWN_TIMEOUT_SECONDS: float = 10.0


def check_no_live_default_singleton(self_engine: Any) -> None:
    """
    Raise if the default singleton engine is alive.

    Parameters
    ----------
    self_engine
        The engine instance being constructed.

    Raises
    ------
    RuntimeError
        If a :class:`DefaultSingletonEngine` is currently alive and
        ``self_engine`` is not itself a :class:`DefaultSingletonEngine`.
    """
    if isinstance(self_engine, DefaultSingletonEngine):
        return
    with _state.lock:
        if _state.instance is not None:
            raise RuntimeError(
                f"Cannot construct {type(self_engine).__name__} while the "
                'default GPU engine (e.g. `.collect(engine="gpu")`) is '
                "active. While the default engine is in use, no explicit "
                "streaming engines may exist. Shut down the default engine "
                "first by calling `DefaultSingletonEngine.shutdown()`."
            )


class DefaultSingletonEngine(SPMDEngine):
    """
    Process-wide single-GPU singleton specialization of :class:`~cudf_polars.engine.spmd.SPMDEngine`.

    At most one live instance exists per process. Use :meth:`get_or_create`
    to obtain it and :meth:`shutdown` to tear it down.

    Always constructs a single-rank communicator and uses default RapidsMPF,
    executor, and engine settings from the environment.

    Users needing custom configuration should construct an engine explicitly.
    See :class:`~cudf_polars.engine.ray.RayEngine`, :class:`~cudf_polars.engine.dask.DaskEngine`,
    and :class:`~cudf_polars.engine.spmd.SPMDEngine`.

    Examples
    --------
    Constructed automatically when using ``engine="gpu"``:

    >>> result = df.lazy().collect(engine="gpu")  # doctest: +SKIP

    Or constructed explicitly:

    >>> engine = DefaultSingletonEngine.get_or_create()  # doctest: +SKIP
    >>> result = df.lazy().collect(engine=engine)  # doctest: +SKIP
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # super().__init__ registered self in _active_engines. If another engine
        # is already alive, tear down this just-built rapidsmpf Context ON THIS
        # THREAD before raising. Otherwise, GC of the partially constructed
        # instance could destroy the Context off-thread, violating rapidsmpf's
        # same-thread teardown invariant.
        active_count = SPMDEngine._active_engine_count()
        if active_count > 1:
            SPMDEngine.shutdown(self)
            raise RuntimeError(
                f"Cannot start the default GPU engine (e.g. "
                f'`.collect(engine="gpu")`) while {active_count - 1} '
                "explicit streaming engine(s) are alive. While "
                "explicit engines are in use, the default engine "
                "cannot also exist. Shut them down first or exit "
                "their `with` blocks."
            )

    @classmethod
    def get_or_create(cls) -> DefaultSingletonEngine:
        """
        Return the live singleton, constructing one if needed.

        Construction runs on a dedicated worker thread so the rapidsmpf
        ``Context`` is born on the same thread that will eventually tear
        it down.

        Raises
        ------
        RuntimeError
            If any other :class:`~cudf_polars.engine.core.StreamingEngine` is currently alive.
        """
        with _state.lock:
            if _state.instance is not None:
                return _state.instance
            if _state.worker is None:
                _state.worker = _DaemonWorker(name="default-singleton-engine")
            worker = _state.worker
        return worker.startup_future.result()

    @staticmethod
    def shutdown() -> None:
        """
        Shut down the live default singleton, if any. Idempotent.

        Submits teardown to the dedicated worker thread, the same thread
        that constructed the rapidsmpf ``Context``, and waits up to
        ``SHUTDOWN_TIMEOUT_SECONDS`` seconds.
        """
        with _state.lock:
            instance = _state.instance
            worker = _state.worker
        if instance is None:
            return
        assert worker is not None
        future = worker.shutdown()
        try:
            future.result(timeout=SHUTDOWN_TIMEOUT_SECONDS)
        except TimeoutError:
            pass
        else:
            # _teardown_engine already cleared _state.instance and _state.worker.
            return

        # Timeout fallback: the worker is hung mid-teardown and did not
        # clear the singleton slots itself. Clear them here so a fresh
        # get_or_create() call can spawn a new worker.
        #
        # The leaked instance is intentionally left in _active_engines.
        # That prevents subsequent get_or_create() calls, or any other
        # streaming engine, from starting while the previous rapidsmpf
        # Context remains in an indeterminate state.
        with _state.lock:
            if _state.worker is worker:
                _state.instance = None
                _state.worker = None
        warnings.warn(
            f"DefaultSingletonEngine shutdown did not complete within "
            f"{SHUTDOWN_TIMEOUT_SECONDS}s; the worker thread is leaked "
            "and rapidsmpf resources may not have been released. "
            "No new streaming engine can be created in this process "
            "until the leaked worker eventually returns.",
            stacklevel=2,
        )


# Register once at module import. The hook is a no-op when no engine is
# live, so it costs nothing if the user never touches the default engine.
atexit.register(DefaultSingletonEngine.shutdown)
