# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test-only observations tied to individual upstream Polars pytest items."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from threading import Lock
from typing import TYPE_CHECKING, ParamSpec, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable
    from contextvars import Token

P = ParamSpec("P")
T = TypeVar("T")


@dataclass
class Observation:
    """Thread-safe observations owned by a single pytest item."""

    signals: dict[str, bool] = field(
        default_factory=lambda: dict.fromkeys(
            ("fallback", "gpu_attempted", "gpu_executed"), False
        )
    )
    active: bool = True
    in_call: bool = False
    lock: Lock = field(default_factory=Lock)
    context_token: Token[Observation | bool | None] | None = None

    def set_in_call(self, in_call: bool) -> None:  # noqa: FBT001
        """Mark whether the item's call phase is running."""
        with self.lock:
            self.in_call = in_call

    def record(self, name: str) -> bool:
        """Ignore callbacks from setup, teardown, or a completed item."""
        # Session- and module-scoped fixtures run their queries during the
        # setup of whichever item requests them first, so only the call phase
        # reflects the item's own GPU usage.
        with self.lock:
            if not self.active or not self.in_call:
                return False
            self.signals[name] = True
            return True

    def snapshot(self) -> dict[str, str]:
        """Return consistent, report-ready signal values."""
        with self.lock:
            return {name: str(value).lower() for name, value in self.signals.items()}


# Rust worker threads do not inherit ContextVars. Within each xdist process only
# one item runs at a time; callbacks capture its record before changing context.
_active: Observation | None = None
_active_lock = Lock()
_bound: ContextVar[Observation | None | bool] = ContextVar(
    "cudf_polars_query_observation", default=False
)


def begin_observation() -> Observation:
    """Install the worker process's active pytest observation."""
    global _active  # noqa: PLW0603
    with _active_lock:
        if _active is not None:
            raise RuntimeError("Overlapping cudf-polars pytest items")
        _active = Observation()
        _active.context_token = _bound.set(_active)
        return _active


def end_observation(observation: Observation) -> None:
    """Retire an item without redirecting its callbacks to another item."""
    global _active  # noqa: PLW0603
    with _active_lock, observation.lock:
        observation.active = False
        if _active is observation:
            _active = None
    if observation.context_token is not None:
        _bound.reset(observation.context_token)
        observation.context_token = None


def _observation() -> Observation | None:
    bound = _bound.get()
    if bound is False:
        with _active_lock:
            return _active
    return bound if isinstance(bound, Observation) else None


def bind_observation(callback: Callable[P, T]) -> Callable[P, T]:  # noqa: UP047
    """Carry query ownership across Polars callback/thread boundaries."""
    observation = _observation()

    @wraps(callback)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> T:
        token = _bound.set(observation)
        try:
            return callback(*args, **kwargs)
        finally:
            _bound.reset(token)

    return wrapped


def record_signal(name: str, signal: ContextVar[bool]) -> None:
    """Preserve standalone signals while attributing injected pytest items."""
    observation = _observation()
    if observation is not None and not observation.record(name):
        return
    if observation is None and _bound.get() is None:
        with _active_lock:
            if _active is not None:
                return
    signal.set(True)
