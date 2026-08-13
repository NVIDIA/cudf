# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Quent telemetry tracing."""

from __future__ import annotations

import collections
import logging
import threading
from typing import TYPE_CHECKING, Any

try:
    import structlog
except ImportError:  # pragma: no cover; requires no structlog
    _HAS_STRUCTLOG = False
else:
    _HAS_STRUCTLOG = True

if TYPE_CHECKING:
    from cudf_polars.quent._types import Event

QUENT_SCOPE = "QUENT"


class QuentLogger:
    """
    An in-memory buffer for Quent events.

    Parameters
    ----------
    maxlen
        The maximum number of events to buffer.
    """

    def __init__(self, maxlen: int = 100_000) -> None:
        self._buffer: collections.deque[dict[str, Any]] = collections.deque(
            maxlen=maxlen
        )
        self._lock = threading.Lock()

    if _HAS_STRUCTLOG:  # pragma: no cover; depends on structlog

        def _get_logger(self, **initial_values: Any) -> Any:
            return structlog.wrap_logger(
                DequeLogger(self._buffer, self._lock),
                processors=[
                    structlog.processors.add_log_level,
                    structlog.processors.TimeStamper(fmt="iso"),
                    collect_to_deque,
                ],
                wrapper_class=structlog.stdlib.BoundLogger,
                **initial_values,
            )
    else:

        def _get_logger(
            self, **initial_values: Any
        ) -> Any:  # pragma: no cover; depends on no structlog
            return logging.getLogger(__name__)

    def emit(self, event: Event) -> None:
        logger = self._get_logger()
        kwargs = {}
        if _HAS_STRUCTLOG:
            kwargs = {"scope": QUENT_SCOPE}
        logger.info(event.to_dict(), **kwargs)

    def drain(self) -> list[dict[str, Any]]:
        with self._lock:
            events = list(self._buffer)
            self._buffer.clear()
        return events


class DequeLogger:
    def __init__(self, deque: collections.deque[dict[str, Any]], lock: threading.Lock):
        self._deque = deque
        self._lock = lock

    def msg(self, **kwargs: Any) -> None:
        with self._lock:
            self._deque.append(kwargs)

    # aliases for log levels
    log = debug = info = warn = warning = error = critical = fatal = msg


def collect_to_deque(
    logger: structlog.types.WrappedLogger,
    method_name: str,
    event_dict: structlog.types.EventDict,
) -> structlog.types.EventDict:
    return event_dict
