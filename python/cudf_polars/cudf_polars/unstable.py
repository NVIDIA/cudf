# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""
Utilities for marking unstable functionality.

Based on Polars' util: <https://github.com/pola-rs/polars/blob/8711285f9b7f4139d0e617b856223c9272945c2d/py-polars/src/polars/_utils/unstable.py>
"""

from __future__ import annotations

import inspect
import os
import warnings
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

import polars.exceptions

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import ParamSpec

    P = ParamSpec("P")
    T = TypeVar("T")

__all__: list[str] = ["UnstableWarning", "issue_unstable_warning", "unstable"]

_PKG_DIR = str(Path(__file__).parent)


def _find_stacklevel() -> int:
    """Find the first call-stack frame outside the cudf_polars package."""
    frame = inspect.currentframe()
    n = 0
    try:
        while frame:
            if inspect.getfile(frame).startswith(_PKG_DIR):
                frame = frame.f_back
                n += 1
            else:
                break
    finally:
        # break reference cycle so frames are freed immediately, see
        # https://docs.python.org/3/library/inspect.html#inspect.Traceback
        del frame
    return n


_UNSTABLE_SUFFIX = (
    " It may be changed at any point without it being considered a breaking change."
)

_UNSTABLE_DOCSTRING_BLOCK = (
    "\n\nWarns\n"
    "-----\n"
    "UnstableWarning\n"
    "    This functionality is considered unstable." + _UNSTABLE_SUFFIX + "\n"
)


class UnstableWarning(polars.exceptions.UnstableWarning):
    """Warning issued when unstable cudf-polars functionality is used."""


def issue_unstable_warning(message: str | None = None) -> None:
    """
    Issue a warning for use of unstable functionality.

    The warning is only emitted when the ``CUDF_POLARS_WARN_UNSTABLE`` environment
    variable is set to ``1``; it is silent otherwise.

    Parameters
    ----------
    message
        The message associated with the warning. A standard suffix is always
        appended noting that the API may change without notice.
    """
    if os.environ.get("CUDF_POLARS_WARN_UNSTABLE", "0") != "1":
        return

    if message is None:
        message = "this functionality is considered unstable."
    message += _UNSTABLE_SUFFIX
    warnings.warn(message, UnstableWarning, stacklevel=_find_stacklevel())


def unstable() -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator to mark a function or method as unstable."""

    def decorate(function: Callable[P, T]) -> Callable[P, T]:
        @wraps(function)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            issue_unstable_warning(f"`{function.__name__}` is considered unstable.")
            return function(*args, **kwargs)

        wrapper.__doc__ = (function.__doc__ or "") + _UNSTABLE_DOCSTRING_BLOCK
        wrapper.__signature__ = inspect.signature(function)  # type: ignore[attr-defined]
        return wrapper

    return decorate
