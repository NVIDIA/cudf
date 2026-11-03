# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for reliable resource cleanup."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


def run_cleanup_steps(message: str, *steps: Callable[[], object]) -> None:
    """Run every cleanup step and group any failures."""
    exceptions: list[Exception] = []
    for step in steps:
        try:
            step()
        except Exception as error:
            exceptions.append(error)
    if exceptions:
        raise ExceptionGroup(message, exceptions)
