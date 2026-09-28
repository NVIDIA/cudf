# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test-only telemetry for Polars CPU fallback."""

from __future__ import annotations

from contextvars import ContextVar

fallback_used: ContextVar[bool] = ContextVar("cudf_polars_fallback_used", default=False)


def record_fallback() -> None:
    """Record that the active test's query fell back to Polars CPU execution."""
    fallback_used.set(True)
