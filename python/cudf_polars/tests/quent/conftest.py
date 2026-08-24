# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixtures for Quent telemetry tracing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import cudf_polars.quent

if TYPE_CHECKING:
    from cudf_polars.quent import QuentContext


@pytest.fixture
def quent_context() -> QuentContext:
    """A Quent Context with a QueryGroup and Query set."""
    return cudf_polars.quent.QuentContext(
        query_group=cudf_polars.quent.QueryGroup(instance_name="test_query_group"),
        query=cudf_polars.quent.Query(instance_name="test_query"),
    )
