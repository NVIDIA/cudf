# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixtures for schema-generated Quent telemetry."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import cudf_polars.quent

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def quent_context(tmp_path: Path) -> cudf_polars.quent.QuentConfig:
    return cudf_polars.quent.QuentConfig(
        output_root=str(tmp_path / "quent"),
        query=cudf_polars.quent.QuentQueryConfig(
            query_group_name="test_query_group",
            query_name="test_query",
        ),
    )
