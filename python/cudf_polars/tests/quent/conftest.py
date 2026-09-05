# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixtures for Quent telemetry tracing."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import pytest

import cudf_polars.quent
from cudf_polars.quent._types import Channel, Memory, Processor

if TYPE_CHECKING:
    from cudf_polars.quent import QuentContext


@pytest.fixture
def processor() -> Processor:
    return Processor(pool_id=uuid.uuid4())


@pytest.fixture
def device_memory() -> Memory:
    return Memory(
        instance_name="device",
        resource_type_name="memory",
        parent_group_id=uuid.uuid4(),
    )


@pytest.fixture
def disk_to_device_channel(device_memory: Memory) -> Channel:
    filesystem = Memory(
        instance_name="filesystem",
        resource_type_name="filesystem",
        parent_group_id=uuid.uuid4(),
    )
    return Channel(
        instance_name="disk -> device",
        resource_type_name="DiskToDevice",
        parent_group_id=uuid.uuid4(),
        source=filesystem,
        target=device_memory,
    )


@pytest.fixture
def quent_context() -> QuentContext:
    """A Quent Context with a QueryGroup and Query set."""
    return cudf_polars.quent.QuentContext(
        query_group=cudf_polars.quent.QueryGroup(instance_name="test_query_group"),
        query=cudf_polars.quent.Query(instance_name="test_query"),
    )
