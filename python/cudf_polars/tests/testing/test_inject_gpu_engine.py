# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest

from cudf_polars.testing.inject_gpu_engine import (
    partition_items_by_shard,
    validate_shard_options,
)


def make_items(num_items: int) -> list[pytest.Item]:
    """Create minimal pytest-item stand-ins for sharding tests."""
    return cast(
        "list[pytest.Item]",
        [SimpleNamespace(nodeid=f"test_{index}") for index in range(num_items)],
    )


def test_partition_items_by_shard_is_exhaustive_and_disjoint() -> None:
    items = make_items(100)

    shard_nodeids = [
        {item.nodeid for item in partition_items_by_shard(items, shard_id, 3)[0]}
        for shard_id in range(3)
    ]

    assert set().union(*shard_nodeids) == {item.nodeid for item in items}
    assert not (shard_nodeids[0] & shard_nodeids[1])
    assert not (shard_nodeids[0] & shard_nodeids[2])
    assert not (shard_nodeids[1] & shard_nodeids[2])


def test_partition_items_by_shard_is_deterministic() -> None:
    items = make_items(100)

    first, _ = partition_items_by_shard(items, 1, 3)
    second, _ = partition_items_by_shard(items, 1, 3)

    assert first == second


def test_partition_items_by_shard_defaults_to_the_full_suite() -> None:
    items = make_items(10)

    selected, deselected = partition_items_by_shard(items, 0, 1)

    assert selected == items
    assert not deselected


@pytest.mark.parametrize(
    "shard_id,num_shards",
    [
        (0, 0),
        (1, 1),
        (3, 3),
    ],
)
def test_validate_shard_options_rejects_invalid_values(
    shard_id: int, num_shards: int
) -> None:
    with pytest.raises(pytest.UsageError):
        validate_shard_options(shard_id, num_shards)
