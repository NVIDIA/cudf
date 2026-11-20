# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import random
import threading
import uuid
from types import SimpleNamespace
from typing import Any

import pytest

import polars as pl

import cudf_polars.engine.core
from cudf_polars.dsl.ir import Union
from cudf_polars.dsl.translate import Translator
from cudf_polars.dsl.traversal import traversal
from cudf_polars.streaming.actor_graph.collectives import (
    ReserveOpIDs,
    collective_id_scope,
    common,
    reserve_op_id,
)


def _sort_ir():
    q = pl.LazyFrame({"a": [3, 1, 2]}).sort("a")
    return Translator(q._ldf.visit(), pl.GPUEngine()).translate_ir()


def test_collective_id_pool_allocation_order(monkeypatch) -> None:
    monkeypatch.setattr(common, "Shuffler", SimpleNamespace(max_concurrent_shuffles=8))
    pool = common._CollectiveIDPool()
    reference = dict.fromkeys(range(8))
    acquired: set[int] = set()
    rng = random.Random(0)

    # Rank agreement depends on retaining the old allocation order even when
    # IDs are released out of order and then reserved again.
    for _ in range(1000):
        if acquired and (not reference or rng.random() < 0.5):
            op_id = rng.choice(sorted(acquired))
            acquired.remove(op_id)
            reference[op_id] = None
            pool.release_unsafe(op_id)
        else:
            expected, _ = reference.popitem()
            actual = pool.acquire_unsafe()
            assert actual == expected
            assert actual not in acquired
            acquired.add(actual)


def test_collective_id_pool_exhaustion_and_release(monkeypatch) -> None:
    monkeypatch.setattr(common, "Shuffler", SimpleNamespace(max_concurrent_shuffles=3))
    pool = common._CollectiveIDPool()
    with pytest.raises(ValueError, match="Restoring ID that exists"):
        pool.release_unsafe(0)
    assert [pool.acquire_unsafe() for _ in range(3)] == [2, 1, 0]
    with pytest.raises(ValueError, match="Cannot shuffle more than 3 times"):
        pool.acquire_unsafe()
    pool.release_unsafe(1)
    with pytest.raises(ValueError, match="Restoring ID that exists"):
        pool.release_unsafe(1)
    assert pool.acquire_unsafe() == 1


def _reserved_ids(ir) -> list[int]:
    with ReserveOpIDs(ir) as collective_id_map:
        return [i for ids in collective_id_map.values() for i in ids]


def test_collective_id_scope_ignores_held_ids() -> None:
    with collective_id_scope(), reserve_op_id() as expected:
        pass
    # Another query (possibly on another communicator) holds an ID.
    with (
        collective_id_scope(),
        reserve_op_id(),
        collective_id_scope(),
        reserve_op_id() as op_id,
    ):
        assert op_id == expected


def test_collective_id_scope_ignores_history() -> None:
    ir = _sort_ir()
    with collective_id_scope():
        expected = _reserved_ids(ir)
    assert len(expected) > 1

    with collective_id_scope():
        # IDs are not released in reverse order of reservation, so earlier
        # queries permute the pool.
        _reserved_ids(ir)
        assert _reserved_ids(ir) != expected
        with collective_id_scope():
            assert _reserved_ids(ir) == expected


def test_collective_id_scope_per_thread() -> None:
    with collective_id_scope(), reserve_op_id() as expected:
        pass
    held = threading.Event()
    release = threading.Event()

    def hold() -> None:
        with collective_id_scope(), reserve_op_id():
            held.set()
            release.wait()

    thread = threading.Thread(target=hold)
    thread.start()
    try:
        held.wait()
        with collective_id_scope(), reserve_op_id() as op_id:
            assert op_id == expected
    finally:
        release.set()
        thread.join()


def test_evaluate_on_rank_uses_fresh_scope(monkeypatch) -> None:
    with collective_id_scope(), reserve_op_id() as expected:
        pass
    seen: list[int] = []

    def fake_evaluate_on_rank(*args, **kwargs):
        with reserve_op_id() as op_id:
            seen.append(op_id)

    monkeypatch.setattr(
        cudf_polars.engine.core, "_evaluate_on_rank", fake_evaluate_on_rank
    )
    unused: Any = None
    with reserve_op_id():
        cudf_polars.engine.core.evaluate_on_rank(
            unused, unused, unused, unused, unused, query_id=uuid.uuid4()
        )
    assert seen == [expected]


@pytest.mark.parametrize("nchildren", [2, 3])
def test_union_reserves_one_id_per_child(nchildren) -> None:
    q = pl.concat([pl.LazyFrame({"a": [i]}) for i in range(nchildren)])
    ir = Translator(q._ldf.visit(), pl.GPUEngine()).translate_ir()
    (union,) = (node for node in traversal([ir]) if isinstance(node, Union))
    with collective_id_scope(), ReserveOpIDs(ir) as collective_id_map:
        assert len(collective_id_map[union]) == nchildren
