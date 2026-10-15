# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
import uuid
from typing import Any

import polars as pl

import cudf_polars.engine.core
from cudf_polars.dsl.translate import Translator
from cudf_polars.streaming.actor_graph.collectives import (
    ReserveOpIDs,
    collective_id_scope,
    reserve_op_id,
)


def _sort_ir():
    q = pl.LazyFrame({"a": [3, 1, 2]}).sort("a")
    return Translator(q._ldf.visit(), pl.GPUEngine()).translate_ir()


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
