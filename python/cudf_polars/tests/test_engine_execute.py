# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests of the engine's execute path."""

from __future__ import annotations

import sys
import types
import uuid
from collections.abc import Sized

import pytest

import polars as pl
from polars.testing.asserts import assert_frame_equal

from cudf_polars.dsl.translate import Translator
from cudf_polars.engine import persisted_result, rank_local_store
from cudf_polars.engine.persisted_result import (
    PersistedHandle,
    PersistedSource,
    _PersistedLoader,
)


def _source_lf() -> pl.LazyFrame:
    return pl.LazyFrame({"a": [1, 2, 3], "b": [4, 5, 6]})


def _unsupported_lf() -> pl.LazyFrame:
    """A query whose translation records an error (unsupported on the GPU)."""
    return (
        pl.LazyFrame({"orderby": [1, 4, 8, 10], "values": [1, 2, 3, 4]})
        .rolling("orderby", period="4i")
        .agg(pl.col("values"))
    )


def test_persisted_source_emits_partition_whole():
    """This rank's partition is emitted as a single chunk (no row-splitting)."""
    df = pl.DataFrame({"a": list(range(10))})
    src = PersistedSource({0: lambda: df, 1: lambda: df}, df.schema)
    out = src(None, None, None, None, rank=1, nranks=2)
    assert isinstance(out, Sized)
    chunks = list(out)
    assert len(out) == 1 == len(chunks)
    assert_frame_equal(chunks[0], df)


def test_persisted_source_absent_rank_is_empty():
    """A rank with no partition emits exactly one empty same-schema chunk."""
    df = pl.DataFrame({"a": list(range(10))})
    src = PersistedSource({0: lambda: df}, df.schema)
    chunks = list(src(None, None, None, None, rank=2, nranks=4))
    assert len(chunks) == 1
    assert chunks[0].height == 0
    assert chunks[0].columns == ["a"]


def test_persisted_source_projection_and_predicate():
    """Projection and predicate are applied to the emitted partition."""
    df = pl.DataFrame({"a": list(range(10)), "b": list(range(10, 20))})
    src = PersistedSource({0: lambda: df}, df.schema)
    out = pl.concat(list(src(["a"], pl.col("a") > 6, None, None, rank=0, nranks=1)))
    assert out.columns == ["a"]
    assert out["a"].to_list() == [7, 8, 9]


def test_persisted_source_zero_column_projection_raises():
    """A zero-column projection (e.g. bare pl.len()) is rejected rather than
    silently dropping the partition's row count via select([]) -> (0, 0)."""
    df = pl.DataFrame({"a": list(range(10))})
    src = PersistedSource({0: lambda: df}, df.schema)
    chunks = src([], None, None, None, rank=0, nranks=1)
    with pytest.raises(NotImplementedError, match="zero-column projection"):
        list(chunks)


@pytest.mark.parametrize("deduplicate_replicated", [False, True])
def test_evaluate_and_persist_deduplicate_replicated(
    monkeypatch, deduplicate_replicated
):
    """SPMD (dedup=False) stores a duplicated output whole and flags it as such;
    Dask/Ray (dedup=True) empty non-root ranks and store a non-duplicate."""
    evaluated = object()  # stand-in for this rank's GPU partition
    deduped = object()  # stand-in for the emptied non-root partition
    drop_calls = []

    def _fake_drop(df, rank, metadata):
        drop_calls.append((df, rank, metadata))
        return deduped

    stored = {}

    class _Store:
        def put(self, query_id, rank, df, *, duplicated):
            stored["df"] = df
            stored["duplicated"] = duplicated

    # The query output is duplicated (metadata[-1].duplicated is True).
    metadata = [types.SimpleNamespace(duplicated=True)]
    monkeypatch.setattr(
        persisted_result, "evaluate_on_rank", lambda *a, **k: (evaluated, metadata)
    )
    monkeypatch.setattr(persisted_result, "drop_if_replicated", _fake_drop)
    monkeypatch.setattr(rank_local_store, "open_store", lambda uid: _Store())

    # comm.rank != 0 is where drop_if_replicated would empty a duplicated output.
    comm = types.SimpleNamespace(rank=1)
    persisted_result.evaluate_and_persist(
        "uid",
        None,
        comm,
        None,
        None,
        None,
        uuid.uuid4(),
        deduplicate_replicated=deduplicate_replicated,
    )

    if deduplicate_replicated:
        # Dask/Ray: duplicated output emptied on non-root, stored as a non-duplicate
        # (client concatenation yields a single copy).
        assert stored["df"] is deduped
        assert len(drop_calls) == 1
        assert stored["duplicated"] is False
    else:
        # SPMD: the partition is stored whole and flagged duplicated so a re-scan
        # can re-advertise it; drop_if_replicated is never called.
        assert stored["df"] is evaluated
        assert drop_calls == []
        assert stored["duplicated"] is True


@pytest.mark.parametrize("duplicated", [False, True])
def test_persisted_source_output_duplicated_reflects_store(duplicated):
    """PersistedSource.output_duplicated() reports the stored duplicated flag so the
    scan node can re-advertise ``duplicated`` for a persisted duplicated output."""
    uid = "test-output-duplicated"
    query_id = uuid.uuid4()
    store = rank_local_store.open_store(uid)
    try:
        # Dummy partitions (object() stands in for a GPU DataFrame); only the
        # duplicated flag matters here.
        store.put(query_id, 0, object(), duplicated=duplicated)
        store.put(query_id, 1, object(), duplicated=duplicated)
        loaders = {
            r: _PersistedLoader(PersistedHandle(uid, query_id, r)) for r in (0, 1)
        }
        src = PersistedSource(loaders, {"a": pl.Int64})

        assert src.output_duplicated(rank=1, nranks=2) is duplicated
        assert src.output_duplicated(rank=0, nranks=2) is duplicated
        # Single-rank run reads every partition; still reflects the flag.
        assert src.output_duplicated(rank=0, nranks=1) is duplicated
    finally:
        rank_local_store.close_store(uid)


def test_execute_lazy_roundtrip(streaming_engine):
    """execute().lazy() round-trips through the producing engine."""
    lf = _source_lf()
    result = streaming_engine.execute(lf)
    collected = result.lazy().collect(engine=streaming_engine)
    assert_frame_equal(collected, lf.collect())


def test_execute_lazy_filter(streaming_engine):
    """execute().lazy() supports chained filter operations."""
    lf = _source_lf()
    result = streaming_engine.execute(lf)
    collected = result.lazy().filter(pl.col("a") > 1).collect(engine=streaming_engine)
    assert_frame_equal(collected, lf.filter(pl.col("a") > 1).collect())


def test_execute_lazy_projection(streaming_engine):
    """execute().lazy() supports projection pushdown."""
    lf = _source_lf()
    result = streaming_engine.execute(lf)
    collected = result.lazy().select("a").collect(engine=streaming_engine)
    assert_frame_equal(collected, lf.select("a").collect())


def test_execute_chains_into_another_execute(streaming_engine):
    """A persisted result's lazy() frame can be fed back into execute(), not just collect()."""
    lf = _source_lf()
    first = streaming_engine.execute(lf)
    second = streaming_engine.execute(first.lazy().filter(pl.col("a") > 1))
    collected = second.lazy().collect(engine=streaming_engine)
    assert_frame_equal(collected, lf.filter(pl.col("a") > 1).collect())


def test_execute_n_partitions(streaming_engine):
    """The result tracks one persisted partition per rank."""
    result = streaming_engine.execute(_source_lf())
    assert len(result._ranks) == streaming_engine._nranks


def test_execute_unsupported_raises(streaming_engine):
    """execute() rejects a query with a translation error before dispatching it."""
    with pytest.raises(NotImplementedError, match="unsupported operations"):
        streaming_engine.execute(_unsupported_lf())


def test_spmd_execute_collect_consumes_result(spmd_engine):
    """Reads are move-on-read: a result collects once, then a re-collect raises."""
    lf = _source_lf()
    result = spmd_engine.execute(lf)
    collected = result.lazy().collect(engine=spmd_engine)
    assert_frame_equal(collected, lf.collect())
    # The partition was consumed by the first collect; a second scan raises.
    with pytest.RaisesGroup(
        pytest.RaisesExc(RuntimeError, match="consumed on read"),
        allow_unwrapped=True,
        flatten_subgroups=True,
    ):
        result.lazy().collect(engine=spmd_engine)


def test_spmd_execute_self_join_raises(spmd_engine):
    """A self-join scans the result twice; move-on-read makes the second scan raise."""
    lf = _source_lf()
    result = spmd_engine.execute(lf)
    reused = result.lazy()
    # Polars does not dedup a reused plugin source, so the self-join scans the
    # partition twice; the second read finds it already consumed. Re-scan support
    # is tracked as future work (see https://github.com/rapidsai/cudf/issues/23115).
    with pytest.RaisesGroup(
        pytest.RaisesExc(RuntimeError, match="consumed on read"),
        allow_unwrapped=True,
        flatten_subgroups=True,
    ):
        reused.join(reused, on="a", how="inner").collect(engine=spmd_engine)


def test_spmd_execute_reset_invalidates_result(spmd_engine):
    """_reset() frees a live result's partition before teardown; reuse raises."""
    lf = _source_lf()
    result = spmd_engine.execute(lf)
    spmd_engine._reset()
    with pytest.RaisesGroup(
        pytest.RaisesExc(RuntimeError, match="store no longer exists"),
        allow_unwrapped=True,
        flatten_subgroups=True,
    ):
        result.lazy().collect(engine=spmd_engine)
    # The engine is still usable after reset.
    again = spmd_engine.execute(lf).lazy().collect(engine=spmd_engine)
    assert_frame_equal(again, lf.collect())


def test_execute_default_engine_collect_raises(streaming_engine):
    """A persisted result is not collectable off the producing engine."""
    result = streaming_engine.execute(_source_lf())
    with pytest.raises(Exception):  # noqa: B017 - GPU frame can't cross to the client
        result.lazy().collect()


def _stored_count(query_id, *, dask_worker=None):
    """Number of this query's persisted partitions across the process's stores."""
    return sum(
        len(store._partitions.get(query_id, {}))
        for store in rank_local_store._stores.values()
    )


def _install_evaluate_and_persist_fault(fail_rank):
    """
    Wrap this process's ``evaluate_and_persist`` to raise on ``fail_rank`` after storing.

    Shipped to the worker/actor process (Dask ``client.run`` / Ray ``_run``) so it
    patches that process's own module. The original is invoked first, so the
    collective completes and the partition is stored on *every* rank before the
    designated rank raises - reproducing "one rank fails after the others finished
    storing".
    """
    orig = persisted_result.evaluate_and_persist

    def wrapper(*args, **kwargs):
        rank = orig(*args, **kwargs)
        comm = args[2]  # (uid, ctx, comm, py_executor, ir, config_options, query_id)
        if comm.rank == fail_rank:
            raise RuntimeError("injected worker failure after evaluate_and_persist")
        return rank

    persisted_result._test_orig_evaluate_and_persist = orig
    persisted_result.evaluate_and_persist = wrapper


def _restore_evaluate_and_persist():
    """Undo :func:`_install_evaluate_and_persist_fault` on this worker (idempotent)."""
    orig = getattr(persisted_result, "_test_orig_evaluate_and_persist", None)
    if orig is not None:
        persisted_result.evaluate_and_persist = orig
        del persisted_result._test_orig_evaluate_and_persist


def test_dask_execute_release_reclaims_worker_partitions(dask_engine):
    """release() drops the query's worker-persisted partitions (leak fix)."""
    result = dask_engine.execute(_source_lf())
    query_id = result._query_id
    client = dask_engine._dask_ctx.client

    before = client.run(_stored_count, query_id)
    assert sum(before.values()) == dask_engine._nranks

    result.release()
    after = client.run(_stored_count, query_id)
    assert sum(after.values()) == 0

    result.release()  # idempotent, no error


def test_dask_backend_eager_drop_swallows_failure():
    """drop_persisted swallows a failing Client.run (teardown still closes the store)."""
    pytest.importorskip("distributed")
    from cudf_polars.engine.dask import DaskPersistedBackend

    class _FailingClient:
        def run(self, *args, **kwargs):
            raise RuntimeError("workers unreachable")

    ctx = types.SimpleNamespace(client=_FailingClient(), rapidsmpf_id="uid")
    backend = DaskPersistedBackend(ctx)

    backend.drop_persisted(uuid.uuid4())  # must not raise; no cluster needed


def test_dask_execute_reset_drops_partitions(dask_engine):
    """_reset() drops this engine's persisted partitions before teardown."""
    result = dask_engine.execute(_source_lf())
    query_id = result._query_id
    client = dask_engine._dask_ctx.client
    assert sum(client.run(_stored_count, query_id).values()) == dask_engine._nranks

    dask_engine._reset()

    assert sum(client.run(_stored_count, query_id).values()) == 0


def test_dask_execute_partial_failure_drops_all_partitions(dask_engine):
    """
    A rank failing after others stored their partition leaves nothing behind.

    Drives the backend directly so the ``query_id`` is known. A fault makes one
    worker raise *after* storing (every rank has stored by then), so a correct
    cleanup path must broadcast a drop for the query to all workers before
    re-raising - otherwise the successful ranks would orphan their GPU partitions.
    """
    pytest.importorskip("distributed")
    from cudf_polars.engine.dask import DaskPersistedBackend

    client = dask_engine._dask_ctx.client
    lf = _source_lf()
    translator = Translator(lf._ldf.visit(), dask_engine)
    ir = translator.translate_ir()
    query_id = uuid.uuid4()
    backend = DaskPersistedBackend(dask_engine._dask_ctx)

    client.run(_install_evaluate_and_persist_fault, dask_engine._nranks - 1)
    try:
        with pytest.raises(Exception):  # noqa: B017 - injected worker failure
            backend.execute_persisted(ir, translator.config_options, query_id)
    finally:
        client.run(_restore_evaluate_and_persist)

    # Every rank stored before the fault, so a working cleanup must have dropped
    # them all (the failure-path broadcast is idempotent across workers).
    counts = client.run(_stored_count, query_id)
    assert sum(counts.values()) == 0


def test_ray_execute_partial_failure_drops_all_partitions(ray_engine):
    """
    An actor failing after others stored their partition leaves nothing behind.

    Ray analogue of the Dask partial-failure test: a fault makes one actor raise
    *after* storing (every rank has stored by then), so a correct cleanup path
    must broadcast a drop for the query to all actors before re-raising -
    otherwise the successful ranks would orphan their GPU partitions.
    """
    ray = pytest.importorskip("ray")
    from ray import cloudpickle

    from cudf_polars.engine.ray import RayPersistedBackend

    lf = _source_lf()
    translator = Translator(lf._ldf.visit(), ray_engine)
    ir = translator.translate_ir()
    query_id = uuid.uuid4()
    actors = ray_engine.rank_actors
    backend = RayPersistedBackend(ray_engine._store_uid, actors)

    # The fault helpers live in this test module, which the actor processes can't
    # import (No module named 'tests'). Tell Ray's cloudpickle to serialize them
    # by value instead of by reference so the actors don't need to import them.
    test_module = sys.modules[__name__]
    cloudpickle.register_pickle_by_value(test_module)
    try:
        ray.get(
            [
                a._run.remote(
                    _install_evaluate_and_persist_fault, ray_engine._nranks - 1
                )
                for a in actors
            ]
        )
        try:
            with pytest.raises(Exception):  # noqa: B017 - injected actor failure
                backend.execute_persisted(ir, translator.config_options, query_id)
        finally:
            ray.get([a._run.remote(_restore_evaluate_and_persist) for a in actors])

        # Every rank stored before the fault, so a working cleanup must have
        # dropped them all (the failure-path broadcast is idempotent).
        counts = ray.get([a._run.remote(_stored_count, query_id) for a in actors])
    finally:
        cloudpickle.unregister_pickle_by_value(test_module)

    assert sum(counts) == 0


def test_ray_backend_eager_drop_swallows_failure():
    """drop_persisted swallows a failing remote call (teardown still closes the store)."""
    pytest.importorskip("ray")

    from cudf_polars.engine.ray import RayPersistedBackend

    class _FailingMethod:
        def remote(self, *args, **kwargs):
            raise RuntimeError("actors unreachable")

    class _FailingActor:
        drop_persisted = _FailingMethod()

    backend = RayPersistedBackend("uid", [_FailingActor()])

    backend.drop_persisted(uuid.uuid4())  # must not raise; no cluster needed
