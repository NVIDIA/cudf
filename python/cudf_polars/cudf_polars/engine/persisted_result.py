# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Persisted-result scan, loader, backend hook, and result for `engine.execute()`.

Developer notes
---------------
The lifecycle of a persisted result spans three phases. Each rank keeps its own
output partition GPU-resident in a process-local store keyed by
``(query_id, rank)`` (see :mod:`cudf_polars.engine.rank_local_store`), so nothing
crosses the process boundary until the caller explicitly collects.

Execute (`engine.execute(lf)` -> :class:`PersistedQueryResult`):

* The engine translates `lf` to IR, creates a `query_id`, and asks the
  backend to execute the query on each rank.
* Each rank evaluates its partition and stores the surviving output locally.
* The engine returns a `PersistedQueryResult` referencing the producing ranks.

Collect (:meth:`PersistedQueryResult.lazy` -> `collect(engine=...)`):

* :meth:`~PersistedQueryResult.lazy` exposes the stored partitions as a Polars
  IO-plugin `LazyFrame` so downstream query nodes can chain onto it normally.
* Collecting with the producing engine scans each owning rank and removes its
  partition from the rank-local store.

Release (GC finalizer / :meth:`~PersistedQueryResult.release` / context manager):

* A live result holds a :func:`weakref.finalize` that calls :meth:`PersistedBackend.drop_persisted`,
  broadcasting a store ``drop`` for the ``query_id`` to every rank. It is idempotent and never raises,
  so a collected, released, or reset result all clean up safely.
"""

from __future__ import annotations

import abc
import dataclasses
import uuid
import weakref
from typing import TYPE_CHECKING

import polars as pl
from polars.io.plugins import register_io_source

from cudf_polars.dsl.translate import Translator
from cudf_polars.engine import rank_local_store
from cudf_polars.engine.core import (
    drop_if_replicated,
    evaluate_on_rank,
    is_duplicated_output,
    raise_for_translation_errors,
)
from cudf_polars.streaming.rank_aware_source import RankAwareSource, SizedChunks

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping
    from concurrent.futures import ThreadPoolExecutor

    from polars._typing import SchemaDict

    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.streaming.core.context import Context

    from cudf_polars.containers import DataFrame
    from cudf_polars.dsl.ir import IR
    from cudf_polars.engine.core import StreamingEngine
    from cudf_polars.streaming.parallel import ConfigOptions
    from cudf_polars.utils.config import StreamingExecutor

    # A zero-argument loader returning this rank's partition (host or GPU frame).
    Loader = Callable[[], "pl.DataFrame | DataFrame"]


def evaluate_and_persist(
    uid: str,
    ctx: Context,
    comm: Communicator,
    py_executor: ThreadPoolExecutor,
    ir: IR,
    config_options: ConfigOptions[StreamingExecutor],
    query_id: uuid.UUID,
    *,
    deduplicate_replicated: bool,
) -> int:
    """
    Evaluate ``ir`` on this rank and store the GPU-persisted result.

    Runs the collective evaluation using this process's rapidsmpf context and
    puts the partition in this engine's store (``uid``) under
    ``(query_id, comm.rank)``. Returns this rank's index (nothing crosses the
    process boundary).

    Parameters
    ----------
    uid
        Identifier of the producing engine's store on this process.
    ctx
        This process's rapidsmpf streaming context.
    comm
        This process's rapidsmpf communicator.
    py_executor
        Thread pool used to drive the streaming evaluation.
    ir
        Pre-lowered root IR node to evaluate on this rank.
    config_options
        Executor configuration for the evaluation.
    query_id
        Identifier of the query; keys the stored partition.
    deduplicate_replicated
        Whether to empty a duplicated output on non-root ranks so the partitions
        can be concatenated into a single copy on collect.

    Returns
    -------
    This rank's index within the cluster (``comm.rank``).
    """
    gpu_df, metadata = evaluate_on_rank(
        ctx, comm, py_executor, ir, config_options, query_id=query_id
    )
    if deduplicate_replicated:
        gpu_df = drop_if_replicated(gpu_df, comm.rank, metadata)
    rank_local_store.open_store(uid).put(
        query_id,
        comm.rank,
        gpu_df,
        duplicated=is_duplicated_output(metadata) and not deduplicate_replicated,
    )
    return comm.rank


@dataclasses.dataclass(frozen=True)
class PersistedHandle:
    """
    Handle to a persisted partition owned by a specific rank.

    Attributes
    ----------
    uid
        Unique identifier of the persisted partition.
    query_id
        Identifier of the query that produced the partition.
    rank
        Rank that owns the persisted partition.
    """

    uid: str
    query_id: uuid.UUID
    rank: int


class _PersistedLoader:
    """
    Loader that reads a rank's persisted partition from its rank-local store.

    Holds ``owner`` (the :class:`PersistedQueryResult`) so a ``LazyFrame``
    created via :meth:`PersistedQueryResult.lazy` keeps the result, and thus
    its persisted partitions, alive until the frame is dropped. ``owner`` is
    discarded when the loader is pickled into the IR for a worker/actor, since
    the owning process only needs the handle to access its local store.

    Parameters
    ----------
    handle
        Locates the partition to read: its store ``uid``, ``query_id``, and the
        owning ``rank``.
    owner
        The :class:`PersistedQueryResult` that produced the partition, retained
        only to keep it (and its persisted partitions) alive for the lifetime of
        a derived ``LazyFrame``. Dropped on pickling (see :meth:`__reduce__`).
    """

    def __init__(self, handle: PersistedHandle, owner: object = None) -> None:
        self._handle = handle
        self._owner = owner

    def __reduce__(self) -> tuple:
        return (_PersistedLoader, (self._handle, None))

    def __call__(self) -> DataFrame:
        h = self._handle
        try:
            store = rank_local_store.require_store(h.uid)
        except KeyError:
            raise RuntimeError(
                "The persisted query result's store no longer exists, so it "
                "cannot be collected. This happens when the engine has been "
                "reset or shut down since engine.execute() produced the result. "
                "Call engine.execute() again for a fresh result."
            ) from None
        return store.pop(h.query_id, h.rank)

    def is_duplicated(self) -> bool:
        """
        Whether this rank's persisted partition is part of a duplicated output.

        Read-only, so it is safe to probe before the (consuming) load. Returns
        ``False`` if the store is gone, letting the load surface the real error.
        """
        h = self._handle
        try:
            store = rank_local_store.require_store(h.uid)
        except KeyError:
            return False
        return store.is_duplicated(h.query_id, h.rank)


def _raising(exc: Exception) -> Iterator[pl.DataFrame | DataFrame]:
    """
    Return an iterator that re-raises ``exc`` when first advanced.

    Deferring a validation error to iteration time lets the default Polars
    engine surface it as a ``ComputeError`` (its handling for a failing IO
    source) rather than as a raw exception at plan-build time.
    """

    def gen() -> Iterator[pl.DataFrame | DataFrame]:
        raise exc
        yield  # pragma: no cover - unreachable; makes this a generator

    return gen()


def _project(
    frame: pl.DataFrame | DataFrame,
    with_columns: list[str] | None,
    predicate: pl.Expr | None,
) -> pl.DataFrame | DataFrame:
    """Apply projection and (host-only) predicate pushdown to a partition."""
    if with_columns is not None:
        frame = frame.select(with_columns)
    if predicate is not None:
        if not isinstance(frame, pl.DataFrame):
            raise RuntimeError(
                "A persisted query result can only be collected with the engine "
                "that produced it, not the default host Polars engine or a "
                "different engine. Call collect(engine=<producing engine>)."
            )
        frame = frame.filter(predicate)
    return frame


class PersistedSource(RankAwareSource):
    """
    Rank-aware source backed by an optional loader for each rank.

    Parameters
    ----------
    loaders
        Mapping from rank to a zero-argument loader producing that rank's
        partition. Loaders are invoked lazily and only on their owning rank, so
        remote partitions are not materialized by the caller. A loader may
        return either a host :class:`polars.DataFrame` or a GPU-resident
        :class:`~cudf_polars.containers.DataFrame`.
    schema
        Output schema, used to construct the empty frame emitted by ranks
        without a loader.
    """

    def __init__(
        self,
        loaders: Mapping[int, Loader],
        schema: SchemaDict,
    ) -> None:
        self._loaders = loaders
        self._schema = schema

    @classmethod
    def register(
        cls,
        loaders: Mapping[int, Loader],
        schema: SchemaDict,
    ) -> pl.LazyFrame:
        """
        Build a :class:`PersistedSource` and register it as a ``LazyFrame``.

        Parameters
        ----------
        loaders
            Mapping from rank to a zero-argument partition loader.
        schema
            Output schema for registration and empty-frame construction.

        Returns
        -------
        A :class:`~polars.LazyFrame` backed by the per-rank partitions.
        """
        return register_io_source(
            cls(loaders, schema),  # type: ignore[arg-type]
            schema=schema,
        )

    def __call__(
        self,
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        n_rows: int | None,
        batch_size: int | None,
        rank: int | None = None,
        nranks: int | None = None,
    ) -> SizedChunks:
        """Emit this rank's partition(s) as :class:`SizedChunks` (see :class:`RankAwareSource`)."""
        if n_rows is not None:
            # cudf-polars rejects a pushed-down row limit during translation, so
            # this only happens under the default Polars engine.
            return SizedChunks(
                1,
                _raising(
                    NotImplementedError(
                        "PersistedSource does not support a pushed-down row "
                        "limit (head/limit/tail)."
                    )
                ),
            )
        if with_columns is not None and len(with_columns) == 0:
            return SizedChunks(
                1,
                _raising(
                    NotImplementedError(
                        "PersistedSource does not support a zero-column "
                        "projection (for example collecting only pl.len() from a "
                        "persisted result). Select at least one column."
                    )
                ),
            )
        if nranks is not None and nranks > 1:
            unreachable = sorted(r for r in self._loaders if not 0 <= r < nranks)
            if unreachable:
                return SizedChunks(
                    1,
                    _raising(
                        ValueError(
                            f"PersistedSource has partitions for ranks "
                            f"{unreachable} that no worker can emit at world size "
                            f"nranks={nranks}."
                        )
                    ),
                )

        if rank is None or nranks is None or nranks == 1:
            # Single-rank run: emit every partition (concatenated downstream).
            loaders = list(self._loaders.values())
        elif rank in self._loaders:
            loaders = [self._loaders[rank]]
        else:
            loaders = []
        if not loaders:
            # Absent rank: emit an empty same-schema (host) frame.
            empty = pl.DataFrame(schema=self._schema)
            empty = empty.select(with_columns) if with_columns is not None else empty
            return SizedChunks(1, iter((empty,)))

        # One chunk per partition.
        def chunks() -> Iterator[pl.DataFrame | DataFrame]:
            for loader in loaders:
                yield _project(loader(), with_columns, predicate)

        return SizedChunks(len(loaders), chunks())

    def output_duplicated(self, rank: int = 0, nranks: int = 1) -> bool:
        """Whether this rank's persisted partition is part of a duplicated output (see base)."""
        if nranks <= 1:
            # Single-rank run: emits every partition (see __call__).
            loaders = list(self._loaders.values())
        elif rank in self._loaders:
            loaders = [self._loaders[rank]]
        else:
            loaders = []
        return any(
            loader.is_duplicated()
            for loader in loaders
            if isinstance(loader, _PersistedLoader)
        )


class PersistedBackend(abc.ABC):
    """Engine hook for running the produce/drop steps on every rank-process."""

    @abc.abstractmethod
    def execute_persisted(
        self,
        ir: IR,
        config_options: ConfigOptions[StreamingExecutor],
        query_id: uuid.UUID,
    ) -> list[int]:
        """
        Run :func:`evaluate_and_persist` on every rank-process.

        Parameters
        ----------
        ir
            Pre-lowered root IR node to evaluate on every rank-process.
        config_options
            Executor configuration for the evaluation.
        query_id
            Identifier of the query; keys the stored partitions.

        Returns
        -------
        The ranks that produced a partition.
        """

    @abc.abstractmethod
    def drop_persisted(self, query_id: uuid.UUID) -> None:
        """
        Remove ``query_id``'s partitions from every rank-process.

        Must be a no-op where the partitions no longer exist.

        Parameters
        ----------
        query_id
            Identifier of the query whose partitions to remove.
        """


class PersistedQueryResult:
    """
    Distributed query result whose partitions remain on their producing ranks.

    Returned by ``engine.execute()``. It must be collected or executed only with
    the engine that produced it, never with a different engine (including the
    default host Polars engine, which cannot read the GPU-resident partitions).
    Using another engine is unsupported and not currently guarded against.

    Collection is one-shot. Once collected, a result cannot be collected
    again; attempting to do so raises an exception.

    If never collected, the partitions are released when this result, and any
    ``LazyFrame`` derived from :meth:`lazy`, is garbage-collected. They may
    also be released explicitly via :meth:`release` or the context-manager
    protocol.

    Parameters
    ----------
    backend
        Engine hook used to release persisted partitions.
    uid
        Store identifier used to locate the persisted partitions.
    query_id
        Identifier of the query that produced the partitions.
    ranks
        Ranks that produced a partition.
    schema
        Output schema as a ``{column_name: polars_dtype}`` mapping.
    """

    def __init__(
        self,
        backend: PersistedBackend,
        uid: str,
        query_id: uuid.UUID,
        ranks: list[int],
        schema: dict[str, pl.DataType],
    ) -> None:
        self._uid = uid
        self._query_id = query_id
        self._ranks = ranks
        self._schema = schema
        self._finalizer = weakref.finalize(self, backend.drop_persisted, query_id)

    def release(self) -> None:
        """
        Release the persisted partitions now (idempotent).

        Invalidates any :class:`~polars.LazyFrame` previously returned by :meth:`lazy`.
        """
        self._finalizer()

    def __enter__(self) -> PersistedQueryResult:
        """Return self; :meth:`release` runs on exit."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Release the persisted partitions on scope exit."""
        self.release()

    def lazy(self) -> pl.LazyFrame:
        """
        Return a :class:`~polars.LazyFrame` backed by the persisted partitions.

        Collecting with the producing engine runs the scan on each owning process,
        which **moves** its partition out of the local store. The result can be
        collected only once. Collecting the returned dataframe a second time raises.

        Returns
        -------
        LazyFrame with one partition per original rank.
        """
        loaders = {
            rank: _PersistedLoader(
                PersistedHandle(self._uid, self._query_id, rank), self
            )
            for rank in self._ranks
        }
        return PersistedSource.register(loaders, self._schema)


def execute_persisted_query(
    engine: StreamingEngine,
    lf: pl.LazyFrame,
    backend: PersistedBackend,
    uid: str,
) -> PersistedQueryResult:
    """
    Translate ``lf``, evaluate it via ``backend``, and return a persisted result.

    Parameters
    ----------
    engine
        The producing streaming engine, used to translate ``lf`` to IR.
    lf
        The lazy query to execute.
    backend
        Engine hook that evaluates and stores each rank's partition.
    uid
        The producing engine's store identifier.

    Returns
    -------
    A persisted query result referencing the producing ranks.
    """
    translator = Translator(lf._ldf.visit(), engine)
    ir = translator.translate_ir()
    raise_for_translation_errors(translator)
    query_id = uuid.uuid4()
    ranks = backend.execute_persisted(ir, translator.config_options, query_id)
    schema = {name: dtype.polars_type for name, dtype in ir.schema.items()}
    return PersistedQueryResult(backend, uid, query_id, ranks, schema)
