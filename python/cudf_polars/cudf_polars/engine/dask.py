# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""RapidsMPF streaming engine running on a Dask distributed cluster."""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import logging
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import distributed
import distributed.system
import pynvml
import ucxx._lib.libucxx as ucx_api

import polars as pl

import rmm.mr
from rapidsmpf import bootstrap
from rapidsmpf.communicator.ucxx import barrier, get_root_ucxx_address, new_communicator
from rapidsmpf.config import Options
from rapidsmpf.progress_thread import ProgressThread
from rapidsmpf.statistics import Statistics
from rapidsmpf.streaming.core.context import Context

import cudf_polars.quent
import cudf_polars.quent._logging
import cudf_polars.quent._types
from cudf_polars.engine import persisted_result, rank_local_store
from cudf_polars.engine.core import (
    ClusterInfo,
    StreamingEngine,
    check_reserved_keys,
    drop_if_replicated,
    evaluate_on_rank,
    reset_statistics_from_options,
    resolve_rapidsmpf_options,
)
from cudf_polars.engine.hardware_binding import (
    HardwareBindingPolicy,
    bind_to_gpu,
)
from cudf_polars.engine.persisted_result import (
    PersistedBackend,
    execute_persisted_query,
)
from cudf_polars.quent._context import LocalQuentContext
from cudf_polars.unstable import unstable
from cudf_polars.utils.config import DaskContext, MemoryResourceConfig

if TYPE_CHECKING:
    from collections.abc import Callable

    from cudf_streaming.channel_metadata import ChannelMetadata
    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.rmm_resource_adaptor import RmmResourceAdaptor

    from cudf_polars.dsl.ir import IR
    from cudf_polars.engine.core import T
    from cudf_polars.engine.options import StreamingOptions
    from cudf_polars.engine.persisted_result import PersistedQueryResult
    from cudf_polars.streaming.parallel import ConfigOptions
    from cudf_polars.utils.config import StreamingExecutor


def _get_visible_gpu_ids() -> list[str]:
    """
    Return the list of visible GPU identifiers.

    Reads ``CUDA_VISIBLE_DEVICES`` if set, otherwise queries NVML for the
    total device count and returns ``["0", "1", ...]``.
    """
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None:
        return [d.strip() for d in cvd.split(",") if d.strip()]
    pynvml.nvmlInit()
    return [str(i) for i in range(pynvml.nvmlDeviceGetCount())]


_nanny_preload_counter = 0


def dask_setup(nanny: distributed.Nanny) -> None:
    """
    Nanny preload: assign one GPU per worker via ``CUDA_VISIBLE_DEVICES``.

    The name ``dask_setup`` is required by Dask's preload protocol, it is
    discovered by name via ``--preload-nanny``. The function runs inside the
    Nanny process *before* the worker subprocess is spawned, so the
    environment variable is inherited by the worker.

    GPUs are assigned in a round-robin fashion across workers on the same
    node. Each worker is bound to a single GPU, but GPUs may be shared across
    multiple workers if there are more workers than available GPUs.

    Usage::

        dask worker SCHEDULER:8786 --nworkers N --nthreads 1 \
            --preload-nanny cudf_polars.engine.dask

    Parameters
    ----------
    nanny
        The :class:`distributed.Nanny` instance (injected by Dask).
    """
    if not isinstance(nanny, distributed.Nanny):
        raise TypeError(
            "dask_setup() must be used with --preload-nanny, not --preload. "
            f"Expected a Nanny instance, got {type(nanny).__name__}."
        )
    global _nanny_preload_counter  # noqa: PLW0603
    gpu_ids = _get_visible_gpu_ids()
    nanny.env["CUDA_VISIBLE_DEVICES"] = gpu_ids[_nanny_preload_counter % len(gpu_ids)]
    _nanny_preload_counter += 1


@dataclasses.dataclass
class _WorkerContext:
    """Per-worker GPU resources stored on each Dask worker."""

    comm: Communicator | None
    ctx: Context | None
    py_executor: ThreadPoolExecutor | None
    base_mr: rmm.mr.DeviceMemoryResource | None
    quent_logger: cudf_polars.quent._logging.QuentLogger | None
    quent_worker: cudf_polars.quent._types.Worker
    statistics: Statistics
    mr: RmmResourceAdaptor | None = None  # set after `Context` is built (below).


def _worker_evaluate_persisted(
    ir: IR,
    config_options: ConfigOptions[StreamingExecutor],
    *,
    uid: str,
    query_id: uuid.UUID,
    dask_worker: distributed.Worker | None = None,
) -> int:
    """
    Evaluate a query on this worker, keeping its partition GPU-resident.

    Parameters
    ----------
    ir
        Pre-lowered root IR node.
    config_options
        Executor configuration (``dask_context`` is already stripped).
    uid
        Unique identifier for the cluster instance.
    query_id
        Unique identifier for the query.
    dask_worker
        Injected by ``distributed`` when called via :meth:`distributed.Client.run`.

    Returns
    -------
    This worker's rank index.
    """
    assert dask_worker is not None
    mp_ctx: _WorkerContext = getattr(dask_worker, f"_cudf_polars_mp_context_{uid}")
    if mp_ctx.ctx is None or mp_ctx.comm is None or mp_ctx.py_executor is None:
        raise RuntimeError(
            "_setup_worker must be called before _worker_evaluate_persisted"
        )
    return persisted_result.evaluate_and_persist(
        uid,
        mp_ctx.ctx,
        mp_ctx.comm,
        mp_ctx.py_executor,
        ir,
        config_options,
        query_id,
        # Partitions are gathered to the client and concatenated, so a duplicated
        # output is deduplicated to a single copy.
        deduplicate_replicated=True,
    )


class DaskPersistedBackend(PersistedBackend):
    """Persisted-result backend for Dask."""

    def __init__(self, dask_context: DaskContext) -> None:
        self._client = dask_context.client
        self._uid = dask_context.rapidsmpf_id
        self._addresses: tuple[str, ...] = ()

    def execute_persisted(
        self,
        ir: IR,
        config_options: ConfigOptions[StreamingExecutor],
        query_id: uuid.UUID,
    ) -> list[int]:
        """Run the query on every worker (see :class:`PersistedBackend`)."""
        worker_config = config_options.drop_unserializable()
        # {worker_address: rank}; the partition for that rank stays there.
        try:
            rank_map = self._client.run(
                functools.partial(_worker_evaluate_persisted, uid=self._uid),
                ir,
                worker_config,
                query_id=query_id,
            )
        except Exception:
            # A worker may have failed after others already stored their
            # partition. _addresses is still empty (client.run raised before
            # returning), so drop_persisted broadcasts to every worker.
            self.drop_persisted(query_id)
            raise
        self._addresses = tuple(rank_map.keys())
        return list(rank_map.values())

    def drop_persisted(self, query_id: uuid.UUID) -> None:
        """Drop the query's partitions on every worker (see :class:`PersistedBackend`)."""
        # Empty tuple -> None so client.run targets all workers (failure path).
        targets = list(self._addresses) or None
        # Suppress failures so cleanup (GC finalizer / explicit release) never
        # raises when the workers are already gone (engine shutdown or worker death).
        with contextlib.suppress(Exception):
            self._client.run(
                rank_local_store.drop_query, self._uid, query_id, workers=targets
            )


def _setup_root(
    nranks: int,
    rapidsmpf_options_as_bytes: bytes,
    *,
    uid: str,
    hardware_binding: HardwareBindingPolicy,
    memory_resource_config: MemoryResourceConfig | None,
    dask_worker: distributed.Worker | None = None,
    engine_id: uuid.UUID,
    worker_id: uuid.UUID,
    quent_context: cudf_polars.quent.QuentContext | None,
) -> bytes:
    """
    Initialize the root rank on one Dask worker.

    Creates the UCXX communicator for rank 0 and stores partial state on the
    worker. The root UCXX address is returned so it can be forwarded to all
    other workers in phase 2.

    Parameters
    ----------
    nranks
        Total number of workers.
    rapidsmpf_options_as_bytes
        Serialized RapidsMPF options.
    uid
        Unique identifier for this cluster instance, used to namespace the
        per-worker attribute so multiple contexts can coexist on a worker.
    hardware_binding
        Policy controlling topology-aware hardware binding.
    memory_resource_config
        Optional RMM memory resource configuration. If ``None``, defaults to
        :class:`rmm.mr.CudaAsyncMemoryResource`.
    dask_worker
        Injected by ``distributed`` when called via :meth:`distributed.Client.run`.
    engine_id
        Unique identifier for the engine this worker belongs to.
    worker_id
        Unique identifier for this worker.
    quent_context: QuentContext | None
        Quent context to use for this worker, if quent is enabled.

    Returns
    -------
    Serialized root UCXX address for communicator bootstrap.
    """
    assert dask_worker is not None
    options = Options.deserialize(rapidsmpf_options_as_bytes)
    bind_to_gpu(hardware_binding)
    memory_resource_config = memory_resource_config or MemoryResourceConfig.default()
    base_mr = memory_resource_config.create_memory_resource()
    statistics = Statistics.from_options(options)
    comm = new_communicator(
        nranks=nranks,
        ucx_worker=None,
        root_ucxx_address=None,
        options=options,
        progress_thread=ProgressThread(statistics),
    )

    quent_worker = cudf_polars.quent._types.Worker(
        id=worker_id,
        engine=cudf_polars.quent.Engine(id=engine_id),
        instance_name=f"rank-{comm.rank}",
    )

    if quent_context is not None:
        quent_logger: cudf_polars.quent._logging.QuentLogger | None = (
            cudf_polars.quent._logging.QuentLogger()
        )
    else:
        quent_logger = None

    setattr(
        dask_worker,
        f"_cudf_polars_mp_context_{uid}",
        _WorkerContext(
            comm=comm,
            ctx=None,
            py_executor=None,
            base_mr=base_mr,
            quent_worker=quent_worker,
            quent_logger=quent_logger,
            statistics=statistics,
        ),
    )
    return get_root_ucxx_address(comm)


def _setup_worker(
    root_ucxx_address_as_bytes: bytes,
    nranks: int,
    rapidsmpf_options_as_bytes: bytes,
    *,
    uid: str,
    hardware_binding: HardwareBindingPolicy,
    memory_resource_config: MemoryResourceConfig | None,
    worker_ids: list[uuid.UUID],
    engine_id: uuid.UUID,
    num_py_executors: int,
    quent_context: cudf_polars.quent.QuentContext | None,
    dask_worker: distributed.Worker | None = None,
) -> None:
    """
    Complete communicator bootstrap and create the streaming context.

    Must be called concurrently on all workers (including the root) so that
    the barrier can be reached by every rank simultaneously.

    Parameters
    ----------
    root_ucxx_address_as_bytes
        Serialized UCXX address returned by :func:`_setup_root`.
    nranks
        Total number of workers.
    rapidsmpf_options_as_bytes
        Serialized RapidsMPF options.
    uid
        Unique identifier for this cluster instance, used to namespace the
        per-worker attribute so multiple contexts can coexist on a worker.
    hardware_binding
        Policy controlling topology-aware hardware binding.
    memory_resource_config
        Optional RMM memory resource configuration. If ``None``, defaults to
        :class:`rmm.mr.CudaAsyncMemoryResource`.
    worker_ids
        List of Quent worker UUIDs indexed by rank. Each worker picks
        its own ID after the barrier using ``comm.rank``.
    engine_id
        Quent engine UUID that owns this worker.
        :meth:`cudf_polars.utils.config.MemoryResourceConfig.default`.
    dask_worker
        Injected by ``distributed`` when called via :meth:`distributed.Client.run`.
    num_py_executors
        Number of Python executors to use for this worker.
    quent_context
        Quent context to use for this worker, if quent is enabled.

    """
    assert dask_worker is not None
    options = Options.deserialize(rapidsmpf_options_as_bytes)
    attr = f"_cudf_polars_mp_context_{uid}"
    mp_ctx: _WorkerContext | None = getattr(dask_worker, attr, None)

    if mp_ctx is None:
        # Non-root worker: create communicator now.
        bind_to_gpu(hardware_binding)
        memory_resource_config = (
            memory_resource_config or MemoryResourceConfig.default()
        )
        base_mr = memory_resource_config.create_memory_resource()
        root_addr = ucx_api.UCXAddress.create_from_buffer(root_ucxx_address_as_bytes)
        statistics = Statistics.from_options(options)
        comm = new_communicator(
            nranks=nranks,
            ucx_worker=None,
            root_ucxx_address=root_addr,
            options=options,
            progress_thread=ProgressThread(statistics),
        )
    else:
        # Root worker: comm and mr were created in _setup_root.
        assert mp_ctx.base_mr is not None
        assert mp_ctx.comm is not None
        base_mr = mp_ctx.base_mr
        comm = mp_ctx.comm
        statistics = mp_ctx.statistics

    barrier(comm)
    worker_id = worker_ids[comm.rank]
    quent_worker = cudf_polars.quent._types.Worker(
        id=worker_id,
        engine=cudf_polars.quent.Engine(id=engine_id),
        instance_name=f"rank-{comm.rank}",
    )
    ctx = Context.from_options(comm.logger, base_mr, options, statistics)
    # Set the current RMM device resource so all temporary allocations
    # in libcudf also use the same memory resource.
    mr = ctx.br().device_mr_adaptor()
    rmm.mr.set_current_device_resource(mr)
    py_executor = ThreadPoolExecutor(
        max_workers=num_py_executors,
        thread_name_prefix="dask-executor",
    )

    if quent_context is not None:
        quent_logger: cudf_polars.quent._logging.QuentLogger | None = (
            cudf_polars.quent._logging.QuentLogger()
        )
    else:
        quent_logger = None

    mp_ctx = _WorkerContext(
        comm=comm,
        ctx=ctx,
        py_executor=py_executor,
        base_mr=base_mr,
        mr=mr,
        quent_worker=quent_worker,
        quent_logger=quent_logger,
        statistics=statistics,
    )
    setattr(dask_worker, attr, mp_ctx)
    if mp_ctx.quent_logger is not None:
        mp_ctx.quent_logger.emit(quent_worker._init())


def _teardown_worker(
    *, uid: str, dask_worker: distributed.Worker | None = None
) -> list[dict[str, Any]]:
    """
    Emit Worker.exit, then release per-worker GPU resources.

    Shuts down the thread pool, drops the streaming context and communicator,
    and removes the worker attribute.

    Parameters
    ----------
    uid
        Unique identifier for the cluster instance to tear down.
    dask_worker
        Injected by ``distributed`` when called via :meth:`distributed.Client.run`.
    """
    assert dask_worker is not None
    attr = f"_cudf_polars_mp_context_{uid}"
    mp_ctx: _WorkerContext | None = getattr(dask_worker, attr, None)
    traces = []
    if mp_ctx is not None:
        if mp_ctx.quent_worker is not None and mp_ctx.quent_logger is not None:
            mp_ctx.quent_logger.emit(mp_ctx.quent_worker._exit())
            traces = mp_ctx.quent_logger.drain()

        # Drop this engine's persisted partitions before the Context is torn down,
        # so they don't outlive their allocator.
        rank_local_store.close_store(uid)
        if mp_ctx.py_executor is not None:
            mp_ctx.py_executor.shutdown(wait=True, cancel_futures=True)
        # Shut down the Context explicitly on the same thread that
        # constructed it.
        try:
            if mp_ctx.ctx is not None:
                mp_ctx.ctx.shutdown()
        finally:
            mp_ctx.ctx = None
            mp_ctx.comm = None
            mp_ctx.base_mr = None
            mp_ctx.mr = None
            delattr(dask_worker, attr)

    return traces


def _reset_worker(
    rapidsmpf_options_as_bytes: bytes,
    *,
    uid: str,
    dask_worker: distributed.Worker | None = None,
) -> None:
    """
    Rebuild the streaming Context with new options.

    Must be called collectively on all workers. A barrier ensures no
    worker tears down its Context while peers may still be using it.

    Parameters
    ----------
    rapidsmpf_options_as_bytes
        Serialized :class:`Options` to install.
    uid
        Cluster instance identifier used to look up the per-worker context.
    dask_worker
        Injected by ``distributed`` when called via :meth:`distributed.Client.run`.
    """
    assert dask_worker is not None
    attr = f"_cudf_polars_mp_context_{uid}"
    mp_ctx: _WorkerContext | None = getattr(dask_worker, attr, None)
    if mp_ctx is None:
        raise RuntimeError(f"_reset_worker called before _setup_worker for uid={uid}")
    assert mp_ctx.comm is not None
    assert mp_ctx.ctx is not None
    assert mp_ctx.base_mr is not None
    # Collective: all ranks idle before any rank tears down its Context.
    if mp_ctx.comm.nranks > 1:
        barrier(mp_ctx.comm)
    # Drop this engine's persisted partitions before the Context is torn down, so they
    # don't outlive their allocator. This invalidates any live QueryResult from execute().
    rank_local_store.close_store(uid)
    # Explicit shutdown is thread-affine. ``distributed.worker.run``
    # dispatches sync work onto the worker's event-loop thread, which is
    # the same thread that built the Context in ``_setup_worker``.
    mp_ctx.ctx.shutdown()
    mp_ctx.ctx = None
    options = Options.deserialize(rapidsmpf_options_as_bytes)
    mp_ctx.statistics = reset_statistics_from_options(mp_ctx.statistics, options)
    mp_ctx.statistics.clear()
    mp_ctx.ctx = Context.from_options(
        mp_ctx.comm.logger, mp_ctx.base_mr, options, mp_ctx.statistics
    )
    mp_ctx.mr = mp_ctx.ctx.br().device_mr_adaptor()
    rmm.mr.set_current_device_resource(mp_ctx.mr)


def _get_statistics(
    *, clear: bool, uid: str, dask_worker: distributed.Worker | None = None
) -> tuple[int, Statistics]:
    """
    Return this worker's ``(rank, Statistics)`` pair.

    The rank is used on the client to produce a rank-ordered list.

    Parameters
    ----------
    clear
        If ``True``, clear this worker's statistics after capturing a copy.
    uid
        Cluster instance identifier used to look up the per-worker context.
    dask_worker
        Injected by ``distributed`` when called via :meth:`distributed.Client.run`.

    Returns
    -------
    Pair of ``(rank, Statistics)`` for this worker.
    """
    assert dask_worker is not None
    mp_ctx: _WorkerContext = getattr(dask_worker, f"_cudf_polars_mp_context_{uid}")
    assert mp_ctx.comm is not None
    assert mp_ctx.ctx is not None
    stats = mp_ctx.statistics
    if clear:
        # Return a deep copy so it survives the in-place clear of `stats`.
        detached = stats.copy()
        stats.clear()
        return mp_ctx.comm.rank, detached
    return mp_ctx.comm.rank, stats


def _worker_evaluate(
    ir: IR,
    config_options: ConfigOptions[StreamingExecutor],
    *,
    uid: str,
    collect_metadata: bool = False,
    query_id: uuid.UUID,
    quent_context: cudf_polars.quent.QuentContext | None = None,
    dask_worker: distributed.Worker | None = None,
) -> tuple[int, pl.DataFrame, list[ChannelMetadata] | None]:
    """
    Lower and execute a Polars IR query on this Dask worker's GPU.

    IR lowering is performed collectively across all workers: rank 0
    collects scan statistics and allgathers them, then every worker
    lowers the graph independently.

    Parameters
    ----------
    ir
        pre-lowered root IR node.
    config_options
        Executor configuration (``dask_context`` is already stripped).
    uid
        Unique identifier for the cluster instance, used to look up the
        per-worker context attribute.
    collect_metadata
        Whether to collect channel metadata.
    query_id
        Unique identifier for the query, propagated into actor traces.
    dask_worker
        Injected by ``distributed`` when called via :meth:`distributed.Client.run`.
    quent_context
        The client's current quent context.

    Returns
    -------
    rank
        This worker's rank index.
    result
        This worker's output partition as a Polars DataFrame.
    metadata
        Collected channel metadata if ``collect_metadata`` is ``True``,
        otherwise ``None``.
    """
    assert dask_worker is not None
    mp_ctx: _WorkerContext = getattr(dask_worker, f"_cudf_polars_mp_context_{uid}")
    if mp_ctx.ctx is None or mp_ctx.comm is None or mp_ctx.py_executor is None:
        raise RuntimeError("_setup_worker must be called before _worker_evaluate")
    local_quent_context: LocalQuentContext | None = None
    if quent_context is not None:
        assert mp_ctx.quent_logger is not None
        local_quent_context = LocalQuentContext(
            context=quent_context,
            worker=mp_ctx.quent_worker,
            logger=mp_ctx.quent_logger,
        )
    # evaluate_on_rank always collects metadata internally so we can read
    # metadata[-1].duplicated to decide whether to suppress this rank's output.
    # The client concatenates each rank's result, so without this dedup an
    # output marked duplicated=True would appear N times. The external
    # collect_metadata parameter still controls whether the collected list is
    # returned to the client (see the return statement), which is the cost we
    # care about saving when the caller doesn't need the metadata.
    gpu_df, metadata = evaluate_on_rank(
        mp_ctx.ctx,
        mp_ctx.comm,
        mp_ctx.py_executor,
        ir,
        config_options,
        local_quent_context=local_quent_context,
        query_id=query_id,
    )
    gpu_df = drop_if_replicated(gpu_df, mp_ctx.comm.rank, metadata)
    return mp_ctx.comm.rank, gpu_df.to_polars(), metadata if collect_metadata else None


def drain_quent_events(
    dask_worker: distributed.Worker | None = None,
    *,
    uid: str,
) -> list[dict[str, Any]]:
    """Drain Quent events from the Dask worker."""
    assert dask_worker is not None

    mp_ctx: _WorkerContext = getattr(dask_worker, f"_cudf_polars_mp_context_{uid}")
    if mp_ctx.quent_worker is not None and mp_ctx.quent_logger is not None:
        return mp_ctx.quent_logger.drain()
    return []


def evaluate_pipeline_dask_mode(
    ir: IR,
    config_options: ConfigOptions[StreamingExecutor],
    *,
    collect_metadata: bool = False,
    query_id: uuid.UUID,
) -> tuple[pl.DataFrame, list[ChannelMetadata] | None]:
    """
    Evaluate a RapidsMPF streaming pipeline in Dask mode.

    The pre-lowered IR is dispatched to every Dask worker via
    :meth:`distributed.Client.run`.  Each worker collectively lowers the
    graph (rank 0 gathers statistics; all ranks allgather them) and then
    executes the resulting pipeline on its local GPU.  Per-worker outputs
    are concatenated on the client before being returned.

    Parameters
    ----------
    ir
        The pre-lowered IR node.
    config_options
        Executor configuration, including the ``dask_context`` handle.
    collect_metadata
        Whether to collect runtime metadata.
    query_id
        A unique identifier for the query.

    Returns
    -------
    result
        Concatenated output from all Dask workers as a Polars DataFrame.
    metadata
        Collected channel metadata if ``collect_metadata`` is ``True``,
        otherwise ``None``.

    Raises
    ------
    RuntimeError
        If ``config_options.executor.dask_context`` is ``None``.
    """
    if config_options.executor.dask_context is None:
        raise RuntimeError("dask_context must be set when cluster='dask'")

    dask_context = config_options.executor.dask_context

    quent_context = config_options.executor.quent_context
    if quent_context is not None:
        quent_logger = dask_context.quent_logger
        assert quent_logger is not None
        quent_context._emit_query_group_events(quent_logger)
        quent_context._emit_query_events(quent_logger)

    worker_config = config_options.drop_unserializable()
    result_map = dask_context.client.run(
        functools.partial(_worker_evaluate, uid=dask_context.rapidsmpf_id),
        ir,
        worker_config,
        collect_metadata=collect_metadata,
        quent_context=quent_context,
        query_id=query_id,
    )

    ranked: list[tuple[int, pl.DataFrame]] = []
    metadata_collector: list[ChannelMetadata] = []
    for rank, df, md in result_map.values():
        ranked.append((rank, df))
        if md is not None:
            metadata_collector.extend(md)

    if quent_context is not None:
        quent_logger = dask_context.quent_logger
        assert quent_logger is not None
        quent_context._emit_query_exit_events(quent_logger)

    ranked.sort(key=lambda p: p[0])
    dfs = [df for _, df in ranked]
    return pl.concat(dfs), metadata_collector or None


class DaskEngine(StreamingEngine):
    """
    Multi-GPU Polars engine for Dask distributed execution backed by RapidsMPF.

    Bootstraps a RapidsMPF UCXX cluster on top of a Dask distributed cluster
    and returns an engine that can be passed to ``LazyFrame.collect(engine=engine)``.

    If ``dask_client`` is provided, it is used directly and its lifetime is
    managed by the caller. If ``dask_client`` is ``None``, a
    :class:`distributed.LocalCluster` is created automatically (one worker
    per visible GPU) and torn down by :meth:`shutdown`.

    Prefer the context-manager form in scripts: it guarantees that workers are
    torn down even if an exception is raised. In interactive environments such
    as Jupyter notebooks, the direct form lets the cluster persist across
    multiple cells without tearing it down after every query.

    Parameters
    ----------
    dask_client
        An existing :class:`~distributed.Client` to use. If ``None``, a
        :class:`~distributed.LocalCluster` (one worker per visible GPU)
        and a new client are created and owned by this engine.
    rapidsmpf_options
        RapidsMPF options forwarded to every worker. If ``None``, defaults to
        ``Options(get_environment_variables())``.
    executor_options
        Executor-specific options (e.g. ``max_rows_per_partition``).
    engine_options
        Engine-specific keyword arguments (e.g. ``raise_on_fail``,
        ``parquet_options``).

    Raises
    ------
    RuntimeError
        If called from within an ``rrun`` cluster.
    TypeError
        If ``executor_options`` or ``engine_options`` contains a reserved key.

    Examples
    --------
    Context-manager style:

    >>> with DaskEngine() as engine:  # doctest: +SKIP
    ...     result = pl.LazyFrame({"a": [1, 2, 3]}).collect(engine=engine)

    Bring-your-own client:

    >>> from distributed import Client
    >>> with Client("scheduler-address:8786") as dc:  # doctest: +SKIP
    ...     with DaskEngine(dask_client=dc) as engine:
    ...         result = pl.LazyFrame({"a": [1, 2, 3]}).collect(engine=engine)

    Jupyter / manual style:

    >>> engine = DaskEngine()  # doctest: +SKIP
    >>> result = pl.LazyFrame({"a": [1, 2, 3]}).collect(engine=engine)  # doctest: +SKIP
    >>> engine.shutdown()  # doctest: +SKIP

    Notes
    -----
    When using a pre-configured cluster that already performs its own hardware
    binding (e.g. :class:`dask_cuda.LocalCUDACluster`, which pins CPU affinity
    and sets ``CUDA_VISIBLE_DEVICES`` per worker), disable some or all of the
    built-in binding to avoid conflicts:

    >>> from cudf_polars.engine.hardware_binding import HardwareBindingPolicy
    >>> with DaskEngine(  # doctest: +SKIP
    ...     dask_client=dc,
    ...     engine_options={
    ...         "hardware_binding": HardwareBindingPolicy(enabled=False),
    ...     },
    ... ) as engine:
    ...     ...

    For manually launched Dask clusters, use the nanny preload to assign
    one GPU per worker before the worker process spawns::

        dask worker SCHEDULER:8786 --nworkers N --nthreads 1 \
            --preload-nanny cudf_polars.engine.dask

    Then connect from the client::

        >>> from distributed import Client  # doctest: +SKIP
        >>> with Client("SCHEDULER:8786") as dc:  # doctest: +SKIP
        ...     with DaskEngine(dask_client=dc) as engine:
        ...         result = lf.collect(engine=engine)
    """

    def __init__(
        self,
        *,
        dask_client: distributed.Client | None = None,
        rapidsmpf_options: Options | None = None,
        executor_options: dict[str, Any] | None = None,
        engine_options: dict[str, Any] | None = None,
    ) -> None:
        executor_options = executor_options or {}
        engine_options = engine_options or {}

        quent_context: cudf_polars.quent.QuentContext | None = executor_options.get(
            "quent_context"
        )
        if quent_context is not None:
            self._quent_logger = cudf_polars.quent._logging.QuentLogger()
        else:
            self._quent_logger = None

        if bootstrap.is_running_with_rrun():
            raise RuntimeError(
                "DaskEngine must not be created from within an rrun cluster. Instead "
                "launch the rrun cluster separately and let this client connect to its "
                "cluster nodes."
            )

        check_reserved_keys(executor_options, engine_options)
        hw_binding = engine_options.get("hardware_binding", HardwareBindingPolicy())

        mr_config: MemoryResourceConfig | None = engine_options.get(
            "memory_resource_config", None
        )

        self.rapidsmpf_options = resolve_rapidsmpf_options(rapidsmpf_options)
        rapidsmpf_options_as_bytes = self.rapidsmpf_options.serialize()

        # TODO: there's no reason our API needs a plain dict[str, Any] rather than
        # a typed config object here.
        if quent_context is not None:
            executor_options.setdefault("quent_context", quent_context)
            assert self._quent_logger is not None
            quent_context._emit_engine_init_events(self._quent_logger)
            rapidsmpf_id = str(quent_context.engine.id)
            engine_id = quent_context.engine.id
        else:
            rapidsmpf_id = str(uuid.uuid4())
            engine_id = uuid.UUID(rapidsmpf_id)

        owned_cluster: Any = None
        owned_client: distributed.Client | None = None
        if dask_client is None:
            gpu_ids = _get_visible_gpu_ids()

            worker_spec: dict[str, Any] = {}
            for i, gpu_id in enumerate(gpu_ids):
                worker_spec[str(gpu_id)] = {
                    "cls": distributed.Nanny,
                    "options": {
                        "nthreads": 1,
                        # Set worker subprocess log level to WARNING
                        # (is INFO by default).
                        "silence_logs": logging.WARNING,
                        # We oversubscribe the system memory limit on multi-gpu systems.
                        # In general, Dask won't be aware of what we're doing with we're
                        # doing with host memory, so just giving each worker access to
                        # all of it seems like the option with the fewest downsides.
                        "memory_limit": distributed.system.MEMORY_LIMIT,
                        "env": {
                            "CUDA_VISIBLE_DEVICES": gpu_ids[i],
                        },
                    },
                }
            # Set scheduler/client log level to WARNING in the main
            # process (is INFO by default).
            owned_cluster = distributed.SpecCluster(
                workers=worker_spec, silence_logs=logging.WARNING
            )
            owned_client = distributed.Client(owned_cluster)
            dask_client = owned_client

        workers_info = dask_client.scheduler_info(n_workers=-1)["workers"]
        nranks = len(workers_info)
        if nranks == 0:
            raise RuntimeError("No workers found in the Dask cluster.")
        worker_addresses = list(workers_info.keys())
        root_worker = worker_addresses[0]

        worker_ids = [uuid.uuid4() for _ in range(nranks)]

        # Phase 1: initialize root communicator on one worker.
        root_result = dask_client.run(
            functools.partial(
                _setup_root,
                uid=rapidsmpf_id,
                hardware_binding=hw_binding,
                memory_resource_config=mr_config,
                worker_id=worker_ids[0],
                engine_id=engine_id,
            ),
            nranks,
            rapidsmpf_options_as_bytes,
            quent_context=quent_context,
            workers=[root_worker],
        )
        root_ucxx_address_as_bytes: bytes = root_result[root_worker]

        # Phase 2: complete bootstrap on all workers concurrently.
        # All workers call barrier() so they must all run simultaneously.
        # Each worker picks its own worker_id from the list using comm.rank.
        dask_client.run(
            functools.partial(
                _setup_worker,
                uid=rapidsmpf_id,
                hardware_binding=hw_binding,
                memory_resource_config=mr_config,
                worker_ids=worker_ids,
                engine_id=engine_id,
            ),
            root_ucxx_address_as_bytes,
            nranks,
            rapidsmpf_options_as_bytes,
            quent_context=quent_context,
            num_py_executors=executor_options.get("num_py_executors", 8),
        )

        dask_ctx = DaskContext(
            client=dask_client,
            rapidsmpf_id=rapidsmpf_id,
            quent_logger=self._quent_logger,
            owned_client=owned_client,
            owned_cluster=owned_cluster,
        )
        self._dask_context: DaskContext | None = dask_ctx
        super().__init__(
            nranks=nranks,
            executor_options={
                **executor_options,
                "cluster": "dask",
                "dask_context": dask_ctx,
            },
            engine_options={**engine_options, "memory_resource": None},
        )

    def _reset(
        self,
        *,
        rapidsmpf_options: Options | None = None,
        executor_options: dict[str, Any] | None = None,
        engine_options: dict[str, Any] | None = None,
    ) -> None:
        """Reset the engine; see :meth:`StreamingEngine._reset` for the contract."""
        if self._dask_context is None:
            raise RuntimeError("Cannot reset a shut-down engine")
        super()._reset(
            rapidsmpf_options=rapidsmpf_options,
            executor_options=executor_options,
            engine_options=engine_options,
        )
        executor_options = executor_options or {}
        existing_executor_options = self.config.get("executor_options", {})
        if isinstance(existing_executor_options, dict):
            existing_quent_context = existing_executor_options.get("quent_context")
            if existing_quent_context is not None:
                executor_options.setdefault("quent_context", existing_quent_context)
        engine_options = engine_options or {}

        rapidsmpf_options_as_bytes = resolve_rapidsmpf_options(
            rapidsmpf_options
        ).serialize()

        ctx = self._dask_context
        # Reset all worker Contexts collectively. ``client.run`` blocks
        # until every worker's reset returns; the per-worker barrier
        # inside :func:`_reset_worker` synchronizes the teardown across
        # workers.
        ctx.client.run(
            functools.partial(_reset_worker, uid=ctx.rapidsmpf_id),
            rapidsmpf_options_as_bytes,
        )

        # Re-run ``StreamingEngine.__init__`` on the existing instance to
        # reconfigure the polars ``GPUEngine`` layer (``self.config``,
        # ``self.device``, etc.) with the new options. Pass the existing
        # ``self._exit_stack`` so any registered callbacks survive.
        StreamingEngine.__init__(
            self,
            nranks=self._nranks,
            executor_options={
                **executor_options,
                "cluster": "dask",
                "dask_context": ctx,
            },
            engine_options={**engine_options, "memory_resource": None},
            exit_stack=self._exit_stack,
        )

    @classmethod
    def from_options(
        cls,
        options: StreamingOptions,
        *,
        dask_client: distributed.Client | None = None,
    ) -> DaskEngine:
        """
        Create a :class:`DaskEngine` from a :class:`~cudf_polars.engine.options.StreamingOptions` object.

        This is the recommended way to construct a ``DaskEngine`` for typical
        use. All RapidsMPF, executor, and engine options are read from
        ``options``; unset fields fall back to environment variables and then
        to built-in defaults.

        Parameters
        ----------
        options
            Unified streaming configuration.
        dask_client
            An existing :class:`distributed.Client` to use. If ``None``, a
            :class:`distributed.LocalCluster` is created automatically.

        Returns
        -------
        A new :class:`DaskEngine` instance.

        Examples
        --------
        >>> from cudf_polars.engine.options import StreamingOptions
        >>> opts = StreamingOptions(num_streaming_threads=4, fallback_mode="silent")
        >>> with DaskEngine.from_options(opts) as engine:  # doctest: +SKIP
        ...     result = pl.LazyFrame({"a": [1, 2, 3]}).collect(engine=engine)
        """
        return cls(
            dask_client=dask_client,
            rapidsmpf_options=options.to_rapidsmpf_options(),
            executor_options=options.to_executor_options(),
            engine_options=options.to_engine_options(),
        )

    @property
    def _dask_ctx(self) -> DaskContext:
        if self._dask_context is None:
            raise RuntimeError("dask_context is not available after shutdown")
        return self._dask_context

    def gather_cluster_info(self) -> list[ClusterInfo]:
        """
        Collect diagnostic information from every rank.

        Returns
        -------
        List of :class:`~cudf_polars.engine.core.ClusterInfo`, one per rank.
        """
        return list(self._dask_ctx.client.run(ClusterInfo.local).values())

    def gather_statistics(self, *, clear: bool = False) -> list[Statistics]:
        """
        Collect statistics from every rank via ``client.run``.

        Parameters
        ----------
        clear
            If ``True``, clear each rank's statistics after gathering.

        Returns
        -------
        List of :class:`~rapidsmpf.statistics.Statistics`, one per rank,
        ordered by rank index.
        """
        results = self._dask_ctx.client.run(
            functools.partial(
                _get_statistics, clear=clear, uid=self._dask_ctx.rapidsmpf_id
            )
        )
        # `client.run` returns a dict keyed by worker address in non-deterministic
        # order; sort by the rank the worker reports.
        return [s for _, s in sorted(results.values(), key=lambda p: p[0])]

    def shutdown(self) -> None:
        """
        Shut down all Dask workers' GPU resources.

        Drains buffered Quent events from all workers before tearing down,
        then emits ``Engine.exit`` on the client.

        If the cluster and client were created by this engine, they are also
        closed. Safe to call more than once. Must be called on the same thread
        that created the engine.

        Raises
        ------
        ExceptionGroup
            If one or more workers raise an unexpected exception during teardown.
        """
        if self._dask_context is None:
            return  # already shut down
        ctx = self._dask_context
        self._dask_context = None
        exceptions: list[Exception] = []
        quent_context: cudf_polars.quent.QuentContext | None = self.config[
            "executor_options"
        ].get("quent_context")
        try:
            # Teardown emits Worker.exit, then we drain all buffered events
            # (including the exit event) from workers.
            traces_map = ctx.client.run(
                functools.partial(_teardown_worker, uid=ctx.rapidsmpf_id)
            )

            for traces in traces_map.values():
                self._quent_events_raw.extend(traces)

        except Exception as e:
            exceptions.append(e)
        finally:
            if quent_context is not None:
                assert self._quent_logger is not None
                quent_context._emit_engine_exit_events(self._quent_logger)
            if ctx.owned_client is not None:
                ctx.owned_client.close()
            if ctx.owned_cluster is not None:
                ctx.owned_cluster.close()
            super().shutdown()

        # gather the client-side events.
        if self._quent_logger is not None:
            self._quent_events_raw.extend(self._quent_logger.drain())
        # final inplace sort of the events.
        self._quent_events_raw.sort(key=lambda x: x["timestamp"])

        if exceptions:
            raise ExceptionGroup("Worker teardown failed", exceptions)

    def _run(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> list[T]:
        return list(self._dask_ctx.client.run(func, *args, **kwargs).values())

    @unstable()
    def execute(self, lf: pl.LazyFrame) -> PersistedQueryResult:
        """
        Execute a :class:`~polars.LazyFrame` and return a distributed result.

        Unlike :meth:`~polars.LazyFrame.collect`, worker outputs remain as separate
        partitions rather than being concatenated into a single dataframe. Call
        ``.lazy()`` on the returned result to build further queries.

        Parameters
        ----------
        lf
            The lazy query to execute.

        Returns
        -------
        A persisted query result; each rank's partition stays GPU-resident on
        the worker that produced it.

        Examples
        --------
        >>> with DaskEngine() as engine:  # doctest: +SKIP
        ...     result = engine.execute(pl.scan_parquet("data/*.parquet"))
        ...     df = result.lazy().filter(pl.col("x") > 0).collect(engine=engine)
        """
        backend = DaskPersistedBackend(self._dask_ctx)
        return execute_persisted_query(self, lf, backend, self._dask_ctx.rapidsmpf_id)
