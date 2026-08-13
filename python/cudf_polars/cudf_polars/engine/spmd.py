# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""RapidsMPF streaming-engine using the SPMD Cluster style."""

from __future__ import annotations

import contextlib
import dataclasses
import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, cast

import pylibcudf as plc
import rmm.mr
from cudf_streaming.partition_utils import (
    packed_data_from_cudf_packed_columns,
    unpack_and_concat,
)
from pylibcudf.contiguous_split import pack
from rapidsmpf import bootstrap
from rapidsmpf.coll import AllGather
from rapidsmpf.communicator.single import (
    new_communicator as single_communicator,
)
from rapidsmpf.communicator.ucxx import barrier
from rapidsmpf.progress_thread import ProgressThread
from rapidsmpf.statistics import Statistics
from rapidsmpf.streaming.core.context import Context

import cudf_polars.quent
import cudf_polars.quent._logging
from cudf_polars.containers import DataFrame, DataType
from cudf_polars.engine import persisted_result, rank_local_store
from cudf_polars.engine.core import (
    ClusterInfo,
    StreamingEngine,
    all_gather_host_data,
    check_reserved_keys,
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
from cudf_polars.quent._types import Worker
from cudf_polars.streaming.actor_graph.collectives.common import reserve_op_id
from cudf_polars.streaming.actor_graph.utils import set_memory_resource
from cudf_polars.unstable import unstable
from cudf_polars.utils.config import (
    MemoryResourceConfig,
    SPMDContext,
    StreamingExecutor,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    import polars as pl

    from cudf_streaming.channel_metadata import ChannelMetadata
    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.config import Options
    from rapidsmpf.rmm_resource_adaptor import RmmResourceAdaptor

    from cudf_polars.dsl.ir import IR
    from cudf_polars.engine.core import T
    from cudf_polars.engine.options import StreamingOptions
    from cudf_polars.engine.persisted_result import PersistedQueryResult
    from cudf_polars.streaming.parallel import ConfigOptions
    from cudf_polars.utils.config import StreamingExecutor


def evaluate_pipeline_spmd_mode(
    ir: IR,
    config_options: ConfigOptions[StreamingExecutor],
    *,
    collect_metadata: bool = False,
    query_id: uuid.UUID,
) -> tuple[DataFrame, list[ChannelMetadata] | None]:
    """
    Build and evaluate a RapidsMPF streaming pipeline in SPMD mode.

    In SPMD mode every rank executes the same Python/Polars script
    independently.  Each rank owns its local DataFrames, which are
    treated as rank-local partitions of a larger distributed dataset and
    fed directly into the pipeline.  Collective operations (shuffles,
    all-gathers, etc.) coordinate across ranks to produce a globally
    consistent result.

    IR lowering is performed collectively on the workers: rank 0
    collects scan statistics and allgathers them, then every rank
    lowers the graph independently.

    Parameters
    ----------
    ir
        The pre-lowered IR node.
    config_options
        Executor configuration, including the rapidsmpf context and the
        Python thread-pool executor used to drive the actor network.
    collect_metadata
        Whether to collect runtime metadata.
    query_id
        A unique identifier for the query.

    Returns
    -------
    The GPU-resident output :class:`~cudf_polars.containers.DataFrame` and,
    if ``collect_metadata`` is True, the list of channel metadata objects;
    otherwise ``None``.
    """
    if config_options.executor.spmd_context is None:
        raise RuntimeError("spmd_context must be set for SPMD mode")
    comm = config_options.executor.spmd_context.comm
    context = config_options.executor.spmd_context.context
    py_executor = config_options.executor.spmd_context.py_executor

    quent_context = config_options.executor.quent_context
    local_quent_context: LocalQuentContext | None = None
    if quent_context is not None:
        quent_logger = config_options.executor.spmd_context.quent_logger
        assert quent_logger is not None
        quent_context._emit_query_group_events(quent_logger)
        quent_context._emit_query_events(quent_logger)
        local_quent_context = LocalQuentContext(
            context=quent_context,
            worker=Worker(
                id=config_options.executor.spmd_context.worker_id,
                engine=quent_context.engine,
                instance_name=f"rank-{comm.rank}",
            ),
            logger=quent_logger,
        )

    df, metadata = evaluate_on_rank(
        context,
        comm,
        py_executor,
        ir,
        config_options,
        local_quent_context=local_quent_context,
        query_id=query_id,
    )
    if quent_context is not None:
        assert config_options.executor.spmd_context.quent_logger is not None
        quent_context._emit_query_exit_events(
            config_options.executor.spmd_context.quent_logger
        )
    return df, metadata if collect_metadata else None


def allgather_polars_dataframe(
    *,
    engine: SPMDEngine,
    local_df: pl.DataFrame,
    op_id: int,
) -> pl.DataFrame:
    """
    AllGather a rank-local DataFrame so every rank receives the full result.

    Each rank contributes its local ``local_df`` partition and receives the
    concatenation of all ranks' partitions in rank order. This is the SPMD
    equivalent of a distributed ``collect``: after the call, every rank holds
    the same complete dataset.

    Parameters
    ----------
    engine
        The active :class:`SPMDEngine`.
    local_df
        Rank-local DataFrame to contribute.
    op_id
        Operation ID for this AllGather collective. Must be identical on every
        rank. For example, use :func:`~cudf_polars.streaming.actor_graph.collectives.common.reserve_op_id` to obtain a collision-free
        ID from the same pool used internally by cudf-polars. Avoid passing
        hardcoded integers.

    Returns
    -------
    DataFrame containing rows from all ranks, ordered by rank.

    Raises
    ------
    RuntimeError
        If ``engine`` has already been shut down.
    """
    comm = engine.comm
    ctx = engine.context
    stream = ctx.br().stream_pool.get_stream()
    col_names = local_df.columns
    dtypes = [DataType(dtype) for dtype in local_df.dtypes]

    plc_table = plc.Table.from_arrow(local_df, stream=stream)

    packed_data = packed_data_from_cudf_packed_columns(
        pack(plc_table, stream),
        stream,
        ctx.br(),
    )
    # Bulk AllGather: each rank contributes once (sequence_number=0)
    allgather = AllGather(comm, op_id, ctx.br())
    try:
        allgather.insert(0, packed_data)
    finally:
        allgather.insert_finished()
    results = allgather.wait_and_extract(ordered=True)

    # Deserialize and concatenate each rank's contribution
    plc_result = unpack_and_concat(results, stream, ctx.br())

    # pylibcudf Table -> pl.DataFrame (restore column names)
    return DataFrame.from_table(
        plc_result,
        col_names,
        dtypes,
        stream,
    ).to_polars()


def synchronize_quent_context(
    *,
    comm: Communicator,
    context: Context,
) -> cudf_polars.quent.QuentContext:
    """
    Ensure all ranks use the same Quent engine ID.

    Rank 0 selects the engine ID (from its local ``quent_context``), then all
    ranks participate in an AllGather so every process converges on that value.
    """
    if comm.rank == 0:
        quent_context = cudf_polars.quent.QuentContext()
        data = quent_context.serialize()
    else:
        data = b""

    if comm.nranks == 1:
        # skip the collective
        return cudf_polars.quent.QuentContext()

    with reserve_op_id() as op_id:
        all_data = all_gather_host_data(comm, context.br(), op_id, data)

    return cudf_polars.quent.QuentContext.deserialize(all_data[0])


class SPMDEngine(StreamingEngine):
    """
    Multi-GPU Polars engine for SPMD executions.

    Bootstraps a RapidsMPF SPMD context and returns a matching engine.

    **SPMD execution model**

    SPMD (Single Program, Multiple Data) is a parallel programming model where each
    process runs the *same* Python script independently on its own slice of data.
    When launched with the RapidsMPF launcher `rrun`, multiple identical processes
    are started. Each process owns a rank-local :class:`~polars.LazyFrame`
    representing its partition of the distributed dataset. Collective operations,
    such as shuffles, all-gathers, and joins, coordinate across ranks to produce
    a globally consistent result.

    Prefer :meth:`from_options` for typical use. Pass a :class:`~cudf_polars.engine.options.StreamingOptions`
    instance for a unified, typed interface. The ``__init__`` parameters (``rapidsmpf_options``,
    ``executor_options``, ``engine_options``) are intended for advanced use when
    fine-grained control is needed.

    This class is the primary entry point for SPMD execution. It:

    - Bootstraps a communicator connecting all ranks. When launched with ``rrun``
      this is a full UCXX communicator. When running as a normal single Python
      process (no ``rrun``) it falls back to a lightweight single-rank communicator
      that requires no external communication library (no UCXX, Ray, or Dask).
    - Creates a RapidsMPF :class:`~rapidsmpf.streaming.core.context.Context`
      that owns GPU memory and a CUDA-stream pool.

    All resources (communicator, stream pool, thread-pool) are released when
    :meth:`~SPMDEngine.shutdown` is called or the engine is used as a context
    manager.

    **Memory resource**

    ``SPMDEngine`` captures the configured device memory resource at construction
    and hands it to the RapidsMPF ``Context``, which wraps it in an internal
    tracking ``RmmResourceAdaptor`` (exposed via ``BufferResource.device_mr_adaptor()``).
    That tracking adaptor is installed as the current device resource so libcudf
    temporary allocations and the RapidsMPF ``Context`` share the same resource;
    the previous current resource is restored on shutdown.

    To use a custom allocator, call ``rmm.mr.set_current_device_resource(your_mr)``
    before constructing ``SPMDEngine``. Do not pre-wrap it in ``RmmResourceAdaptor``.

    .. code-block:: python

        import rmm

        # Optional: install a pool allocator before constructing SPMDEngine.
        # rmm.mr.set_current_device_resource(
        #     rmm.mr.PoolMemoryResource(rmm.mr.CudaMemoryResource())
        # )
        with SPMDEngine(...) as engine:
            ...

    **DataFrame and LazyFrame semantics**

    Because every rank runs an independent Python process, a :class:`~polars.DataFrame`
    is always *rank-local* i.e. it contains only that rank's partition of the distributed
    dataset.  This is true whether the DataFrame originates from a file reader or from
    Python literals.

    File-based sources (``scan_parquet``, ``scan_csv``, ...) distribute their work
    automatically: the engine assigns disjoint file- or row-group ranges to each rank,
    so different ranks produce different data.

    An in-memory ``DataFrame`` (or one produced by a previous ``collect``) is already
    rank-local by construction.  Each rank processes its own copy in full; the engine
    does **not** re-slice it across ranks.  In particular, the two patterns below are
    equivalent:

    .. code-block:: python

        # One-step: scan and transform in a single pipeline
        result = pl.scan_parquet(...).pipe(transform).collect(engine=engine)

        # Two-step: collect an intermediate result, then transform
        intermediate = pl.scan_parquet(...).collect(engine=engine)
        result = intermediate.lazy().pipe(transform).collect(engine=engine)

    In both cases rank k operates on exactly the data it read from parquet. The
    intermediate ``collect`` simply materializes the data in memory; it does not
    change which rows belong to which rank.

    **Query symmetry requirement**

    Every rank must issue the *same* sequence of Polars queries in the *same*
    order.  Collective operations (shuffles, all-gathers, joins) are matched
    across ranks by a monotonically increasing operation ID; if one rank calls
    a collective that another rank does not, all ranks will deadlock.  This means
    your driver script must be fully deterministic: avoid rank-conditional
    ``collect`` calls, early exits, or any branching that would cause different
    ranks to execute different query graphs.

    Parameters
    ----------
    comm
        An already-bootstrapped communicator. When provided, the bootstrap step
        is skipped and the caller retains ownership; the communicator is **not**
        closed on shutdown. Pass this to share a single communicator across multiple
        engine lifetimes (e.g. a session-scoped pytest fixture).
        When ``None`` (default) a new communicator is bootstrapped automatically.
    rapidsmpf_options
        RapidsMPF-specific options. Defaults to the reading ``RAPIDSMPF_*``
        environment variables.
    executor_options
        Executor-specific options (e.g. ``max_rows_per_partition``).
    engine_options
        Engine-specific keyword arguments (e.g. ``raise_on_fail``, ``parquet_options``).

    Raises
    ------
    TypeError
        If ``executor_options`` or ``engine_options`` contains a reserved key.

    Notes
    -----
    Calls :func:`~cudf_polars.engine.hardware_binding.bind_to_gpu` at construction
    time, before RMM and communicator initialisation, so that CPU affinity, NUMA
    memory policy, and ``UCX_NET_DEVICES`` are set as early as possible. By default,
    binding is skipped under ``rrun`` (which already performs its own binding),
    see ``HardwareBindingPolicy.skip_under_rrun``.

    If a bootstrapped communicator is provided, the attached statistics
    object is used for all statistics logging, and enabled/disabled
    according to any options provided in ``rapidsmpf_options``.

    Examples
    --------
    Context-manager style (recommended for scripts):

    >>> with SPMDEngine() as engine:  # doctest: +SKIP
    ...     result = (
    ...         df.lazy().group_by("a").agg(pl.col("b").sum()).collect(engine=engine)
    ...     )
    ...     full = allgather_polars_dataframe(engine=engine, local_df=result, op_id=0)

    Direct style (Jupyter / long-lived clusters):

    >>> engine = SPMDEngine()  # doctest: +SKIP
    >>> result = df.lazy().collect(engine=engine)  # doctest: +SKIP
    >>> engine.shutdown()  # doctest: +SKIP
    """

    def __init__(
        self,
        *,
        comm: Communicator | None = None,
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

        check_reserved_keys(executor_options, engine_options)
        hw_binding = cast(
            "HardwareBindingPolicy",
            engine_options.get("hardware_binding", HardwareBindingPolicy()),
        )
        bind_to_gpu(hw_binding)

        self.rapidsmpf_options = resolve_rapidsmpf_options(rapidsmpf_options)
        mr_config: MemoryResourceConfig = engine_options.get(
            "memory_resource_config", MemoryResourceConfig.default()
        )
        base_mr = mr_config.create_memory_resource()
        if comm is None:
            statistics = Statistics.from_options(self.rapidsmpf_options)
            if bootstrap.is_running_with_rrun():
                comm = bootstrap.create_ucxx_comm(
                    progress_thread=ProgressThread(statistics),
                    type=bootstrap.BackendType.AUTO,
                    options=self.rapidsmpf_options,
                )
            else:
                comm = single_communicator(
                    progress_thread=ProgressThread(statistics),
                    options=self.rapidsmpf_options,
                )
        else:
            statistics = reset_statistics_from_options(
                comm.progress_thread.statistics, self.rapidsmpf_options
            )
        # else: caller-provided comm; the caller retains ownership

        self._base_mr: rmm.mr.DeviceMemoryResource = base_mr
        self._mr: RmmResourceAdaptor  # set after `Context` is built (below).
        self._comm: Communicator | None = comm
        self._ctx: Context | None = None
        self._py_executor: ThreadPoolExecutor | None = None
        self._store_uid = uuid.uuid4().hex
        exit_stack = contextlib.ExitStack()

        # TODO: there's no reason our API needs a plain dict[str, Any] rather than
        # a typed config object here.
        try:
            # Register `_cleanup_ctx`, which shuts down whatever `self._ctx` points
            # to at engine shutdown time, i.e. the `Context` from the latest reset.
            self._ctx = Context.from_options(
                comm.logger, base_mr, self.rapidsmpf_options, statistics
            )
            # `Context` wraps `base_mr` in its `BufferResource`'s internal
            # tracking `RmmResourceAdaptor`. Capture it as `self._mr` and
            # install it as the current device resource so libcudf temporary
            # allocations share the same resource and are tracked.
            self._mr = self._ctx.br().device_mr_adaptor()
            exit_stack.enter_context(set_memory_resource(self._mr))
            exit_stack.callback(self._cleanup_ctx)

            if quent_context is not None:
                executor_options["quent_context"] = quent_context
                assert self._quent_logger is not None
                quent_context._emit_engine_init_events(self._quent_logger)
                engine_id = quent_context.engine.id
            else:
                engine_id = uuid.uuid4()

            self._quent_worker = Worker(
                id=uuid.uuid4(),
                engine=cudf_polars.quent.Engine(id=engine_id),
                instance_name=f"rank-{self.rank}",  # relies on self.comm
            )

            # Register after `_cleanup_ctx` so on teardown (LIFO) the
            # executor shuts down first. `wait=True` is safe because
            # rapidsmpf's `run_actor_network` awaits its only submitted
            # future so by the time we reach shutdown the executor has no
            # in-flight work and wait returns immediately.
            self._py_executor = ThreadPoolExecutor(
                max_workers=cast("int", executor_options.get("num_py_executors", 8)),
                thread_name_prefix="spmd-executor",
            )
            exit_stack.callback(
                self._py_executor.shutdown, wait=True, cancel_futures=True
            )

            super().__init__(
                nranks=comm.nranks,
                executor_options={
                    **executor_options,
                    "cluster": "spmd",
                    "spmd_context": SPMDContext(
                        comm=comm,
                        engine_id=engine_id,
                        worker_id=self._quent_worker.id,
                        quent_logger=self._quent_logger,
                        context=self._ctx,
                        py_executor=self._py_executor,
                    ),
                },
                engine_options={
                    **engine_options,
                    "memory_resource": self._ctx.br().device_mr,
                },
                exit_stack=exit_stack,
            )

            if self._quent_logger is not None:
                self._quent_logger.emit(self._quent_worker._init())
        except Exception:
            exit_stack.close()
            raise

    def _cleanup_ctx(self) -> None:
        """
        Shut down the current ``self._ctx`` if any; called from exit-stack.

        ``Context.shutdown()`` is idempotent on the rapidsmpf C++ side, so this is
        safe even if a prior ``_reset`` already shut down a now-replaced Context.
        """
        if self._ctx is not None:
            self._ctx.shutdown()
            self._ctx = None

    @classmethod
    def from_options(cls, options: StreamingOptions) -> SPMDEngine:
        """
        Create an :class:`SPMDEngine` from a :class:`~cudf_polars.engine.options.StreamingOptions` object.

        This is the recommended way to construct an ``SPMDEngine`` for typical
        use. All RapidsMPF, executor, and engine options are read from
        ``options``; unset fields fall back to environment variables and then
        to built-in defaults.

        Parameters
        ----------
        options
            Unified streaming configuration.

        Returns
        -------
        A new :class:`SPMDEngine` instance.

        Examples
        --------
        >>> from cudf_polars.engine.options import StreamingOptions
        >>> opts = StreamingOptions(num_streaming_threads=8, fallback_mode="silent")
        >>> with SPMDEngine.from_options(opts) as engine:  # doctest: +SKIP
        ...     result = df.lazy().collect(engine=engine)
        """
        return cls(
            rapidsmpf_options=options.to_rapidsmpf_options(),
            executor_options=options.to_executor_options(),
            engine_options=options.to_engine_options(),
        )

    def _drop_persisted(self) -> None:
        """Drop this engine's persisted partitions from the rank-local store."""
        rank_local_store.close_store(self._store_uid)

    def _reset(
        self,
        *,
        rapidsmpf_options: Options | None = None,
        executor_options: dict[str, Any] | None = None,
        engine_options: dict[str, Any] | None = None,
    ) -> None:
        """
        Reset the engine; see :meth:`StreamingEngine._reset` for the contract.

        Must be called collectively on all ranks. A barrier ensures no
        rank tears down its Context while peers may still be using it.
        """
        if self._ctx is None:
            raise RuntimeError("Cannot reset a shut-down engine")
        assert self._comm is not None
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
        quent_context: cudf_polars.quent.QuentContext | None = executor_options.get(
            "quent_context"
        )
        rapidsmpf_options = resolve_rapidsmpf_options(rapidsmpf_options)

        # Collective: synchronize all ranks before tearing down the Context.
        if self._comm.nranks > 1:
            barrier(self._comm)
        # Free persisted partitions before the Context is torn down.
        self._drop_persisted()
        # Same-thread shutdown, _reset runs on the thread that built the
        # Context (the test driver's main thread). The per-engine RMM
        # resource is kept alive across resets, see :meth:`_cleanup_ctx`.
        self._ctx.shutdown()

        statistics = reset_statistics_from_options(
            self._comm.progress_thread.statistics, rapidsmpf_options
        )
        statistics.clear()

        self._ctx = Context.from_options(
            self._comm.logger, self._base_mr, rapidsmpf_options, statistics
        )
        # Refresh `self._mr` and the current device resource to the new
        # Context's tracking adaptor (the original adaptor was tied to the
        # now-defunct Context). The original ``set_memory_resource`` exit
        # callback still restores the pre-engine MR at engine shutdown.
        self._mr = self._ctx.br().device_mr_adaptor()
        rmm.mr.set_current_device_resource(self._mr)

        if quent_context is not None:
            quent_context = synchronize_quent_context(
                comm=self._comm,
                context=self._ctx,
            )
            executor_options["quent_context"] = quent_context
            engine_id = quent_context.engine.id
        else:
            engine_id = uuid.uuid4()

        # Re-run ``StreamingEngine.__init__`` on the existing instance to
        # reconfigure the polars ``GPUEngine`` layer (``self.config``,
        # ``self.device``, etc.) with the new options. Pass the existing
        # ``self._exit_stack`` so any registered callbacks (notably
        # ``_cleanup_ctx`` and ``set_memory_resource``) survive.
        StreamingEngine.__init__(
            self,
            nranks=self._comm.nranks,
            executor_options={
                **executor_options,
                "cluster": "spmd",
                "spmd_context": SPMDContext(
                    comm=self._comm,
                    context=self._ctx,
                    py_executor=self.py_executor,
                    engine_id=engine_id,
                    worker_id=self._quent_worker.id,
                    quent_logger=self._quent_logger,
                ),
            },
            engine_options={
                **engine_options,
                "memory_resource": self._ctx.br().device_mr,
            },
            exit_stack=self._exit_stack,
        )

    @property
    def rank(self) -> int:
        """
        Rank index within the cluster (zero-based).

        Returns
        -------
        Rank index.

        Raises
        ------
        RuntimeError
            If called after :meth:`shutdown`.
        """
        return self.comm.rank

    @property
    def comm(self) -> Communicator:
        """
        The active RapidsMPF communicator.

        Returns
        -------
        Active communicator.

        Raises
        ------
        RuntimeError
            If called after :meth:`shutdown`.
        """
        if self._comm is None:
            raise RuntimeError("comm is not available after shutdown")
        return self._comm

    @property
    def context(self) -> Context:
        """
        The active RapidsMPF streaming context.

        Returns
        -------
        Active streaming context.

        Raises
        ------
        RuntimeError
            If called after :meth:`shutdown`.
        """
        if self._ctx is None:
            raise RuntimeError("context is not available after shutdown")
        return self._ctx

    @property
    def py_executor(self) -> ThreadPoolExecutor:
        """
        The thread-pool executor used to drive the actor network.

        Returns
        -------
        Active Python thread-pool executor.

        Raises
        ------
        RuntimeError
            If called after :meth:`shutdown`.
        """
        if self._py_executor is None:
            raise RuntimeError("py_executor is not available after shutdown")
        return self._py_executor

    def gather_cluster_info(self) -> list[ClusterInfo]:
        """
        Collect diagnostic information from every rank.

        This is a collective operation, every rank must call it.

        Returns
        -------
        List of :class:`~cudf_polars.engine.core.ClusterInfo`, one per rank.
        """
        data = json.dumps(dataclasses.asdict(ClusterInfo.local())).encode()
        with reserve_op_id() as op_id:
            results = all_gather_host_data(self.comm, self.context.br(), op_id, data)
        return [ClusterInfo(**json.loads(r)) for r in results]

    def gather_statistics(self, *, clear: bool = False) -> list[Statistics]:
        """
        Collect statistics from every rank via an all-gather.

        This is a collective operation, every rank must call it.

        Parameters
        ----------
        clear
            If ``True``, clear each rank's statistics after gathering.

        Returns
        -------
        List of :class:`~rapidsmpf.statistics.Statistics`, one per rank,
        ordered by rank index.
        """
        # Serialize before the optional clear so the returned stats still carry data.
        data = self.context.statistics().serialize()
        with reserve_op_id() as op_id:
            results = all_gather_host_data(self.comm, self.context.br(), op_id, data)
        if clear:
            self.context.statistics().clear()
        return [Statistics.deserialize(r) for r in results]

    def shutdown(self) -> None:
        """
        Shut down the engine and release all owned resources.

        Idempotent: safe to call more than once. Must be called on the same
        thread that created the engine.
        """
        if self._ctx is None:
            return  # already shut down

        # Free persisted partitions before _cleanup_ctx tears down the Context.
        self._drop_persisted()

        # Order matters: ``super().shutdown()`` closes ``self._exit_stack``,
        # which invokes ``self._cleanup_ctx``. That requires ``self._ctx`` to
        # still be set so the rapidsmpf Context can be shut down correctly.
        # But, super().shutdown() clears self.config, so we need to emit the
        # quent traces before that.
        # Clear the references only after shutdown completes.

        if self._quent_logger is not None:
            self._quent_logger.emit(self._quent_worker._exit())
        quent_context: cudf_polars.quent.QuentContext | None = self.config[
            "executor_options"
        ].get("quent_context")
        if quent_context is not None:
            assert self._quent_logger is not None
            quent_context._emit_engine_exit_events(self._quent_logger)

        super().shutdown()

        self._comm = None
        self._ctx = None
        # TODO: Figure out multi-rank handling.
        if self._quent_logger is not None:
            self._quent_events_raw.extend(self._quent_logger.drain())
        self._quent_events_raw.sort(key=lambda x: x["timestamp"])
        self._py_executor = None

    def _run(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> list[T]:
        data = json.dumps(func(*args, **kwargs)).encode()
        with reserve_op_id() as op_id:
            results = all_gather_host_data(self.comm, self.context.br(), op_id, data)

        return [json.loads(r) for r in results]

    @unstable()
    def execute(self, lf: pl.LazyFrame) -> PersistedQueryResult:
        """
        Execute a :class:`~polars.LazyFrame` and return a GPU-resident result.

        Unlike :meth:`~polars.LazyFrame.collect`, this method does not copy the
        result to host memory. The returned :class:`~cudf_polars.engine.persisted_result.PersistedQueryResult`
        keeps each rank's partition GPU-resident in its process. Call its
        :meth:`~cudf_polars.engine.persisted_result.PersistedQueryResult.lazy` to
        chain further operations without an intermediate host round-trip.

        This is a collective operation: every rank must call it with an
        equivalent query.

        Parameters
        ----------
        lf
            The lazy query to execute.

        Returns
        -------
        A persisted query result; each rank's partition stays GPU-resident in
        the process that produced it.

        Examples
        --------
        >>> with SPMDEngine() as engine:  # doctest: +SKIP
        ...     result = engine.execute(pl.scan_parquet("data/*.parquet"))
        ...     # Chain further work without copying to host:
        ...     df = result.lazy().filter(pl.col("x") > 0).collect(engine=engine)
        """
        backend = SpmdPersistedBackend(
            self._store_uid, self.context, self.comm, self.py_executor
        )
        return execute_persisted_query(self, lf, backend, self._store_uid)


class SpmdPersistedBackend(PersistedBackend):
    """Persisted-result backend for SPMD."""

    def __init__(
        self,
        uid: str,
        ctx: Context,
        comm: Communicator,
        py_executor: ThreadPoolExecutor,
    ) -> None:
        self._uid = uid
        self._ctx = ctx
        self._comm = comm
        self._py_executor = py_executor

    def execute_persisted(
        self,
        ir: IR,
        config_options: ConfigOptions[StreamingExecutor],
        query_id: uuid.UUID,
    ) -> list[int]:
        """Evaluate and store this rank's partition (see :class:`PersistedBackend`)."""
        rank = persisted_result.evaluate_and_persist(
            self._uid,
            self._ctx,
            self._comm,
            self._py_executor,
            ir,
            config_options,
            query_id,
            # SPMD collects rank-locally: each rank reads its own partition, so a
            # duplicated output must stay whole on every rank (not deduplicated).
            deduplicate_replicated=False,
        )
        return [rank]

    def drop_persisted(self, query_id: uuid.UUID) -> None:
        """Drop this rank's partition from this engine's store (see :class:`PersistedBackend`)."""
        rank_local_store.drop_query(self._uid, query_id)
