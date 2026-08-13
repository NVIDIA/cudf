# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""RapidsMPF streaming engine running on a Ray cluster."""

from __future__ import annotations

import contextlib
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, cast

import ray
import ray.exceptions
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
from cudf_polars.quent._types import Worker
from cudf_polars.unstable import unstable
from cudf_polars.utils.config import MemoryResourceConfig, RayContext

if TYPE_CHECKING:
    from collections.abc import Callable

    from ray.actor import ActorHandle

    from cudf_streaming.channel_metadata import ChannelMetadata
    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.rmm_resource_adaptor import RmmResourceAdaptor

    from cudf_polars.dsl.ir import IR
    from cudf_polars.engine.core import T
    from cudf_polars.engine.options import StreamingOptions
    from cudf_polars.engine.persisted_result import PersistedQueryResult
    from cudf_polars.streaming.parallel import ConfigOptions
    from cudf_polars.utils.config import StreamingExecutor


class RayPersistedBackend(PersistedBackend):
    """Persisted-result backend for Ray."""

    def __init__(self, uid: str, rank_actors: list[ActorHandle[RankActor]]) -> None:
        self._uid = uid
        self._rank_actors = rank_actors

    def execute_persisted(
        self,
        ir: IR,
        config_options: ConfigOptions[StreamingExecutor],
        query_id: uuid.UUID,
    ) -> list[int]:
        """Run the query on every actor (see :class:`PersistedBackend`)."""
        actor_config_options = config_options.drop_unserializable()
        ir_ref = ray.put(ir)
        try:
            return ray.get(
                [
                    actor.execute_persisted.remote(
                        self._uid, ir_ref, actor_config_options, query_id=query_id
                    )
                    for actor in self._rank_actors
                ]
            )
        except Exception:
            # An actor may have failed after others already stored their
            # partition. Drop the query on every actor (idempotent) to avoid
            # orphaning GPU partitions, then re-raise. drop_persisted suppresses
            # its own failures, so no outer guard is needed here.
            self.drop_persisted(query_id)
            raise

    def drop_persisted(self, query_id: uuid.UUID) -> None:
        """Drop the query's partitions on every actor (see :class:`PersistedBackend`)."""
        # Suppress failures so cleanup (GC finalizer / explicit release) never
        # raises when the actors are already gone (engine shutdown or actor death).
        with contextlib.suppress(Exception):
            ray.get(
                [
                    actor.drop_persisted.remote(self._uid, query_id)
                    for actor in self._rank_actors
                ]
            )


def evaluate_pipeline_ray_mode(
    ir: IR,
    config_options: ConfigOptions[StreamingExecutor],
    *,
    collect_metadata: bool = False,
    query_id: uuid.UUID,
) -> tuple[pl.DataFrame, list[ChannelMetadata] | None]:
    """
    Evaluate a RapidsMPF streaming pipeline in Ray mode.

    The pre-lowered IR is dispatched to every :class:`RankActor` in the
    Ray cluster.  Each actor collectively lowers the graph (rank 0
    gathers statistics; all ranks allgather them) and then executes the
    resulting pipeline on its local GPU.  Per-rank outputs are
    concatenated on the client before being returned.

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
    result
        Concatenated output from all Ray actors as a Polars DataFrame.
    metadata
        Collected channel metadata if ``collect_metadata`` is ``True``,
        otherwise ``None``.

    Raises
    ------
    RuntimeError
        If ``config_options.executor.ray_context`` is not set.
    """
    if config_options.executor.ray_context is None:
        raise RuntimeError("ray_context must be set when cluster='ray'")
    rank_actors = config_options.executor.ray_context.rank_actors
    actor_config_options = config_options.drop_unserializable()
    quent_context = config_options.executor.quent_context
    if quent_context is not None:
        quent_logger = config_options.executor.ray_context.quent_logger
        assert quent_logger is not None
        quent_context._emit_query_group_events(quent_logger)
        quent_context._emit_query_events(quent_logger)

    # Serialize the IR into the Ray object store so actors fetch by reference
    # instead of receiving N copies.
    ir_ref = ray.put(ir)
    # ray.get() returns results in the same order as the input list of object refs,
    # guaranteeing that result[i] corresponds to rank_actors[i] (rank order).
    result = ray.get(
        [
            rank.evaluate_polars_ir.remote(
                ir_ref,
                actor_config_options,
                collect_metadata=collect_metadata,
                quent_context=config_options.executor.quent_context,
                query_id=query_id,
            )
            for rank in rank_actors
        ]
    )
    dfs: list[pl.DataFrame] = []
    metadata_collector: list[ChannelMetadata] = []
    for df, md in result:
        dfs.append(df)
        if md is not None:
            metadata_collector.extend(md)

    if quent_context is not None:
        quent_logger = config_options.executor.ray_context.quent_logger
        assert quent_logger is not None
        quent_context._emit_query_exit_events(quent_logger)
    return pl.concat(dfs), metadata_collector or None


@ray.remote(
    max_restarts=0,
    max_task_retries=0,
    num_cpus=0,
    num_gpus=1,
)
class RankActor:
    """
    Ray actor that owns one GPU and participates in a RapidsMPF cluster.

    Each actor manages its own memory resource, communicator, streaming context,
    etc. Collectively, the actors form an SPMD execution cluster used by the
    client-side Ray integration.

    Parameters
    ----------
    nranks
        Total number of actors, typically one per GPU in the Ray cluster.
    rapidsmpf_options_as_bytes
        Serialized RapidsMPF options produced by
        :meth:`rapidsmpf.config.Options.serialize`.
    num_py_executors
        Maximum number of threads for the actor's Python thread-pool executor.
        ``None`` lets :class:`~concurrent.futures.ThreadPoolExecutor` choose.
    hardware_binding
        Policy controlling topology-aware hardware binding.
    memory_resource_config
        Optional RMM memory resource configuration. If ``None``, defaults to
        :meth:`cudf_polars.utils.config.MemoryResourceConfig.default`.

    Notes
    -----
    Calls :func:`~cudf_polars.engine.hardware_binding.bind_to_gpu` at construction
    time, before RMM and communicator initialisation, so that CPU affinity, NUMA
    memory policy, and ``UCX_NET_DEVICES`` are set as early as possible.
    """

    def __init__(
        self,
        *,
        nranks: int,
        rapidsmpf_options_as_bytes: bytes,
        num_py_executors: int,
        hardware_binding: HardwareBindingPolicy,
        memory_resource_config: MemoryResourceConfig | None,
        worker_id: uuid.UUID,
        engine: cudf_polars.quent.Engine,
        quent_enabled: bool,
    ) -> None:
        bind_to_gpu(hardware_binding)
        memory_resource_config = (
            memory_resource_config or MemoryResourceConfig.default()
        )
        self._base_mr: rmm.mr.DeviceMemoryResource | None = (
            memory_resource_config.create_memory_resource()
        )
        self._mr: RmmResourceAdaptor | None = (
            None  # set after `Context` is built (below).
        )
        self._rapidsmpf_options: Options = Options.deserialize(
            rapidsmpf_options_as_bytes
        )
        self._rapidsmpf_statistics = Statistics.from_options(self._rapidsmpf_options)
        self._nranks: int = nranks
        self._py_executor = ThreadPoolExecutor(
            max_workers=num_py_executors,
            thread_name_prefix="ray-executor",
        )
        self._comm: Communicator | None = None
        self._ctx: Context | None = None
        if quent_enabled:
            self._quent_logger: cudf_polars.quent._logging.QuentLogger | None = (
                cudf_polars.quent._logging.QuentLogger()
            )
        else:
            self._quent_logger = None
        self._quent_worker = Worker(
            id=worker_id,
            engine=engine,
            instance_name=f"RankActor-{worker_id.hex[:8]}",
        )
        if self._quent_logger is not None:
            self._quent_logger.emit(self._quent_worker._init())

    def setup_root(self) -> bytes:
        """
        Initialize this actor as the root rank.

        The root actor creates a new UCXX communicator and returns the
        serialized root address, which must be passed to :meth:`setup_worker`
        on all actors to complete communicator setup.

        Returns
        -------
        Serialized UCXX root address for communicator bootstrap.
        """
        self._comm = new_communicator(
            nranks=self._nranks,
            ucx_worker=None,
            root_ucxx_address=None,
            options=self._rapidsmpf_options,
            progress_thread=ProgressThread(self._rapidsmpf_statistics),
        )
        return get_root_ucxx_address(self._comm)

    def setup_worker(self, root_ucxx_address_as_bytes: bytes) -> None:
        """
        Complete communicator bootstrap and create the streaming context.

        This method must be called concurrently on all actors, including the
        root. Non-root actors connect to the root using the provided UCXX
        address. Once all ranks have joined, the actors synchronize with a
        barrier and create their RapidsMPF streaming contexts.

        Parameters
        ----------
        root_ucxx_address_as_bytes
            Serialized UCXX root address returned by :meth:`setup_root`.
        """
        if self._comm is None:
            root_ucxx_address = ucx_api.UCXAddress.create_from_buffer(
                root_ucxx_address_as_bytes
            )
            self._comm = new_communicator(
                nranks=self._nranks,
                ucx_worker=None,
                root_ucxx_address=root_ucxx_address,
                options=self._rapidsmpf_options,
                progress_thread=ProgressThread(self._rapidsmpf_statistics),
            )
        barrier(self._comm)
        assert self._base_mr is not None
        self._ctx = Context.from_options(
            self._comm.logger,
            self._base_mr,
            self._rapidsmpf_options,
            self._rapidsmpf_statistics,
        )
        # Replace the plain base MR with the Context's internal tracking
        # adaptor so subsequent uses of `self._mr` (including the next
        # `Context.from_options` call in `reset`) share the same tracking
        # adaptor; also install it as the current RMM device resource so
        # libcudf temporary allocations use the same memory resource.
        self._mr = self._ctx.br().device_mr_adaptor()
        rmm.mr.set_current_device_resource(self._mr)

    def reset(self, *, rapidsmpf_options_as_bytes: bytes) -> None:
        """
        Rebuild the streaming Context with new options.

        Must be called collectively on all actors. A barrier ensures no
        rank tears down its Context while peers may still be using it.

        Parameters
        ----------
        rapidsmpf_options_as_bytes
            Serialized :class:`Options` to install.
        """
        if self._ctx is None:
            raise RuntimeError("reset() requires setup_worker() to have run")
        assert self._comm is not None
        # Collective: all ranks idle before any rank tears down its Context.
        if self._comm.nranks > 1:
            barrier(self._comm)
        # Drop persisted partitions before tearing down the Context.
        rank_local_store.close_all()
        self._ctx.shutdown()
        self._ctx = None
        self._rapidsmpf_options = Options.deserialize(rapidsmpf_options_as_bytes)
        self._rapidsmpf_statistics = reset_statistics_from_options(
            self._rapidsmpf_statistics, self._rapidsmpf_options
        )
        self._rapidsmpf_statistics.clear()
        assert self._base_mr is not None
        self._ctx = Context.from_options(
            self._comm.logger,
            self._base_mr,
            self._rapidsmpf_options,
            self._rapidsmpf_statistics,
        )
        # Refresh `self._mr` and the current RMM device resource to point at
        # the new Context's tracking adaptor; the previous adaptor was tied
        # to the now-shutdown Context.
        self._mr = self._ctx.br().device_mr_adaptor()
        rmm.mr.set_current_device_resource(self._mr)

    def _exit(self) -> list[dict[str, Any]]:
        # Emit the Exit event on the worker.
        # Maybe generalize this to all application-level things,
        # followed by framework (ray) level things.
        if self._quent_worker is not None and self._quent_logger is not None:
            self._quent_logger.emit(self._quent_worker._exit())
            return self._drain_quent_events()
        return []

    def shutdown(self) -> None:
        """
        Release actor-owned resources and exit the process.

        Raises `ray.exceptions.RayActorError`.
        """
        self._py_executor.shutdown(wait=True, cancel_futures=True)
        # Release resources in dependency order before exit_actor() terminates
        # the process. Shut down the Context explicitly on the same thread
        # that constructed it.
        try:
            # Drop persisted partitions before tearing down the Context.
            rank_local_store.close_all()
            if self._ctx is not None:
                self._ctx.shutdown()
        finally:
            self._ctx = None
            self._comm = None
            self._mr = None
            self._base_mr = None
            ray.actor.exit_actor()

    def get_info(self) -> ClusterInfo:
        """
        Return diagnostic information about actor placement.

        Returns
        -------
        Diagnostic information about this actor's placement and state.
        """
        return ClusterInfo.local()

    def get_statistics(self, *, clear: bool = False) -> Statistics:
        """
        Return this rank's :class:`~rapidsmpf.statistics.Statistics` object.

        The returned object is pickled by Ray when sent to the client, so the
        caller receives a detached copy.

        Parameters
        ----------
        clear
            If ``True``, clear this rank's statistics after returning a copy.

        Returns
        -------
        The rank's Statistics object (a detached copy if ``clear`` is True).
        """
        assert self._ctx is not None
        stats = self._ctx.statistics()
        if clear:
            # Return a deep copy so it survives the in-place clear of `stats`.
            detached = stats.copy()
            stats.clear()
            return detached
        return stats

    def evaluate_polars_ir(
        self,
        ir: IR,
        config_options: ConfigOptions[StreamingExecutor],
        *,
        collect_metadata: bool,
        quent_context: cudf_polars.quent.QuentContext | None,
        query_id: uuid.UUID,
    ) -> tuple[pl.DataFrame, list[ChannelMetadata] | None]:
        """
        Lower and execute a Polars IR query on this actor's GPU.

        The pre-lowered IR is collectively lowered across all actors
        (rank 0 gathers scan statistics, all ranks allgather them, then
        each rank lowers independently) and executed as a RapidsMPF
        actor network.

        Parameters
        ----------
        ir
            The pre-lowered IR tree.
        config_options
            Executor configuration forwarded from the client.
        collect_metadata
            If ``True``, collect channel metadata during execution.
        quent_context
            The client's current quent context.
        query_id
            Unique identifier for the query, propagated into actor traces.

        Returns
        -------
        result
            This rank's output partition as a Polars DataFrame.
        metadata
            Collected channel metadata if ``collect_metadata`` is ``True``,
            otherwise ``None``.

        Raises
        ------
        RuntimeError
            If :meth:`setup_worker` has not been called first.
        """
        if self._ctx is None or self._comm is None:
            raise RuntimeError("setup_worker must be called before evaluate_polars_ir")
        # Ray transfers the returned Polars DataFrame back to the client via the
        # object store (pickle / Arrow IPC). The DataFrame is already on CPU at
        # this point (to_polars() copies the result off-GPU), so no GPU memory
        # crosses process boundaries.
        local_quent_context: LocalQuentContext | None = None
        if quent_context is not None:
            assert self._quent_logger is not None
            local_quent_context = LocalQuentContext(
                context=quent_context,
                worker=self._quent_worker,
                logger=self._quent_logger,
            )
        # evaluate_on_rank always collects metadata internally so we can read
        # metadata[-1].duplicated to decide whether to suppress this rank's
        # output. The client concatenates each rank's result, so without this
        # dedup an output marked duplicated=True would appear N times. The
        # external collect_metadata parameter still controls whether the
        # collected list is returned to the client (see the return statement),
        # which is the cost we care about saving when the caller doesn't need
        # the metadata.
        gpu_df, metadata = evaluate_on_rank(
            self._ctx,
            self._comm,
            self._py_executor,
            ir,
            config_options,
            local_quent_context=local_quent_context,
            query_id=query_id,
        )
        gpu_df = drop_if_replicated(gpu_df, self._comm.rank, metadata)
        return gpu_df.to_polars(), metadata if collect_metadata else None

    def execute_persisted(
        self,
        uid: str,
        ir: IR,
        config_options: ConfigOptions[StreamingExecutor],
        *,
        query_id: uuid.UUID,
    ) -> int:
        """
        Execute a query, keeping this rank's partition GPU-resident on this actor.

        Parameters
        ----------
        uid
            The producing engine's store uid on this actor.
        ir
            The pre-lowered IR tree.
        config_options
            Executor configuration forwarded from the client.
        query_id
            Unique identifier for the query.

        Returns
        -------
        This actor's communicator rank.

        Raises
        ------
        RuntimeError
            If :meth:`setup_worker` has not been called first.
        """
        if self._ctx is None or self._comm is None:
            raise RuntimeError("setup_worker must be called before execute_persisted")
        return persisted_result.evaluate_and_persist(
            uid,
            self._ctx,
            self._comm,
            self._py_executor,
            ir,
            config_options,
            query_id,
            # Partitions are gathered to the client and concatenated, so a
            # duplicated output is deduplicated to a single copy.
            deduplicate_replicated=True,
        )

    def drop_persisted(self, uid: str, query_id: uuid.UUID) -> None:
        """Drop this query's partitions from this engine's store (idempotent)."""
        rank_local_store.drop_query(uid, query_id)

    def _drain_quent_events(self) -> list[dict[str, Any]]:
        """
        Return and clear all buffered Quent events from this actor.

        Parameters
        ----------
        emit_exit
            If True, emit Worker.exit before draining so that the exit
            event is included in the returned buffer.
        """
        if self._quent_logger is None:
            return []
        return self._quent_logger.drain()

    def _run(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        return func(*args, **kwargs)


def get_num_gpus_in_ray_cluster() -> int:
    """
    Return the number of free GPUs available for Ray actor creation.

    Raises
    ------
    RuntimeError
        If not all GPUs in the Ray cluster are free at startup.
    RuntimeError
        If no GPUs are available in the Ray cluster.
    """
    total_gpus = int(ray.cluster_resources().get("GPU", 0.0))
    # Note: available_resources() returns a snapshot and is inherently racy.
    # This is only a best-effort guard, another process could claim GPUs between
    # this check and actor creation. That is fine here, because the snapshot
    # simply determines the number of ranks used for this cluster instance.
    free_gpus = int(ray.available_resources().get("GPU", 0.0))
    if total_gpus != free_gpus:
        raise RuntimeError(
            "Ray execution expects all GPUs in the Ray cluster to be available at startup"
        )
    if free_gpus == 0:
        raise RuntimeError("No available GPUs in the Ray cluster at startup")
    return free_gpus


class RayEngine(StreamingEngine):
    """
    Multi-GPU Polars engine for Ray cluster execution.

    Creates a RapidsMPF Ray cluster and returns an engine that can be passed
    to ``LazyFrame.collect(engine=engine)``.

    Prefer :meth:`from_options` for typical use. Pass a :class:`~cudf_polars.engine.options.StreamingOptions`
    instance for a unified, typed interface. The ``__init__`` parameters (``rapidsmpf_options``,
    ``executor_options``, ``engine_options``) are intended for advanced use when
    fine-grained control is needed.

    Prefer the context-manager form in scripts: it guarantees that actors and
    Ray are shut down even if an exception is raised. In interactive environments
    such as Jupyter notebooks, the direct form lets the cluster persist across
    multiple cells without tearing it down after every query.

    If Ray is not already initialized, :func:`ray.init` is called here and
    :func:`ray.shutdown` is called by :meth:`shutdown`. If Ray is already
    initialized, cluster lifetime remains managed by the caller.

    Parameters
    ----------
    rapidsmpf_options
        RapidsMPF-specific options. Defaults to the reading ``RAPIDSMPF_*``
        environment variables.
    executor_options
        Executor-specific options (e.g. ``max_rows_per_partition``).
    engine_options
        Engine-specific keyword arguments (e.g. ``raise_on_fail``,
        ``parquet_options``).
    ray_init_options
        Keyword arguments forwarded to :func:`ray.init` when Ray is not
        already initialized.
    num_ranks
        Number of ranks (Ray actors) to create. When ``None`` (the default),
        one rank is created per available GPU using Ray's GPU scheduling,
        which provides placement guarantees and topology-aware hardware
        binding.
        When set, bypasses Ray's GPU resource accounting
        so that actors do not contend for GPU resource slots. This allows
        multiple ``RayEngine`` instances to share a single Ray cluster
        and enables oversubscribed execution on limited GPU hardware.
        Hardware binding is disabled implicitly but the caller must
        pass ``engine_options={"allow_gpu_sharing": True}`` explicitly
        to acknowledge the multi-tenant GPU semantics.

        Note, oversubscription does not increase throughput. When multiple
        ranks share a GPU, they compete for the same compute and
        memory resources, which may increase memory pressure and
        reduce overall performance. This option is primarily useful
        for testing multi-rank code paths on machines with fewer
        GPUs than ranks, and for downstream projects that need to
        validate distributed logic in resource-constrained CI
        environments.

    Raises
    ------
    RuntimeError
        If called from within an ``rrun`` cluster.
    RuntimeError
        If not all GPUs in the Ray cluster are free at startup
        (only when ``num_ranks`` is ``None``).
    RuntimeError
        If no GPUs are available in the Ray cluster
        (only when ``num_ranks`` is ``None``).
    TypeError
        If ``executor_options`` or ``engine_options`` contains a reserved key.
    ValueError
        If ``num_ranks`` is set but ``engine_options["allow_gpu_sharing"] == False``
    ValueError
        If ``num_ranks`` is set to a value less than 1.

    Examples
    --------
    Context-manager style:

    >>> with RayEngine() as engine:  # doctest: +SKIP
    ...     result = pl.LazyFrame({"a": [1, 2, 3]}).collect(engine=engine)

    Jupyter / manual style:

    >>> engine = RayEngine()  # doctest: +SKIP
    >>> result = pl.LazyFrame({"a": [1, 2, 3]}).collect(engine=engine)  # doctest: +SKIP
    >>> engine.shutdown()  # doctest: +SKIP
    """

    def __init__(
        self,
        *,
        rapidsmpf_options: Options | None = None,
        executor_options: dict[str, Any] | None = None,
        engine_options: dict[str, Any] | None = None,
        ray_init_options: dict[str, Any] | None = None,
        num_ranks: int | None = None,
    ) -> None:
        executor_options = executor_options or {}
        engine_options = engine_options or {}
        ray_init_options = ray_init_options or {}

        if bootstrap.is_running_with_rrun():
            raise RuntimeError(
                "RayEngine must not be created from within an rrun cluster. Instead "
                "launch the rrun cluster separately and let this client connect to its "
                "cluster nodes."
            )

        check_reserved_keys(executor_options, engine_options)

        quent_context: cudf_polars.quent.QuentContext | None = executor_options.get(
            "quent_context"
        )
        if quent_context is not None:
            self._quent_logger = cudf_polars.quent._logging.QuentLogger()
        else:
            self._quent_logger = None

        if quent_context is not None:
            executor_options.setdefault("quent_context", quent_context)
            assert self._quent_logger is not None
            quent_context._emit_engine_init_events(self._quent_logger)
            engine = quent_context.engine
        else:
            engine = cudf_polars.quent.Engine(id=uuid.uuid4())

        # This engine's store uid, used to key its partitions in each actor's process rank-local store.
        self._store_uid = uuid.uuid4().hex

        if num_ranks is not None:
            if num_ranks < 1:
                raise ValueError(f"num_ranks must be >= 1 (got {num_ranks})")
            if not engine_options.get("allow_gpu_sharing", False):
                raise ValueError(
                    "num_ranks requires engine_options['allow_gpu_sharing']=True"
                )
            hw_binding = HardwareBindingPolicy(enabled=False)
        else:
            hw_binding = engine_options.get("hardware_binding", HardwareBindingPolicy())

        mr_config: MemoryResourceConfig | None = engine_options.get(
            "memory_resource_config", None
        )

        self.rapidsmpf_options = resolve_rapidsmpf_options(rapidsmpf_options)
        rapidsmpf_options_as_bytes = self.rapidsmpf_options.serialize()

        exit_stack = contextlib.ExitStack()
        if not ray.is_initialized():
            # Prevent Ray from overriding CUDA_VISIBLE_DEVICES to "" when a worker
            # process starts with zero visible GPUs (e.g., the driver process itself).
            # In the future, this behavior will become the default in Ray.
            os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
            ray.init(**ray_init_options)
            # Ensure Ray is shut down when RayEngine shuts down.
            exit_stack.callback(ray.shutdown)

        try:
            # Override num_gpus=0 when num_ranks is set so Ray doesn't gate
            # actor scheduling on GPU resources; .options() with no overrides
            # is a no-op for the default path.
            actor_options: dict[str, Any] = (
                {"num_gpus": 0} if num_ranks is not None else {}
            )
            nranks = (
                num_ranks if num_ranks is not None else get_num_gpus_in_ray_cluster()
            )
            worker_ids = [uuid.uuid4() for _ in range(nranks)]

            rank_actors: list[ActorHandle[RankActor]] = [
                RankActor.options(**actor_options).remote(  # type: ignore[attr-defined]
                    nranks=nranks,
                    rapidsmpf_options_as_bytes=rapidsmpf_options_as_bytes,
                    num_py_executors=cast(
                        "int",
                        executor_options.get("num_py_executors", 8),
                    ),
                    hardware_binding=hw_binding,
                    memory_resource_config=mr_config,
                    worker_id=worker_id,
                    engine=engine,
                    quent_enabled=quent_context is not None,
                )
                for worker_id in worker_ids
            ]

            root_ucxx_address_as_bytes = ray.get(rank_actors[0].setup_root.remote())
            # Call setup_worker on all actors concurrently, including the root.
            # The root skips communicator creation and proceeds directly to the barrier.
            # Non-root actors create their communicators and then join the barrier.
            ray.get(
                [
                    rank.setup_worker.remote(root_ucxx_address_as_bytes)
                    for rank in rank_actors
                ]
            )

            self._rank_actors: list[ActorHandle[RankActor]] | None = rank_actors
            super().__init__(
                nranks=nranks,
                executor_options={
                    **executor_options,
                    "cluster": "ray",
                    "ray_context": RayContext(rank_actors, self._quent_logger),
                },
                engine_options=engine_options,
                exit_stack=exit_stack,
            )
        except Exception:
            exit_stack.close()
            raise

    def _reset(
        self,
        *,
        rapidsmpf_options: Options | None = None,
        executor_options: dict[str, Any] | None = None,
        engine_options: dict[str, Any] | None = None,
    ) -> None:
        """Reset the engine; see :meth:`StreamingEngine._reset` for the contract."""
        if self._rank_actors is None:
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

        # Reset all actor Contexts collectively. ``ray.get`` blocks until
        # every actor's reset returns; the per-actor barrier inside
        # :meth:`RankActor.reset` synchronizes the teardown across ranks.
        ray.get(
            [
                rank.reset.remote(
                    rapidsmpf_options_as_bytes=rapidsmpf_options_as_bytes,
                )
                for rank in self._rank_actors
            ]
        )

        # Re-run ``StreamingEngine.__init__`` on the existing instance to
        # reconfigure the polars ``GPUEngine`` layer (``self.config``,
        # ``self.device``, etc.) with the new options. Pass the existing
        # ``self._exit_stack`` so any registered shutdown callbacks
        # (e.g. ``ray.shutdown`` if this engine bootstrapped Ray itself)
        # remain registered.
        StreamingEngine.__init__(
            self,
            nranks=len(self._rank_actors),
            executor_options={
                **executor_options,
                "cluster": "ray",
                "ray_context": RayContext(self._rank_actors, self._quent_logger),
            },
            engine_options=engine_options,
            exit_stack=self._exit_stack,
        )

    @classmethod
    def from_options(
        cls,
        options: StreamingOptions,
        *,
        ray_init_options: dict[str, object] | None = None,
    ) -> RayEngine:
        """
        Create a :class:`RayEngine` from a :class:`~cudf_polars.engine.options.StreamingOptions` object.

        This is the recommended way to construct a ``RayEngine`` for typical
        use. All RapidsMPF, executor, and engine options are read from
        ``options``; unset fields fall back to environment variables and then
        to built-in defaults.

        Parameters
        ----------
        options
            Unified streaming configuration.
        ray_init_options
            Keyword arguments forwarded to :func:`ray.init` when Ray is not
            already initialized. These are Ray infrastructure settings and are
            kept separate from streaming behavior options.

        Returns
        -------
        A new :class:`RayEngine` instance.

        Examples
        --------
        >>> from cudf_polars.engine.options import StreamingOptions
        >>> opts = StreamingOptions(num_streaming_threads=4, fallback_mode="silent")
        >>> with RayEngine.from_options(opts) as engine:  # doctest: +SKIP
        ...     result = pl.LazyFrame({"a": [1, 2, 3]}).collect(engine=engine)
        """
        return cls(
            rapidsmpf_options=options.to_rapidsmpf_options(),
            executor_options=options.to_executor_options(),
            engine_options=options.to_engine_options(),
            ray_init_options=ray_init_options,
        )

    @property
    def rank_actors(self) -> list[ActorHandle[RankActor]]:
        """
        List of Ray rank actor handles.

        Raises
        ------
        RuntimeError
            If called after :meth:`shutdown`.
        """
        if self._rank_actors is None:
            raise RuntimeError("rank_actors is not available after shutdown")
        return self._rank_actors

    def gather_cluster_info(self) -> list[ClusterInfo]:
        """
        Collect diagnostic information from every rank.

        Returns
        -------
        List of :class:`~cudf_polars.engine.core.ClusterInfo`, one per rank.
        """
        return ray.get([rank.get_info.remote() for rank in self.rank_actors])

    def gather_statistics(self, *, clear: bool = False) -> list[Statistics]:
        """
        Collect statistics from every rank via Ray.

        Parameters
        ----------
        clear
            If ``True``, clear each rank's statistics after gathering.

        Returns
        -------
        List of :class:`~rapidsmpf.statistics.Statistics`, one per rank,
        ordered by rank index.
        """
        return ray.get(
            [rank.get_statistics.remote(clear=clear) for rank in self.rank_actors]
        )

    def shutdown(self) -> None:
        """
        Shut down all rank actors and release resources.

        If Ray was initialized by this engine, also calls :func:`ray.shutdown`.
        Safe to call more than once.

        Raises
        ------
        ExceptionGroup
            If one or more actors raise an unexpected exception during shutdown.
        """
        if self._rank_actors is None:
            return  # already shut down; idempotent
        exceptions: list[Exception] = []
        quent_context: cudf_polars.quent.QuentContext | None = self.config[
            "executor_options"
        ].get("quent_context")
        try:
            # If Ray is no longer initialized (for example, if ``ray.shutdown()`` was
            # called before ``RayEngine.shutdown()``), the actors are gone as well.
            # Calling ``.remote()`` in this state would trigger Ray's ``auto_init_hook``
            # and start a new cluster.
            if not ray.is_initialized():
                return

            exit_refs = [a._exit.remote() for a in self._rank_actors]
            for ref in exit_refs:
                try:
                    exit_events = ray.get(ref)
                except ray.exceptions.RayActorError:
                    pass  # expected: exit_actor() terminates the process immediately
                except Exception as e:
                    exceptions.append(e)
                else:
                    self._quent_events_raw.extend(exit_events)

            refs = [a.shutdown.remote() for a in self._rank_actors]
            for ref in refs:
                try:
                    ray.get(ref)
                except ray.exceptions.RayActorError:
                    pass  # expected: exit_actor() terminates the process immediately
                except Exception as e:
                    exceptions.append(e)

            if exceptions:
                raise ExceptionGroup("Actor shutdown failed", exceptions)
        finally:
            if quent_context is not None:
                assert self._quent_logger is not None
                quent_context._emit_engine_exit_events(self._quent_logger)
            self._rank_actors = None
            super().shutdown()

        # gather the client-side events.
        if self._quent_logger is not None:
            self._quent_events_raw.extend(self._quent_logger.drain())
        # final inplace sort of the events.
        self._quent_events_raw.sort(key=lambda x: x["timestamp"])

    def _run(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> list[T]:
        return ray.get(
            [rank._run.remote(func, *args, **kwargs) for rank in self.rank_actors]
        )

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
        Distributed query result backed by per-rank GPU-persisted partitions
        kept in the producing actors.

        Examples
        --------
        >>> with RayEngine() as engine:  # doctest: +SKIP
        ...     result = engine.execute(pl.scan_parquet("data/*.parquet"))
        ...     df = result.lazy().filter(pl.col("x") > 0).collect(engine=engine)
        """
        backend = RayPersistedBackend(self._store_uid, self.rank_actors)
        return execute_persisted_query(self, lf, backend, self._store_uid)
