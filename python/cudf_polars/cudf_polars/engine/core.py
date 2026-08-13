# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multi-GPU frontend core."""

from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import socket
import threading
import uuid
import weakref
from typing import TYPE_CHECKING, Any, ClassVar, Self, TypeVar

import cuda.core

import polars as pl

import pylibcudf as plc
from cudf_streaming.table_chunk import TableChunk
from rapidsmpf.coll import AllGather
from rapidsmpf.config import Options, get_environment_variables
from rapidsmpf.memory.packed_data import PackedData
from rapidsmpf.statistics import Statistics
from rapidsmpf.streaming.core.actor import run_actor_network

from cudf_polars.containers import DataFrame
from cudf_polars.dsl.ir import IRExecutionContext
from cudf_polars.dsl.utils.io import (
    attach_cached_parquet_metadata,
    prefetch_parquet_file_metadata_for_ir,
)
from cudf_polars.quent._plan import build_plan
from cudf_polars.streaming.actor_graph.collectives import ReserveOpIDs
from cudf_polars.streaming.actor_graph.collectives.common import reserve_op_id
from cudf_polars.streaming.actor_graph.core import generate_network
from cudf_polars.streaming.actor_graph.tracing import log_query_plan
from cudf_polars.streaming.actor_graph.utils import empty_table_chunk
from cudf_polars.streaming.base import StatsCollector
from cudf_polars.streaming.parallel import lower_ir_graph_with_node_map
from cudf_polars.streaming.statistics import collect_statistics
from cudf_polars.streaming.utils import _concat
from cudf_polars.utils.config import get_total_device_memory

if TYPE_CHECKING:
    from collections.abc import Callable, MutableMapping
    from concurrent.futures import Executor, ThreadPoolExecutor

    import rapidsmpf.config
    from cudf_streaming.channel_metadata import ChannelMetadata
    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.memory.buffer_resource import BufferResource
    from rapidsmpf.streaming.core.context import Context

    import cudf_polars.quent
    import cudf_polars.quent._logging
    from cudf_polars.dsl.ir import IR
    from cudf_polars.dsl.translate import Translator
    from cudf_polars.quent._context import LocalQuentContext
    from cudf_polars.streaming.base import PartitionInfo
    from cudf_polars.streaming.parallel import ConfigOptions
    from cudf_polars.utils.config import StreamingExecutor


T = TypeVar("T")


def reset_statistics_from_options(
    statistics: Statistics, options: Options
) -> Statistics:
    """
    Reset the enabled state of a statistics object from options.

    Parameters
    ----------
    statistics
        Statistics to reset.
    options
        Options providing new enabled setting.

    Returns
    -------
    Statistics
        Reset statistics object.

    Notes
    -----
    Does not clear the statistics.
    """
    if Statistics.from_options(options).enabled:
        statistics.enable()
    else:
        statistics.disable()
    return statistics


def resolve_rapidsmpf_options(rapidsmpf_options: Options | None) -> Options:
    """
    Resolve ``rapidsmpf_options`` and apply cross-frontend defaults.

    If ``None`` is passed, constructs an ``Options`` instance from
    environment variables. Then applies defaults that should be consistent
    across SPMD, Ray, and Dask. Defaults are set via
    ``Options.insert_if_absent``, so explicit values or environment
    variables always take precedence.

    Defaults applied:

    - ``num_streaming_threads=4``: moderate worker count for the rapidsmpf
      streaming runtime, shared across frontends.

    Parameters
    ----------
    rapidsmpf_options
        Existing options to resolve, or ``None`` to construct from environment
        variables.

    Returns
    -------
    Options
        Resolved options with cross-frontend defaults applied.
    """
    if rapidsmpf_options is None:
        rapidsmpf_options = Options(get_environment_variables())

    rapidsmpf_options.insert_if_absent({"num_streaming_threads": "4"})
    return rapidsmpf_options


@dataclasses.dataclass(frozen=True)
class ClusterInfo:
    """
    Diagnostic information about a single rank in the cluster.

    Attributes
    ----------
    pid
        Process ID of the current rank.
    hostname
        Hostname of the machine running this rank.
    cuda_visible_devices
        Value of ``CUDA_VISIBLE_DEVICES``, or ``None`` if unset.
    gpu_uuid
        UUID of the current CUDA device.
    device_memory
        Total device memory in bytes, or ``None`` if unknown.
    """

    pid: int
    hostname: str
    cuda_visible_devices: str | None
    gpu_uuid: str
    device_memory: int | None = None

    @classmethod
    def local(cls) -> ClusterInfo:
        """
        Build a :class:`ClusterInfo` for the current process and GPU.

        Returns
        -------
        Diagnostic information for this rank.
        """
        return cls(
            pid=os.getpid(),
            hostname=socket.gethostname(),
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            gpu_uuid=cuda.core.Device().uuid,
            device_memory=get_total_device_memory(),
        )


class StreamingEngine(pl.GPUEngine):
    """
    Base class for multi-GPU Polars engines.

    The engine manages the lifecycle of a streaming execution and can
    be used as a context manager. On exit, :meth:`shutdown` is called.

    Notes
    -----
    The engine must be created and shut down on the same thread. In particular,
    destruction and context manager exit must occur on the thread that created
    the instance.

    Parameters
    ----------
    nranks
        Number of ranks (workers or GPUs) in the cluster.
    executor_options
        Executor-specific options (e.g. ``max_rows_per_partition``).
    engine_options
        Engine-specific keyword arguments (e.g. ``raise_on_fail``,
        ``parquet_options``).
    exit_stack
        A :class:`contextlib.ExitStack` whose registered contexts are closed
        when :meth:`shutdown` is called. If ``None``, an empty stack is created.
    """

    _quent_logger: cudf_polars.quent._logging.QuentLogger | None
    rapidsmpf_options: rapidsmpf.config.Options
    # Process-wide registry of every live :class:`StreamingEngine`. Used by
    # :class:`DefaultSingletonEngine` to enforce that no other engine is
    # alive when the singleton is constructed.
    _active_engines: ClassVar[weakref.WeakSet[StreamingEngine]] = weakref.WeakSet()
    _active_engines_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(
        self,
        *,
        nranks: int,
        executor_options: dict[str, Any],
        engine_options: dict[str, Any],
        exit_stack: contextlib.ExitStack | None = None,
    ):
        # Refuse to construct if a ``DefaultSingletonEngine`` is alive
        # (no-op for the singleton itself).
        from cudf_polars.engine.default_singleton_engine import (
            check_no_live_default_singleton,
        )

        check_no_live_default_singleton(self)
        self._nranks = nranks
        self._quent_events_raw: list[dict[str, Any]] = []  # populated on shutdown
        self._exit_stack: contextlib.ExitStack | None = (
            exit_stack or contextlib.ExitStack()
        )

        # Gather `min_device_size` from the cluster
        cluster_infos: list[ClusterInfo] = self.gather_cluster_info()
        device_memories = [info.device_memory for info in cluster_infos]
        executor_options["min_device_size"] = (
            None
            if any(dm is None for dm in device_memories)
            else min(device_memories, default=None)  # type: ignore[type-var]  # (None entries excluded by prior check)
        )

        # allow_gpu_sharing is consumed here since polars' GPUEngine doesn't
        # accept it.
        engine_options = dict(engine_options)
        allow_gpu_sharing = engine_options.pop("allow_gpu_sharing", False)
        super().__init__(
            executor="streaming",
            executor_options=executor_options,
            **engine_options,
        )
        if nranks > 1 and not allow_gpu_sharing:
            uuids = [info.gpu_uuid for info in cluster_infos]
            if len(uuids) != len(set(uuids)):
                raise RuntimeError(
                    "Multiple ranks share the same GPU (UUID collision detected). "
                    f"UUIDs: {uuids}. Set allow_gpu_sharing=True to allow this."
                )
        with StreamingEngine._active_engines_lock:
            StreamingEngine._active_engines.add(self)

    @classmethod
    def _active_engine_count(cls) -> int:
        """
        Return the number of currently-live :class:`StreamingEngine` instances.

        "Live" means constructed and not yet shut down (or garbage collected).
        The count is process-wide and shared across all subclasses.

        Returns
        -------
        Number of live engines, including ``self`` if called on a live
        instance.
        """
        with StreamingEngine._active_engines_lock:
            return len(StreamingEngine._active_engines)

    @property
    def nranks(self) -> int:
        """
        Number of ranks (for example GPUs or workers) in the cluster.

        Local execution without a cluster returns 1.

        Returns
        -------
        Number of ranks.
        """
        return self._nranks

    def gather_cluster_info(self) -> list[ClusterInfo]:
        """
        Collect diagnostic information from every rank.

        Returns
        -------
        List of :class:`ClusterInfo`, one per rank.
        """
        raise NotImplementedError

    def gather_statistics(self, *, clear: bool = False) -> list[Statistics]:
        """
        Collect statistics from every rank.

        Parameters
        ----------
        clear
            If ``True``, clear each rank's statistics after gathering.

        Returns
        -------
        List of :class:`~rapidsmpf.statistics.Statistics`, one per rank,
        ordered by rank index.
        """
        raise NotImplementedError

    def global_statistics(self, *, clear: bool = False) -> Statistics:
        """
        Collect statistics from every rank and merge them into a single global statistics.

        Parameters
        ----------
        clear
            If ``True``, clear each rank's statistics after gathering.

        Returns
        -------
        A merged :class:`~rapidsmpf.statistics.Statistics`: per-stat counts
        and values are summed, maxima are reduced with ``max``. Formatters
        are taken from rank 0.
        """
        return Statistics.merge(self.gather_statistics(clear=clear))

    def _reset(
        self,
        *,
        rapidsmpf_options: Options | None = None,
        executor_options: dict[str, Any] | None = None,
        engine_options: dict[str, Any] | None = None,
    ) -> None:
        """
        Reset the engine with new options, keeping cluster resources alive.

        The following inputs are fixed at construction time and cannot change:
          - ``num_ranks``
          - ``num_py_executors`` (in ``executor_options``)
          - ``hardware_binding`` (in ``engine_options``)
          - ``memory_resource_config`` (in ``engine_options``)

        Subclasses must override this method. The override should:
          1. Raise :class:`RuntimeError` if the engine is already shut down.
          2. Call ``super()._reset(...)`` to apply the universal option validation below.
          3. Perform the backend-specific rebuild.

        Parameters
        ----------
        rapidsmpf_options
            New :class:`Options` for each rank's :class:`Context`.
            ``None`` is treated as an empty dict.
        executor_options
            New executor options for the polars ``GPUEngine`` layer.
            ``None`` is treated as an empty dict.
        engine_options
            New engine options for the polars ``GPUEngine`` layer.
            ``None`` is treated as an empty dict.

        Raises
        ------
        ValueError
            If ``executor_options`` or ``engine_options`` contains a
            construction-time-only key (see list above), or if a
            reserved key is set (via :func:`check_reserved_keys`).
        """
        executor_options = executor_options or {}
        engine_options = engine_options or {}
        check_reserved_keys(executor_options, engine_options)

        _disallowed_exec = {"num_py_executors"} & executor_options.keys()
        if _disallowed_exec:
            raise ValueError(
                f"executor_options keys {sorted(_disallowed_exec)} cannot be "
                "changed via _reset(). Construct a fresh engine instead."
            )
        _disallowed_engine = {
            "hardware_binding",
            "memory_resource_config",
        } & engine_options.keys()
        if _disallowed_engine:
            raise ValueError(
                f"engine_options keys {sorted(_disallowed_engine)} cannot be "
                "changed via _reset(). Construct a fresh engine instead."
            )

    def shutdown(self) -> None:
        """
        Shut down engine and release all owned resources.

        Idempotent: safe to call more than once. Must be called on the same
        thread that created the engine.
        """
        if self._exit_stack is None:
            return  # already shut down
        try:
            self._exit_stack.close()
        finally:
            self._exit_stack = None
            self.device = None
            self.memory_resource = None
            self.config = {}
            with StreamingEngine._active_engines_lock:
                StreamingEngine._active_engines.discard(self)

    def __enter__(self) -> Self:
        """Enter the context manager, returning ``self``."""
        return self

    def __exit__(self, *_: object) -> None:
        """Exit the context manager, calling :meth:`shutdown`."""
        self.shutdown()

    @property
    def _quent_events(self) -> list[dict[str, Any]]:
        """Return all Quent telemetry events collected during the engine's lifecycle."""
        # Not ready to make this public yet.
        return [x["event"] for x in self._quent_events_raw]

    def _run(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> list[T]:
        """
        Execute a function on all ranks.

        Parameters
        ----------
        func
            Function to execute.
        args
            Arguments to pass to the function.
        kwargs
            Keyword arguments to pass to the function.

        Returns
        -------
        List of results from calling ``func``, one per rank.
        """
        raise NotImplementedError


def _find_memory_error(exc: BaseException) -> MemoryError | None:
    """Recursively search for MemoryErrors."""
    if isinstance(exc, MemoryError):
        return exc
    elif isinstance(exc, BaseExceptionGroup):
        for sub in exc.exceptions:
            if (mem_error := _find_memory_error(sub)) is not None:
                return mem_error
    return None


def execute_ir_on_rank(
    ctx: Context,
    comm: Communicator,
    ir: IR,
    ir_context: IRExecutionContext,
    partition_info: MutableMapping[IR, PartitionInfo],
    config_options: ConfigOptions[StreamingExecutor],
    stats: StatsCollector,
    collective_id_map: dict[IR, list[int]],
) -> tuple[DataFrame, list[ChannelMetadata]]:
    """
    Execute a Polars IR query on a single rank's GPU.

    Shared implementation used by the frontends. Each frontend acquires its local
    ``ctx``, ``comm``, and ``py_executor`` from its own per-rank state and delegates
    to this function for the actual execution.

    Parameters
    ----------
    ctx
        The active RapidsMPF streaming context for this rank.
    comm
        The active RapidsMPF communicator for this rank.
    ir
        Root IR node describing the query.
    ir_context
        Execution context reused across scan-task execution.
    partition_info
        Per-node partition metadata.
    config_options
        Executor configuration forwarded from the client.
    stats
        Statistics collector.
    collective_id_map
        Mapping from IR nodes to their pre-allocated collective operation IDs.

    Returns
    -------
    result
        This rank's output fragment as a GPU-resident :class:`~cudf_polars.containers.DataFrame`.
    metadata
        Collected channel metadata.
    """
    metadata_collector: list[ChannelMetadata] = []

    nodes, output = generate_network(
        ctx,
        comm,
        ir,
        partition_info,
        config_options,
        stats,
        ir_context=ir_context,
        collective_id_map=collective_id_map,
        metadata_collector=metadata_collector,
    )

    try:
        run_actor_network(ctx, actors=nodes)
    except (MemoryError, BaseExceptionGroup) as e:
        if (mem_error := _find_memory_error(e)) is not None:
            target_partition_size = config_options.executor.target_partition_size
            hint = (
                f"Try lowering `target_partition_size` (current {target_partition_size}) "
                f"and/or RAPIDSMPF_SPILL_DEVICE_LIMIT (default '80%') to reduce peak memory."
                f"\nSee https://docs.rapids.ai/api/cudf/stable/cudf_polars/memory_errors/ "
                f"for troubleshooting guidance."
                f"\nOriginal error:\n{mem_error}"
            )
            raise MemoryError(hint) from e
        else:
            raise

    messages = output.release()
    chunks = [
        TableChunk.from_message(msg, br=ctx.br()).make_available_and_spill(
            ctx.br(), allow_overbooking=True
        )
        for msg in messages
    ]
    if chunks:
        dfs = [
            DataFrame.from_table(
                chunk.table_view(),
                list(ir.schema.keys()),
                list(ir.schema.values()),
                chunk.stream,
            )
            for chunk in chunks
        ]
        df = _concat(*dfs, context=ir_context)
    else:
        stream = ir_context.get_cuda_stream()
        chunk = empty_table_chunk(ir, ctx, stream)
        df = DataFrame.from_table(
            chunk.table_view(),
            list(ir.schema.keys()),
            list(ir.schema.values()),
            stream,
        )
    return df, metadata_collector


_RESERVED_EXECUTOR_KEYS: frozenset[str] = frozenset(
    {"cluster", "spmd_context", "ray_context", "dask_context"}
)
_RESERVED_ENGINE_KEYS: frozenset[str] = frozenset({"memory_resource", "executor"})


def check_reserved_keys(
    executor_options: dict[str, Any],
    engine_options: dict[str, Any],
) -> None:
    """
    Raise :exc:`TypeError` if any reserved keys are present in the option dicts.

    Parameters
    ----------
    executor_options
        Executor-specific options to validate.
    engine_options
        Engine-specific options to validate.

    Raises
    ------
    TypeError
        If ``executor_options`` contains any reserved key.
    TypeError
        If ``engine_options`` contains any reserved key.
    """
    if bad := _RESERVED_EXECUTOR_KEYS & executor_options.keys():
        raise TypeError(f"executor_options may not contain reserved keys: {bad}")
    if bad := _RESERVED_ENGINE_KEYS & engine_options.keys():
        raise TypeError(f"engine_options may not contain reserved keys: {bad}")


def all_gather_host_data(
    comm: Communicator,
    br: BufferResource,
    op_id: int,
    data: bytes | bytearray,
) -> list[bytes]:
    """
    Gather host data from every rank using an AllGather collective.

    Each rank contributes a buffer of host bytes; every rank receives back
    an ordered list containing the contributions from all ranks (index `i`
    holds the bytes sent by rank `i`).

    This function is blocking: all ranks must call it, and each rank
    waits until the collective completes. The input buffer is copied
    and cannot be stream-ordered.

    Parameters
    ----------
    comm
        The communicator shared by all participating ranks.
    br
        Buffer resource for memory allocation.
    op_id
        Unique operation identifier for this collective.
    data
        Host-side buffer to broadcast from this rank.  Accepts any object
        that implements the buffer protocol (``bytes``, ``bytearray``,
        ``memoryview``, etc.).

    Returns
    -------
    List of bytes, one element per rank, ordered by rank index.
    """
    allgather = AllGather(comm=comm, op_id=op_id, br=br)
    # TODO: Make AllGather (bulk) a context manager so this becomes
    # with AllGather(...) as ag:
    #     ag.insert(0, PackedData.from_host_bytes(data, br))
    # results = ag.wait_and_extract(ordered=True)
    try:
        allgather.insert(0, PackedData.from_host_bytes(data, br))
    finally:
        allgather.insert_finished()
    results = allgather.wait_and_extract(ordered=True)
    return [r.to_host_bytes() for r in results]


def allgather_stats(
    comm: Communicator,
    br: BufferResource,
    ir: IR,
    config_options: ConfigOptions[StreamingExecutor],
    executor: Executor,
) -> StatsCollector:
    """
    Collect scan statistics on rank 0 and distribute to all ranks.

    When ``comm.nranks == 1`` the allgather is skipped and statistics are
    collected locally.

    Parameters
    ----------
    comm
        Communicator shared by all participating ranks.
    br
        Buffer resource for the allgather allocation.
    ir
        Root of the pre-lowered IR graph (same object on every rank).
    config_options
        Executor configuration.
    executor: concurrent.futures.Executor
        Executor to use for IO operations. This function does not start
        or shutdown the executor.

    Returns
    -------
    A :class:`StatsCollector` valid for the local rank's IR node objects.
    """
    if comm.nranks == 1:
        return collect_statistics(ir, config_options, executor)

    if comm.rank == 0:
        stats = collect_statistics(ir, config_options, executor)
        data = json.dumps(stats.serialize(ir)).encode()
    else:
        data = b""

    with reserve_op_id() as op_id:
        all_data = all_gather_host_data(comm, br, op_id, data)

    if comm.rank == 0:
        return stats
    return StatsCollector.deserialize(json.loads(all_data[0]), ir)


def evaluate_on_rank(
    ctx: Context,
    comm: Communicator,
    py_executor: ThreadPoolExecutor,
    ir: IR,
    config_options: ConfigOptions[StreamingExecutor],
    *,
    collect_metadata: bool = False,
    local_quent_context: LocalQuentContext | None = None,
    query_id: uuid.UUID,
) -> tuple[DataFrame, list[ChannelMetadata]]:
    """
    Evaluate a polars IR plan on a single rank.

    This is the main worker-side entry point for multi-rank execution.
    It performs the following steps collectively across all ranks:

    1. Collect statistics (on rank 0 and allgather)
    2. Lower the IR graph
    3. Reserve collective operation IDs
    4. Execute the lowered pipeline

    Parameters
    ----------
    ctx
        The active RapidsMPF streaming context for this rank.
    comm
        The active RapidsMPF communicator for this rank.
    py_executor
        Thread-pool executor used to drive the actor network.
    ir
        Root of the **pre-lowered** IR graph.
    config_options
        Executor configuration forwarded from the client.
    collect_metadata
        Whether to collect channel metadata during execution.
    local_quent_context
        The local Quent context for this rank, or ``None`` when tracing is
        disabled.
    query_id
        A unique identifier for the query.

    Returns
    -------
    result
        This rank's output fragment as a GPU-resident :class:`~cudf_polars.containers.DataFrame`.
    metadata
        Collected channel metadata.
    """
    stats = allgather_stats(comm, ctx.br(), ir, config_options, py_executor)

    lowering, node_map = lower_ir_graph_with_node_map(
        ir, config_options, stats, rank=comm.rank, nranks=comm.nranks
    )
    optimized = lowering.optimized
    ir = lowering.lowered
    partition_info = lowering.partition_info
    if config_options.executor.quent_context is not None:
        assert local_quent_context is not None
        logical_plan_id = optimized.get_stable_plan_id()
        plan, ops, ports, logical_op_by_id = build_plan(
            optimized,
            config_options,
            query=local_quent_context.context.query,
            plan_id=logical_plan_id,
            worker=local_quent_context.worker,
            instance_name="logical",
            parent_plan=None,
            parent_operators_by_node_id=None,
        )
        if comm.rank == 0:
            local_quent_context.context._emit_plan_declarations(
                local_quent_context.logger, plan, ops, ports
            )

    if comm.rank == 0:
        log_query_plan(ir, config_options)

    if config_options.executor.quent_context is not None:
        assert local_quent_context is not None
        physical_plan_id = uuid.uuid4()
        local_quent_context.context._emit_physical_plan_events(
            local_quent_context.logger,
            ir,
            config_options,
            plan_id=physical_plan_id,
            worker=local_quent_context.worker,
            parent_plan=plan,
            node_map=node_map,
            logical_op_by_id=logical_op_by_id,
        )
    ir_context = IRExecutionContext(
        py_executor, get_cuda_stream=ctx.br().stream_pool.get_stream, query_id=query_id
    )

    if config_options.parquet_options.prefetch_file_metadata:
        cached_parquet_info_map = prefetch_parquet_file_metadata_for_ir(
            ir,
            ir_context.py_executor,
            stats=stats,
        )
        attach_cached_parquet_metadata(ir, cached_parquet_info_map)

    with ReserveOpIDs(ir, config_options) as collective_id_map:
        return execute_ir_on_rank(
            ctx,
            comm,
            ir,
            ir_context,
            partition_info,
            config_options,
            stats,
            collective_id_map,
        )


def is_duplicated_output(metadata: list[ChannelMetadata] | None) -> bool:
    """
    Return whether a query's output is duplicated across ranks.

    A duplicated output is an identical, complete copy held on every rank (for
    example the result of a global sort/limit), signalled by the ``duplicated``
    channel flag on the final output.

    Parameters
    ----------
    metadata
        Channel metadata for the query.

    Returns
    -------
    ``True`` if the output is an identical copy held on every rank.
    """
    return bool(metadata and metadata[-1].duplicated)


def drop_if_replicated(
    df: DataFrame, rank: int, metadata: list[ChannelMetadata] | None
) -> DataFrame:
    """
    Drop a duplicated output on non-root ranks.

    Parameters
    ----------
    df
        This rank's output partition.
    rank
        This rank's index within the cluster.
    metadata
        Channel metadata for the query.

    Returns
    -------
    ``df`` or a freshly-allocated empty same-schema frame.
    """
    if rank != 0 and is_duplicated_output(metadata):
        return DataFrame.from_table(
            plc.copying.empty_like(df.table, stream=df.stream),
            df.column_names,
            df.dtypes,
            df.stream,
        )
    return df


def raise_for_translation_errors(translator: Translator) -> None:
    """
    Raise if the translator recorded unsupported operations.

    Parameters
    ----------
    translator
        The translator whose :attr:`~cudf_polars.dsl.translate.Translator.errors`
        are checked.

    Raises
    ------
    NotImplementedError
        If the query contains operations unsupported on the GPU.
    """
    error = translator.unsupported_operations_error()
    if error is not None:
        raise error
