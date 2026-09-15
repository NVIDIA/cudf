# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Quent telemetry tracing."""

from __future__ import annotations

import dataclasses
import json
import threading
import uuid
from typing import TYPE_CHECKING

from cudf_polars.quent._plan import (
    build_parent_operators_map,
    build_plan,
)
from cudf_polars.quent._types import (
    Channel,
    Engine,
    Implementation,
    Memory,
    Network,
    Processor,
    Query,
    QueryGroup,
    ThreadPool,
)
from cudf_polars.utils.config import get_total_device_memory

if TYPE_CHECKING:
    from typing import Self

    from cudf_polars.containers import DataFrame
    from cudf_polars.dsl.ir import IR
    from cudf_polars.quent._logging import QuentLogger
    from cudf_polars.quent._types import (
        Operator,
        Plan,
        Port,
        Task,
        Worker,
    )
    from cudf_polars.utils.config import ConfigOptions, StreamingExecutor

__all__ = [
    "LocalQuentContext",
    "ProcessorRegistry",
    "QuentContext",
    "QuentIRExecutionContext",
]


class ProcessorRegistry:
    """
    Engine/worker-scoped registry of dynamically declared Quent Processors.

    One registry is owned by the object that owns the Python
    :class:`~concurrent.futures.ThreadPoolExecutor` (e.g. ``SPMDEngine``,
    a Dask worker, or a Ray actor).

    Processors (thread resources) are declared on-demand in ``get_or_declare_processor``.
    Call ``_emit_processor_exit_events`` on engine shutdown to emit finalizing/exit events
    for all declared processors.
    """

    def __init__(self) -> None:
        self._processors: dict[int, Processor] = {}
        self._lock = threading.Lock()

    def get_or_declare_processor(
        self, logger: QuentLogger, thread_ident: int, pool_id: uuid.UUID
    ) -> Processor:
        """Get (or declare a new) Quent Processor for a CPU thread."""
        with self._lock:
            if thread_ident in self._processors:
                return self._processors[thread_ident]
            processor = Processor(pool_id=pool_id)
            self._processors[thread_ident] = processor

        logger.emit(processor.initializing())
        logger.emit(processor.operating())
        return processor

    def _emit_processor_exit_events(self, logger: QuentLogger) -> None:
        """Emit finalizing/exit events for all declared processors."""
        with self._lock:
            processors = list(self._processors.values())

        for processor in processors:
            logger.emit(processor.finalizing())
            logger.emit(processor.exit())


@dataclasses.dataclass(frozen=True, kw_only=True)
class QuentContext:
    """
    A Quent context that is globally valid for a query.

    Parameters
    ----------
    engine
        A Quent Engine object. By default, a new Engine object is created
        with the cudf-polars Implementation.
    query_group
        A Quent QueryGroup object. By default, a new QueryGroup with no
        instance name is created.

        This query group is used for all queries executed by this engine.
    query
        A Query Query object. By default, a new Query with no instance name
        is created.
    """

    engine: Engine = dataclasses.field(default_factory=Engine)
    query_group: QueryGroup = dataclasses.field(default_factory=QueryGroup)
    query: Query = dataclasses.field(default_factory=Query)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_query_group_cache_", set())

    def _serialize(self) -> bytes:
        """
        Serialize identity fields for transmission between ranks.

        Only the engine, query group, and query identifiers (plus
        implementation name/version) are included. Custom attributes are
        event-only and are not round-tripped.

        See Also
        --------
        QuentContext._deserialize
        """
        payload = {
            "engine": {
                "id": int(self.engine.id),
                "implementation": {
                    "name": self.engine.implementation.name,
                    "version": self.engine.implementation.version,
                },
            },
            "query_group": {
                "id": int(self.query_group.id),
                "instance_name": self.query_group.instance_name,
            },
            "query": {
                "id": int(self.query.id),
                "instance_name": self.query.instance_name,
            },
        }
        return json.dumps(payload).encode()

    @classmethod
    def _deserialize(cls, data: bytes) -> Self:
        """
        Reconstruct a QuentContext from serialized bytes.

        See Also
        --------
        QuentContext._serialize
        """
        payload = json.loads(data)
        return cls(
            engine=Engine(
                id=uuid.UUID(int=int(payload["engine"]["id"])),
                implementation=Implementation(
                    name=payload["engine"]["implementation"]["name"],
                    version=payload["engine"]["implementation"]["version"],
                ),
            ),
            query_group=QueryGroup(
                id=uuid.UUID(int=int(payload["query_group"]["id"])),
                instance_name=payload["query_group"]["instance_name"],
            ),
            query=Query(
                id=uuid.UUID(int=int(payload["query"]["id"])),
                instance_name=payload["query"]["instance_name"],
            ),
        )

    def query_for(self, query_id: uuid.UUID) -> Query:
        """
        Build a per-collect Quent Query with a unique id.

        Parameters
        ----------
        query_id: uuid.UUID
            The unique ID for the query.

        Returns
        -------
        A new Quent Query with the given ID and the same instance name as the
        engine-scoped query.
        """
        return Query(id=query_id, instance_name=self.query.instance_name)

    @property
    def _query_group_cache(self) -> set[uuid.UUID]:
        return self._query_group_cache_  # type: ignore[attr-defined]

    def _emit_engine_init_events(self, logger: QuentLogger) -> None:
        """Emit a Quent Engine init event."""
        logger.emit(self.engine._init())

    def _emit_engine_exit_events(self, logger: QuentLogger) -> None:
        """Emit a Quent Engine exit event."""
        logger.emit(self.engine._exit())

    def _emit_query_group_events(self, logger: QuentLogger) -> None:
        """
        Emit a Quent QueryGroup declaration event.

        This ensures that a declaration event is only emitted once per
        query group.
        """
        if self.query_group.id in self._query_group_cache:
            return
        self._query_group_cache.add(self.query_group.id)
        logger.emit(self.query_group._declare(engine=self.engine))

    def _emit_query_events(self, logger: QuentLogger, query: Query) -> None:
        """
        Emit Quent Query events.

        This includes events for 'Declare', 'Init', and 'Planning'.
        """
        logger.emit(query._init(query_group=self.query_group))
        logger.emit(query._planning())
        logger.emit(query._executing())

    def _emit_query_exit_events(self, logger: QuentLogger, query: Query) -> None:
        """Emit a Quent Query exit event."""
        logger.emit(query._exit())

    def _emit_plan_declarations(
        self,
        logger: QuentLogger,
        plan: Plan,
        operators: list[Operator],
        ports: list[Port],
    ) -> None:
        """Emit declaration events for a plan and all its operators and ports."""
        logger.emit(plan.declare())
        for operator in operators:
            logger.emit(operator.declare())
        for port in ports:
            logger.emit(port.declare())

    def _emit_plan_events(
        self,
        logger: QuentLogger,
        ir: IR,
        config_options: ConfigOptions[StreamingExecutor],
        plan_id: uuid.UUID,
        worker: Worker,
        query: Query | None = None,
        instance_name: str = "logical",
        parent_plan: Plan | None = None,
        parent_operators_by_node_id: dict[str, list[Operator]] | None = None,
    ) -> dict[str, Operator]:
        """
        Build and emit declaration events for a plan.

        Serializes ``ir`` into a :class:`SerializablePlan`, constructs the
        operators, ports and edges, emits the declarations, and returns a
        mapping from each node's stable ID to its :class:`Operator`.

        Parameters
        ----------
        logger: QuentLogger
            The quent logger, which buffers the events in memory.
        ir
            Root of the IR graph for this plan.
        config_options
            Executor configuration.
        plan_id
            Unique ID for this plan.
        worker
            The Quent worker executing the plan.
        query
            The Quent query this plan belongs to (``None`` for physical
            plans that hang off a parent plan instead).
        instance_name
            Human-readable plan name (e.g. ``"logical"`` or ``"physical"``).
        parent_plan
            If this plan was derived from another, the parent plan.
        parent_operators_by_node_id
            Optional mapping from node stable ID to parent
            :class:`Operator` objects (used for physical plans).

        Returns
        -------
        Mapping from node stable ID to the :class:`Operator` created for
        that node.
        """
        plan, ops, ports, op_by_id = build_plan(
            ir,
            config_options,
            query=query,
            plan_id=plan_id,
            worker=worker,
            instance_name=instance_name,
            parent_plan=parent_plan,
            parent_operators_by_node_id=parent_operators_by_node_id,
        )
        self._emit_plan_declarations(logger, plan, ops, ports)
        return op_by_id

    def _emit_physical_plan_events(
        self,
        logger: QuentLogger,
        ir: IR,
        config_options: ConfigOptions[StreamingExecutor],
        plan_id: uuid.UUID,
        worker: Worker,
        *,
        parent_plan: Plan,
        node_map: dict[str, list[str]],
        logical_op_by_id: dict[str, Operator],
    ) -> dict[str, Operator]:
        """
        Build and emit declaration events for a physical plan.

        Derives parent-operator linkage from the lowering ``node_map`` and
        delegates to :func:`emit_plan_events`.

        Parameters
        ----------
        logger: QuentLogger
            The quent logger, which buffers the events in memory.
        ir
            Root of the **lowered** IR graph.
        config_options
            Executor configuration.
        plan_id
            Unique ID for the physical plan.
        worker
            The Quent worker executing the physical plan.
        parent_plan
            Logical plan this physical plan was derived from.
        node_map
            Mapping from physical stable IDs to the logical stable IDs
            they were derived from (as returned by
            :func:`~cudf_polars.experimental.parallel.lower_ir_graph_with_node_map`).
        logical_op_by_id
            Mapping from logical stable ID to its :class:`Operator`
            (as returned by :func:`emit_plan_events` for the logical plan).

        Returns
        -------
        Mapping from physical stable ID to the :class:`Operator` created
        for that node.
        """
        parent_operators_by_node_id = build_parent_operators_map(
            node_map, logical_op_by_id
        )
        return self._emit_plan_events(
            logger,
            ir,
            config_options,
            plan_id,
            worker,
            instance_name="physical",
            parent_plan=parent_plan,
            parent_operators_by_node_id=parent_operators_by_node_id,
        )

    def _emit_task_begin_events(
        self,
        ir_type: type[IR],
        quent_task: Task,
        quent_ir_execution_context: QuentIRExecutionContext,
        input_frames_bytes: int,
    ) -> None:
        """
        Emit begin events for a Quent Task.

        Parameters
        ----------
        ir_type: type[IR]
            The IR type of the operator.
        quent_task: Task
            The Quent Task to emit events for.
        quent_ir_execution_context: QuentIRExecutionContext
            The Quent IR execution context.
        input_frames_bytes: int
            The total size of the input dataframes in bytes.

        Notes
        -----
        This emits the following events:

        - queueing
        - allocating (with the Quent Processor for the current thread)
        - loading (I/O nodes only)
        - computing (non-I/O nodes only)
        """
        quent_processor = quent_ir_execution_context.get_or_declare_processor(
            thread_ident=threading.get_ident(),
        )
        quent_ir_execution_context.logger.emit(quent_task.queueing())
        quent_ir_execution_context.logger.emit(
            quent_task.allocating(resource_id=quent_processor.id)
        )
        if ir_type.is_io_node:
            quent_ir_execution_context.logger.emit(
                quent_task.loading(
                    use_thread=quent_processor,
                    use_channel=quent_ir_execution_context.worker_resources.disk_to_device_channel,
                    channel_capacity_bytes=input_frames_bytes,
                    use_memory=quent_ir_execution_context.worker_resources.device_memory,
                    memory_capacity_bytes=input_frames_bytes,
                )
            )
        else:
            quent_ir_execution_context.logger.emit(
                quent_task.computing(
                    use_thread=quent_processor,
                    use_memory=quent_ir_execution_context.worker_resources.device_memory,
                    input_bytes=input_frames_bytes,
                    memory_capacity_bytes=input_frames_bytes,
                )
            )

    def _emit_task_end_events(
        self,
        ir_type: type[IR],
        quent_task: Task,
        quent_ir_execution_context: QuentIRExecutionContext,
        result: DataFrame | None,
    ) -> None:
        """
        Build and emit Quent events for the end of an IR node's evaluation.

        The timestamp here represents when the IR node completed **host**-side
        processing.  Work work may be happening asynchronously on the GPU.

        Parameters
        ----------
        ir_type: type[IR]
            The IR type of the operator.
        quent_task: Task
            The Quent Task to emit events for.
        quent_ir_execution_context: QuentIRExecutionContext
            The Quent IR execution context.
        result
            The output dataframe returned from the IR node. This will be ``None``
            if an exception was raised while evaluating the IR node.
        ir_execution_context
            The IR execution context. To emit any events, this must have a
            quent_ir_execution_context bound.

        Notes
        -----
        This emits the following events:

        - computing (I/O nodes only)
        - exit
        """
        if quent_ir_execution_context is None:  # pragma: no cover;
            return

        if result is not None:
            output_capacity_bytes = result._size_bytes
        else:  # pragma: no cover;
            output_capacity_bytes = 0
        if ir_type.is_io_node:
            quent_processor = quent_ir_execution_context.get_or_declare_processor(
                thread_ident=threading.get_ident(),
            )
            quent_ir_execution_context.logger.emit(
                quent_task.computing(
                    use_thread=quent_processor,
                    use_memory=quent_ir_execution_context.worker_resources.device_memory,
                    memory_capacity_bytes=output_capacity_bytes,
                )
            )
        quent_ir_execution_context.logger.emit(quent_task.exit())

        # TODO: Figure out how to emit some chunk/task-level statistics.
        # We can't do it directly on the Task object, because that (seems to)
        # break operator-level aggregation like duration_s.


@dataclasses.dataclass(kw_only=True)
class WorkerResources:
    """A simple container for per-worker Quent resources."""

    thread_pool: ThreadPool
    processor_registry: ProcessorRegistry
    device_memory: Memory
    filesystem: Memory
    disk_to_device_channel: Channel
    device_memory_capacity: int
    network: Network
    link_channels: dict[int, Channel]

    @classmethod
    def build(
        cls,
        instance_suffix: str,
        engine_id: uuid.UUID,
        worker_id: uuid.UUID,
        rank: int,
        nranks: int,
    ) -> Self:
        processor_registry = ProcessorRegistry()
        device_memory = Memory(
            instance_name=f"{instance_suffix} device memory",
            resource_type_name="memory",
            parent_group_id=engine_id,
        )
        filesystem = Memory(
            instance_name=f"{instance_suffix} filesystem",
            resource_type_name="filesystem",
            parent_group_id=worker_id,
        )
        disk_to_device_channel = Channel(
            instance_name=f"{instance_suffix} disk -> device",
            resource_type_name="DiskToDevice",
            parent_group_id=worker_id,
            source=filesystem,
            target=device_memory,
        )
        thread_pool = ThreadPool(worker_id=worker_id)

        # Network / Link Channels
        network = Network(engine_id=engine_id)
        link_channels: dict[int, Channel] = {}
        for target_rank in range(nranks):
            if target_rank == rank:
                continue
            link = Channel(
                instance_name=f"rank-{rank} -> rank-{target_rank}",
                resource_type_name="Link",
                parent_group_id=network.id,
                source=device_memory,
                target=device_memory,
            )
            link_channels[target_rank] = link

        return cls(
            processor_registry=processor_registry,
            device_memory=device_memory,
            filesystem=filesystem,
            disk_to_device_channel=disk_to_device_channel,
            thread_pool=thread_pool,
            device_memory_capacity=get_total_device_memory() or 0,
            network=network,
            link_channels=link_channels,
        )

    def declare(self, logger: QuentLogger) -> None:
        logger.emit(self.device_memory.initializing())
        logger.emit(self.device_memory.operating(self.device_memory_capacity))
        logger.emit(self.filesystem.initializing())
        # Filesystem capacity is unknown; declare as unbounded.
        logger.emit(self.filesystem.operating(None))
        logger.emit(self.disk_to_device_channel.initializing())
        # Channel capacity is a rate bound; unbounded when unknown.
        logger.emit(self.disk_to_device_channel.operating(None))
        logger.emit(self.thread_pool.declare())

        logger.emit(self.network.declare())
        for link in self.link_channels.values():
            logger.emit(link.initializing())
            logger.emit(link.operating(None))

    def finalize(self, logger: QuentLogger) -> None:
        self.processor_registry._emit_processor_exit_events(logger)

        logger.emit(self.disk_to_device_channel.finalizing())
        logger.emit(self.disk_to_device_channel.exit())
        logger.emit(self.disk_to_device_channel.source.finalizing())
        logger.emit(self.disk_to_device_channel.source.exit())
        logger.emit(self.device_memory.finalizing())
        logger.emit(self.device_memory.exit())

        # Network / Link Channels
        for link in self.link_channels.values():
            logger.emit(link.finalizing())
            logger.emit(link.exit())


@dataclasses.dataclass(kw_only=True)
class LocalQuentContext:
    """
    A Quent Context that is only ever used on the local worker rank.

    This can contain non-serializable objects (like a ``QuentLogger``)
    and entities that are only valid on the local rank.

    The ``query`` is per-collect: each ``.collect()`` derives a fresh
    :class:`Query` from its unique ``query_id`` (see
    :meth:`QuentContext.query_for`), rather than reusing the shared
    ``context.query``.

    The ``worker_resources`` (device memory, disk-to-device channel, network,
    link channels, thread pool, and processor registry) are engine/worker-scoped:
    they are declared once at worker setup via :class:`WorkerResources` and
    injected into each per-collect context, rather than being declared per query.
    """

    context: QuentContext
    query: Query
    worker: Worker
    logger: QuentLogger
    worker_resources: WorkerResources

    def get_or_declare_processor(
        self,
        thread_ident: int,
    ) -> Processor:
        """Get (or declare a new) Quent Processor for a CPU thread."""
        return self.worker_resources.processor_registry.get_or_declare_processor(
            self.logger,
            thread_ident=thread_ident,
            pool_id=self.worker_resources.thread_pool.id,
        )


@dataclasses.dataclass(kw_only=True)
class QuentIRExecutionContext(LocalQuentContext):
    """Like ``LocalQuentContext``, but with a Quent Operator bound too."""

    quent_operator: Operator

    @classmethod
    def from_execution_context(
        cls, execution_context: LocalQuentContext, quent_operator: Operator
    ) -> Self:
        """Create a ``QuentIRExecutionContext`` from a ``LocalQuentContext``."""
        return cls(
            quent_operator=quent_operator,
            context=execution_context.context,
            query=execution_context.query,
            worker=execution_context.worker,
            logger=execution_context.logger,
            worker_resources=execution_context.worker_resources,
        )
