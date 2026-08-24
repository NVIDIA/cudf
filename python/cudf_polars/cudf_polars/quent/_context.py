# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Quent telemetry tracing."""

from __future__ import annotations

import dataclasses
import json
import uuid
from typing import TYPE_CHECKING

from cudf_polars.quent._plan import (
    build_parent_operators_map,
    build_plan,
)
from cudf_polars.quent._types import (
    Attribute,
    Engine,
    Implementation,
    Query,
    QueryGroup,
)

if TYPE_CHECKING:
    from typing import Self

    from cudf_polars.dsl.ir import IR
    from cudf_polars.quent._logging import QuentLogger
    from cudf_polars.quent._types import (
        Operator,
        Plan,
        Port,
        Worker,
    )
    from cudf_polars.utils.config import ConfigOptions, StreamingExecutor

__all__ = [
    "QuentContext",
]


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

    def serialize(self) -> bytes:
        """
        Serialize a QuentContext, for transmission between ranks.

        See Also
        --------
        QuentContext.deserialize
        """
        # This might be serialized between ranks.
        payload = {
            "engine": {
                "id": int(self.engine.id),
                "implementation": {
                    "name": self.engine.implementation.name,
                    "version": self.engine.implementation.version,
                    "custom_attributes": [
                        attribute.serialize()
                        for attribute in self.engine.implementation.custom_attributes
                    ],
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
    def deserialize(cls, data: bytes) -> Self:
        """
        Deserialize a QuentContext from bytes.

        See Also
        --------
        QuentContext.serialize
        """
        payload = json.loads(data)
        return cls(
            engine=Engine(
                id=uuid.UUID(int=int(payload["engine"]["id"])),
                implementation=Implementation(
                    name=payload["engine"]["implementation"]["name"],
                    version=payload["engine"]["implementation"]["version"],
                    custom_attributes=[
                        Attribute.deserialize(attribute)
                        for attribute in payload["engine"]["implementation"][
                            "custom_attributes"
                        ]
                    ],
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

    def _emit_query_events(self, logger: QuentLogger) -> None:
        """
        Emit Quent Query events.

        This includes events for 'Declare', 'Init', and 'Planning'.
        """
        logger.emit(self.query._init(query_group=self.query_group))
        logger.emit(self.query._planning())
        logger.emit(self.query._executing())

    def _emit_query_exit_events(self, logger: QuentLogger) -> None:
        """Emit a Quent Query exit event."""
        logger.emit(self.query._exit())

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


@dataclasses.dataclass(frozen=True, kw_only=True)
class LocalQuentContext:
    """
    A Quent Context that is only ever used on the local worker rank.

    This can contain non-serializable objects (like a ``QuentLogger``)
    and entities that are only valid on the local rank.
    """

    context: QuentContext
    worker: Worker
    logger: QuentLogger
