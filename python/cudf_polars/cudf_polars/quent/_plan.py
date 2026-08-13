# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Quent telemetry tracing."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

from cudf_polars.quent._types import (
    Attribute,
    Edge,
    Operator,
    Plan,
    Port,
    new_quent_id,
)
from cudf_polars.streaming.explain import SerializablePlan

if TYPE_CHECKING:
    import uuid

    from cudf_polars.dsl.ir import IR
    from cudf_polars.quent._types import Query, Worker
    from cudf_polars.utils.config import ConfigOptions, StreamingExecutor

_JOIN_TYPES = frozenset({"Join", "ConditionalJoin"})


def build_plan(
    ir: IR,
    config_options: ConfigOptions[StreamingExecutor],
    query: Query | None,
    plan_id: uuid.UUID,
    worker: Worker,
    *,
    instance_name: str = "logical",
    parent_plan: Plan | None = None,
    parent_operators_by_node_id: dict[str, list[Operator]] | None = None,
) -> tuple[Plan, list[Operator], list[Port], dict[str, Operator]]:
    """
    Build a Quent plan, including operators, edges, and ports.

    Parameters
    ----------
    ir
        The cudf-polars IR.
    config_options
        The configuration options for the streaming executor.
    query
        The Quent query this plan belongs to.
    plan_id
        Unique ID for this plan.
    worker
        The Quent worker.
    instance_name
        Human-readable plan name (e.g. ``"logical"`` or ``"physical"``).
    parent_plan
        If this plan was derived from another (e.g. a physical plan derived
        from a logical plan), the parent plan.
    parent_operators_by_node_id
        Optional mapping from node stable ID to the list of parent
        :class:`Operator` objects from a parent plan.  Used to populate
        :attr:`Operator.parent_operators` for physical-plan operators
        that were derived from logical-plan operators during lowering.
    """
    serializable_plan = SerializablePlan.from_ir(ir, config_options=config_options)
    parent_ops = parent_operators_by_node_id or {}
    operator_by_ir_id: dict[str, Operator] = {}
    port_lookup: dict[tuple[uuid.UUID, str], Port] = {}
    operators: list[Operator] = []
    all_ports: list[Port] = []
    edges: list[Edge] = []
    plan = Plan(
        id=plan_id,
        query=query,
        parent_plan=parent_plan,
        instance_name=instance_name,
        edges=edges,
        worker=worker,
    )

    for node_id in sorted(serializable_plan.nodes.keys(), key=int):
        serializable_node = serializable_plan.nodes[node_id]

        operator_id = new_quent_id()
        # TODO: Include serializable_node.properties as custom attributes
        # We need to handle serialization of lists and dicts properly.
        custom_attributes = [
            Attribute(name="node_id", value=node_id),
        ]
        operator = Operator(
            id=operator_id,
            plan=plan,
            parent_operators=parent_ops.get(node_id, []),
            instance_name=serializable_node.type,
            type_name=serializable_node.type,
            custom_attributes=custom_attributes,
        )
        operator_by_ir_id[node_id] = operator
        operators.append(operator)

        for port_name in port_names_for_node(
            len(serializable_node.children), serializable_node.type
        ):
            port = Port(new_quent_id(), operator=operator, instance_name=port_name)
            all_ports.append(port)
            port_lookup[(operator_id, port_name)] = port

    for node_id in sorted(serializable_plan.nodes.keys(), key=int):
        serializable_node = serializable_plan.nodes[node_id]
        operator = operator_by_ir_id[node_id]
        input_port_names = port_names_for_node(
            len(serializable_node.children), serializable_node.type
        )[1:]

        for i, child_id in enumerate(serializable_node.children):
            child_operator = operator_by_ir_id[child_id]
            source = port_lookup[(child_operator.id, "out")]
            target = port_lookup[(operator.id, input_port_names[i])]
            edges.append(Edge(source=source, target=target))

    op_by_id = dict(
        zip(
            sorted(serializable_plan.nodes.keys(), key=int),
            operators,
            strict=True,
        )
    )
    return plan, operators, all_ports, op_by_id


@functools.cache
def port_names_for_node(n_children: int, node_type: str) -> tuple[str, ...]:
    """Determine port names for an IR node based on its children count and type."""
    if n_children == 0:
        return ("out",)
    elif n_children == 1:
        return (
            "out",
            "in",
        )
    elif n_children == 2 and node_type in _JOIN_TYPES:
        return (
            "out",
            "left",
            "right",
        )
    else:
        return ("out", *tuple(f"in_{i}" for i in range(n_children)))


def build_parent_operators_map(
    node_map: dict[str, list[str]],
    logical_op_by_id: dict[str, Operator],
) -> dict[str, list[Operator]]:
    """
    Map physical node IDs to their logical-plan parent operators.

    Parameters
    ----------
    node_map
        Mapping from physical (post-lowering) stable IDs to the
        logical (pre-lowering) stable IDs they were derived from.
    logical_op_by_id
        Mapping from logical stable ID to its :class:`Operator`.

    Returns
    -------
    Mapping from physical stable ID to the list of parent :class:`Operator`
    objects, with an empty list for entries with no parents.
    """
    return {
        physical_sid: [
            logical_op_by_id[sid] for sid in logical_sids if sid in logical_op_by_id
        ]
        for physical_sid, logical_sids in node_map.items()
    }
