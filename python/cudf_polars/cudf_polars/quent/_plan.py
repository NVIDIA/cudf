# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Quent telemetry tracing."""

from __future__ import annotations

import functools
import json
import uuid
from typing import TYPE_CHECKING, Any, Literal, overload

from cudf_polars.dsl.traversal import traversal
from cudf_polars.streaming.explain import SerializablePlan

if TYPE_CHECKING:
    import cudf_polars_quent as quent_bindings

    from cudf_polars.dsl.ir import IR
    from cudf_polars.quent._runtime import QuentSession
    from cudf_polars.utils.config import ConfigOptions, StreamingExecutor

_JOIN_TYPES = frozenset({"Join", "ConditionalJoin"})


def _emit_operator_details(
    operator: quent_bindings.OperatorHandle,
    node_type: str,
    properties: dict[str, Any],
) -> None:
    """Emit the schema-defined detail event for one operator."""
    match node_type:
        case "Scan":
            operator.scan_details(
                values={
                    "typ": str(properties["typ"]),
                    "prefix": str(properties["prefix"]),
                    "predicate": _json_optional(properties["predicate"]),
                }
            )
        case "StreamingScan":
            operator.streaming_scan_details(
                values={
                    "typ": str(properties["typ"]),
                    "task_count": int(properties["task_count"]),
                    "prefix": str(properties["prefix"]),
                    "predicate": _json_optional(properties["predicate"]),
                }
            )
        case "Join":
            operator.join_details(values=_join_details(properties))
        case "JoinWithPrefilter":
            operator.join_with_prefilter_details(
                values={
                    **_join_details(properties),
                    "prefilters": [
                        {
                            "type_name": str(prefilter["type"]),
                            "target_side": str(prefilter["target_side"]),
                            "target_on": _strings(prefilter["target_on"]),
                            "domain_on": _strings(prefilter["domain_on"]),
                            "nulls_equal": bool(prefilter["nulls_equal"]),
                            "domain": {
                                "type_name": str(prefilter["domain"]["type"]),
                                "side": (
                                    str(prefilter["domain"]["side"])
                                    if "side" in prefilter["domain"]
                                    else None
                                ),
                            },
                        }
                        for prefilter in properties["prefilters"]
                    ],
                }
            )
        case "PushdownFilterHint":
            operator.pushdown_filter_hint_details(
                values={
                    "target_on": _strings(properties["target_on"]),
                    "domain_on": _strings(properties["domain_on"]),
                    "nulls_equal": bool(properties["nulls_equal"]),
                    "placement": str(properties["placement"]),
                }
            )
        case "GroupBy":
            operator.group_by_details(values={"keys": _strings(properties["keys"])})
        case "Shuffle":
            operator.shuffle_details(values={"keys": _strings(properties["keys"])})
        case "Sort":
            operator.sort_details(
                values={
                    "by": _strings(properties["by"]),
                    "order": _strings(properties["order"]),
                }
            )
        case "Filter":
            operator.filter_details(
                values={
                    "predicate": str(properties["predicate"]),
                    "expression": json.dumps(
                        {
                            key: value
                            for key, value in properties.items()
                            if key != "predicate"
                        },
                        sort_keys=True,
                        default=str,
                    ),
                }
            )
        case "Select":
            operator.select_details(values={"columns": _strings(properties["columns"])})
        case "HStack":
            operator.hstack_details(values={"columns": _strings(properties["columns"])})


def _join_details(properties: dict[str, Any]) -> quent_bindings.JoinDetailsDict:
    return {
        "how": str(properties["how"]),
        "left_on": _strings(properties["left_on"]),
        "right_on": _strings(properties["right_on"]),
    }


def _strings(value: Any) -> list[str]:
    return [str(item) for item in value]


@overload
def _json_optional(value: None) -> None: ...


@overload
def _json_optional(value: Any) -> str: ...


def _json_optional(value: Any) -> str | None:
    return None if value is None else json.dumps(value, sort_keys=True, default=str)


def emit_plan(
    session: QuentSession,
    ir: IR,
    config_options: ConfigOptions[StreamingExecutor],
    query_id: uuid.UUID,
    plan_id: uuid.UUID,
    worker_id: uuid.UUID | None,
    *,
    instance_name: Literal["logical", "physical"] = "logical",
    parent_plan_id: uuid.UUID | None = None,
    parent_operators_by_node_id: dict[str, list[uuid.UUID]] | None = None,
    emit: bool = True,
) -> dict[str, uuid.UUID]:
    """
    Build and potentially emit one plan using deterministic entity UUIDs.

    This is usable on both lowered and pre-lowered IR graphs.

    Parameters
    ----------
    session
        The QuentSession from the local quent context.
    ir
        The root node of the IR graph.
    config_options
        The config options for the streaming executor.
    query_id, plan_id, worker_id
        Unique identifiers for the query, plan, and worker.
    instance_name
        The name indicating whether this is a logical or physical plan.
    parent_plan_id
        The ID of the parent plan, if any. For example, the ID of the
        pre-lowered plan for a physical plan.
    parent_operators_by_node_id
        A mapping from node IDs to their parent operator IDs.
    emit
        Whether to emit the plan. This can be used to only emit the logical
        plan (which is identical across all ranks) once.

    Returns
    -------
    A mapping from node IDs to their operator IDs.
    """
    serializable_plan = SerializablePlan.from_ir(ir, config_options=config_options)
    parent_ops = parent_operators_by_node_id or {}
    operator_by_ir_id: dict[str, uuid.UUID] = {}
    port_lookup: dict[tuple[uuid.UUID, str], uuid.UUID] = {}
    for node_id in sorted(serializable_plan.nodes.keys(), key=int):
        serializable_node = serializable_plan.nodes[node_id]
        operator_id = uuid.uuid5(plan_id, f"operator:{node_id}")
        operator_by_ir_id[node_id] = operator_id
        for port_name in port_names_for_node(
            len(serializable_node.children), serializable_node.type
        ):
            port_lookup[(operator_id, port_name)] = uuid.uuid5(
                operator_id, f"port:{port_name}"
            )
    if not emit:  # pragma: no cover; multi-rank
        return operator_by_ir_id

    edges: list[quent_bindings.PlanEdgeDict] = []
    for node_id in sorted(serializable_plan.nodes.keys(), key=int):
        serializable_node = serializable_plan.nodes[node_id]
        operator_id = operator_by_ir_id[node_id]
        input_port_names = port_names_for_node(
            len(serializable_node.children), serializable_node.type
        )[1:]
        for i, child_id in enumerate(serializable_node.children):
            child_operator_id = operator_by_ir_id[child_id]
            edges.append(
                {
                    "source": port_lookup[(child_operator_id, "out")],
                    "target": port_lookup[(operator_id, input_port_names[i])],
                }
            )

    context = session.binding_context
    context.plan_observer().handle(plan_id).declared(
        instance_name=instance_name,
        query=query_id,
        parent_plan=parent_plan_id,
        worker=worker_id,
        edges=edges,
    )
    for node_id in sorted(serializable_plan.nodes.keys(), key=int):
        serializable_node = serializable_plan.nodes[node_id]
        operator_id = operator_by_ir_id[node_id]
        operator = context.operator_observer().handle(operator_id)
        operator.declared(
            plan=plan_id,
            parent_operators=parent_ops.get(node_id, []),
            instance_name=f"{serializable_node.type}-{operator_id.hex[:8]}",
            type_name=serializable_node.type,
            node_id=node_id,
        )
        _emit_operator_details(
            operator, serializable_node.type, serializable_node.properties
        )
        for port_name in port_names_for_node(
            len(serializable_node.children), serializable_node.type
        ):
            context.port_observer().handle(
                port_lookup[(operator_id, port_name)]
            ).declared(operator=operator_id, instance_name=port_name)
    return operator_by_ir_id


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
    logical_op_by_id: dict[str, uuid.UUID],
) -> dict[str, list[uuid.UUID]]:
    """
    Map physical node IDs to their logical-plan parent operators.

    Parameters
    ----------
    node_map
        Mapping from physical (post-lowering) stable IDs to the
        logical (pre-lowering) stable IDs they were derived from.
    logical_op_by_id
        Mapping from logical stable ID to its operator UUID.

    Returns
    -------
    Mapping from physical stable ID to parent operator UUIDs, with an empty
    list for entries with no parents.
    """
    return {
        physical_sid: [
            logical_op_by_id[sid] for sid in logical_sids if sid in logical_op_by_id
        ]
        for physical_sid, logical_sids in node_map.items()
    }


def build_quent_operator_map(
    ir: IR,
    physical_op_by_id: dict[str, uuid.UUID],
) -> dict[IR, uuid.UUID]:
    """Build a map from IR nodes to their physical-plan operator UUIDs."""
    result: dict[IR, uuid.UUID] = {}
    for node in traversal([ir]):
        stable_id = str(node.get_stable_id())
        if stable_id in physical_op_by_id:
            result[node] = physical_op_by_id[stable_id]
    return result
