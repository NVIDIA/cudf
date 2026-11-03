# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Quent telemetry tracing."""

from __future__ import annotations

import functools
import json
import os
import uuid
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeVar

from cudf_polars.dsl.expressions.base import Col, NamedExpr
from cudf_polars.dsl.expressions.binaryop import BinOp
from cudf_polars.dsl.expressions.literal import Literal as LiteralExpr
from cudf_polars.dsl.ir import Filter, GroupBy, HStack, Join, Scan, Select, Sort
from cudf_polars.dsl.traversal import traversal
from cudf_polars.streaming.filter_hint import (
    JoinInputDomain,
    JoinWithPrefilter,
    PushdownFilterHint,
)
from cudf_polars.streaming.io import StreamingScan
from cudf_polars.streaming.join_filter_pushdown import CompositeCandidate
from cudf_polars.streaming.shuffle import Shuffle

if TYPE_CHECKING:
    import cudf_polars_quent as quent_bindings

    from cudf_polars.dsl.expressions.base import Expr
    from cudf_polars.dsl.ir import IR
    from cudf_polars.quent._runtime import QuentSession
    from cudf_polars.streaming.filter_hint import Prefilter
    from cudf_polars.streaming.join_filter_pushdown import JoinFilterPushdownDecision
    from cudf_polars.streaming.plan_metadata import PlanMetadata
    from cudf_polars.typing import Schema

_JOIN_TYPES = frozenset({"Join", "ConditionalJoin"})
# These type variables / protocol ensures that the callable and dictionary values
# emitted from each `_operator_details` dispatch are compatible.
DetailsT = TypeVar("DetailsT")
DetailsT_contra = TypeVar("DetailsT_contra", contravariant=True)


class DetailsEmitter(Protocol[DetailsT_contra]):
    """Callable that emits one schema-defined operator-details payload."""

    def __call__(self, *, values: DetailsT_contra) -> None: ...


type OperatorDetails[DetailsT] = tuple[DetailsEmitter[DetailsT], DetailsT]


@functools.singledispatch
def _operator_details(
    node: IR, operator: quent_bindings.OperatorHandle
) -> OperatorDetails[Any] | None:
    """
    Build the specific emitter and details dictionary for an operator.

    This dispatches based on the type of the node. By default, no emitter /
    details dictionary are returned. Nodes with specific details can register
    their own emitter and details dictionary. These details are included in the
    emitted Quent plan.

    Parameters
    ----------
    node
        The IR node being evaluated.
    operator
        The Quent operator handle for the node.

    Returns
    -------
    A tuple containing the emitter callable and the details dictionary.
    """
    return None


@_operator_details.register(Scan)
def _(
    node: Scan, operator: quent_bindings.OperatorHandle
) -> OperatorDetails[quent_bindings.ScanDetailsDict]:
    return (
        operator.scan_details,
        {
            "typ": node.typ,
            "prefix": os.path.commonprefix(node.paths),
            "predicate": _json_expr(node.predicate),
        },
    )


@_operator_details.register(StreamingScan)
def _(
    node: StreamingScan, operator: quent_bindings.OperatorHandle
) -> OperatorDetails[quent_bindings.StreamingScanDetailsDict]:
    return (
        operator.streaming_scan_details,
        {
            "typ": node.base_scan.typ,
            "task_count": len(node.tasks),
            "prefix": os.path.commonprefix(node.base_scan.paths),
            "predicate": _json_expr(node.base_scan.predicate),
        },
    )


@_operator_details.register(Join)
def _(
    node: Join, operator: quent_bindings.OperatorHandle
) -> OperatorDetails[quent_bindings.JoinDetailsDict]:
    return operator.join_details, _join_details(node)


@_operator_details.register(JoinWithPrefilter)
def _(
    node: JoinWithPrefilter, operator: quent_bindings.OperatorHandle
) -> OperatorDetails[quent_bindings.JoinWithPrefilterDetailsDict]:
    return (
        operator.join_with_prefilter_details,
        {
            **_join_details(node),
            "prefilters": [_prefilter_details(value) for value in node.prefilters],
        },
    )


@_operator_details.register(PushdownFilterHint)
def _(
    node: PushdownFilterHint, operator: quent_bindings.OperatorHandle
) -> OperatorDetails[quent_bindings.PushdownFilterHintDetailsDict]:
    return (
        operator.pushdown_filter_hint_details,
        {
            "target_on": [value.name for value in node.target_on],
            "domain_on": [value.name for value in node.domain_on],
            "nulls_equal": node.nulls_equal,
            "placement": node.placement,
        },
    )


@_operator_details.register(GroupBy)
def _(
    node: GroupBy, operator: quent_bindings.OperatorHandle
) -> OperatorDetails[quent_bindings.KeysDetailsDict]:
    return operator.group_by_details, {"keys": [value.name for value in node.keys]}


@_operator_details.register(Shuffle)
def _(
    node: Shuffle, operator: quent_bindings.OperatorHandle
) -> OperatorDetails[quent_bindings.KeysDetailsDict]:
    return operator.shuffle_details, {"keys": [value.name for value in node.keys]}


@_operator_details.register(Sort)
def _(
    node: Sort, operator: quent_bindings.OperatorHandle
) -> OperatorDetails[quent_bindings.SortDetailsDict]:
    return (
        operator.sort_details,
        {
            "by": [value.name for value in node.by],
            "order": [value.name for value in node.order],
        },
    )


@_operator_details.register(Filter)
def _(
    node: Filter, operator: quent_bindings.OperatorHandle
) -> OperatorDetails[quent_bindings.FilterDetailsDict]:
    expression = _json_expr(node.mask.value)
    assert expression is not None
    return (
        operator.filter_details,
        {
            "predicate": node.mask.name,
            "expression": expression,
        },
    )


@_operator_details.register(Select)
def _(
    node: Select, operator: quent_bindings.OperatorHandle
) -> OperatorDetails[quent_bindings.ColumnsDetailsDict]:
    return operator.select_details, {"columns": [value.name for value in node.exprs]}


@_operator_details.register(HStack)
def _(
    node: HStack, operator: quent_bindings.OperatorHandle
) -> OperatorDetails[quent_bindings.ColumnsDetailsDict]:
    return operator.hstack_details, {"columns": [value.name for value in node.columns]}


def _emit_operator_details(node: IR, operator: quent_bindings.OperatorHandle) -> None:
    """Build and emit schema-defined details for an operator, when available."""
    details = _operator_details(node, operator)
    if details is not None:
        emit, values = details
        emit(values=values)


def _join_details(node: Join) -> quent_bindings.JoinDetailsDict:
    return {
        "how": node.options[0],
        "left_on": [value.name for value in node.left_on],
        "right_on": [value.name for value in node.right_on],
    }


def _prefilter_details(
    prefilter: Prefilter,
) -> quent_bindings.PrefilterDetailsDict:
    domain = prefilter.domain
    return {
        "type_name": type(prefilter).__name__,
        "target_side": prefilter.target_side,
        "target_on": [value.name for value in prefilter.target_on],
        "domain_on": [value.name for value in prefilter.domain_on],
        "nulls_equal": prefilter.nulls_equal,
        "domain": {
            "type_name": type(domain).__name__,
            "side": domain.side if isinstance(domain, JoinInputDomain) else None,
        },
    }


def _json_expr(expr: Expr | NamedExpr | None) -> str | None:
    return (
        None
        if expr is None
        else json.dumps(_serialize_expr(expr), sort_keys=True, default=str)
    )


def _serialize_expr(expr: Expr | NamedExpr) -> dict[str, Any]:
    match expr:
        case NamedExpr(name=name, value=value):
            return {"type": "NamedExpr", "name": name, "value": _serialize_expr(value)}
        case Col(name=name):
            return {"type": "Col", "name": name}
        case LiteralExpr(value=value):
            return {
                "type": "Literal",
                "value": {
                    "type": type(value).__name__,
                    "value": value.isoformat()
                    if hasattr(value, "isoformat")
                    else value
                    if isinstance(value, int | float | bool)
                    else str(value),
                },
            }
        case BinOp():
            return {
                "type": "BinOp",
                "op": expr.op.name,
                "left": _serialize_expr(expr.children[0]),
                "right": _serialize_expr(expr.children[1]),
            }
        case _:
            return {"type": type(expr).__name__}


def _dataframe_schema(schema: Schema) -> quent_bindings.DataFrameSchemaDict:
    columns: list[quent_bindings.ColumnSchemaDict] = []
    for name, dtype in schema.items():
        column: quent_bindings.ColumnSchemaDict = {
            "name": name,
            "dtype": dtype.id().name,
        }
        columns.append(column)
    return {"columns": columns}


def _emit_plan_detail(
    details: JoinFilterPushdownDecision,
    operator: quent_bindings.OperatorHandle,
) -> None:
    """Emit one typed detail payload collected while building the plan."""
    decision = details.decision
    candidate = decision.candidate
    values: quent_bindings.JoinFilterPushdownDetailsDict = {
        "threshold": details.threshold,
        "reason": decision.reason,
        "mode": None,
        "target_side": None,
        "target_key": None,
        "domain_key": None,
        "estimated_target_rows": None,
        "estimated_domain_rows": None,
        "estimated_target_cost": None,
        "estimated_domain_cost": None,
        "target_node_type": None,
        "domain_node_type": None,
        "constraint_key": None,
        "estimated_constraint_rows": None,
        "estimated_constraint_cost": None,
    }
    if candidate is not None:
        values.update(
            {
                "mode": candidate.mode,
                "target_side": candidate.target_side,
                "target_key": candidate.target_key.name,
                "domain_key": candidate.domain_key.name,
                "estimated_target_rows": candidate.target.rows,
                "estimated_domain_rows": candidate.domain.rows,
                "estimated_target_cost": candidate.target.cost,
                "estimated_domain_cost": candidate.domain.cost,
                "target_node_type": type(candidate.target.node).__name__,
                "domain_node_type": type(candidate.domain.node).__name__,
            }
        )
        if isinstance(candidate, CompositeCandidate):
            values.update(
                {
                    "constraint_key": candidate.target_constraint_key.name,
                    "estimated_constraint_rows": candidate.constraint_domain.rows,
                    "estimated_constraint_cost": candidate.constraint_domain.cost,
                }
            )
    operator.join_filter_pushdown_details(values=values)


def emit_plan(
    session: QuentSession,
    ir: IR,
    query_id: uuid.UUID,
    plan_id: uuid.UUID,
    worker_id: uuid.UUID | None,
    *,
    instance_name: Literal["logical", "physical"] = "logical",
    parent_plan_id: uuid.UUID | None = None,
    parent_operators_by_node_id: dict[str, list[uuid.UUID]] | None = None,
    plan_metadata: PlanMetadata | None = None,
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
    query_id, plan_id, worker_id
        Unique identifiers for the query, plan, and worker.
    instance_name
        The name indicating whether this is a logical or physical plan.
    parent_plan_id
        The ID of the parent plan, if any. For example, the ID of the
        pre-lowered plan for a physical plan.
    parent_operators_by_node_id
        A mapping from node IDs to their parent operator IDs.
    plan_metadata
        Details collected while optimizing the plan.
    emit
        Whether to emit the plan. This can be used to only emit the logical
        plan (which is identical across all ranks) once.

    Returns
    -------
    A mapping from node IDs to their operator IDs.
    """
    parent_ops = parent_operators_by_node_id or {}
    nodes = sorted(traversal([ir]), key=lambda node: node.get_stable_id())
    operator_by_ir_id: dict[str, uuid.UUID] = {}
    port_lookup: dict[tuple[uuid.UUID, str], uuid.UUID] = {}
    for node in nodes:
        node_id = str(node.get_stable_id())
        operator_id = uuid.uuid5(plan_id, f"operator:{node_id}")
        operator_by_ir_id[node_id] = operator_id
        for port_name in port_names_for_node(len(node.children), type(node).__name__):
            port_lookup[(operator_id, port_name)] = uuid.uuid5(
                operator_id, f"port:{port_name}"
            )
    if not emit:  # pragma: no cover; multi-rank
        return operator_by_ir_id

    edges: list[quent_bindings.PlanEdgeDict] = []
    for node in nodes:
        node_id = str(node.get_stable_id())
        operator_id = operator_by_ir_id[node_id]
        input_port_names = port_names_for_node(len(node.children), type(node).__name__)[
            1:
        ]
        for i, child in enumerate(node.children):
            child_id = str(child.get_stable_id())
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
    for node in nodes:
        node_id = str(node.get_stable_id())
        operator_id = operator_by_ir_id[node_id]
        operator = context.operator_observer().handle(operator_id)
        operator.declared(
            plan=plan_id,
            parent_operators=parent_ops.get(node_id, []),
            instance_name=f"{type(node).__name__}-{operator_id.hex[:8]}",
            type_name=type(node).__name__,
            node_id=node_id,
            schemas={
                "input_schemas": [
                    _dataframe_schema(child.schema) for child in node.children
                ],
                "output_schema": _dataframe_schema(node.schema),
            },
        )
        _emit_operator_details(node, operator)
        if plan_metadata is not None:
            for details in plan_metadata.operator_details(node):
                _emit_plan_detail(details, operator)
        for port_name in port_names_for_node(len(node.children), type(node).__name__):
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
