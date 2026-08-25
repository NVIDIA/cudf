# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Downstream partitioning requests for streaming actor-graph construction.

A partitioning request is attached to an IR node when a downstream consumer
may benefit from that node producing a specific partitioning. These requests
use "partitioning" in the same broad sense as ``ChannelMetadata.Partitioning``:
rows may be strictly partitioned by equality keys, ordered by key values, or
both.

Requests are planning-time information. They do not describe or guarantee the
actual partitioning of the node's output. Runtime partitioning metadata is
tracked separately in ``ChannelMetadata``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

from cudf_polars.dsl import expr
from cudf_polars.dsl.ir import (
    Filter,
    GroupBy,
    Join,
    MapFunction,
    Projection,
    Select,
    Slice,
    Sort,
)
from cudf_polars.dsl.traversal import post_traversal
from cudf_polars.dsl.utils.column_domain import column_domain_bindings
from cudf_polars.utils.sorting import sort_order

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import pylibcudf as plc

    from cudf_polars.dsl.ir import IR


@dataclass(frozen=True)
class NamedOrderKey:
    """Named sort key with pylibcudf ordering options."""

    name: str
    order: plc.types.Order
    null_order: plc.types.NullOrder


@dataclass(frozen=True)
class StrictPartitioningRequest:
    """Request for upstream output to strictly partition equal keys."""

    keys: tuple[str, ...]


@dataclass(frozen=True)
class OrderPartitioningRequest:
    """Request for upstream output to be ordered by a key sequence."""

    keys: tuple[NamedOrderKey, ...]
    strict_key_count: int | None = None


PartitioningRequest: TypeAlias = StrictPartitioningRequest | OrderPartitioningRequest


def collect_partitioning_requests(
    ir: IR,
) -> dict[IR, tuple[PartitioningRequest, ...]]:
    """
    Collect downstream partitioning requests for each IR node.

    The returned mapping answers "which partitionings could make downstream
    consumers cheaper if this node produced them?" A request is therefore
    aspirational: it is not evidence that the data is currently sorted, hash
    partitioned, or otherwise partitioned that way.
    """
    requests: dict[IR, tuple[PartitioningRequest, ...]] = {}
    # Reverse post-order ensures every downstream consumer is processed before
    # any shared upstream producer whose requests must be propagated further.
    for node in reversed(list(post_traversal([ir]))):
        child_requests = _direct_child_requests(node)
        child_requests.extend(_propagated_child_requests(node, requests))
        for child, request in child_requests:
            requests[child] = _merge_candidate_request(requests.get(child, ()), request)
    return requests


def _direct_child_requests(ir: IR) -> list[tuple[IR, PartitioningRequest]]:
    """Create child requests implied by partitioning-aware operators."""
    if isinstance(ir, Sort):
        names = _column_names(ir.by)
        if names is not None:
            return [(ir.children[0], _order_request(names, ir.order, ir.null_order))]

    if isinstance(ir, MapFunction) and ir.name == "hint_sorted":
        column_names, descending, nulls_last = ir.options
        order, null_order = sort_order(
            descending, nulls_last=nulls_last, num_keys=len(column_names)
        )
        return [(ir.children[0], _order_request(column_names, order, null_order))]

    if isinstance(ir, Join) and ir.options[0] != "Cross":
        left_keys = _column_names(ir.left_on)
        right_keys = _column_names(ir.right_on)
        if left_keys is not None and right_keys is not None:
            return [
                (ir.children[0], StrictPartitioningRequest(left_keys)),
                (ir.children[1], StrictPartitioningRequest(right_keys)),
            ]

    if isinstance(ir, GroupBy) and not ir.maintain_order:
        keys = _column_names(ir.keys)
        if keys is not None:
            return [(ir.children[0], StrictPartitioningRequest(keys))]

    return []


def _propagated_child_requests(
    node: IR, requests: dict[IR, tuple[PartitioningRequest, ...]]
) -> list[tuple[IR, PartitioningRequest]]:
    """Push compatible downstream requests through single-child operators."""
    child_requests: list[tuple[IR, PartitioningRequest]] = []
    node_requests = requests.get(node)
    if (
        node_requests is not None
        and len(node.children) == 1
        and isinstance(node, (Projection, Select, Filter, Slice, GroupBy))
    ):
        remapping = {
            output_name: binding.name
            for output_name, binding in column_domain_bindings(node).items()
        }
        child_requests.extend(
            (node.children[0], remapped)
            for node_request in node_requests
            if (remapped := _remap_request(node_request, remapping)) is not None
        )
    return child_requests


def _merge_candidate_request(
    existing_requests: tuple[PartitioningRequest, ...],
    request: PartitioningRequest,
) -> tuple[PartitioningRequest, ...]:
    """Merge compatible requests while preserving incompatible candidates."""
    candidates: list[PartitioningRequest] = []
    new_request = request
    insertion_index: int | None = None
    for existing_request in existing_requests:
        if (merged := _merge_requests(existing_request, new_request)) is None:
            candidates.append(existing_request)
        else:
            new_request = merged
            if insertion_index is None:
                # Candidate order is not a priority. Keep the first compatible
                # merge in place so request collection remains deterministic.
                insertion_index = len(candidates)
    if insertion_index is None:
        candidates.append(new_request)
    else:
        candidates.insert(insertion_index, new_request)
    return tuple(candidates)


def _order_request(
    names: tuple[str, ...],
    orders: Sequence[plc.types.Order],
    null_orders: Sequence[plc.types.NullOrder],
) -> OrderPartitioningRequest:
    return OrderPartitioningRequest(
        tuple(
            NamedOrderKey(name, order, null_order)
            for name, order, null_order in zip(names, orders, null_orders, strict=True)
        )
    )


def _column_names(named_exprs: tuple[expr.NamedExpr, ...]) -> tuple[str, ...] | None:
    """Return column names when every expression is a direct column reference."""
    names = []
    for named_expr in named_exprs:
        if not isinstance(named_expr.value, expr.Col):
            return None
        names.append(named_expr.value.name)
    return tuple(names)


def _remap_request(
    request: PartitioningRequest, remapping: Mapping[str, str]
) -> PartitioningRequest | None:
    """Rewrite request column names through a child-to-parent name mapping."""
    if isinstance(request, StrictPartitioningRequest):
        remapped_names = []
        for name in request.keys:
            if (new_name := remapping.get(name)) is None:
                return None
            remapped_names.append(new_name)
        return StrictPartitioningRequest(tuple(remapped_names))

    remapped_keys = []
    for key in request.keys:
        new_name = remapping.get(key.name)
        if new_name is None:
            return None
        remapped_keys.append(NamedOrderKey(new_name, key.order, key.null_order))
    return OrderPartitioningRequest(tuple(remapped_keys), request.strict_key_count)


def _merge_requests(
    left: PartitioningRequest, right: PartitioningRequest
) -> PartitioningRequest | None:
    """Merge compatible requests, or keep both candidates if incompatible."""
    if isinstance(left, StrictPartitioningRequest):
        if isinstance(right, StrictPartitioningRequest):
            if _is_prefix(left.keys, right.keys):
                return left
            if _is_prefix(right.keys, left.keys):
                return right
            return None
        return _merge_order_with_strict(right, left)

    if isinstance(right, StrictPartitioningRequest):
        return _merge_order_with_strict(left, right)

    if _is_prefix(left.keys, right.keys):
        keys = right.keys
    elif _is_prefix(right.keys, left.keys):
        keys = left.keys
    else:
        return None
    return OrderPartitioningRequest(
        keys, _merge_strict_key_count(left.strict_key_count, right.strict_key_count)
    )


def _merge_order_with_strict(
    order_request: OrderPartitioningRequest,
    strict_request: StrictPartitioningRequest,
) -> OrderPartitioningRequest | None:
    """Fold strict-key requirements into compatible ordering requests."""
    order_names = tuple(key.name for key in order_request.keys)
    if _is_prefix(strict_request.keys, order_names) or _is_prefix(
        order_names, strict_request.keys
    ):
        strict_key_count = min(len(strict_request.keys), len(order_names))
        return OrderPartitioningRequest(
            order_request.keys,
            _merge_strict_key_count(order_request.strict_key_count, strict_key_count),
        )
    return None


def _merge_strict_key_count(*counts: int | None) -> int | None:
    return max((count for count in counts if count is not None), default=None)


def _is_prefix(left: tuple[object, ...], right: tuple[object, ...]) -> bool:
    return len(left) <= len(right) and left == right[: len(left)]
