# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Partitioning hints for streaming actor-graph construction."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, TypeAlias

import pylibcudf as plc

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
from cudf_polars.streaming.repartition import Repartition

if TYPE_CHECKING:
    from collections.abc import Mapping

    from cudf_polars.dsl.ir import IR
    from cudf_polars.streaming.base import PartitionInfo


@dataclasses.dataclass(frozen=True)
class NamedOrderKey:
    """Named sort key with logical Polars ordering options."""

    name: str
    descending: bool
    nulls_last: bool


@dataclasses.dataclass(frozen=True)
class HashPartitioningHint:
    """
    Hint that rows should be co-located by equality keys.

    ``peer_partition_count`` is a planning-time estimate of the partition
    count that may align with a downstream peer input.
    """

    keys: tuple[str, ...]
    peer_partition_count: int | None = None


@dataclasses.dataclass(frozen=True)
class OrderPartitioningHint:
    """
    Hint that rows should be ordered by a key sequence.

    ``peer_partition_count`` is a planning-time estimate of the partition
    count that may align with a downstream peer input.
    """

    keys: tuple[NamedOrderKey, ...]
    peer_partition_count: int | None = None


PartitioningHint: TypeAlias = HashPartitioningHint | OrderPartitioningHint
_MaybeHint: TypeAlias = PartitioningHint | None

__all__ = [
    "HashPartitioningHint",
    "NamedOrderKey",
    "OrderPartitioningHint",
    "PartitioningHint",
    "collect_partitioning_hints",
]


def collect_partitioning_hints(
    ir: IR,
    partition_info: Mapping[IR, PartitionInfo],
) -> dict[IR, PartitioningHint]:
    """
    Collect physical-layout hints for each IR node.

    Parameters
    ----------
    ir
        The IR to collect partitioning hints for.
    partition_info : Mapping[IR, PartitionInfo]
        The partition information for each IR node.

    Returns
    -------
    The physical-layout hints for each IR node.

    Notes
    -----
    The returned hints describe layouts that downstream consumers would prefer,
    if practical. These hints are not a strict runtime contract.
    """
    hints: dict[IR, _MaybeHint] = {}
    for node in reversed(list(post_traversal([ir]))):
        for child, hint in _direct_child_hints(node, partition_info):
            _record_hint(hints, child, hint)

        if (current_hint := hints.get(node)) is not None:
            for child, child_hint in _propagate_hint(node, current_hint):
                _record_hint(hints, child, child_hint)

    return {node: hint for node, hint in hints.items() if hint is not None}


def _direct_child_hints(
    ir: IR,
    partition_info: Mapping[IR, PartitionInfo],
) -> tuple[tuple[IR, PartitioningHint], ...]:
    """Return hints created directly by *ir* for its children."""
    if isinstance(ir, Sort):
        hint = _sort_hint(ir)
        return () if hint is None else ((ir.children[0], hint),)

    if isinstance(ir, Join) and ir.options[0] != "Cross":
        left_keys = _column_names(ir.left_on)
        right_keys = _column_names(ir.right_on)
        if left_keys is None or right_keys is None:
            return ()
        peer_partition_count = max(
            partition_info[ir.children[0]].count,
            partition_info[ir.children[1]].count,
        )
        return (
            (ir.children[0], HashPartitioningHint(left_keys, peer_partition_count)),
            (ir.children[1], HashPartitioningHint(right_keys, peer_partition_count)),
        )

    if isinstance(ir, GroupBy) and not ir.maintain_order:
        keys = _column_names(ir.keys)
        return () if keys is None else ((ir.children[0], HashPartitioningHint(keys)),)

    return ()


def _propagate_hint(
    ir: IR, hint: PartitioningHint
) -> tuple[tuple[IR, PartitioningHint], ...]:
    """Propagate *hint* through nodes whose children may benefit from it."""
    if len(ir.children) != 1:
        return ()

    (child,) = ir.children
    if isinstance(ir, (Filter, Projection, Select, Slice)):
        remapped = _remap_hint(hint, _child_zero_remapping(ir))
    elif isinstance(ir, GroupBy):
        # Only group-key outputs remap to the child; aggregate outputs stop here.
        remapped = _remap_hint(hint, _child_zero_remapping(ir))
    elif isinstance(ir, Repartition) or (
        isinstance(ir, MapFunction) and ir.name in {"hint_sorted", "rechunk"}
    ):
        remapped = _remap_hint(hint, {name: name for name in ir.schema})
    elif isinstance(ir, MapFunction) and ir.name == "row_index":
        remapped = _remap_hint(hint, {name: name for name in child.schema})
    else:
        remapped = None

    return () if remapped is None else ((child, remapped),)


def _sort_hint(ir: Sort) -> OrderPartitioningHint | None:
    names = _column_names(ir.by)
    if names is None:
        return None
    return OrderPartitioningHint(
        tuple(
            NamedOrderKey(
                name,
                order == plc.types.Order.DESCENDING,
                (order == plc.types.Order.ASCENDING)
                == (null_order == plc.types.NullOrder.AFTER),
            )
            for name, order, null_order in zip(
                names, ir.order, ir.null_order, strict=True
            )
        )
    )


def _column_names(named_exprs: tuple[expr.NamedExpr, ...]) -> tuple[str, ...] | None:
    names = []
    for named_expr in named_exprs:
        if not isinstance(named_expr.value, expr.Col):
            return None
        names.append(named_expr.value.name)
    return tuple(names)


def _remap_hint(
    hint: PartitioningHint, remapping: Mapping[str, str]
) -> PartitioningHint | None:
    if isinstance(hint, HashPartitioningHint):
        remapped_names = []
        for name in hint.keys:
            if (new_name := remapping.get(name)) is None:
                return None
            remapped_names.append(new_name)
        return dataclasses.replace(hint, keys=tuple(remapped_names))

    remapped_keys = []
    for key in hint.keys:
        new_name = remapping.get(key.name)
        if new_name is None:
            return None
        remapped_keys.append(dataclasses.replace(key, name=new_name))
    return dataclasses.replace(hint, keys=tuple(remapped_keys))


def _child_zero_remapping(ir: IR) -> dict[str, str]:
    return {
        output_name: binding.name
        for output_name, binding in column_domain_bindings(ir).items()
        if binding.child_index == 0
    }


def _record_hint(
    hints: dict[IR, _MaybeHint],
    node: IR,
    hint: PartitioningHint,
) -> None:
    if node not in hints:
        hints[node] = hint
    elif (current := hints[node]) is not None:
        hints[node] = _merge_hints(current, hint)


def _merge_hints(
    left: PartitioningHint, right: PartitioningHint
) -> PartitioningHint | None:
    peer_partition_count = _merge_peer_partition_count(left, right)
    if isinstance(left, HashPartitioningHint) and isinstance(
        right, HashPartitioningHint
    ):
        hash_keys = _shortest_common_prefix(left.keys, right.keys)
        return (
            None
            if hash_keys is None
            else HashPartitioningHint(
                hash_keys, peer_partition_count=peer_partition_count
            )
        )

    if isinstance(left, OrderPartitioningHint) and isinstance(
        right, OrderPartitioningHint
    ):
        order_keys = _longest_common_extension(left.keys, right.keys)
        return (
            None
            if order_keys is None
            else OrderPartitioningHint(
                order_keys, peer_partition_count=peer_partition_count
            )
        )

    if isinstance(left, OrderPartitioningHint):
        order_hint = left
        assert isinstance(right, HashPartitioningHint)
        hash_hint = right
    else:
        assert isinstance(right, OrderPartitioningHint)
        order_hint = right
        hash_hint = left
    order_names = tuple(key.name for key in order_hint.keys)
    if _is_prefix(hash_hint.keys, order_names) or _is_prefix(
        order_names, hash_hint.keys
    ):
        return dataclasses.replace(
            order_hint, peer_partition_count=peer_partition_count
        )
    return None


def _merge_peer_partition_count(
    left: PartitioningHint, right: PartitioningHint
) -> int | None:
    counts = [
        count
        for count in (left.peer_partition_count, right.peer_partition_count)
        if count is not None
    ]
    return max(counts) if counts else None


def _shortest_common_prefix(
    left: tuple[str, ...], right: tuple[str, ...]
) -> tuple[str, ...] | None:
    if _is_prefix(left, right):
        return left
    if _is_prefix(right, left):
        return right
    return None


def _longest_common_extension(
    left: tuple[NamedOrderKey, ...], right: tuple[NamedOrderKey, ...]
) -> tuple[NamedOrderKey, ...] | None:
    if _is_prefix(left, right):
        return right
    if _is_prefix(right, left):
        return left
    return None


def _is_prefix(left: tuple[object, ...], right: tuple[object, ...]) -> bool:
    return len(left) <= len(right) and left == right[: len(left)]
