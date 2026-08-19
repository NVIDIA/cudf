# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Partitioning hints for streaming actor-graph construction."""

from __future__ import annotations

from dataclasses import dataclass
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

if TYPE_CHECKING:
    from collections.abc import Mapping

    from cudf_polars.dsl.ir import IR


@dataclass(frozen=True)
class NamedOrderKey:
    """Named sort key with logical Polars ordering options."""

    name: str
    descending: bool
    nulls_last: bool


@dataclass(frozen=True)
class StrictPartitioningHint:
    """Hint that a downstream consumer wants strict partitioning by keys."""

    keys: tuple[str, ...]


@dataclass(frozen=True)
class OrderPartitioningHint:
    """Hint that upstream rows should be ordered, optionally with a strict prefix."""

    keys: tuple[NamedOrderKey, ...]
    strict_key_count: int | None = None


PartitioningHint: TypeAlias = StrictPartitioningHint | OrderPartitioningHint


def collect_partitioning_hints(ir: IR) -> dict[IR, tuple[PartitioningHint, ...]]:
    """Collect upstream partitioning hints for an IR graph."""
    hints: dict[IR, tuple[PartitioningHint, ...]] = {}
    for node in reversed(list(post_traversal([ir]))):
        child_hints = _direct_child_hints(node)
        child_hints.extend(_propagated_child_hints(node, hints))
        for child, child_hint in child_hints:
            hints[child] = _merge_candidate_hint(hints.get(child, ()), child_hint)
    return hints


def _direct_child_hints(ir: IR) -> list[tuple[IR, PartitioningHint]]:
    """Create hints implied directly by operators that consume a partitioning."""
    if isinstance(ir, Sort):
        names = _column_names(ir.by)
        if names is not None:
            return [
                (
                    ir.children[0],
                    _order_hint(
                        names,
                        tuple(
                            order == plc.types.Order.DESCENDING for order in ir.order
                        ),
                        tuple(
                            (order == plc.types.Order.ASCENDING)
                            == (null_order == plc.types.NullOrder.AFTER)
                            for order, null_order in zip(
                                ir.order, ir.null_order, strict=True
                            )
                        ),
                    ),
                )
            ]

    if isinstance(ir, MapFunction) and ir.name == "hint_sorted":
        return [(ir.children[0], _order_hint(*ir.options))]

    if isinstance(ir, Join) and ir.options[0] != "Cross":
        left_keys = _column_names(ir.left_on)
        right_keys = _column_names(ir.right_on)
        if left_keys is not None and right_keys is not None:
            return [
                (ir.children[0], StrictPartitioningHint(left_keys)),
                (ir.children[1], StrictPartitioningHint(right_keys)),
            ]

    if isinstance(ir, GroupBy) and not ir.maintain_order:
        keys = _column_names(ir.keys)
        if keys is not None:
            return [(ir.children[0], StrictPartitioningHint(keys))]

    return []


def _propagated_child_hints(
    node: IR, hints: dict[IR, tuple[PartitioningHint, ...]]
) -> list[tuple[IR, PartitioningHint]]:
    """Push compatible downstream hints through single-child operators."""
    child_hints: list[tuple[IR, PartitioningHint]] = []
    node_hints = hints.get(node)
    if (
        node_hints is not None
        and len(node.children) == 1
        and isinstance(node, (Projection, Select, Filter, Slice, GroupBy))
    ):
        remapping = {
            output_name: binding.name
            for output_name, binding in column_domain_bindings(node).items()
            if binding.child_index == 0
        }
        child_hints.extend(
            (node.children[0], remapped)
            for node_hint in node_hints
            if (remapped := _remap_hint(node_hint, remapping)) is not None
        )
    return child_hints


def _merge_candidate_hint(
    existing_hints: tuple[PartitioningHint, ...], child_hint: PartitioningHint
) -> tuple[PartitioningHint, ...]:
    """Merge one hint into compatible candidates and preserve incompatible ones."""
    candidates: list[PartitioningHint] = []
    new_hint = child_hint
    insertion_index: int | None = None
    for existing_hint in existing_hints:
        if (merged := _merge_hints(existing_hint, new_hint)) is None:
            candidates.append(existing_hint)
        else:
            new_hint = merged
            if insertion_index is None:
                insertion_index = len(candidates)
    if insertion_index is None:
        candidates.append(new_hint)
    else:
        candidates.insert(insertion_index, new_hint)
    return tuple(candidates)


def _order_hint(
    names: tuple[str, ...],
    descending: tuple[bool, ...],
    nulls_last: tuple[bool, ...],
) -> OrderPartitioningHint:
    return OrderPartitioningHint(
        tuple(
            NamedOrderKey(name, desc, null_last)
            for name, desc, null_last in zip(names, descending, nulls_last, strict=True)
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
    """Rewrite hint column names through a child-to-parent name mapping."""
    if isinstance(hint, StrictPartitioningHint):
        remapped_names = []
        for name in hint.keys:
            if (new_name := remapping.get(name)) is None:
                return None
            remapped_names.append(new_name)
        return StrictPartitioningHint(tuple(remapped_names))

    remapped_keys = []
    for key in hint.keys:
        new_name = remapping.get(key.name)
        if new_name is None:
            return None
        remapped_keys.append(NamedOrderKey(new_name, key.descending, key.nulls_last))
    return OrderPartitioningHint(tuple(remapped_keys), hint.strict_key_count)


def _merge_hints(
    left: PartitioningHint, right: PartitioningHint
) -> PartitioningHint | None:
    """Merge compatible hints, or return None if both should remain candidates."""
    if isinstance(left, StrictPartitioningHint):
        if isinstance(right, StrictPartitioningHint):
            if _is_prefix(left.keys, right.keys):
                return left
            if _is_prefix(right.keys, left.keys):
                return right
            return None
        return _merge_order_with_strict(right, left)

    if isinstance(right, StrictPartitioningHint):
        return _merge_order_with_strict(left, right)

    if _is_prefix(left.keys, right.keys):
        keys = right.keys
    elif _is_prefix(right.keys, left.keys):
        keys = left.keys
    else:
        return None
    return OrderPartitioningHint(
        keys, _merge_strict_key_count(left.strict_key_count, right.strict_key_count)
    )


def _merge_order_with_strict(
    order_hint: OrderPartitioningHint, strict_hint: StrictPartitioningHint
) -> OrderPartitioningHint | None:
    """Fold strict-key requirements into compatible ordering hints."""
    order_names = tuple(key.name for key in order_hint.keys)
    if _is_prefix(strict_hint.keys, order_names) or _is_prefix(
        order_names, strict_hint.keys
    ):
        strict_key_count = min(len(strict_hint.keys), len(order_names))
        return OrderPartitioningHint(
            order_hint.keys,
            _merge_strict_key_count(order_hint.strict_key_count, strict_key_count),
        )
    return None


def _merge_strict_key_count(*counts: int | None) -> int | None:
    count = max((count for count in counts if count is not None), default=0)
    return count or None


def _is_prefix(left: tuple[object, ...], right: tuple[object, ...]) -> bool:
    return len(left) <= len(right) and left == right[: len(left)]
