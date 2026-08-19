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
    from collections.abc import Iterator, Mapping

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


def collect_partitioning_hints(ir: IR) -> dict[IR, PartitioningHint]:
    """Collect non-conflicting upstream partitioning hints for each IR node."""
    hints: dict[IR, PartitioningHint | None] = {}
    for node in reversed(list(post_traversal([ir]))):
        for child, child_hint in _construct_child_hints(node):
            _record_hint(hints, child, child_hint)

        node_hint = hints.get(node)
        if (
            node_hint is None
            or len(node.children) != 1
            or not isinstance(node, (Projection, Select, Filter, Slice, GroupBy))
        ):
            continue

        remapped = _remap_hint(
            node_hint,
            {
                output_name: binding.name
                for output_name, binding in column_domain_bindings(node).items()
                if binding.child_index == 0
            },
        )
        if remapped is not None:
            _record_hint(hints, node.children[0], remapped)

    return {node: hint for node, hint in hints.items() if hint is not None}


def _construct_child_hints(ir: IR) -> Iterator[tuple[IR, PartitioningHint]]:
    """Construct hints for the upstream children of *ir*."""
    if isinstance(ir, Sort):
        names = _column_names(ir.by)
        if names is not None:
            yield (
                ir.children[0],
                _order_hint(
                    names,
                    tuple(order == plc.types.Order.DESCENDING for order in ir.order),
                    tuple(
                        (order == plc.types.Order.ASCENDING)
                        == (null_order == plc.types.NullOrder.AFTER)
                        for order, null_order in zip(
                            ir.order, ir.null_order, strict=True
                        )
                    ),
                ),
            )

    elif isinstance(ir, MapFunction) and ir.name == "hint_sorted":
        yield ir.children[0], _order_hint(*ir.options)

    elif isinstance(ir, Join) and ir.options[0] != "Cross":
        left_keys = _column_names(ir.left_on)
        right_keys = _column_names(ir.right_on)
        if left_keys is not None and right_keys is not None:
            yield ir.children[0], StrictPartitioningHint(left_keys)
            yield ir.children[1], StrictPartitioningHint(right_keys)

    elif isinstance(ir, GroupBy) and not ir.maintain_order:
        keys = _column_names(ir.keys)
        if keys is not None:
            yield ir.children[0], StrictPartitioningHint(keys)


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


def _record_hint(
    hints: dict[IR, PartitioningHint | None],
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
