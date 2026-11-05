# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for tracking column value domains between IR nodes."""

from __future__ import annotations

from dataclasses import dataclass
from functools import singledispatch
from typing import TYPE_CHECKING

from cudf_polars.dsl import expr
from cudf_polars.dsl.ir import (
    Distinct,
    Filter,
    GroupBy,
    HConcat,
    HStack,
    Join,
    Projection,
    Rolling,
    Select,
    Slice,
    Sort,
)
from cudf_polars.streaming.filter_hint import PushdownFilterHint
from cudf_polars.streaming.over import Over

if TYPE_CHECKING:
    from collections.abc import Mapping

    from cudf_polars.dsl.ir import IR

__all__ = [
    "ColumnBinding",
    "ColumnLineage",
    "ColumnRef",
    "column_domain_bindings",
]


@dataclass(frozen=True)
class ColumnBinding:
    """An output column copied directly from one child column."""

    child_index: int
    name: str


@dataclass(frozen=True)
class ColumnRef:
    """A named column produced by an IR node."""

    node: IR
    name: str


@dataclass(frozen=True)
class ColumnLineage:
    """Persistent value-domain lineage, sharing suffixes across DAG branches."""

    column: ColumnRef
    source: ColumnLineage | None = None
    source_child_index: int | None = None
    """Child edge leading to ``source``, or None if there is no source."""


@singledispatch
def column_domain_bindings(node: IR) -> Mapping[str, ColumnBinding]:
    """
    Map output columns to child columns supplying their values unchanged.

    For every ``output_name -> ColumnBinding(child_index, input_name)`` binding,
    the output contains values selected directly from the bound input column.
    The binding does not imply that row order, multiplicity, or cardinality is
    preserved.

    If a name in ``node.schema`` does not appear in the mapping it means
    that it was not possible to derive a relationship between the domain of
    the output and input values for that column.
    """
    return {}


@column_domain_bindings.register(Select)
@column_domain_bindings.register(Over)
def _(node: Select | Over) -> Mapping[str, ColumnBinding]:
    return {
        item.name: ColumnBinding(0, item.value.name)
        for item in node.exprs
        if isinstance(item.value, expr.Col)
    }


@column_domain_bindings.register(HStack)
def _(node: HStack) -> Mapping[str, ColumnBinding]:
    child = node.children[0]
    replaced = {item.name for item in node.columns}
    return {
        name: ColumnBinding(0, name) for name in child.schema if name not in replaced
    } | {
        item.name: ColumnBinding(0, item.value.name)
        for item in node.columns
        if isinstance(item.value, expr.Col)
    }


@column_domain_bindings.register(GroupBy)
def _(node: GroupBy) -> Mapping[str, ColumnBinding]:
    return {
        key.name: ColumnBinding(0, key.value.name)
        for key in node.keys
        if isinstance(key.value, expr.Col)
    }


@column_domain_bindings.register(Rolling)
def _(node: Rolling) -> Mapping[str, ColumnBinding]:
    assert isinstance(node.index.value, expr.Col)
    return {node.index.name: ColumnBinding(0, node.index.value.name)}


@column_domain_bindings.register(Join)
def _(node: Join) -> Mapping[str, ColumnBinding]:
    # JoinWithPrefilter appends domain children after the two join inputs.
    left, right = node.children[:2]
    how = node.options[0]
    if how in ("Semi", "Anti"):
        return {
            name: ColumnBinding(0, name) for name in node.schema if name in left.schema
        }
    if how == "Left":
        return {
            name: ColumnBinding(0, name) for name in left.schema if name in node.schema
        }
    if how not in ("Cross", "Inner", "Right"):
        return {}

    suffix = node.options[3]
    bindings = (
        {name: ColumnBinding(0, name) for name in left.schema}
        if how in ("Cross", "Inner")
        else {}
    )
    if how == "Inner" and node.options[4] and node.options[5].startswith("right"):
        for left_key, right_key in zip(node.left_on, node.right_on, strict=True):
            if (
                isinstance(left_key.value, expr.Col)
                and isinstance(right_key.value, expr.Col)
                and left_key.name in node.schema
            ):
                bindings[left_key.name] = ColumnBinding(1, right_key.value.name)
    left_names = set(left.schema)
    if how == "Right" and node.options[4]:
        left_names.difference_update(key.name for key in node.left_on)
    for name in right.schema:
        output_name = f"{name}{suffix}" if name in left_names else name
        if output_name in node.schema:
            bindings[output_name] = ColumnBinding(1, name)
    return bindings


@column_domain_bindings.register(HConcat)
def _(node: HConcat) -> Mapping[str, ColumnBinding]:
    if not node.streaming_safe:
        return {}
    sources: dict[str, list[int]] = {}
    for child_index, child in enumerate(node.children):
        for name in child.schema.keys() & node.schema.keys():
            sources.setdefault(name, []).append(child_index)
    return {
        name: ColumnBinding(child_indices[0], name)
        for name, child_indices in sources.items()
        if len(child_indices) == 1
    }


@column_domain_bindings.register(Distinct)
@column_domain_bindings.register(Filter)
@column_domain_bindings.register(Projection)
@column_domain_bindings.register(Slice)
@column_domain_bindings.register(Sort)
def _(
    node: Distinct | Filter | Projection | Slice | Sort,
) -> Mapping[str, ColumnBinding]:
    child = node.children[0]
    return {
        name: ColumnBinding(0, name) for name in node.schema if name in child.schema
    }


@column_domain_bindings.register(PushdownFilterHint)
def _(node: PushdownFilterHint) -> Mapping[str, ColumnBinding]:
    target = node.children[0]
    return {
        name: ColumnBinding(0, name) for name in node.schema if name in target.schema
    }
