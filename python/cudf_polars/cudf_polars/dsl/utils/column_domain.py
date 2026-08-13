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
    HStack,
    Join,
    Projection,
    Select,
    Slice,
    Sort,
)

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
    """A direct binding to a named column on a specific child edge."""

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
    Map output columns to child columns containing their value domains.

    For every ``output_name -> ColumnBinding(child_index, input_name)`` binding,
    every value appearing in ``node[output_name]`` is guaranteed to appear in
    ``node.children[child_index][input_name]``. Row order, multiplicity, and
    cardinality are not preserved.

    If a name in ``node.schema`` does not appear in the mapping it means
    that it was not possible to derive a relationship between the domain of
    the output and input values for that column.
    """
    return {}


@column_domain_bindings.register(Select)
def _(node: Select) -> Mapping[str, ColumnBinding]:
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


@column_domain_bindings.register(Join)
def _(node: Join) -> Mapping[str, ColumnBinding]:
    left, right = node.children
    how = node.options[0]
    if how in ("Semi", "Anti"):
        return {
            name: ColumnBinding(0, name) for name in node.schema if name in left.schema
        }
    if how != "Inner":
        return {}

    bindings = {name: ColumnBinding(0, name) for name in left.schema}
    suffix = node.options[3]
    for name in right.schema:
        output_name = f"{name}{suffix}" if name in left.schema else name
        if output_name in node.schema:
            bindings[output_name] = ColumnBinding(1, name)
    return bindings


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
