# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import polars as pl

import pylibcudf as plc

from cudf_polars.containers import DataType
from cudf_polars.dsl import expr
from cudf_polars.dsl.ir import (
    DataFrameScan,
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
from cudf_polars.dsl.utils.column_domain import (
    ColumnBinding,
    column_domain_bindings,
)

I64 = DataType(pl.Int64())
BOOL = DataType(pl.Boolean())


def make_scan(*names: str) -> DataFrameScan:
    frame = pl.DataFrame({name: [1] for name in names})
    return DataFrameScan(dict.fromkeys(names, I64), frame._df, None)


def col(name: str) -> expr.Col:
    return expr.Col(I64, name)


def named_col(output: str, source: str) -> expr.NamedExpr:
    return expr.NamedExpr(output, col(source))


def test_source_has_no_column_domain_bindings() -> None:
    assert column_domain_bindings(make_scan("a")) == {}


def test_select_binds_aliases_and_omits_derived_columns() -> None:
    child = make_scan("a", "b")
    node = Select(
        {"renamed": I64, "derived": I64},
        (
            named_col("renamed", "a"),
            expr.NamedExpr("derived", expr.Literal(I64, 1)),
        ),
        True,  # noqa: FBT003
        child,
    )

    assert column_domain_bindings(node) == {
        "renamed": ColumnBinding(0, "a"),
    }


def test_hstack_binds_passthrough_alias_and_override() -> None:
    child = make_scan("a", "b")
    node = HStack(
        {"a": I64, "b": I64, "alias": I64},
        (
            expr.NamedExpr("a", expr.Literal(I64, 1)),
            named_col("alias", "b"),
        ),
        True,  # noqa: FBT003
        child,
    )

    assert column_domain_bindings(node) == {
        "b": ColumnBinding(0, "b"),
        "alias": ColumnBinding(0, "b"),
    }


def test_groupby_binds_only_direct_keys() -> None:
    child = make_scan("a", "b")
    node = GroupBy(
        {"key": I64, "value": I64},
        (named_col("key", "a"),),
        (named_col("value", "b"),),
        False,  # noqa: FBT003
        None,
        child,
    )

    assert column_domain_bindings(node) == {
        "key": ColumnBinding(0, "a"),
    }


def test_inner_join_binds_left_right_and_suffixed_columns() -> None:
    left = make_scan("key", "left_value")
    right = make_scan("key", "right_value")
    node = Join(
        {
            "key": I64,
            "left_value": I64,
            "key_right": I64,
            "right_value": I64,
        },
        (named_col("key", "key"),),
        (named_col("key", "key"),),
        ("Inner", False, (0, 1), "_right", False, "none"),
        left,
        right,
    )

    assert column_domain_bindings(node) == {
        "key": ColumnBinding(0, "key"),
        "left_value": ColumnBinding(0, "left_value"),
        "key_right": ColumnBinding(1, "key"),
        "right_value": ColumnBinding(1, "right_value"),
    }


def test_inner_join_omits_coalesced_right_key() -> None:
    left = make_scan("key", "left_value")
    right = make_scan("key", "right_value")
    node = Join(
        {"key": I64, "left_value": I64, "right_value": I64},
        (named_col("key", "key"),),
        (named_col("key", "key"),),
        ("Inner", False, None, "_right", True, "none"),
        left,
        right,
    )

    assert column_domain_bindings(node) == {
        "key": ColumnBinding(0, "key"),
        "left_value": ColumnBinding(0, "left_value"),
        "right_value": ColumnBinding(1, "right_value"),
    }


def test_semi_join_binds_only_left_columns() -> None:
    left = make_scan("key", "value")
    right = make_scan("key")
    node = Join(
        left.schema,
        (named_col("key", "key"),),
        (named_col("key", "key"),),
        ("Semi", False, None, "_right", False, "none"),
        left,
        right,
    )

    assert column_domain_bindings(node) == {
        "key": ColumnBinding(0, "key"),
        "value": ColumnBinding(0, "value"),
    }


def test_outer_join_has_no_column_domain_bindings() -> None:
    left = make_scan("key")
    right = make_scan("other")
    node = Join(
        {**left.schema, **right.schema},
        (named_col("key", "key"),),
        (named_col("other", "other"),),
        ("Left", False, None, "_right", False, "none"),
        left,
        right,
    )

    assert column_domain_bindings(node) == {}


def test_passthrough_nodes_bind_same_named_columns() -> None:
    child = make_scan("a", "b")
    mask = expr.NamedExpr("mask", expr.Literal(BOOL, True))  # noqa: FBT003
    nodes = (
        Filter(child.schema, mask, child),
        Projection({"b": I64}, child),
        Slice(child.schema, 0, 1, child),
        Distinct(
            child.schema,
            plc.stream_compaction.DuplicateKeepOption.KEEP_ANY,
            None,
            (0, 1),
            False,  # noqa: FBT003
            child,
        ),
        Sort(
            child.schema,
            (named_col("a", "a"),),
            (plc.types.Order.ASCENDING,),
            (plc.types.NullOrder.AFTER,),
            False,  # noqa: FBT003
            (0, 1),
            child,
        ),
    )

    for node in nodes:
        assert column_domain_bindings(node) == {
            name: ColumnBinding(0, name) for name in node.schema
        }
