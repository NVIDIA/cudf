# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pickle
from typing import TYPE_CHECKING

import pytest

import polars as pl

import pylibcudf as plc

from cudf_polars.containers import DataType
from cudf_polars.dsl import expr
from cudf_polars.dsl.expressions.base import ExecutionContext

if TYPE_CHECKING:
    from cudf_polars.dsl.nodebase import Node


@pytest.mark.parametrize(
    "node",
    [
        expr.Col(DataType(pl.Int64()), "foo"),
        expr.Literal(DataType(pl.Int64()), 42),
        expr.Cast(
            DataType(pl.Float64()),
            True,  # noqa: FBT003
            expr.Col(DataType(pl.Int64), "foo"),
        ),
        expr.BinOp(
            DataType(pl.Int64()),
            plc.binaryop.BinaryOperator.ADD,
            expr.Col(DataType(pl.Int64()), "foo"),
            expr.Literal(DataType(pl.Int64()), 1),
        ),
        expr.GroupedWindow(
            DataType(pl.Float64),
            ("groups_to_rows", True, False, False),
            [
                expr.NamedExpr(
                    "foo",
                    expr.Agg(
                        DataType(pl.Float64),
                        "sum",
                        None,
                        ExecutionContext.WINDOW,
                        expr.Col(DataType(pl.Int64()), "bar"),
                    ),
                )
            ],
            expr.NamedExpr("foo", expr.Col(DataType(pl.Float64()), "foo")),
            1,
            expr.Col(DataType(pl.Int64()), "foo"),
            expr.Col(DataType(pl.Int64()), "bar"),
            expr.Col(DataType(pl.Int64()), "baz"),
        ),
    ],
)
def test_reconstruct(node: Node):
    assert node.reconstruct(node.children) == node


@pytest.mark.parametrize(
    "node",
    [
        expr.Col(DataType(pl.Int64()), "foo"),
        expr.Literal(DataType(pl.Int64()), 42),
        expr.BinOp(
            DataType(pl.Int64()),
            plc.binaryop.BinaryOperator.ADD,
            expr.Col(DataType(pl.Int64()), "foo"),
            expr.Literal(DataType(pl.Int64()), 1),
        ),
    ],
)
def test_pickle(node: Node):
    unpickled = pickle.loads(pickle.dumps(node))
    assert unpickled == node


@pytest.mark.parametrize(
    "node",
    [
        expr.Col(DataType(pl.Int64()), "foo"),
        expr.Literal(DataType(pl.Int64()), 42),
    ],
)
def test_get_stable_id(node: Node):
    node_id = node.get_stable_id()
    assert isinstance(node_id, int)
    # Second call should return the cached value
    assert node.get_stable_id() == node_id
    # External id is a truncation of the full digest used for composition.
    assert node_id == int.from_bytes(node._get_stable_digest()[:4], "big")
    assert len(node._get_stable_digest()) == 16


def test_get_stable_plan_id() -> None:
    node = expr.BinOp(
        DataType(pl.Int64()),
        plc.binaryop.BinaryOperator.ADD,
        expr.Col(DataType(pl.Int64()), "foo"),
        expr.Literal(DataType(pl.Int64()), 1),
    )

    plan_id = node.get_stable_plan_id()
    # Second call should return the cached value
    assert node.get_stable_plan_id() == plan_id

    # stability
    node2 = expr.BinOp(
        DataType(pl.Int64()),
        plc.binaryop.BinaryOperator.ADD,
        expr.Col(DataType(pl.Int64()), "foo"),
        expr.Literal(DataType(pl.Int64()), 1),
    )
    assert node2.get_stable_plan_id() == plan_id
    assert node.children[0].get_stable_id() == node2.children[0].get_stable_id()

    # And uniqueness
    node3 = node.children[0]
    assert node3.get_stable_plan_id() != plan_id


def test_all_pointwise_walks_descendants() -> None:
    i64 = DataType(pl.Int64())
    col = expr.Col(i64, "a")
    lit = expr.Literal(i64, 1)
    agg = expr.Agg(i64, "sum", None, ExecutionContext.FRAME, col)
    pointwise_over_agg = expr.Cast(DataType(pl.Float64()), True, agg)  # noqa: FBT003

    assert col.all_pointwise()
    assert expr.BinOp(i64, plc.binaryop.BinaryOperator.ADD, col, lit).all_pointwise()
    assert not agg.all_pointwise()
    assert not expr.NamedExpr("a", agg).all_pointwise()
    # Root is pointwise, descendant is not.
    assert pointwise_over_agg.is_pointwise
    assert not pointwise_over_agg.all_pointwise()
