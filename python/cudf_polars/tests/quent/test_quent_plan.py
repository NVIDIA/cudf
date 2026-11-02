# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Quent plan emission."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import polars as pl

import pylibcudf as plc

from cudf_polars.containers import DataType
from cudf_polars.dsl.expressions.base import Col, NamedExpr
from cudf_polars.dsl.expressions.binaryop import BinOp
from cudf_polars.dsl.expressions.literal import Literal
from cudf_polars.dsl.ir import Filter, GroupBy, HStack, Join, Scan, Select, Sort
from cudf_polars.quent._plan import (
    _dataframe_schema,
    _emit_plan_detail,
    _operator_details,
    port_names_for_node,
)
from cudf_polars.streaming.filter_hint import (
    ExternalDomain,
    JoinInputDomain,
    JoinWithPrefilter,
    Prefilter,
    PushdownFilterHint,
)
from cudf_polars.streaming.io import StreamingScan
from cudf_polars.streaming.join_filter_pushdown import (
    Decision,
    JoinFilterPushdownDecision,
)
from cudf_polars.streaming.shuffle import Shuffle


@pytest.mark.parametrize(
    "node_type, method_name, expected",
    [
        (
            Scan,
            "scan_details",
            {"typ": "parquet", "prefix": "scan-", "predicate": None},
        ),
        (
            StreamingScan,
            "streaming_scan_details",
            {
                "typ": "csv",
                "task_count": 2,
                "prefix": "stream-",
                "predicate": None,
            },
        ),
        (
            Join,
            "join_details",
            {"how": "inner", "left_on": ["a"], "right_on": ["b"]},
        ),
        (
            JoinWithPrefilter,
            "join_with_prefilter_details",
            {
                "how": "left",
                "left_on": ["a"],
                "right_on": ["b"],
                "prefilters": [
                    {
                        "type_name": "Prefilter",
                        "target_side": "left",
                        "target_on": ["a"],
                        "domain_on": ["b"],
                        "nulls_equal": True,
                        "domain": {"type_name": "ExternalDomain", "side": None},
                    },
                    {
                        "type_name": "Prefilter",
                        "target_side": "right",
                        "target_on": ["b"],
                        "domain_on": ["a"],
                        "nulls_equal": False,
                        "domain": {"type_name": "JoinInputDomain", "side": "left"},
                    },
                ],
            },
        ),
        (
            PushdownFilterHint,
            "pushdown_filter_hint_details",
            {
                "target_on": ["a"],
                "domain_on": ["b"],
                "nulls_equal": True,
                "placement": "pushed_down",
            },
        ),
        (
            GroupBy,
            "group_by_details",
            {"keys": ["a"]},
        ),
        (
            Shuffle,
            "shuffle_details",
            {"keys": ["a"]},
        ),
        (
            Sort,
            "sort_details",
            {"by": ["a"], "order": ["ascending"]},
        ),
        (
            Filter,
            "filter_details",
            {
                "predicate": "mask",
                "expression": (
                    '{"left": {"name": "a", "type": "Col"}, "op": "GREATER", '
                    '"right": {"type": "Literal", "value": {"type": "int", '
                    '"value": 1}}, "type": "BinOp"}'
                ),
            },
        ),
        (
            Select,
            "select_details",
            {"columns": ["a"]},
        ),
        (
            HStack,
            "hstack_details",
            {"columns": ["a"]},
        ),
    ],
)
def test_operator_details(
    node_type: type,
    method_name: str,
    expected: dict,
) -> None:
    operator = MagicMock()
    node = MagicMock(spec=node_type)
    dtype = DataType(pl.Int64())
    a = NamedExpr("a", Col(dtype, "a"))
    b = NamedExpr("b", Col(dtype, "b"))

    if node_type is Scan:
        node.typ = "parquet"
        node.paths = ["scan-a", "scan-b"]
        node.predicate = None
    elif node_type is StreamingScan:
        node.base_scan = SimpleNamespace(
            typ="csv", paths=["stream-a", "stream-b"], predicate=None
        )
        node.tasks = [object(), object()]
    elif node_type is Join:
        node.options = ("inner",)
        node.left_on = [a]
        node.right_on = [b]
    elif node_type is JoinWithPrefilter:
        node.options = ("left",)
        node.left_on = [a]
        node.right_on = [b]
        node.prefilters = [
            Prefilter("left", (a,), ExternalDomain(), (b,), nulls_equal=True),
            Prefilter(
                "right",
                (b,),
                JoinInputDomain("left"),
                (a,),
                nulls_equal=False,
            ),
        ]
    elif node_type is PushdownFilterHint:
        node.target_on = [a]
        node.domain_on = [b]
        node.nulls_equal = True
        node.placement = "pushed_down"
    elif node_type in (GroupBy, Shuffle):
        node.keys = [a]
    elif node_type is Sort:
        node.by = [a]
        node.order = [SimpleNamespace(name="ascending")]
    elif node_type is Filter:
        node.mask = SimpleNamespace(
            name="mask",
            value=BinOp(
                DataType(pl.Boolean()),
                plc.binaryop.BinaryOperator.GREATER,
                Col(dtype, "a"),
                Literal(dtype, 1),
            ),
        )
    elif node_type is Select:
        node.exprs = [a]
    elif node_type is HStack:
        node.columns = [a]
    else:  # pragma: no cover
        raise AssertionError(f"Missing test setup for {node_type}")

    details = _operator_details(node, operator)
    assert details is not None
    emit, values = details

    assert emit == getattr(operator, method_name)
    assert values == expected


def test_dataframe_schema() -> None:
    schema = {"a": DataType(pl.Int64()), "b": DataType(pl.String())}

    assert _dataframe_schema(schema) == {
        "columns": [
            {"name": "a", "dtype": "INT64"},
            {"name": "b", "dtype": "STRING"},
        ]
    }


def test_emit_rejected_join_filter_pushdown_details() -> None:
    operator = MagicMock()
    details = JoinFilterPushdownDecision(
        threshold=0.25,
        decision=Decision(reason="not_inner_join"),
    )

    _emit_plan_detail(details, operator)

    operator.join_filter_pushdown_details.assert_called_once_with(
        values={
            "threshold": 0.25,
            "reason": "not_inner_join",
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
    )


@pytest.mark.parametrize(
    "n_children, node_type, expected",
    [
        (0, "Scan", ("out",)),
        (1, "Select", ("out", "in")),
        (2, "Join", ("out", "left", "right")),
        (2, "ConditionalJoin", ("out", "left", "right")),
        (2, "Union", ("out", "in_0", "in_1")),
        (3, "Union", ("out", "in_0", "in_1", "in_2")),
    ],
)
def test_port_names_for_node(
    n_children: int,
    node_type: str,
    expected: tuple[str, ...],
) -> None:
    assert port_names_for_node(n_children, node_type) == expected
