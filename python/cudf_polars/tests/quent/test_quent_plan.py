# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Quent plan emission."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from cudf_polars.quent._plan import (
    _emit_operator_details,
    port_names_for_node,
)


@pytest.mark.parametrize(
    "node_type, properties, method_name, expected",
    [
        (
            "Scan",
            {"typ": "parquet", "prefix": "scan", "predicate": None},
            "scan_details",
            {"typ": "parquet", "prefix": "scan", "predicate": None},
        ),
        (
            "StreamingScan",
            {
                "typ": "csv",
                "task_count": 2,
                "prefix": "stream",
                "predicate": {"column": "a"},
            },
            "streaming_scan_details",
            {
                "typ": "csv",
                "task_count": 2,
                "prefix": "stream",
                "predicate": '{"column": "a"}',
            },
        ),
        (
            "Join",
            {"how": "inner", "left_on": ["a"], "right_on": ["b"]},
            "join_details",
            {"how": "inner", "left_on": ["a"], "right_on": ["b"]},
        ),
        (
            "JoinWithPrefilter",
            {
                "how": "left",
                "left_on": ["a"],
                "right_on": ["b"],
                "prefilters": [
                    {
                        "type": "bloom",
                        "target_side": "left",
                        "target_on": ["a"],
                        "domain_on": ["b"],
                        "nulls_equal": True,
                        "domain": {"type": "min-max"},
                    },
                    {
                        "type": "bloom",
                        "target_side": "right",
                        "target_on": ["b"],
                        "domain_on": ["a"],
                        "nulls_equal": False,
                        "domain": {"type": "set", "side": "left"},
                    },
                ],
            },
            "join_with_prefilter_details",
            {
                "how": "left",
                "left_on": ["a"],
                "right_on": ["b"],
                "prefilters": [
                    {
                        "type_name": "bloom",
                        "target_side": "left",
                        "target_on": ["a"],
                        "domain_on": ["b"],
                        "nulls_equal": True,
                        "domain": {"type_name": "min-max", "side": None},
                    },
                    {
                        "type_name": "bloom",
                        "target_side": "right",
                        "target_on": ["b"],
                        "domain_on": ["a"],
                        "nulls_equal": False,
                        "domain": {"type_name": "set", "side": "left"},
                    },
                ],
            },
        ),
        (
            "PushdownFilterHint",
            {
                "target_on": ["a", 1],
                "domain_on": ["b"],
                "nulls_equal": True,
                "placement": "build",
            },
            "pushdown_filter_hint_details",
            {
                "target_on": ["a", "1"],
                "domain_on": ["b"],
                "nulls_equal": True,
                "placement": "build",
            },
        ),
        (
            "GroupBy",
            {"keys": ["a", 1]},
            "group_by_details",
            {"keys": ["a", "1"]},
        ),
        (
            "Shuffle",
            {"keys": ["a", 1]},
            "shuffle_details",
            {"keys": ["a", "1"]},
        ),
        (
            "Sort",
            {"by": ["a"], "order": ["ascending"]},
            "sort_details",
            {"by": ["a"], "order": ["ascending"]},
        ),
        (
            "Select",
            {"columns": ["a", 1]},
            "select_details",
            {"columns": ["a", "1"]},
        ),
        (
            "HStack",
            {"columns": ["a", 1]},
            "hstack_details",
            {"columns": ["a", "1"]},
        ),
    ],
)
def test_emit_operator_details(
    node_type: str,
    properties: dict,
    method_name: str,
    expected: dict,
) -> None:
    operator = MagicMock()

    _emit_operator_details(operator, node_type, properties)

    getattr(operator, method_name).assert_called_once_with(values=expected)


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
