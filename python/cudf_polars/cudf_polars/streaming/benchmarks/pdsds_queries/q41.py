# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 41."""

from __future__ import annotations

import operator as op
from functools import reduce
from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.streaming.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.streaming.benchmarks.utils import QueryResult, get_data

if TYPE_CHECKING:
    from cudf_polars.streaming.benchmarks.utils import RunConfig

Q_NUM = 41

MANUFACTURER_ID_START = 765


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 41."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=41,
        qualification=run_config.qualification,
    )

    manufact = params["manufact"]
    rules = params["rules"]

    # Build SQL conditions for each rule (8 rules total)
    formatted_rules = [
        f"(i_category = '{rule['category']}' and\n"
        f"        (i_color = '{rule['colors'][0]}' or i_color = '{rule['colors'][1]}') and\n"
        f"        (i_units = '{rule['units'][0]}' or i_units = '{rule['units'][1]}') and\n"
        f"        (i_size = '{rule['sizes'][0]}' or i_size = '{rule['sizes'][1]}')\n"
        f"        )"
        for rule in rules
    ]

    # Split rules into two groups (4 rules each for the two OR blocks)
    rules_block1 = " or\n        ".join(formatted_rules[:4])
    rules_block2 = " or\n        ".join(formatted_rules[4:])

    return f"""
    SELECT Distinct(i_product_name)
    FROM   item i1
    WHERE  i_manufact_id BETWEEN {manufact} AND {manufact} + 40
        AND (SELECT Count(*) AS item_cnt
                FROM   item
                WHERE  ( i_manufact = i1.i_manufact
                        AND ( {rules_block1} ) )
                        OR ( i_manufact = i1.i_manufact
                            AND ( {rules_block2} ) )) > 0
    ORDER  BY i_product_name
    LIMIT 100;

    """


def polars_impl(run_config: RunConfig) -> QueryResult:
    """Query 41."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=41,
        qualification=run_config.qualification,
    )

    manufact = params["manufact"]
    rules = params["rules"]

    item = get_data(run_config.dataset_path, "item", run_config.suffix)

    # Convert rules from dict format to tuple format and build expressions
    rule_exprs = [
        (
            (pl.col("i_category") == rule["category"])
            & pl.col("i_color").is_in(rule["colors"])
            & pl.col("i_units").is_in(rule["units"])
            & pl.col("i_size").is_in(rule["sizes"])
        )
        for rule in rules
    ]
    subquery_conditions = reduce(op.or_, rule_exprs)

    manufacturers_with_criteria = (
        item.filter(subquery_conditions).select("i_manufact").unique()
    )

    sort_by = {"i_product_name": False}
    limit = 100
    return QueryResult(
        frame=(
            item.filter(pl.col("i_manufact_id").is_between(manufact, manufact + 40))
            .join(manufacturers_with_criteria, on="i_manufact", how="inner")
            .select("i_product_name")
            .unique()
            .sort(sort_by.keys(), nulls_last=True)
            .limit(limit)
        ),
        sort_by=list(sort_by.items()),
        limit=limit,
    )
