# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 22."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.streaming.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.streaming.benchmarks.utils import QueryResult, get_data

if TYPE_CHECKING:
    from cudf_polars.streaming.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 22."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=22,
        qualification=run_config.qualification,
    )

    dms = params["dms"]

    return f"""
    SELECT i_product_name,
                   i_brand,
                   i_class,
                   i_category,
                   Avg(inv_quantity_on_hand) qoh
    FROM   inventory,
           date_dim,
           item,
           warehouse
    WHERE  inv_date_sk = d_date_sk
           AND inv_item_sk = i_item_sk
           AND inv_warehouse_sk = w_warehouse_sk
           AND d_month_seq BETWEEN {dms} AND {dms} + 11
    GROUP  BY rollup( i_product_name, i_brand, i_class, i_category )
    ORDER  BY qoh,
              i_product_name,
              i_brand,
              i_class,
              i_category
    LIMIT 100;
    """


def level(  # noqa: D103
    base_data: pl.LazyFrame, agg_exprs: list[pl.Expr], group_cols: list[str]
) -> pl.LazyFrame:
    if group_cols:
        lf = base_data.group_by(group_cols).agg(agg_exprs)
    else:
        lf = base_data.select(agg_exprs)
    missing = [
        c
        for c in ["i_product_name", "i_brand", "i_class", "i_category"]
        if c not in group_cols
    ]
    if missing:
        lf = lf.with_columns([pl.lit(None, dtype=pl.String).alias(c) for c in missing])
    return lf.select(
        [
            "i_product_name",
            "i_brand",
            "i_class",
            "i_category",
            "qoh",
        ]
    )


def polars_impl(run_config: RunConfig) -> QueryResult:
    """Query 22."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=22,
        qualification=run_config.qualification,
    )

    dms = params["dms"]

    inventory = get_data(run_config.dataset_path, "inventory", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)
    warehouse = get_data(run_config.dataset_path, "warehouse", run_config.suffix)
    base_data = (
        inventory.join(date_dim, left_on="inv_date_sk", right_on="d_date_sk")
        .join(item, left_on="inv_item_sk", right_on="i_item_sk")
        .join(warehouse, left_on="inv_warehouse_sk", right_on="w_warehouse_sk")
        .filter(pl.col("d_month_seq").is_between(dms, dms + 11))
    )
    agg_exprs = [pl.col("inv_quantity_on_hand").mean().alias("qoh")]

    level1 = level(
        base_data, agg_exprs, ["i_product_name", "i_brand", "i_class", "i_category"]
    )
    level2 = level(base_data, agg_exprs, ["i_product_name", "i_brand", "i_class"])
    level3 = level(base_data, agg_exprs, ["i_product_name", "i_brand"])
    level4 = level(base_data, agg_exprs, ["i_product_name"])
    level5 = level(base_data, agg_exprs, [])

    sort_by = {
        "qoh": False,
        "i_product_name": False,
        "i_brand": False,
        "i_class": False,
        "i_category": False,
    }
    limit = 100

    return QueryResult(
        frame=(
            pl.concat([level1, level2, level3, level4, level5])
            .sort(sort_by.keys(), nulls_last=True)
            .limit(limit)
        ),
        sort_by=list(sort_by.items()),
        limit=limit,
    )
