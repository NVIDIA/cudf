# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 21."""

from __future__ import annotations

from datetime import date, timedelta
from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.streaming.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.streaming.benchmarks.utils import QueryResult, get_data

if TYPE_CHECKING:
    from cudf_polars.streaming.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 21."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=21,
        qualification=run_config.qualification,
    )

    sales_date = params["sales_date"]

    return f"""
    SELECT
             *
    FROM    (
                      SELECT   w_warehouse_name ,
                               i_item_id ,
                               Sum(
                               CASE
                                        WHEN (
                                                          Cast(d_date AS DATE) < Cast ('{sales_date}' AS DATE)) THEN inv_quantity_on_hand
                                        ELSE 0
                               END) AS inv_before ,
                               Sum(
                               CASE
                                        WHEN (
                                                          Cast(d_date AS DATE) >= Cast ('{sales_date}' AS DATE)) THEN inv_quantity_on_hand
                                        ELSE 0
                               END) AS inv_after
                      FROM     inventory ,
                               warehouse ,
                               item ,
                               date_dim
                      WHERE    i_current_price BETWEEN 0.99 AND      1.49
                      AND      i_item_sk = inv_item_sk
                      AND      inv_warehouse_sk = w_warehouse_sk
                      AND      inv_date_sk = d_date_sk
                      AND      d_date BETWEEN (Cast ('{sales_date}' AS DATE) - INTERVAL '30' day) AND      (
                                        cast ('{sales_date}' AS        date) + INTERVAL '30' day)
                      GROUP BY w_warehouse_name,
                               i_item_id) x
    WHERE    (
                      CASE
                               WHEN inv_before > 0 THEN inv_after / inv_before
                               ELSE NULL
                      END) BETWEEN 2.0/3.0 AND      3.0/2.0
    ORDER BY w_warehouse_name ,
             i_item_id
    LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> QueryResult:
    """Query 21."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=21,
        qualification=run_config.qualification,
    )

    sales_date_str = params["sales_date"]

    # Load tables
    inventory = get_data(run_config.dataset_path, "inventory", run_config.suffix)
    warehouse = get_data(run_config.dataset_path, "warehouse", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)

    # Parse sales_date and compute date range
    sales_date_obj = date.fromisoformat(sales_date_str)
    start_date = sales_date_obj - timedelta(days=30)
    end_date = sales_date_obj + timedelta(days=30)

    # Use date literals so comparisons work for both Date and String d_date columns
    sales_date_lit = pl.lit(sales_date_obj)
    start_date_lit = pl.lit(start_date)
    end_date_lit = pl.lit(end_date)
    d_date = pl.col("d_date")

    sort_by = {"w_warehouse_name": False, "i_item_id": False}
    limit = 100

    return QueryResult(
        frame=(
            inventory.join(item, left_on="inv_item_sk", right_on="i_item_sk")
            .join(warehouse, left_on="inv_warehouse_sk", right_on="w_warehouse_sk")
            .join(date_dim, left_on="inv_date_sk", right_on="d_date_sk")
            .filter(
                (pl.col("i_current_price").is_between(0.99, 1.49))
                & d_date.is_between(start_date_lit, end_date_lit, closed="both")
            )
            .with_columns(
                [
                    pl.when(d_date < sales_date_lit)
                    .then(pl.col("inv_quantity_on_hand"))
                    .otherwise(0)
                    .alias("inv_before_amount"),
                    pl.when(d_date >= sales_date_lit)
                    .then(pl.col("inv_quantity_on_hand"))
                    .otherwise(0)
                    .alias("inv_after_amount"),
                ]
            )
            .group_by(["w_warehouse_name", "i_item_id"])
            .agg(
                [
                    pl.col("inv_before_amount").sum().alias("inv_before"),
                    pl.col("inv_after_amount").sum().alias("inv_after"),
                ]
            )
            .filter(
                pl.when(pl.col("inv_before") > 0)
                .then(pl.col("inv_after") / pl.col("inv_before"))
                .otherwise(None)
                .is_between(2.0 / 3.0, 3.0 / 2.0)
            )
            .filter(pl.col("w_warehouse_name").is_not_null())
            .sort(sort_by.keys())
            .limit(limit)
        ),
        sort_by=list(sort_by.items()),
        limit=limit,
    )
