# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Query 12."""

from __future__ import annotations

from datetime import date, timedelta
from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.streaming.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.streaming.benchmarks.pdsds_queries import sql_sum
from cudf_polars.streaming.benchmarks.utils import QueryResult, get_data

if TYPE_CHECKING:
    from cudf_polars.streaming.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 12."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=12,
        qualification=run_config.qualification,
    )

    sdate = params["sdate"]
    categories = params["category"]

    return f"""
    SELECT
             i_item_id ,
             i_item_desc ,
             i_category ,
             i_class ,
             i_current_price ,
             Sum(ws_ext_sales_price)                                                              AS itemrevenue ,
             Sum(ws_ext_sales_price)*100/Sum(Sum(ws_ext_sales_price)) OVER (partition BY i_class) AS revenueratio
    FROM     web_sales ,
             item ,
             date_dim
    WHERE    ws_item_sk = i_item_sk
    AND      i_category IN ({", ".join(f"'{cat}'" for cat in categories)})
    AND      ws_sold_date_sk = d_date_sk
    AND      d_date BETWEEN Cast('{sdate}' AS DATE) AND      (
                      Cast('{sdate}' AS DATE) + INTERVAL '30' day)
    GROUP BY i_item_id ,
             i_item_desc ,
             i_category ,
             i_class ,
             i_current_price
    ORDER BY i_category ,
             i_class ,
             i_item_id ,
             i_item_desc ,
             revenueratio
    LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> QueryResult:
    """Query 12."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=12,
        qualification=run_config.qualification,
    )

    sdate = params["sdate"]
    categories = params["category"]

    # Load tables
    web_sales = get_data(run_config.dataset_path, "web_sales", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)

    # Parse sdate and compute end date
    start_date = date.fromisoformat(sdate)
    end_date = start_date + timedelta(days=30)

    start_date_lit = pl.lit(start_date)
    end_date_lit = pl.lit(end_date)

    return QueryResult(
        frame=(
            web_sales.join(item, left_on="ws_item_sk", right_on="i_item_sk")
            .join(date_dim, left_on="ws_sold_date_sk", right_on="d_date_sk")
            .filter(
                pl.col("i_category").is_in(categories)
                & pl.col("d_date").is_between(
                    start_date_lit, end_date_lit, closed="both"
                )
            )
            .group_by(
                ["i_item_id", "i_item_desc", "i_category", "i_class", "i_current_price"]
            )
            .agg([sql_sum("ws_ext_sales_price").alias("itemrevenue")])
            .with_columns(
                [
                    (
                        pl.col("itemrevenue")
                        * 100
                        / sql_sum("itemrevenue").over("i_class")
                    ).alias("revenueratio")
                ]
            )
            .sort(
                ["i_category", "i_class", "i_item_id", "i_item_desc", "revenueratio"],
                nulls_last=True,
            )
            .limit(100)
        ),
        sort_by=[
            ("i_category", False),
            ("i_class", False),
            ("i_item_id", False),
            ("i_item_desc", False),
            ("revenueratio", False),
        ],
        limit=100,
    )
