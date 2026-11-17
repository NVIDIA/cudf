# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 65."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.streaming.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.streaming.benchmarks.utils import QueryResult, get_data

if TYPE_CHECKING:
    from cudf_polars.streaming.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 65."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=65,
        qualification=run_config.qualification,
    )

    dms = params["dms"]

    return f"""
    SELECT s_store_name,
                   i_item_desc,
                   sc.revenue,
                   i_current_price,
                   i_wholesale_cost,
                   i_brand
    FROM   store,
           item,
           (SELECT ss_store_sk,
                   Avg(revenue) AS ave
            FROM   (SELECT ss_store_sk,
                           ss_item_sk,
                           Sum(ss_sales_price) AS revenue
                    FROM   store_sales,
                           date_dim
                    WHERE  ss_sold_date_sk = d_date_sk
                           AND d_month_seq BETWEEN {dms} AND {dms} + 11
                    GROUP  BY ss_store_sk,
                              ss_item_sk) sa
            GROUP  BY ss_store_sk) sb,
           (SELECT ss_store_sk,
                   ss_item_sk,
                   Sum(ss_sales_price) AS revenue
            FROM   store_sales,
                   date_dim
            WHERE  ss_sold_date_sk = d_date_sk
                   AND d_month_seq BETWEEN {dms} AND {dms} + 11
            GROUP  BY ss_store_sk,
                      ss_item_sk) sc
    WHERE  sb.ss_store_sk = sc.ss_store_sk
           AND sc.revenue <= 0.1 * sb.ave
           AND s_store_sk = sc.ss_store_sk
           AND i_item_sk = sc.ss_item_sk
    ORDER  BY s_store_name,
              i_item_desc,
              revenue,  --added for deterministic ordering
              i_current_price,  --added for deterministic ordering
              i_brand --added for deterministic ordering
    LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> QueryResult:
    """Query 65."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=65,
        qualification=run_config.qualification,
    )

    dms = params["dms"]

    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    store = get_data(run_config.dataset_path, "store", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)

    revenue_by_store_item = (
        store_sales.join(date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk")
        .filter(pl.col("d_month_seq").is_between(dms, dms + 11))
        .group_by(["ss_store_sk", "ss_item_sk"])
        .agg(
            [
                pl.col("ss_sales_price").count().alias("revenue_count"),
                pl.col("ss_sales_price").sum().alias("revenue_sum"),
            ]
        )
        .with_columns(
            pl.when(pl.col("revenue_count") == 0)
            .then(None)
            .otherwise(pl.col("revenue_sum"))
            .alias("revenue")
        )
        .select(["ss_store_sk", "ss_item_sk", "revenue"])
    )

    avg_revenue_by_store = revenue_by_store_item.group_by("ss_store_sk").agg(
        [pl.col("revenue").mean().alias("ave")]
    )

    return QueryResult(
        frame=(
            revenue_by_store_item.join(
                avg_revenue_by_store, on="ss_store_sk", suffix="_sb"
            )
            .filter(pl.col("revenue") <= 0.1 * pl.col("ave"))
            .join(store, left_on="ss_store_sk", right_on="s_store_sk")
            .join(item, left_on="ss_item_sk", right_on="i_item_sk")
            .select(
                [
                    "s_store_name",
                    "i_item_desc",
                    "revenue",
                    "i_current_price",
                    "i_wholesale_cost",
                    "i_brand",
                ]
            )
            .sort(
                [
                    "s_store_name",
                    "i_item_desc",
                    "revenue",
                    "i_current_price",
                    "i_brand",
                ],
                nulls_last=True,
                descending=[False, False, False, False, False],
            )
            .limit(100)
        ),
        sort_by=[
            ("s_store_name", False),
            ("i_item_desc", False),
            ("revenue", False),
            ("i_current_price", False),
            ("i_brand", False),
        ],
        limit=100,
    )
