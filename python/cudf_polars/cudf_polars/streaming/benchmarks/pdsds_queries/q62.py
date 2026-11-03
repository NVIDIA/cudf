# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 62."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.streaming.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.streaming.benchmarks.utils import QueryResult, get_data

if TYPE_CHECKING:
    from cudf_polars.streaming.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 62."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=62,
        qualification=run_config.qualification,
    )

    dms = params["dms"]

    return f"""
    SELECT Substr(w_warehouse_name, 1, 20),
                   sm_type,
                   web_name,
                   Sum(CASE
                         WHEN ( ws_ship_date_sk - ws_sold_date_sk <= 30 ) THEN 1
                         ELSE 0
                       END) AS '30 days',
                   Sum(CASE
                         WHEN ( ws_ship_date_sk - ws_sold_date_sk > 30 )
                              AND ( ws_ship_date_sk - ws_sold_date_sk <= 60 ) THEN 1
                         ELSE 0
                       END) AS '31-60 days',
                   Sum(CASE
                         WHEN ( ws_ship_date_sk - ws_sold_date_sk > 60 )
                              AND ( ws_ship_date_sk - ws_sold_date_sk <= 90 ) THEN 1
                         ELSE 0
                       END) AS '61-90 days',
                   Sum(CASE
                         WHEN ( ws_ship_date_sk - ws_sold_date_sk > 90 )
                              AND ( ws_ship_date_sk - ws_sold_date_sk <= 120 ) THEN
                         1
                         ELSE 0
                       END) AS '91-120 days',
                   Sum(CASE
                         WHEN ( ws_ship_date_sk - ws_sold_date_sk > 120 ) THEN 1
                         ELSE 0
                       END) AS '>120 days'
    FROM   web_sales,
           warehouse,
           ship_mode,
           web_site,
           date_dim
    WHERE  d_month_seq BETWEEN {dms} AND {dms} + 11
           AND ws_ship_date_sk = d_date_sk
           AND ws_warehouse_sk = w_warehouse_sk
           AND ws_ship_mode_sk = sm_ship_mode_sk
           AND ws_web_site_sk = web_site_sk
    GROUP  BY Substr(w_warehouse_name, 1, 20),
              sm_type,
              web_name
    ORDER  BY Substr(w_warehouse_name, 1, 20),
              sm_type,
              web_name
    LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> QueryResult:
    """Query 62."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=62,
        qualification=run_config.qualification,
    )

    dms = params["dms"]

    web_sales = get_data(run_config.dataset_path, "web_sales", run_config.suffix)
    warehouse = get_data(run_config.dataset_path, "warehouse", run_config.suffix)
    ship_mode = get_data(run_config.dataset_path, "ship_mode", run_config.suffix)
    web_site = get_data(run_config.dataset_path, "web_site", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)

    sort_by = {
        "substr(w_warehouse_name, 1, 20)": False,
        "sm_type": False,
        "web_name": False,
    }
    limit = 100
    return QueryResult(
        frame=(
            web_sales.join(date_dim, left_on="ws_ship_date_sk", right_on="d_date_sk")
            .join(warehouse, left_on="ws_warehouse_sk", right_on="w_warehouse_sk")
            .join(ship_mode, left_on="ws_ship_mode_sk", right_on="sm_ship_mode_sk")
            .join(web_site, left_on="ws_web_site_sk", right_on="web_site_sk")
            .filter(pl.col("d_month_seq").is_between(dms, dms + 11))
            .with_columns(
                [
                    (pl.col("ws_ship_date_sk") - pl.col("ws_sold_date_sk")).alias(
                        "shipping_delay"
                    ),
                    pl.col("w_warehouse_name")
                    .str.slice(0, 20)
                    .alias("warehouse_substr"),
                ]
            )
            .with_columns(
                [
                    pl.when(pl.col("shipping_delay") <= 30)
                    .then(1)
                    .otherwise(0)
                    .alias("bucket_30"),
                    pl.when(
                        (pl.col("shipping_delay") > 30)
                        & (pl.col("shipping_delay") <= 60)
                    )
                    .then(1)
                    .otherwise(0)
                    .alias("bucket_31_60"),
                    pl.when(
                        (pl.col("shipping_delay") > 60)
                        & (pl.col("shipping_delay") <= 90)
                    )
                    .then(1)
                    .otherwise(0)
                    .alias("bucket_61_90"),
                    pl.when(
                        (pl.col("shipping_delay") > 90)
                        & (pl.col("shipping_delay") <= 120)
                    )
                    .then(1)
                    .otherwise(0)
                    .alias("bucket_91_120"),
                    pl.when(pl.col("shipping_delay") > 120)
                    .then(1)
                    .otherwise(0)
                    .alias("bucket_over_120"),
                ]
            )
            .group_by(["warehouse_substr", "sm_type", "web_name"])
            .agg(
                [
                    pl.col("bucket_30").sum().alias("30 days"),
                    pl.col("bucket_31_60").sum().alias("31-60 days"),
                    pl.col("bucket_61_90").sum().alias("61-90 days"),
                    pl.col("bucket_91_120").sum().alias("91-120 days"),
                    pl.col("bucket_over_120").sum().alias(">120 days"),
                ]
            )
            .select(
                [
                    pl.col("warehouse_substr").alias("substr(w_warehouse_name, 1, 20)"),
                    "sm_type",
                    "web_name",
                    "30 days",
                    "31-60 days",
                    "61-90 days",
                    "91-120 days",
                    ">120 days",
                ]
            )
            .sort(sort_by.keys(), nulls_last=True)
            .limit(limit)
        ),
        sort_by=list(sort_by.items()),
        limit=limit,
    )
