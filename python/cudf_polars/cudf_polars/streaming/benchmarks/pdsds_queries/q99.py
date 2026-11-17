# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 99."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.streaming.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.streaming.benchmarks.utils import QueryResult, get_data

if TYPE_CHECKING:
    from cudf_polars.streaming.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 99."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=99,
        qualification=run_config.qualification,
    )

    d_month_seq = params["d_month_seq"]

    return f"""
    -- start query 99 in stream 0 using template query99.tpl
    SELECT Substr(w_warehouse_name, 1, 20),
                   sm_type,
                   cc_name,
                   Sum(CASE
                         WHEN ( cs_ship_date_sk - cs_sold_date_sk <= 30 ) THEN 1
                         ELSE 0
                       END) AS '30 days',
                   Sum(CASE
                         WHEN ( cs_ship_date_sk - cs_sold_date_sk > 30 )
                              AND ( cs_ship_date_sk - cs_sold_date_sk <= 60 ) THEN 1
                         ELSE 0
                       END) AS '31-60 days',
                   Sum(CASE
                         WHEN ( cs_ship_date_sk - cs_sold_date_sk > 60 )
                              AND ( cs_ship_date_sk - cs_sold_date_sk <= 90 ) THEN 1
                         ELSE 0
                       END) AS '61-90 days',
                   Sum(CASE
                         WHEN ( cs_ship_date_sk - cs_sold_date_sk > 90 )
                              AND ( cs_ship_date_sk - cs_sold_date_sk <= 120 ) THEN
                         1
                         ELSE 0
                       END) AS '91-120 days',
                   Sum(CASE
                         WHEN ( cs_ship_date_sk - cs_sold_date_sk > 120 ) THEN 1
                         ELSE 0
                       END) AS '>120 days'
    FROM   catalog_sales,
           warehouse,
           ship_mode,
           call_center,
           date_dim
    WHERE  d_month_seq BETWEEN {d_month_seq} AND {d_month_seq} + 11
           AND cs_ship_date_sk = d_date_sk
           AND cs_warehouse_sk = w_warehouse_sk
           AND cs_ship_mode_sk = sm_ship_mode_sk
           AND cs_call_center_sk = cc_call_center_sk
    GROUP  BY Substr(w_warehouse_name, 1, 20),
              sm_type,
              cc_name
    ORDER  BY Substr(w_warehouse_name, 1, 20),
              sm_type,
              cc_name
    LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> QueryResult:
    """Query 99."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=99,
        qualification=run_config.qualification,
    )

    d_month_seq = params["d_month_seq"]
    catalog_sales = get_data(
        run_config.dataset_path, "catalog_sales", run_config.suffix
    )
    warehouse = get_data(run_config.dataset_path, "warehouse", run_config.suffix)
    ship_mode = get_data(run_config.dataset_path, "ship_mode", run_config.suffix)
    call_center = get_data(run_config.dataset_path, "call_center", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)

    return QueryResult(
        frame=(
            catalog_sales.join(
                date_dim, left_on="cs_ship_date_sk", right_on="d_date_sk", how="inner"
            )
            .join(
                warehouse,
                left_on="cs_warehouse_sk",
                right_on="w_warehouse_sk",
                how="inner",
            )
            .join(
                ship_mode,
                left_on="cs_ship_mode_sk",
                right_on="sm_ship_mode_sk",
                how="inner",
            )
            .join(
                call_center,
                left_on="cs_call_center_sk",
                right_on="cc_call_center_sk",
                how="inner",
            )
            .filter(
                pl.col("d_month_seq").is_between(
                    d_month_seq, d_month_seq + 11, closed="both"
                )
            )
            .with_columns(
                [
                    (pl.col("cs_ship_date_sk") - pl.col("cs_sold_date_sk")).alias(
                        "ship_days"
                    ),
                    pl.col("w_warehouse_name")
                    .str.slice(0, 20)
                    .alias("substr(w_warehouse_name, 1, 20)"),
                ]
            )
            .with_columns(
                [
                    pl.when(pl.col("ship_days") <= 30)
                    .then(1)
                    .otherwise(0)
                    .alias("days_30"),
                    pl.when((pl.col("ship_days") > 30) & (pl.col("ship_days") <= 60))
                    .then(1)
                    .otherwise(0)
                    .alias("days_31_60"),
                    pl.when((pl.col("ship_days") > 60) & (pl.col("ship_days") <= 90))
                    .then(1)
                    .otherwise(0)
                    .alias("days_61_90"),
                    pl.when((pl.col("ship_days") > 90) & (pl.col("ship_days") <= 120))
                    .then(1)
                    .otherwise(0)
                    .alias("days_91_120"),
                    pl.when(pl.col("ship_days") > 120)
                    .then(1)
                    .otherwise(0)
                    .alias("days_120_plus"),
                ]
            )
            .group_by(["substr(w_warehouse_name, 1, 20)", "sm_type", "cc_name"])
            .agg(
                [
                    pl.col("days_30").sum().alias("30 days"),
                    pl.col("days_31_60").sum().alias("31-60 days"),
                    pl.col("days_61_90").sum().alias("61-90 days"),
                    pl.col("days_91_120").sum().alias("91-120 days"),
                    pl.col("days_120_plus").sum().alias(">120 days"),
                ]
            )
            .select(
                [
                    "substr(w_warehouse_name, 1, 20)",
                    "sm_type",
                    "cc_name",
                    "30 days",
                    "31-60 days",
                    "61-90 days",
                    "91-120 days",
                    ">120 days",
                ]
            )
            .sort(
                ["substr(w_warehouse_name, 1, 20)", "sm_type", "cc_name"],
                nulls_last=True,
            )
            .limit(100)
        ),
        sort_by=[
            ("substr(w_warehouse_name, 1, 20)", False),
            ("sm_type", False),
            ("cc_name", False),
        ],
        limit=100,
    )
