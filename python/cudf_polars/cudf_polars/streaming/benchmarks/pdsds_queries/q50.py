# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Query 50."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.streaming.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.streaming.benchmarks.utils import QueryResult, get_data

if TYPE_CHECKING:
    from cudf_polars.streaming.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 50."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=50,
        qualification=run_config.qualification,
    )

    year = params["year"]
    month = params["month"]

    return f"""
    SELECT s_store_name,
                   s_company_id,
                   s_street_number,
                   s_street_name,
                   s_street_type,
                   s_suite_number,
                   s_city,
                   s_county,
                   s_state,
                   s_zip,
                   Sum(CASE
                         WHEN ( sr_returned_date_sk - ss_sold_date_sk <= 30 ) THEN 1
                         ELSE 0
                       END) AS '30 days',
                   Sum(CASE
                         WHEN ( sr_returned_date_sk - ss_sold_date_sk > 30 )
                              AND ( sr_returned_date_sk - ss_sold_date_sk <= 60 )
                       THEN 1
                         ELSE 0
                       END) AS '31-60 days',
                   Sum(CASE
                         WHEN ( sr_returned_date_sk - ss_sold_date_sk > 60 )
                              AND ( sr_returned_date_sk - ss_sold_date_sk <= 90 )
                       THEN 1
                         ELSE 0
                       END) AS '61-90 days',
                   Sum(CASE
                         WHEN ( sr_returned_date_sk - ss_sold_date_sk > 90 )
                              AND ( sr_returned_date_sk - ss_sold_date_sk <= 120 )
                       THEN 1
                         ELSE 0
                       END) AS '91-120 days',
                   Sum(CASE
                         WHEN ( sr_returned_date_sk - ss_sold_date_sk > 120 ) THEN 1
                         ELSE 0
                       END) AS '>120 days'
    FROM   store_sales,
           store_returns,
           store,
           date_dim d1,
           date_dim d2
    WHERE  d2.d_year = {year}
           AND d2.d_moy = {month}
           AND ss_ticket_number = sr_ticket_number
           AND ss_item_sk = sr_item_sk
           AND ss_sold_date_sk = d1.d_date_sk
           AND sr_returned_date_sk = d2.d_date_sk
           AND ss_customer_sk = sr_customer_sk
           AND ss_store_sk = s_store_sk
    GROUP  BY s_store_name,
              s_company_id,
              s_street_number,
              s_street_name,
              s_street_type,
              s_suite_number,
              s_city,
              s_county,
              s_state,
              s_zip
    ORDER  BY s_store_name,
              s_company_id,
              s_street_number,
              s_street_name,
              s_street_type,
              s_suite_number,
              s_city,
              s_county,
              s_state,
              s_zip
    LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> QueryResult:
    """Query 50."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=50,
        qualification=run_config.qualification,
    )

    year = params["year"]
    month = params["month"]

    # Load tables
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    store_returns = get_data(
        run_config.dataset_path, "store_returns", run_config.suffix
    )
    store = get_data(run_config.dataset_path, "store", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)

    # Pre-filter date_dim for the target year/month, then join with
    # store_returns early to shrink the fact table before the expensive
    # multi-key join with store_sales.
    d2 = date_dim.filter(
        (pl.col("d_year") == year) & (pl.col("d_moy") == month)
    ).select("d_date_sk")

    group_cols = [
        "s_store_name",
        "s_company_id",
        "s_street_number",
        "s_street_name",
        "s_street_type",
        "s_suite_number",
        "s_city",
        "s_county",
        "s_state",
        "s_zip",
    ]
    sort_by = dict.fromkeys(group_cols, False)
    limit = 100
    diff = pl.col("sr_returned_date_sk") - pl.col("ss_sold_date_sk")

    return QueryResult(
        frame=(
            store_returns.join(d2, left_on="sr_returned_date_sk", right_on="d_date_sk")
            .join(
                store_sales,
                left_on=["sr_ticket_number", "sr_item_sk", "sr_customer_sk"],
                right_on=["ss_ticket_number", "ss_item_sk", "ss_customer_sk"],
            )
            .join(
                date_dim.select("d_date_sk"),
                left_on="ss_sold_date_sk",
                right_on="d_date_sk",
                how="semi",
            )
            .join(store, left_on="ss_store_sk", right_on="s_store_sk")
            .with_columns(
                (diff <= 30).alias("bucket_30"),
                ((diff > 30) & (diff <= 60)).alias("bucket_31_60"),
                ((diff > 60) & (diff <= 90)).alias("bucket_61_90"),
                ((diff > 90) & (diff <= 120)).alias("bucket_91_120"),
                (diff > 120).alias("bucket_120_plus"),
            )
            .group_by(group_cols)
            .agg(
                pl.col("bucket_30").sum().alias("bucket_30_sum"),
                pl.col("bucket_31_60").sum().alias("bucket_31_60_sum"),
                pl.col("bucket_61_90").sum().alias("bucket_61_90_sum"),
                pl.col("bucket_91_120").sum().alias("bucket_91_120_sum"),
                pl.col("bucket_120_plus").sum().alias("bucket_120_plus_sum"),
                pl.col("sr_returned_date_sk").count().alias("return_count"),
            )
            .with_columns(
                pl.when(pl.col("return_count") > 0)
                .then(pl.col("bucket_30_sum"))
                .otherwise(None)
                .alias("30 days"),
                pl.when(pl.col("return_count") > 0)
                .then(pl.col("bucket_31_60_sum"))
                .otherwise(None)
                .alias("31-60 days"),
                pl.when(pl.col("return_count") > 0)
                .then(pl.col("bucket_61_90_sum"))
                .otherwise(None)
                .alias("61-90 days"),
                pl.when(pl.col("return_count") > 0)
                .then(pl.col("bucket_91_120_sum"))
                .otherwise(None)
                .alias("91-120 days"),
                pl.when(pl.col("return_count") > 0)
                .then(pl.col("bucket_120_plus_sum"))
                .otherwise(None)
                .alias(">120 days"),
            )
            .drop(
                "bucket_30_sum",
                "bucket_31_60_sum",
                "bucket_61_90_sum",
                "bucket_91_120_sum",
                "bucket_120_plus_sum",
                "return_count",
            )
            .select(
                *group_cols,
                "30 days",
                "31-60 days",
                "61-90 days",
                "91-120 days",
                ">120 days",
            )
            .sort(group_cols, nulls_last=True)
            .limit(limit)
        ),
        sort_by=list(sort_by.items()),
        limit=limit,
    )
