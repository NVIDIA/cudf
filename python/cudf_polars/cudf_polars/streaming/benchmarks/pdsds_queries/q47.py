# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 47."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.streaming.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.streaming.benchmarks.utils import QueryResult, get_data

if TYPE_CHECKING:
    from cudf_polars.streaming.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 47."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=47,
        qualification=run_config.qualification,
    )

    year = params["year"]

    return f"""
    WITH v1
         AS (SELECT i_category,
                    i_brand,
                    s_store_name,
                    s_company_name,
                    d_year,
                    d_moy,
                    Sum(ss_sales_price)         sum_sales,
                    Avg(Sum(ss_sales_price))
                      OVER (
                        partition BY i_category, i_brand, s_store_name,
                      s_company_name,
                      d_year)
                                                avg_monthly_sales,
                    Rank()
                      OVER (
                        partition BY i_category, i_brand, s_store_name,
                      s_company_name
                        ORDER BY d_year, d_moy) rn
             FROM   item,
                    store_sales,
                    date_dim,
                    store
             WHERE  ss_item_sk = i_item_sk
                    AND ss_sold_date_sk = d_date_sk
                    AND ss_store_sk = s_store_sk
                    AND ( d_year = {year}
                           OR ( d_year = {year} - 1
                                AND d_moy = 12 )
                           OR ( d_year = {year} + 1
                                AND d_moy = 1 ) )
             GROUP  BY i_category,
                       i_brand,
                       s_store_name,
                       s_company_name,
                       d_year,
                       d_moy),
         v2
         AS (SELECT v1.i_category,
                    v1.d_year,
                    v1.d_moy,
                    v1.avg_monthly_sales,
                    v1.sum_sales,
                    v1_lag.sum_sales  psum,
                    v1_lead.sum_sales nsum
             FROM   v1,
                    v1 v1_lag,
                    v1 v1_lead
             WHERE  v1.i_category = v1_lag.i_category
                    AND v1.i_category = v1_lead.i_category
                    AND v1.i_brand = v1_lag.i_brand
                    AND v1.i_brand = v1_lead.i_brand
                    AND v1.s_store_name = v1_lag.s_store_name
                    AND v1.s_store_name = v1_lead.s_store_name
                    AND v1.s_company_name = v1_lag.s_company_name
                    AND v1.s_company_name = v1_lead.s_company_name
                    AND v1.rn = v1_lag.rn + 1
                    AND v1.rn = v1_lead.rn - 1)
    SELECT *
    FROM   v2
    WHERE  d_year = {year}
           AND avg_monthly_sales > 0
           AND CASE
                 WHEN avg_monthly_sales > 0 THEN Abs(sum_sales - avg_monthly_sales)
                                                 /
                                                 avg_monthly_sales
                 ELSE NULL
               END > 0.1
    ORDER  BY sum_sales - avg_monthly_sales,
              3
    LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> QueryResult:
    """Query 47."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=47,
        qualification=run_config.qualification,
    )

    year = params["year"]

    # Load tables
    item = get_data(run_config.dataset_path, "item", run_config.suffix)
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    store = get_data(run_config.dataset_path, "store", run_config.suffix)
    # Step 1: Create CTE v1 equivalent
    v1 = (
        store_sales.join(item, left_on="ss_item_sk", right_on="i_item_sk")
        .join(date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk")
        .join(store, left_on="ss_store_sk", right_on="s_store_sk")
        .filter(
            # Complex date filter: year OR (year-1 + Dec) OR (year+1 + Jan)
            (pl.col("d_year") == year)
            | ((pl.col("d_year") == year - 1) & (pl.col("d_moy") == 12))
            | ((pl.col("d_year") == year + 1) & (pl.col("d_moy") == 1))
        )
        .group_by(
            [
                "i_category",
                "i_brand",
                "s_store_name",
                "s_company_name",
                "d_year",
                "d_moy",
            ]
        )
        .agg(
            [
                pl.col("ss_sales_price").sum().alias("sales_sum"),
                pl.col("ss_sales_price").count().alias("sales_count"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("sales_count") > 0)
                .then(pl.col("sales_sum"))
                .otherwise(None)
                .alias("sum_sales")
            ]
        )
        .drop(["sales_sum", "sales_count"])
        .with_columns(
            [
                # Window function: AVG over partition by category, brand, store, company, year
                pl.col("sum_sales")
                .mean()
                .over(
                    [
                        "i_category",
                        "i_brand",
                        "s_store_name",
                        "s_company_name",
                        "d_year",
                    ]
                )
                .alias("avg_monthly_sales"),
                # Window function: RANK over partition by category, brand, store, company ordered by year, month
                pl.col("d_year")
                .rank(method="ordinal")
                .over(
                    ["i_category", "i_brand", "s_store_name", "s_company_name"],
                    order_by=["d_year", "d_moy"],
                )
                .alias("rn"),
            ]
        )
    )
    # Step 2: Create CTE v2 equivalent (self-joins for lag/lead)
    v2 = (
        v1
        # Join with lag (previous period)
        .join(
            v1.select(
                [
                    pl.col("i_category").alias("i_category_lag"),
                    pl.col("i_brand").alias("i_brand_lag"),
                    pl.col("s_store_name").alias("s_store_name_lag"),
                    pl.col("s_company_name").alias("s_company_name_lag"),
                    pl.col("rn").alias("rn_lag"),
                    pl.col("sum_sales").alias("psum"),
                ]
            ),
            left_on=["i_category", "i_brand", "s_store_name", "s_company_name"],
            right_on=[
                "i_category_lag",
                "i_brand_lag",
                "s_store_name_lag",
                "s_company_name_lag",
            ],
            how="inner",
        )
        .filter(pl.col("rn") == pl.col("rn_lag") + 1)
        # Join with lead (next period)
        .join(
            v1.select(
                [
                    pl.col("i_category").alias("i_category_lead"),
                    pl.col("i_brand").alias("i_brand_lead"),
                    pl.col("s_store_name").alias("s_store_name_lead"),
                    pl.col("s_company_name").alias("s_company_name_lead"),
                    pl.col("rn").alias("rn_lead"),
                    pl.col("sum_sales").alias("nsum"),
                ]
            ),
            left_on=["i_category", "i_brand", "s_store_name", "s_company_name"],
            right_on=[
                "i_category_lead",
                "i_brand_lead",
                "s_store_name_lead",
                "s_company_name_lead",
            ],
            how="inner",
        )
        .filter(pl.col("rn") == pl.col("rn_lead") - 1)
        .select(
            [
                "i_category",
                "d_year",
                "d_moy",
                "avg_monthly_sales",
                "sum_sales",
                "psum",
                "nsum",
            ]
        )
    )
    sort_by = {"d_moy": False}
    limit = 100
    # Step 3: Final query with filters and calculations
    return QueryResult(
        frame=(
            v2.filter(
                (pl.col("d_year") == year)
                & (pl.col("avg_monthly_sales") > 0)
                &
                # Percentage deviation > 10%
                (
                    pl.when(pl.col("avg_monthly_sales") > 0)
                    .then(
                        (pl.col("sum_sales") - pl.col("avg_monthly_sales")).abs()
                        / pl.col("avg_monthly_sales")
                    )
                    .otherwise(None)
                    > 0.1
                )
            )
            .sort(
                by=[
                    pl.col("sum_sales") - pl.col("avg_monthly_sales"),
                    pl.col("d_moy"),
                ],
                descending=[False, False],
                nulls_last=True,
            )
            .select(
                [
                    "i_category",
                    "d_year",
                    "d_moy",
                    "avg_monthly_sales",
                    "sum_sales",
                    "psum",
                    "nsum",
                ]
            )
            .limit(limit)
        ),
        sort_by=list(sort_by.items()),
        limit=limit,
        sort_keys=[
            (pl.col("sum_sales") - pl.col("avg_monthly_sales"), False),
            (pl.col("d_moy"), False),
        ],
    )
