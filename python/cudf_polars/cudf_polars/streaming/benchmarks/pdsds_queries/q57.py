# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 57."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.streaming.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.streaming.benchmarks.utils import QueryResult, get_data

if TYPE_CHECKING:
    from cudf_polars.streaming.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 57."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=57,
        qualification=run_config.qualification,
    )

    year = params["year"]

    return f"""
    WITH v1
         AS (SELECT i_category,
                    i_brand,
                    cc_name,
                    d_year,
                    d_moy,
                    Sum(cs_sales_price)                                    sum_sales
                    ,
                    Avg(Sum(cs_sales_price))
                      OVER (
                        partition BY i_category, i_brand, cc_name, d_year)
                    avg_monthly_sales
                       ,
                    Rank()
                      OVER (
                        partition BY i_category, i_brand, cc_name
                        ORDER BY d_year, d_moy)                            rn
             FROM   item,
                    catalog_sales,
                    date_dim,
                    call_center
             WHERE  cs_item_sk = i_item_sk
                    AND cs_sold_date_sk = d_date_sk
                    AND cc_call_center_sk = cs_call_center_sk
                    AND ( d_year = {year}
                           OR ( d_year = {year} - 1
                                AND d_moy = 12 )
                           OR ( d_year = {year} + 1
                                AND d_moy = 1 ) )
             GROUP  BY i_category,
                       i_brand,
                       cc_name,
                       d_year,
                       d_moy),
         v2
         AS (SELECT v1.i_brand,
                    v1.d_year,
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
                    AND v1. cc_name = v1_lag. cc_name
                    AND v1. cc_name = v1_lead. cc_name
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
    """Query 57."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=57,
        qualification=run_config.qualification,
    )

    year = params["year"]

    item = get_data(run_config.dataset_path, "item", run_config.suffix)
    catalog_sales = get_data(
        run_config.dataset_path, "catalog_sales", run_config.suffix
    )
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    call_center = get_data(run_config.dataset_path, "call_center", run_config.suffix)

    v1 = (
        catalog_sales.join(item, left_on="cs_item_sk", right_on="i_item_sk")
        .join(date_dim, left_on="cs_sold_date_sk", right_on="d_date_sk")
        .join(call_center, left_on="cs_call_center_sk", right_on="cc_call_center_sk")
        .filter(
            (pl.col("d_year") == year)
            | ((pl.col("d_year") == year - 1) & (pl.col("d_moy") == 12))
            | ((pl.col("d_year") == year + 1) & (pl.col("d_moy") == 1))
        )
        .group_by(["i_category", "i_brand", "cc_name", "d_year", "d_moy"])
        .agg(
            [
                pl.col("cs_sales_price").count().alias("count_sales"),
                pl.col("cs_sales_price").sum().alias("sum_sales_raw"),
            ]
        )
        .with_columns(
            pl.when(pl.col("count_sales") == 0)
            .then(None)
            .otherwise(pl.col("sum_sales_raw"))
            .alias("sum_sales")
        )
        .drop("count_sales", "sum_sales_raw")
        .with_columns(
            [
                pl.col("sum_sales")
                .mean()
                .over(["i_category", "i_brand", "cc_name", "d_year"])
                .alias("avg_monthly_sales"),
                pl.col("d_year")
                .rank(method="ordinal")
                .over(
                    ["i_category", "i_brand", "cc_name"], order_by=["d_year", "d_moy"]
                )
                .alias("rn"),
            ]
        )
    )

    v2 = (
        v1.join(
            v1.select(
                [
                    pl.col("i_category").alias("i_category_lag"),
                    pl.col("i_brand").alias("i_brand_lag"),
                    pl.col("cc_name").alias("cc_name_lag"),
                    pl.col("rn").alias("rn_lag"),
                    pl.col("sum_sales").alias("psum"),
                ]
            ),
            left_on=["i_category", "i_brand", "cc_name"],
            right_on=["i_category_lag", "i_brand_lag", "cc_name_lag"],
            how="inner",
        )
        .filter(pl.col("rn") == pl.col("rn_lag") + 1)
        .join(
            v1.select(
                [
                    pl.col("i_category").alias("i_category_lead"),
                    pl.col("i_brand").alias("i_brand_lead"),
                    pl.col("cc_name").alias("cc_name_lead"),
                    pl.col("rn").alias("rn_lead"),
                    pl.col("sum_sales").alias("nsum"),
                ]
            ),
            left_on=["i_category", "i_brand", "cc_name"],
            right_on=["i_category_lead", "i_brand_lead", "cc_name_lead"],
            how="inner",
        )
        .filter(pl.col("rn") == pl.col("rn_lead") - 1)
        .select(["i_brand", "d_year", "avg_monthly_sales", "sum_sales", "psum", "nsum"])
    )

    sort_by = {"avg_monthly_sales": False}
    limit = 100
    return QueryResult(
        frame=(
            v2.filter(
                (pl.col("d_year") == year)
                & (pl.col("avg_monthly_sales") > 0)
                & (
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
                    pl.col("avg_monthly_sales"),
                ],
                descending=[False, False],
                nulls_last=True,
            )
            .limit(limit)
        ),
        sort_by=list(sort_by.items()),
        limit=limit,
        sort_keys=[
            (pl.col("sum_sales") - pl.col("avg_monthly_sales"), False),
            (pl.col("avg_monthly_sales"), False),
        ],
    )
