# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Query 72."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.streaming.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.streaming.benchmarks.utils import COUNT_DTYPE, QueryResult, get_data

if TYPE_CHECKING:
    from cudf_polars.streaming.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 72."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=72,
        qualification=run_config.qualification,
    )

    year = params["year"]
    bp = params["bp"]
    ms = params["ms"]

    return f"""
    SELECT i_item_desc,
                   w_warehouse_name,
                   d1.d_week_seq,
                   Sum(CASE
                         WHEN p_promo_sk IS NULL THEN 1
                         ELSE 0
                       END) no_promo,
                   Sum(CASE
                         WHEN p_promo_sk IS NOT NULL THEN 1
                         ELSE 0
                       END) promo,
                   Count(*) total_cnt
    FROM   catalog_sales
           JOIN inventory
             ON ( cs_item_sk = inv_item_sk )
           JOIN warehouse
             ON ( w_warehouse_sk = inv_warehouse_sk )
           JOIN item
             ON ( i_item_sk = cs_item_sk )
           JOIN customer_demographics
             ON ( cs_bill_cdemo_sk = cd_demo_sk )
           JOIN household_demographics
             ON ( cs_bill_hdemo_sk = hd_demo_sk )
           JOIN date_dim d1
             ON ( cs_sold_date_sk = d1.d_date_sk )
           JOIN date_dim d2
             ON ( inv_date_sk = d2.d_date_sk )
           JOIN date_dim d3
             ON ( cs_ship_date_sk = d3.d_date_sk )
           LEFT OUTER JOIN promotion
                        ON ( cs_promo_sk = p_promo_sk )
           LEFT OUTER JOIN catalog_returns
                        ON ( cr_item_sk = cs_item_sk
                             AND cr_order_number = cs_order_number )
    WHERE  d1.d_week_seq = d2.d_week_seq
           AND inv_quantity_on_hand < cs_quantity
           AND d3.d_date > d1.d_date + INTERVAL '5' day
           AND hd_buy_potential = '{bp}'
           AND d1.d_year = {year}
           AND cd_marital_status = '{ms}'
    GROUP  BY i_item_desc,
              w_warehouse_name,
              d1.d_week_seq
    ORDER  BY total_cnt DESC,
              i_item_desc,
              w_warehouse_name,
              d1.d_week_seq
    LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> QueryResult:
    """Query 72."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=72,
        qualification=run_config.qualification,
    )

    year = params["year"]
    bp = params["bp"]
    ms = params["ms"]

    catalog_sales = get_data(
        run_config.dataset_path, "catalog_sales", run_config.suffix
    )
    inventory = get_data(run_config.dataset_path, "inventory", run_config.suffix)
    warehouse = get_data(run_config.dataset_path, "warehouse", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)
    customer_demographics = get_data(
        run_config.dataset_path, "customer_demographics", run_config.suffix
    )
    household_demographics = get_data(
        run_config.dataset_path, "household_demographics", run_config.suffix
    )
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    promotion = get_data(run_config.dataset_path, "promotion", run_config.suffix)
    catalog_returns = get_data(
        run_config.dataset_path, "catalog_returns", run_config.suffix
    )
    d1_dates = (
        date_dim.filter(pl.col("d_year").is_not_null() & (pl.col("d_year") == year))
        .select(["d_date_sk", "d_week_seq", "d_date"])
        .rename(
            {
                "d_date_sk": "d1_date_sk",
                "d_week_seq": "d1_week_seq",
                "d_date": "d1_date",
            }
        )
    )
    week_seqs_2002 = d1_dates.select("d1_week_seq").unique()
    d2_dates = (
        date_dim.select(["d_date_sk", "d_week_seq"])
        .join(week_seqs_2002, left_on="d_week_seq", right_on="d1_week_seq")
        .rename({"d_date_sk": "d2_date_sk", "d_week_seq": "d2_week_seq"})
    )
    d3_dates = date_dim.select(["d_date_sk", "d_date"]).rename(
        {"d_date_sk": "d3_date_sk", "d_date": "d3_date"}
    )
    filtered_cd = customer_demographics.filter(
        pl.col("cd_marital_status").is_not_null() & (pl.col("cd_marital_status") == ms)
    ).select(["cd_demo_sk"])
    filtered_hd = household_demographics.filter(
        pl.col("hd_buy_potential").is_not_null() & (pl.col("hd_buy_potential") == bp)
    ).select(["hd_demo_sk"])
    filtered_catalog_sales = (
        catalog_sales.join(d1_dates, left_on="cs_sold_date_sk", right_on="d1_date_sk")
        .join(filtered_cd, left_on="cs_bill_cdemo_sk", right_on="cd_demo_sk")
        .join(filtered_hd, left_on="cs_bill_hdemo_sk", right_on="hd_demo_sk")
        .select(
            [
                "cs_item_sk",
                "cs_order_number",
                "cs_quantity",
                "cs_promo_sk",
                "cs_ship_date_sk",
                "d1_week_seq",
                "d1_date",
            ]
        )
    )
    return QueryResult(
        frame=(
            filtered_catalog_sales.join(
                inventory, left_on="cs_item_sk", right_on="inv_item_sk"
            )
            .filter(pl.col("inv_quantity_on_hand") < pl.col("cs_quantity"))
            .join(d2_dates, left_on="inv_date_sk", right_on="d2_date_sk")
            .filter(pl.col("d1_week_seq") == pl.col("d2_week_seq"))
            .join(d3_dates, left_on="cs_ship_date_sk", right_on="d3_date_sk")
            .filter(
                pl.col("d3_date").cast(pl.Datetime("us"))
                > pl.col("d1_date").cast(pl.Datetime("us"))
                + pl.duration(days=5).cast(pl.Duration("us"))
            )
            .join(warehouse, left_on="inv_warehouse_sk", right_on="w_warehouse_sk")
            .join(item, left_on="cs_item_sk", right_on="i_item_sk")
            .with_columns(
                [
                    pl.when(pl.col("cs_promo_sk").is_null())
                    .then(pl.lit(1, dtype=COUNT_DTYPE))
                    .otherwise(pl.lit(0, dtype=COUNT_DTYPE))
                    .alias("no_promo_flag"),
                    pl.when(pl.col("cs_promo_sk").is_not_null())
                    .then(pl.lit(1, dtype=COUNT_DTYPE))
                    .otherwise(pl.lit(0, dtype=COUNT_DTYPE))
                    .alias("promo_flag"),
                ]
            )
            .join(promotion, left_on="cs_promo_sk", right_on="p_promo_sk", how="left")
            .join(
                catalog_returns,
                left_on=["cs_item_sk", "cs_order_number"],
                right_on=["cr_item_sk", "cr_order_number"],
                how="left",
            )
            .group_by(["i_item_desc", "w_warehouse_name", "d1_week_seq"])
            .agg(
                [
                    pl.col("no_promo_flag").sum().alias("no_promo"),
                    pl.col("promo_flag").sum().alias("promo"),
                    pl.len().alias("total_cnt"),
                ]
            )
            .select(
                [
                    "i_item_desc",
                    "w_warehouse_name",
                    pl.col("d1_week_seq").alias("d_week_seq"),
                    pl.col("no_promo"),
                    pl.col("promo"),
                    "total_cnt",
                ]
            )
            .sort(
                ["total_cnt", "i_item_desc", "w_warehouse_name", "d_week_seq"],
                descending=[True, False, False, False],
                nulls_last=True,
            )
            .limit(100)
        ),
        sort_by=[
            ("total_cnt", True),
            ("i_item_desc", False),
            ("w_warehouse_name", False),
            ("d_week_seq", False),
        ],
        limit=100,
    )
