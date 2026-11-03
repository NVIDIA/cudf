# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 71."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.streaming.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.streaming.benchmarks.utils import QueryResult, get_data

if TYPE_CHECKING:
    from cudf_polars.streaming.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 71."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=71,
        qualification=run_config.qualification,
    )

    year = params["year"]
    month = params["month"]
    manager = params["manager"]

    return f"""
    SELECT i_brand_id brand_id,
           i_brand brand,
           t_hour,
           t_minute,
           sum(ext_price) ext_price
    FROM item,
      (SELECT ws_ext_sales_price AS ext_price,
              ws_sold_date_sk AS sold_date_sk,
              ws_item_sk AS sold_item_sk,
              ws_sold_time_sk AS time_sk
       FROM web_sales,
            date_dim
       WHERE d_date_sk = ws_sold_date_sk
         AND d_moy={month}
         AND d_year={year}
       UNION ALL SELECT cs_ext_sales_price AS ext_price,
                        cs_sold_date_sk AS sold_date_sk,
                        cs_item_sk AS sold_item_sk,
                        cs_sold_time_sk AS time_sk
       FROM catalog_sales,
            date_dim
       WHERE d_date_sk = cs_sold_date_sk
         AND d_moy={month}
         AND d_year={year}
       UNION ALL SELECT ss_ext_sales_price AS ext_price,
                        ss_sold_date_sk AS sold_date_sk,
                        ss_item_sk AS sold_item_sk,
                        ss_sold_time_sk AS time_sk
       FROM store_sales,
            date_dim
       WHERE d_date_sk = ss_sold_date_sk
         AND d_moy={month}
         AND d_year={year} ) tmp,
         time_dim
    WHERE sold_item_sk = i_item_sk
      AND i_manager_id={manager}
      AND time_sk = t_time_sk
      AND (t_meal_time = 'breakfast'
           OR t_meal_time = 'dinner')
    GROUP BY i_brand,
             i_brand_id,
             t_hour,
             t_minute
    ORDER BY ext_price DESC NULLS FIRST,
             i_brand_id NULLS FIRST,
             t_hour NULLS FIRST;
    """


def polars_impl(run_config: RunConfig) -> QueryResult:
    """Query 71."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=71,
        qualification=run_config.qualification,
    )

    year = params["year"]
    month = params["month"]
    manager = params["manager"]

    web_sales = get_data(run_config.dataset_path, "web_sales", run_config.suffix)
    catalog_sales = get_data(
        run_config.dataset_path, "catalog_sales", run_config.suffix
    )
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)
    time_dim = get_data(run_config.dataset_path, "time_dim", run_config.suffix)

    target_dates = date_dim.filter(
        (pl.col("d_moy") == month) & (pl.col("d_year") == year)
    ).select("d_date_sk")

    web_component = web_sales.join(
        target_dates, left_on="ws_sold_date_sk", right_on="d_date_sk"
    ).select(
        pl.col("ws_ext_sales_price").alias("ext_price"),
        pl.col("ws_item_sk").alias("sold_item_sk"),
        pl.col("ws_sold_time_sk").alias("time_sk"),
    )

    catalog_component = catalog_sales.join(
        target_dates, left_on="cs_sold_date_sk", right_on="d_date_sk"
    ).select(
        pl.col("cs_ext_sales_price").alias("ext_price"),
        pl.col("cs_item_sk").alias("sold_item_sk"),
        pl.col("cs_sold_time_sk").alias("time_sk"),
    )

    store_component = store_sales.join(
        target_dates, left_on="ss_sold_date_sk", right_on="d_date_sk"
    ).select(
        pl.col("ss_ext_sales_price").alias("ext_price"),
        pl.col("ss_item_sk").alias("sold_item_sk"),
        pl.col("ss_sold_time_sk").alias("time_sk"),
    )

    combined_sales = pl.concat([web_component, catalog_component, store_component])

    filtered_items = item.filter(pl.col("i_manager_id") == manager).select(
        ["i_item_sk", "i_brand_id", "i_brand"]
    )

    filtered_time = time_dim.filter(
        pl.col("t_meal_time").is_in(["breakfast", "dinner"])
    ).select(["t_time_sk", "t_hour", "t_minute"])

    return QueryResult(
        frame=(
            combined_sales.join(
                filtered_items, left_on="sold_item_sk", right_on="i_item_sk"
            )
            .join(filtered_time, left_on="time_sk", right_on="t_time_sk")
            .group_by(["i_brand", "i_brand_id", "t_hour", "t_minute"])
            .agg(
                pl.col("ext_price").sum().alias("ext_price"),
                pl.col("ext_price").count().alias("_ext_price_count"),
            )
            .with_columns(
                pl.when(pl.col("_ext_price_count") > 0)
                .then(pl.col("ext_price"))
                .otherwise(None)
                .alias("ext_price")
            )
            .drop("_ext_price_count")
            .select(
                pl.col("i_brand_id").alias("brand_id"),
                pl.col("i_brand").alias("brand"),
                "t_hour",
                "t_minute",
                "ext_price",
            )
            .sort(
                ["ext_price", "brand_id", "t_hour"],
                descending=[True, False, False],
                nulls_last=False,
            )
        ),
        sort_by=[("ext_price", True), ("brand_id", False), ("t_hour", False)],
        limit=None,
        nulls_last=False,
    )
