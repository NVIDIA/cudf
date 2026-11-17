# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 75."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.streaming.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.streaming.benchmarks.utils import QueryResult, get_data

if TYPE_CHECKING:
    from cudf_polars.streaming.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 75."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=75,
        qualification=run_config.qualification,
    )
    category = params["category"]
    year = params["year"]

    return f"""
        WITH all_sales AS
        ( SELECT d_year ,
                i_brand_id ,
                i_class_id ,
                i_category_id ,
                i_manufact_id ,
                SUM(sales_cnt) AS sales_cnt ,
                SUM(sales_amt) AS sales_amt
        FROM
            (SELECT d_year ,
                    i_brand_id ,
                    i_class_id ,
                    i_category_id ,
                    i_manufact_id ,
                    cs_quantity - COALESCE(cr_return_quantity,0) AS sales_cnt ,
                    cs_ext_sales_price - COALESCE(cr_return_amount,0.0) AS sales_amt
            FROM catalog_sales
            JOIN item ON i_item_sk=cs_item_sk
            JOIN date_dim ON d_date_sk=cs_sold_date_sk
            LEFT JOIN catalog_returns ON (cs_order_number=cr_order_number
                                            AND cs_item_sk=cr_item_sk)
            WHERE i_category='{category}'
            UNION SELECT d_year ,
                        i_brand_id ,
                        i_class_id ,
                        i_category_id ,
                        i_manufact_id ,
                        ss_quantity - COALESCE(sr_return_quantity,0) AS sales_cnt ,
                        ss_ext_sales_price - COALESCE(sr_return_amt,0.0) AS sales_amt
            FROM store_sales
            JOIN item ON i_item_sk=ss_item_sk
            JOIN date_dim ON d_date_sk=ss_sold_date_sk
            LEFT JOIN store_returns ON (ss_ticket_number=sr_ticket_number
                                        AND ss_item_sk=sr_item_sk)
            WHERE i_category='{category}'
            UNION SELECT d_year ,
                        i_brand_id ,
                        i_class_id ,
                        i_category_id ,
                        i_manufact_id ,
                        ws_quantity - COALESCE(wr_return_quantity,0) AS sales_cnt ,
                        ws_ext_sales_price - COALESCE(wr_return_amt,0.0) AS sales_amt
            FROM web_sales
            JOIN item ON i_item_sk=ws_item_sk
            JOIN date_dim ON d_date_sk=ws_sold_date_sk
            LEFT JOIN web_returns ON (ws_order_number=wr_order_number
                                        AND ws_item_sk=wr_item_sk)
            WHERE i_category='{category}') sales_detail
        GROUP BY d_year,
                    i_brand_id,
                    i_class_id,
                    i_category_id,
                    i_manufact_id)
        SELECT prev_yr.d_year AS prev_year ,
            curr_yr.d_year AS year_ ,
            curr_yr.i_brand_id ,
            curr_yr.i_class_id ,
            curr_yr.i_category_id ,
            curr_yr.i_manufact_id ,
            prev_yr.sales_cnt AS prev_yr_cnt ,
            curr_yr.sales_cnt AS curr_yr_cnt ,
            curr_yr.sales_cnt-prev_yr.sales_cnt AS sales_cnt_diff ,
            curr_yr.sales_amt-prev_yr.sales_amt AS sales_amt_diff
        FROM all_sales curr_yr,
            all_sales prev_yr
        WHERE curr_yr.i_brand_id=prev_yr.i_brand_id
        AND curr_yr.i_class_id=prev_yr.i_class_id
        AND curr_yr.i_category_id=prev_yr.i_category_id
        AND curr_yr.i_manufact_id=prev_yr.i_manufact_id
        AND curr_yr.d_year={year}
        AND prev_yr.d_year={year}-1
        AND CAST(curr_yr.sales_cnt AS DECIMAL(17,2))/CAST(prev_yr.sales_cnt AS DECIMAL(17,2))<0.9
        ORDER BY sales_cnt_diff,
                sales_amt_diff
        LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> QueryResult:
    """Query 75."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=75,
        qualification=run_config.qualification,
    )

    category = params["category"]
    year = params["year"]

    catalog_sales = get_data(
        run_config.dataset_path, "catalog_sales", run_config.suffix
    )
    catalog_returns = get_data(
        run_config.dataset_path, "catalog_returns", run_config.suffix
    )
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    store_returns = get_data(
        run_config.dataset_path, "store_returns", run_config.suffix
    )
    web_sales = get_data(run_config.dataset_path, "web_sales", run_config.suffix)
    web_returns = get_data(run_config.dataset_path, "web_returns", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)

    filtered_items = item.filter(pl.col("i_category") == category).select(
        ["i_item_sk", "i_brand_id", "i_class_id", "i_category_id", "i_manufact_id"]
    )

    def build_year_sales(target_year: int) -> pl.LazyFrame:
        """Build all_sales aggregated for a single year."""
        date_yr = date_dim.filter(pl.col("d_year") == target_year)

        cat = (
            catalog_sales.join(
                filtered_items, left_on="cs_item_sk", right_on="i_item_sk"
            )
            .join(date_yr, left_on="cs_sold_date_sk", right_on="d_date_sk")
            .join(
                catalog_returns,
                left_on=["cs_order_number", "cs_item_sk"],
                right_on=["cr_order_number", "cr_item_sk"],
                how="left",
            )
            .with_columns(
                [
                    (
                        pl.col("cs_quantity")
                        - pl.col("cr_return_quantity").fill_null(0)
                    ).alias("sales_cnt"),
                    (
                        pl.col("cs_ext_sales_price")
                        - pl.col("cr_return_amount").fill_null(0.0)
                    ).alias("sales_amt"),
                ]
            )
            .select(
                [
                    "i_brand_id",
                    "i_class_id",
                    "i_category_id",
                    "i_manufact_id",
                    "sales_cnt",
                    "sales_amt",
                ]
            )
        )

        sto = (
            store_sales.join(filtered_items, left_on="ss_item_sk", right_on="i_item_sk")
            .join(date_yr, left_on="ss_sold_date_sk", right_on="d_date_sk")
            .join(
                store_returns,
                left_on=["ss_ticket_number", "ss_item_sk"],
                right_on=["sr_ticket_number", "sr_item_sk"],
                how="left",
            )
            .with_columns(
                [
                    (
                        pl.col("ss_quantity")
                        - pl.col("sr_return_quantity").fill_null(0)
                    ).alias("sales_cnt"),
                    (
                        pl.col("ss_ext_sales_price")
                        - pl.col("sr_return_amt").fill_null(0.0)
                    ).alias("sales_amt"),
                ]
            )
            .select(
                [
                    "i_brand_id",
                    "i_class_id",
                    "i_category_id",
                    "i_manufact_id",
                    "sales_cnt",
                    "sales_amt",
                ]
            )
        )

        web = (
            web_sales.join(filtered_items, left_on="ws_item_sk", right_on="i_item_sk")
            .join(date_yr, left_on="ws_sold_date_sk", right_on="d_date_sk")
            .join(
                web_returns,
                left_on=["ws_order_number", "ws_item_sk"],
                right_on=["wr_order_number", "wr_item_sk"],
                how="left",
            )
            .with_columns(
                [
                    (
                        pl.col("ws_quantity")
                        - pl.col("wr_return_quantity").fill_null(0)
                    ).alias("sales_cnt"),
                    (
                        pl.col("ws_ext_sales_price")
                        - pl.col("wr_return_amt").fill_null(0.0)
                    ).alias("sales_amt"),
                ]
            )
            .select(
                [
                    "i_brand_id",
                    "i_class_id",
                    "i_category_id",
                    "i_manufact_id",
                    "sales_cnt",
                    "sales_amt",
                ]
            )
        )

        return (
            pl.concat([cat, sto, web])
            .unique()
            .group_by(["i_brand_id", "i_class_id", "i_category_id", "i_manufact_id"])
            .agg(
                [
                    pl.col("sales_cnt").sum().alias("sales_cnt"),
                    pl.col("sales_amt").sum().alias("sales_amt"),
                ]
            )
        )

    curr_yr = (
        build_year_sales(year)
        .with_columns(pl.lit(year, dtype=pl.Int64).alias("curr_d_year"))
        .select(
            [
                pl.col("curr_d_year"),
                pl.col("i_brand_id").alias("curr_brand_id"),
                pl.col("i_class_id").alias("curr_class_id"),
                pl.col("i_category_id").alias("curr_category_id"),
                pl.col("i_manufact_id").alias("curr_manufact_id"),
                pl.col("sales_cnt").alias("curr_yr_cnt"),
                pl.col("sales_amt").alias("curr_yr_amt"),
            ]
        )
    )

    prev_yr = (
        build_year_sales(year - 1)
        .with_columns(pl.lit(year - 1, dtype=pl.Int64).alias("prev_d_year"))
        .select(
            [
                pl.col("prev_d_year"),
                pl.col("i_brand_id").alias("prev_brand_id"),
                pl.col("i_class_id").alias("prev_class_id"),
                pl.col("i_category_id").alias("prev_category_id"),
                pl.col("i_manufact_id").alias("prev_manufact_id"),
                pl.col("sales_cnt").alias("prev_yr_cnt"),
                pl.col("sales_amt").alias("prev_yr_amt"),
            ]
        )
    )

    sort_by = {"sales_cnt_diff": False, "sales_amt_diff": False}
    limit = 100
    return QueryResult(
        frame=(
            curr_yr.join(
                prev_yr,
                left_on=[
                    "curr_brand_id",
                    "curr_class_id",
                    "curr_category_id",
                    "curr_manufact_id",
                ],
                right_on=[
                    "prev_brand_id",
                    "prev_class_id",
                    "prev_category_id",
                    "prev_manufact_id",
                ],
            )
            .filter(
                (pl.col("prev_yr_cnt") > 0)
                & (
                    pl.col("curr_yr_cnt").cast(pl.Float64)
                    / pl.col("prev_yr_cnt").cast(pl.Float64)
                    < 0.9
                )
            )
            .with_columns(
                [
                    (pl.col("curr_yr_cnt") - pl.col("prev_yr_cnt")).alias(
                        "sales_cnt_diff"
                    ),
                    (pl.col("curr_yr_amt") - pl.col("prev_yr_amt")).alias(
                        "sales_amt_diff"
                    ),
                ]
            )
            .select(
                [
                    pl.col("prev_d_year").alias("prev_year"),
                    pl.col("curr_d_year").alias("year_"),
                    pl.col("curr_brand_id").alias("i_brand_id"),
                    pl.col("curr_class_id").alias("i_class_id"),
                    pl.col("curr_category_id").alias("i_category_id"),
                    pl.col("curr_manufact_id").alias("i_manufact_id"),
                    pl.col("prev_yr_cnt"),
                    pl.col("curr_yr_cnt"),
                    pl.col("sales_cnt_diff"),
                    pl.col("sales_amt_diff"),
                ]
            )
            .sort(sort_by.keys(), nulls_last=True)
            .limit(limit)
        ),
        sort_by=list(sort_by.items()),
        limit=limit,
    )
