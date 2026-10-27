# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Query 78."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.streaming.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.streaming.benchmarks.pdsds_queries import sql_sum
from cudf_polars.streaming.benchmarks.utils import QueryResult, get_data

if TYPE_CHECKING:
    from cudf_polars.streaming.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 78."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=78,
        qualification=run_config.qualification,
    )
    year = params["year"]

    return f"""
        WITH ws AS
        (SELECT d_year AS ws_sold_year,
                ws_item_sk,
                ws_bill_customer_sk ws_customer_sk,
                sum(ws_quantity) ws_qty,
                sum(ws_wholesale_cost) ws_wc,
                sum(ws_sales_price) ws_sp
        FROM web_sales
        LEFT JOIN web_returns ON wr_order_number=ws_order_number
        AND ws_item_sk=wr_item_sk
        JOIN date_dim ON ws_sold_date_sk = d_date_sk
        WHERE wr_order_number IS NULL
        GROUP BY d_year,
                    ws_item_sk,
                    ws_bill_customer_sk ),
            cs AS
        (SELECT d_year AS cs_sold_year,
                cs_item_sk,
                cs_bill_customer_sk cs_customer_sk,
                sum(cs_quantity) cs_qty,
                sum(cs_wholesale_cost) cs_wc,
                sum(cs_sales_price) cs_sp
        FROM catalog_sales
        LEFT JOIN catalog_returns ON cr_order_number=cs_order_number
        AND cs_item_sk=cr_item_sk
        JOIN date_dim ON cs_sold_date_sk = d_date_sk
        WHERE cr_order_number IS NULL
        GROUP BY d_year,
                    cs_item_sk,
                    cs_bill_customer_sk ),
            ss AS
        (SELECT d_year AS ss_sold_year,
                ss_item_sk,
                ss_customer_sk,
                sum(ss_quantity) ss_qty,
                sum(ss_wholesale_cost) ss_wc,
                sum(ss_sales_price) ss_sp
        FROM store_sales
        LEFT JOIN store_returns ON sr_ticket_number=ss_ticket_number
        AND ss_item_sk=sr_item_sk
        JOIN date_dim ON ss_sold_date_sk = d_date_sk
        WHERE sr_ticket_number IS NULL
        GROUP BY d_year,
                    ss_item_sk,
                    ss_customer_sk )
        SELECT ss_sold_year,
            ss_item_sk,
            ss_customer_sk,
            round((ss_qty*1.00)/(coalesce(ws_qty,0)+coalesce(cs_qty,0)),2) ratio,
            ss_qty store_qty,
            ss_wc store_wholesale_cost,
            ss_sp store_sales_price,
            coalesce(ws_qty,0)+coalesce(cs_qty,0) other_chan_qty,
            coalesce(ws_wc,0)+coalesce(cs_wc,0) other_chan_wholesale_cost,
            coalesce(ws_sp,0)+coalesce(cs_sp,0) other_chan_sales_price
        FROM ss
        LEFT JOIN ws ON (ws_sold_year=ss_sold_year
                        AND ws_item_sk=ss_item_sk
                        AND ws_customer_sk=ss_customer_sk)
        LEFT JOIN cs ON (cs_sold_year=ss_sold_year
                        AND cs_item_sk=ss_item_sk
                        AND cs_customer_sk=ss_customer_sk)
        WHERE (coalesce(ws_qty,0)>0
            OR coalesce(cs_qty, 0)>0)
        AND ss_sold_year={year}
        ORDER BY ss_sold_year,
                ss_item_sk,
                ss_customer_sk,
                ss_qty DESC,
                ss_wc DESC,
                ss_sp DESC,
                other_chan_qty,
                other_chan_wholesale_cost,
                other_chan_sales_price,
                ratio
        LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> QueryResult:
    """Query 78."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=78,
        qualification=run_config.qualification,
    )

    year = params["year"]

    web_sales = get_data(run_config.dataset_path, "web_sales", run_config.suffix)
    web_returns = get_data(run_config.dataset_path, "web_returns", run_config.suffix)
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
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)

    # Filter date_dim to just the target year to reduce data volume.
    # The final query filters ss_sold_year == year and joins on sold_year,
    # so all three CTEs only need data for that single year.
    date_year = date_dim.filter(pl.col("d_year") == year).select("d_date_sk")

    ws = (
        web_sales.join(date_year, left_on="ws_sold_date_sk", right_on="d_date_sk")
        .join(
            web_returns,
            left_on=["ws_order_number", "ws_item_sk"],
            right_on=["wr_order_number", "wr_item_sk"],
            how="anti",
        )
        .group_by(["ws_item_sk", "ws_bill_customer_sk"])
        .agg(
            [
                sql_sum("ws_quantity").alias("ws_qty"),
                sql_sum("ws_wholesale_cost").alias("ws_wc"),
                sql_sum("ws_sales_price").alias("ws_sp"),
            ]
        )
        .rename({"ws_bill_customer_sk": "ws_customer_sk"})
    )
    cs = (
        catalog_sales.join(date_year, left_on="cs_sold_date_sk", right_on="d_date_sk")
        .join(
            catalog_returns,
            left_on=["cs_order_number", "cs_item_sk"],
            right_on=["cr_order_number", "cr_item_sk"],
            how="anti",
        )
        .group_by(["cs_item_sk", "cs_bill_customer_sk"])
        .agg(
            [
                sql_sum("cs_quantity").alias("cs_qty"),
                sql_sum("cs_wholesale_cost").alias("cs_wc"),
                sql_sum("cs_sales_price").alias("cs_sp"),
            ]
        )
        .rename({"cs_bill_customer_sk": "cs_customer_sk"})
    )
    ss = (
        store_sales.join(date_year, left_on="ss_sold_date_sk", right_on="d_date_sk")
        .join(
            store_returns,
            left_on=["ss_ticket_number", "ss_item_sk"],
            right_on=["sr_ticket_number", "sr_item_sk"],
            how="anti",
        )
        .group_by(["ss_item_sk", "ss_customer_sk"])
        .agg(
            [
                sql_sum("ss_quantity").alias("ss_qty"),
                sql_sum("ss_wholesale_cost").alias("ss_wc"),
                sql_sum("ss_sales_price").alias("ss_sp"),
            ]
        )
    )
    result = (
        ss.join(
            ws,
            left_on=["ss_item_sk", "ss_customer_sk"],
            right_on=["ws_item_sk", "ws_customer_sk"],
            how="left",
        )
        .join(
            cs,
            left_on=["ss_item_sk", "ss_customer_sk"],
            right_on=["cs_item_sk", "cs_customer_sk"],
            how="left",
        )
        .filter(
            (pl.col("ws_qty").fill_null(0) > 0) | (pl.col("cs_qty").fill_null(0) > 0)
        )
        .select(
            [
                pl.lit(year, dtype=pl.Int64).alias("ss_sold_year"),
                pl.col("ss_item_sk"),
                pl.col("ss_customer_sk"),
                (
                    pl.col("ss_qty")
                    / (pl.col("ws_qty").fill_null(0) + pl.col("cs_qty").fill_null(0))
                )
                .round(2)
                .alias("ratio"),
                pl.col("ss_qty").alias("store_qty"),
                pl.col("ss_wc").alias("store_wholesale_cost"),
                pl.col("ss_sp").alias("store_sales_price"),
                (pl.col("ws_qty").fill_null(0) + pl.col("cs_qty").fill_null(0)).alias(
                    "other_chan_qty"
                ),
                (pl.col("ws_wc").fill_null(0) + pl.col("cs_wc").fill_null(0)).alias(
                    "other_chan_wholesale_cost"
                ),
                (pl.col("ws_sp").fill_null(0) + pl.col("cs_sp").fill_null(0)).alias(
                    "other_chan_sales_price"
                ),
            ]
        )
    )
    sort_by = {
        "ss_sold_year": False,
        "ss_item_sk": False,
        "ss_customer_sk": False,
        "store_qty": True,
        "store_wholesale_cost": True,
        "store_sales_price": True,
        "other_chan_qty": False,
        "other_chan_wholesale_cost": False,
        "other_chan_sales_price": False,
        "ratio": False,
    }
    limit = 100
    return QueryResult(
        frame=(
            result.sort(
                list(sort_by.keys()),
                descending=list(sort_by.values()),
                nulls_last=True,
            ).limit(limit)
        ),
        sort_by=list(sort_by.items()),
        limit=limit,
    )
