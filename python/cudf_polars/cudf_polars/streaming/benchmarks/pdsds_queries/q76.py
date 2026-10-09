# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Query 76."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.streaming.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.streaming.benchmarks.pdsds_queries import sql_sum
from cudf_polars.streaming.benchmarks.utils import QueryResult, get_data

if TYPE_CHECKING:
    from cudf_polars.streaming.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 76."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=76,
        qualification=run_config.qualification,
    )
    nullcol_ss = params["nullcol_ss"]
    nullcol_ws = params["nullcol_ws"]
    nullcol_cs = params["nullcol_cs"]

    return f"""
    SELECT channel,
                   col_name,
                   d_year,
                   d_qoy,
                   i_category,
                   Count(*)             sales_cnt,
                   Sum(ext_sales_price) sales_amt
    FROM   (SELECT 'store'            AS channel,
                   '{nullcol_ss}'      col_name,
                   d_year,
                   d_qoy,
                   i_category,
                   ss_ext_sales_price ext_sales_price
            FROM   store_sales,
                   item,
                   date_dim
            WHERE  {nullcol_ss} IS NULL
                   AND ss_sold_date_sk = d_date_sk
                   AND ss_item_sk = i_item_sk
            UNION ALL
            SELECT 'web'              AS channel,
                   '{nullcol_ws}' col_name,
                   d_year,
                   d_qoy,
                   i_category,
                   ws_ext_sales_price ext_sales_price
            FROM   web_sales,
                   item,
                   date_dim
            WHERE  {nullcol_ws} IS NULL
                   AND ws_sold_date_sk = d_date_sk
                   AND ws_item_sk = i_item_sk
            UNION ALL
            SELECT 'catalog'          AS channel,
                   '{nullcol_cs}'  col_name,
                   d_year,
                   d_qoy,
                   i_category,
                   cs_ext_sales_price ext_sales_price
            FROM   catalog_sales,
                   item,
                   date_dim
            WHERE  {nullcol_cs} IS NULL
                   AND cs_sold_date_sk = d_date_sk
                   AND cs_item_sk = i_item_sk) foo
    GROUP  BY channel,
              col_name,
              d_year,
              d_qoy,
              i_category
    ORDER  BY channel,
              col_name,
              d_year,
              d_qoy,
              i_category
    LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> QueryResult:
    """Query 76."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=76,
        qualification=run_config.qualification,
    )

    nullcol_ss = params["nullcol_ss"]
    nullcol_ws = params["nullcol_ws"]
    nullcol_cs = params["nullcol_cs"]

    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    web_sales = get_data(run_config.dataset_path, "web_sales", run_config.suffix)
    catalog_sales = get_data(
        run_config.dataset_path, "catalog_sales", run_config.suffix
    )
    item = get_data(run_config.dataset_path, "item", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)

    # Project lookup tables to only the columns needed in each component.
    date_cols = date_dim.select(["d_date_sk", "d_year", "d_qoy"])
    item_cols = item.select(["i_item_sk", "i_category"])

    store_component = (
        store_sales.filter(pl.col(nullcol_ss).is_null())
        .select(["ss_sold_date_sk", "ss_item_sk", "ss_ext_sales_price"])
        .join(date_cols, left_on="ss_sold_date_sk", right_on="d_date_sk")
        .join(item_cols, left_on="ss_item_sk", right_on="i_item_sk")
        .select(
            [
                pl.lit("store").alias("channel"),
                pl.lit(nullcol_ss).alias("col_name"),
                "d_year",
                "d_qoy",
                "i_category",
                pl.col("ss_ext_sales_price").alias("ext_sales_price"),
            ]
        )
    )
    web_component = (
        web_sales.filter(pl.col(nullcol_ws).is_null())
        .select(["ws_sold_date_sk", "ws_item_sk", "ws_ext_sales_price"])
        .join(date_cols, left_on="ws_sold_date_sk", right_on="d_date_sk")
        .join(item_cols, left_on="ws_item_sk", right_on="i_item_sk")
        .select(
            [
                pl.lit("web").alias("channel"),
                pl.lit(nullcol_ws).alias("col_name"),
                "d_year",
                "d_qoy",
                "i_category",
                pl.col("ws_ext_sales_price").alias("ext_sales_price"),
            ]
        )
    )
    catalog_component = (
        catalog_sales.filter(pl.col(nullcol_cs).is_null())
        .select(["cs_sold_date_sk", "cs_item_sk", "cs_ext_sales_price"])
        .join(date_cols, left_on="cs_sold_date_sk", right_on="d_date_sk")
        .join(item_cols, left_on="cs_item_sk", right_on="i_item_sk")
        .select(
            [
                pl.lit("catalog").alias("channel"),
                pl.lit(nullcol_cs).alias("col_name"),
                "d_year",
                "d_qoy",
                "i_category",
                pl.col("cs_ext_sales_price").alias("ext_sales_price"),
            ]
        )
    )
    sort_by = {
        "channel": False,
        "col_name": False,
        "d_year": False,
        "d_qoy": False,
        "i_category": False,
    }
    limit = 100
    return QueryResult(
        frame=(
            pl.concat([store_component, web_component, catalog_component])
            .group_by(["channel", "col_name", "d_year", "d_qoy", "i_category"])
            .agg(
                [
                    pl.len().cast(pl.Int64).alias("sales_cnt"),
                    sql_sum("ext_sales_price").alias("sales_amt"),
                ]
            )
            .select(
                [
                    "channel",
                    "col_name",
                    "d_year",
                    "d_qoy",
                    "i_category",
                    "sales_cnt",
                    "sales_amt",
                ]
            )
            .sort(sort_by.keys(), nulls_last=True)
            .limit(limit)
        ),
        sort_by=list(sort_by.items()),
        limit=limit,
    )
