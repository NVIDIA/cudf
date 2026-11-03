# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 77."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.streaming.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.streaming.benchmarks.utils import QueryResult, get_data

if TYPE_CHECKING:
    from cudf_polars.streaming.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 77."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=77,
        qualification=run_config.qualification,
    )
    sdate = params["sdate"]

    return f"""
    -- start query 77 in stream 0 using template query77.tpl
    WITH ss AS
    (
             SELECT   s_store_sk,
                      Sum(ss_ext_sales_price) AS sales,
                      Sum(ss_net_profit)      AS profit
             FROM     store_sales,
                      date_dim,
                      store
             WHERE    ss_sold_date_sk = d_date_sk
             AND      d_date BETWEEN Cast('{sdate}' AS DATE) AND      (
                               Cast('{sdate}' AS DATE) + INTERVAL '30' day)
             AND      ss_store_sk = s_store_sk
             GROUP BY s_store_sk) , sr AS
    (
             SELECT   s_store_sk,
                      sum(sr_return_amt) AS returns1,
                      sum(sr_net_loss)   AS profit_loss
             FROM     store_returns,
                      date_dim,
                      store
             WHERE    sr_returned_date_sk = d_date_sk
             AND      d_date BETWEEN cast('{sdate}' AS date) AND      (
                               cast('{sdate}' AS date) + INTERVAL '30' day)
             AND      sr_store_sk = s_store_sk
             GROUP BY s_store_sk), cs AS
    (
             SELECT   cs_call_center_sk,
                      sum(cs_ext_sales_price) AS sales,
                      sum(cs_net_profit)      AS profit
             FROM     catalog_sales,
                      date_dim
             WHERE    cs_sold_date_sk = d_date_sk
             AND      d_date BETWEEN cast('{sdate}' AS date) AND      (
                               cast('{sdate}' AS date) + INTERVAL '30' day)
             GROUP BY cs_call_center_sk ), cr AS
    (
             SELECT   cr_call_center_sk,
                      sum(cr_return_amount) AS returns1,
                      sum(cr_net_loss)      AS profit_loss
             FROM     catalog_returns,
                      date_dim
             WHERE    cr_returned_date_sk = d_date_sk
             AND      d_date BETWEEN cast('{sdate}' AS date) AND      (
                               cast('{sdate}' AS date) + INTERVAL '30' day)
             GROUP BY cr_call_center_sk ), ws AS
    (
             SELECT   wp_web_page_sk,
                      sum(ws_ext_sales_price) AS sales,
                      sum(ws_net_profit)      AS profit
             FROM     web_sales,
                      date_dim,
                      web_page
             WHERE    ws_sold_date_sk = d_date_sk
             AND      d_date BETWEEN cast('{sdate}' AS date) AND      (
                               cast('{sdate}' AS date) + INTERVAL '30' day)
             AND      ws_web_page_sk = wp_web_page_sk
             GROUP BY wp_web_page_sk), wr AS
    (
             SELECT   wp_web_page_sk,
                      sum(wr_return_amt) AS returns1,
                      sum(wr_net_loss)   AS profit_loss
             FROM     web_returns,
                      date_dim,
                      web_page
             WHERE    wr_returned_date_sk = d_date_sk
             AND      d_date BETWEEN cast('{sdate}' AS date) AND      (
                               cast('{sdate}' AS date) + INTERVAL '30' day)
             AND      wr_web_page_sk = wp_web_page_sk
             GROUP BY wp_web_page_sk)
    SELECT
             channel ,
             id ,
             sum(sales)   AS sales ,
             sum(returns1) AS returns1 ,
             sum(profit)  AS profit
    FROM     (
                       SELECT    'store channel' AS channel ,
                                 ss.s_store_sk   AS id ,
                                 sales ,
                                 COALESCE(returns1, 0)               AS returns1 ,
                                 (profit - COALESCE(profit_loss,0)) AS profit
                       FROM      ss
                       LEFT JOIN sr
                       ON        ss.s_store_sk = sr.s_store_sk
                       UNION ALL
                       SELECT 'catalog channel' AS channel ,
                              cs_call_center_sk AS id ,
                              sales ,
                              returns1 ,
                              (profit - profit_loss) AS profit
                       FROM   cs ,
                              cr
                       UNION ALL
                       SELECT    'web channel'     AS channel ,
                                 ws.wp_web_page_sk AS id ,
                                 sales ,
                                 COALESCE(returns1, 0)                  returns1 ,
                                 (profit - COALESCE(profit_loss,0)) AS profit
                       FROM      ws
                       LEFT JOIN wr
                       ON        ws.wp_web_page_sk = wr.wp_web_page_sk ) x
    GROUP BY rollup (channel, id)
    ORDER BY channel,
             id
    LIMIT 100;
    """


def _sum_sales_profit(
    lf: pl.LazyFrame,
    date_key: str,
    dates: pl.LazyFrame,
    group_key: str,
    sales_col: str,
    profit_col: str,
) -> pl.LazyFrame:
    return (
        lf.join(dates, left_on=date_key, right_on="d_date_sk")
        .group_by(group_key)
        .agg(
            [
                pl.col(sales_col).count().alias("sales_count"),
                pl.col(sales_col).sum().alias("sales_sum"),
                pl.col(profit_col).count().alias("profit_count"),
                pl.col(profit_col).sum().alias("profit_sum"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("sales_count") == 0)
                .then(None)
                .otherwise(pl.col("sales_sum"))
                .alias("sales"),
                pl.when(pl.col("profit_count") == 0)
                .then(None)
                .otherwise(pl.col("profit_sum"))
                .alias("profit"),
            ]
        )
    )


def _sum_returns_loss(
    lf: pl.LazyFrame,
    date_key: str,
    dates: pl.LazyFrame,
    group_key: str,
    returns_col: str,
    loss_col: str,
) -> pl.LazyFrame:
    return (
        lf.join(dates, left_on=date_key, right_on="d_date_sk")
        .group_by(group_key)
        .agg(
            [
                pl.col(returns_col).count().alias("returns1_count"),
                pl.col(returns_col).sum().alias("returns1_sum"),
                pl.col(loss_col).count().alias("profit_loss_count"),
                pl.col(loss_col).sum().alias("profit_loss_sum"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("returns1_count") == 0)
                .then(None)
                .otherwise(pl.col("returns1_sum"))
                .alias("returns1"),
                pl.when(pl.col("profit_loss_count") == 0)
                .then(None)
                .otherwise(pl.col("profit_loss_sum"))
                .alias("profit_loss"),
            ]
        )
    )


def polars_impl(run_config: RunConfig) -> QueryResult:
    """Query 77."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=77,
        qualification=run_config.qualification,
    )

    sdate = params["sdate"]
    year, month, day = map(int, sdate.split("-"))

    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    store_returns = get_data(
        run_config.dataset_path, "store_returns", run_config.suffix
    )
    catalog_sales = get_data(
        run_config.dataset_path, "catalog_sales", run_config.suffix
    )
    catalog_returns = get_data(
        run_config.dataset_path, "catalog_returns", run_config.suffix
    )
    web_sales = get_data(run_config.dataset_path, "web_sales", run_config.suffix)
    web_returns = get_data(run_config.dataset_path, "web_returns", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    store = get_data(run_config.dataset_path, "store", run_config.suffix)
    web_page = get_data(run_config.dataset_path, "web_page", run_config.suffix)

    start_date = (pl.date(year, month, day)).cast(pl.Datetime("us"))
    end_date = (start_date + pl.duration(days=30)).cast(pl.Datetime("us"))
    filtered_dates = date_dim.filter(
        (pl.col("d_date") >= start_date) & (pl.col("d_date") <= end_date)
    ).select("d_date_sk")

    ss = _sum_sales_profit(
        store_sales.join(store, left_on="ss_store_sk", right_on="s_store_sk"),
        "ss_sold_date_sk",
        filtered_dates,
        "ss_store_sk",
        "ss_ext_sales_price",
        "ss_net_profit",
    ).select(["ss_store_sk", "sales", "profit"])

    sr = _sum_returns_loss(
        store_returns.join(store, left_on="sr_store_sk", right_on="s_store_sk"),
        "sr_returned_date_sk",
        filtered_dates,
        "sr_store_sk",
        "sr_return_amt",
        "sr_net_loss",
    ).select(["sr_store_sk", "returns1", "profit_loss"])

    cs = _sum_sales_profit(
        catalog_sales,
        "cs_sold_date_sk",
        filtered_dates,
        "cs_call_center_sk",
        "cs_ext_sales_price",
        "cs_net_profit",
    ).select(["cs_call_center_sk", "sales", "profit"])

    cr = _sum_returns_loss(
        catalog_returns,
        "cr_returned_date_sk",
        filtered_dates,
        "cr_call_center_sk",
        "cr_return_amount",
        "cr_net_loss",
    ).select(["cr_call_center_sk", "returns1", "profit_loss"])

    ws = _sum_sales_profit(
        web_sales.join(web_page, left_on="ws_web_page_sk", right_on="wp_web_page_sk"),
        "ws_sold_date_sk",
        filtered_dates,
        "ws_web_page_sk",
        "ws_ext_sales_price",
        "ws_net_profit",
    ).select(["ws_web_page_sk", "sales", "profit"])

    wr = _sum_returns_loss(
        web_returns.join(web_page, left_on="wr_web_page_sk", right_on="wp_web_page_sk"),
        "wr_returned_date_sk",
        filtered_dates,
        "wr_web_page_sk",
        "wr_return_amt",
        "wr_net_loss",
    ).select(["wr_web_page_sk", "returns1", "profit_loss"])

    store_channel = ss.join(
        sr, left_on="ss_store_sk", right_on="sr_store_sk", how="left"
    ).select(
        [
            pl.lit("store channel").alias("channel"),
            pl.col("ss_store_sk").alias("id"),
            "sales",
            pl.col("returns1").fill_null(0).alias("returns1"),
            (pl.col("profit") - pl.col("profit_loss").fill_null(0)).alias("profit"),
        ]
    )

    catalog_channel = cs.join(cr, how="cross").select(
        [
            pl.lit("catalog channel").alias("channel"),
            pl.col("cs_call_center_sk").alias("id"),
            "sales",
            "returns1",
            (pl.col("profit") - pl.col("profit_loss")).alias("profit"),
        ]
    )

    web_channel = ws.join(
        wr, left_on="ws_web_page_sk", right_on="wr_web_page_sk", how="left"
    ).select(
        [
            pl.lit("web channel").alias("channel"),
            pl.col("ws_web_page_sk").alias("id"),
            "sales",
            pl.col("returns1").fill_null(0).alias("returns1"),
            (pl.col("profit") - pl.col("profit_loss").fill_null(0)).alias("profit"),
        ]
    )

    combined_channels = pl.concat(
        [store_channel, catalog_channel, web_channel]
    ).with_columns(pl.col("id").cast(pl.Int64))

    level1 = (
        combined_channels.group_by(["channel", "id"])
        .agg(
            [
                pl.col("sales").sum().alias("sales"),
                pl.col("returns1").sum().alias("returns1"),
                pl.col("profit").sum().alias("profit"),
            ]
        )
        .select(["channel", "id", "sales", "returns1", "profit"])
    )

    level2 = (
        combined_channels.group_by("channel")
        .agg(
            [
                pl.col("sales").sum().alias("sales"),
                pl.col("returns1").sum().alias("returns1"),
                pl.col("profit").sum().alias("profit"),
            ]
        )
        .with_columns(pl.lit(None, dtype=pl.Int64).alias("id"))
        .select(["channel", "id", "sales", "returns1", "profit"])
    )

    level3 = (
        combined_channels.select(
            [
                pl.col("sales").sum().alias("sales"),
                pl.col("returns1").sum().alias("returns1"),
                pl.col("profit").sum().alias("profit"),
            ]
        )
        .with_columns(
            [
                pl.lit(None, dtype=pl.Utf8).alias("channel"),
                pl.lit(None, dtype=pl.Int64).alias("id"),
            ]
        )
        .select(["channel", "id", "sales", "returns1", "profit"])
    )

    # TODO: ROLLUP produces two (channel="catalog channel", id=null) rows whose
    # relative order is non-deterministic in DuckDB and differs between float and
    # decimal data. See https://github.com/duckdb/duckdb/issues/21164
    sort_by = {"channel": False, "id": False}
    limit = 100
    return QueryResult(
        frame=(
            pl.concat([level1, level2, level3], how="diagonal")
            .sort(sort_by.keys(), nulls_last=True)
            .limit(limit)
        ),
        sort_by=list(sort_by.items()),
        limit=limit,
    )
