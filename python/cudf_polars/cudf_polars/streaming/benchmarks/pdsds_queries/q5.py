# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Query 5."""

from __future__ import annotations

from datetime import date, timedelta
from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.streaming.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.streaming.benchmarks.pdsds_queries import sql_sum
from cudf_polars.streaming.benchmarks.utils import QueryResult, get_data

if TYPE_CHECKING:
    from cudf_polars.streaming.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 5."""
    params = load_parameters(
        int(run_config.scale_factor), query_id=5, qualification=run_config.qualification
    )

    sales_date = params["sales_date"]

    return f"""
    WITH ssr AS
    (
             SELECT   s_store_id,
                      Sum(sales_price) AS sales,
                      Sum(profit)      AS profit,
                      Sum(return_amt)  AS returns1,
                      Sum(net_loss)    AS profit_loss
             FROM     (
                             SELECT ss_store_sk             AS store_sk,
                                    ss_sold_date_sk         AS date_sk,
                                    ss_ext_sales_price      AS sales_price,
                                    ss_net_profit           AS profit,
                                    Cast(0 AS DECIMAL(7,2)) AS return_amt,
                                    Cast(0 AS DECIMAL(7,2)) AS net_loss
                             FROM   store_sales
                             UNION ALL
                             SELECT sr_store_sk             AS store_sk,
                                    sr_returned_date_sk     AS date_sk,
                                    Cast(0 AS DECIMAL(7,2)) AS sales_price,
                                    Cast(0 AS DECIMAL(7,2)) AS profit,
                                    sr_return_amt           AS return_amt,
                                    sr_net_loss             AS net_loss
                             FROM   store_returns ) salesreturns,
                      date_dim,
                      store
             WHERE    date_sk = d_date_sk
             AND      d_date BETWEEN Cast('{sales_date}' AS DATE) AND      (
                               Cast('{sales_date}' AS DATE) + INTERVAL '14' day)
             AND      store_sk = s_store_sk
             GROUP BY s_store_id) , csr AS
    (
             SELECT   cp_catalog_page_id,
                      sum(sales_price) AS sales,
                      sum(profit)      AS profit,
                      sum(return_amt)  AS returns1,
                      sum(net_loss)    AS profit_loss
             FROM     (
                             SELECT cs_catalog_page_sk      AS page_sk,
                                    cs_sold_date_sk         AS date_sk,
                                    cs_ext_sales_price      AS sales_price,
                                    cs_net_profit           AS profit,
                                    cast(0 AS decimal(7,2)) AS return_amt,
                                    cast(0 AS decimal(7,2)) AS net_loss
                             FROM   catalog_sales
                             UNION ALL
                             SELECT cr_catalog_page_sk      AS page_sk,
                                    cr_returned_date_sk     AS date_sk,
                                    cast(0 AS decimal(7,2)) AS sales_price,
                                    cast(0 AS decimal(7,2)) AS profit,
                                    cr_return_amount        AS return_amt,
                                    cr_net_loss             AS net_loss
                             FROM   catalog_returns ) salesreturns,
                      date_dim,
                      catalog_page
             WHERE    date_sk = d_date_sk
             AND      d_date BETWEEN cast('{sales_date}' AS date) AND      (
                               cast('{sales_date}' AS date) + INTERVAL '14' day)
             AND      page_sk = cp_catalog_page_sk
             GROUP BY cp_catalog_page_id) , wsr AS
    (
             SELECT   web_site_id,
                      sum(sales_price) AS sales,
                      sum(profit)      AS profit,
                      sum(return_amt)  AS returns1,
                      sum(net_loss)    AS profit_loss
             FROM     (
                             SELECT ws_web_site_sk          AS wsr_web_site_sk,
                                    ws_sold_date_sk         AS date_sk,
                                    ws_ext_sales_price      AS sales_price,
                                    ws_net_profit           AS profit,
                                    cast(0 AS decimal(7,2)) AS return_amt,
                                    cast(0 AS decimal(7,2)) AS net_loss
                             FROM   web_sales
                             UNION ALL
                             SELECT          ws_web_site_sk          AS wsr_web_site_sk,
                                             wr_returned_date_sk     AS date_sk,
                                             cast(0 AS decimal(7,2)) AS sales_price,
                                             cast(0 AS decimal(7,2)) AS profit,
                                             wr_return_amt           AS return_amt,
                                             wr_net_loss             AS net_loss
                             FROM            web_returns
                             LEFT OUTER JOIN web_sales
                             ON              (
                                                             wr_item_sk = ws_item_sk
                                             AND             wr_order_number = ws_order_number) ) salesreturns,
                      date_dim,
                      web_site
             WHERE    date_sk = d_date_sk
             AND      d_date BETWEEN cast('{sales_date}' AS date) AND      (
                               cast('{sales_date}' AS date) + INTERVAL '14' day)
             AND      wsr_web_site_sk = web_site_sk
             GROUP BY web_site_id)
    SELECT
             channel ,
             id ,
             sum(sales)   AS sales ,
             sum(returns1) AS returns1 ,
             sum(profit)  AS profit
    FROM     (
                    SELECT 'store channel' AS channel ,
                           'store'
                                  || s_store_id AS id ,
                           sales ,
                           returns1 ,
                           (profit - profit_loss) AS profit
                    FROM   ssr
                    UNION ALL
                    SELECT 'catalog channel' AS channel ,
                           'catalog_page'
                                  || cp_catalog_page_id AS id ,
                           sales ,
                           returns1 ,
                           (profit - profit_loss) AS profit
                    FROM   csr
                    UNION ALL
                    SELECT 'web channel' AS channel ,
                           'web_site'
                                  || web_site_id AS id ,
                           sales ,
                           returns1 ,
                           (profit - profit_loss) AS profit
                    FROM   wsr ) x
    GROUP BY rollup (channel, id)
    ORDER BY channel ,
             id
    LIMIT 100;
    """


def _channel_agg(
    sales: pl.LazyFrame,
    returns: pl.LazyFrame,
    date_dim: pl.LazyFrame,
    entity: pl.LazyFrame,
    *,
    sales_date_key: str,
    returns_date_key: str,
    sales_entity_key: str,
    returns_entity_key: str,
    entity_id_col: str,
    entity_join_key: str,
    sales_price_col: str,
    profit_col: str,
    return_amt_col: str,
    net_loss_col: str,
    start_date: date,
    end_date: date,
) -> pl.LazyFrame:
    """Aggregate sales and returns for one channel via UNION ALL, producing (entity_id, sales, profit, returns1, profit_loss)."""
    target_dates = date_dim.filter(
        pl.col("d_date").is_between(pl.lit(start_date), pl.lit(end_date), closed="both")
    ).select("d_date_sk")

    sales_leg = sales.select(
        pl.col(sales_entity_key).alias("entity_sk"),
        pl.col(sales_date_key).alias("date_sk"),
        pl.col(sales_price_col).alias("sales_price"),
        pl.col(profit_col).alias("profit"),
        pl.lit(0).alias("return_amt"),
        pl.lit(0).alias("net_loss"),
    )
    returns_leg = returns.select(
        pl.col(returns_entity_key).alias("entity_sk"),
        pl.col(returns_date_key).alias("date_sk"),
        pl.lit(0).alias("sales_price"),
        pl.lit(0).alias("profit"),
        pl.col(return_amt_col).alias("return_amt"),
        pl.col(net_loss_col).alias("net_loss"),
    )

    return (
        pl.concat([sales_leg, returns_leg], how="vertical_relaxed")
        .join(target_dates, left_on="date_sk", right_on="d_date_sk")
        .join(entity, left_on="entity_sk", right_on=entity_join_key)
        .group_by(entity_id_col)
        .agg(
            sql_sum("sales_price").alias("sales"),
            sql_sum("profit").alias("profit"),
            sql_sum("return_amt").alias("returns1"),
            sql_sum("net_loss").alias("profit_loss"),
        )
    )


def polars_impl(run_config: RunConfig) -> QueryResult:
    """Query 5."""
    params = load_parameters(
        int(run_config.scale_factor), query_id=5, qualification=run_config.qualification
    )

    sales_date_str = params["sales_date"]
    year, month, day = map(int, sales_date_str.split("-"))

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
    catalog_page = get_data(run_config.dataset_path, "catalog_page", run_config.suffix)
    web_site = get_data(run_config.dataset_path, "web_site", run_config.suffix)

    start_date = date(year, month, day)
    end_date = start_date + timedelta(days=14)

    # For web returns the entity key (ws_web_site_sk) comes from a LEFT JOIN
    # with web_sales (web_returns doesn't carry web_site_sk directly).
    web_returns_with_site = web_returns.join(
        web_sales.select(["ws_item_sk", "ws_order_number", "ws_web_site_sk"]),
        left_on=["wr_item_sk", "wr_order_number"],
        right_on=["ws_item_sk", "ws_order_number"],
        how="left",
    )

    ssr = _channel_agg(
        store_sales,
        store_returns,
        date_dim,
        store,
        sales_date_key="ss_sold_date_sk",
        returns_date_key="sr_returned_date_sk",
        sales_entity_key="ss_store_sk",
        returns_entity_key="sr_store_sk",
        entity_id_col="s_store_id",
        entity_join_key="s_store_sk",
        sales_price_col="ss_ext_sales_price",
        profit_col="ss_net_profit",
        return_amt_col="sr_return_amt",
        net_loss_col="sr_net_loss",
        start_date=start_date,
        end_date=end_date,
    )
    csr = _channel_agg(
        catalog_sales,
        catalog_returns,
        date_dim,
        catalog_page,
        sales_date_key="cs_sold_date_sk",
        returns_date_key="cr_returned_date_sk",
        sales_entity_key="cs_catalog_page_sk",
        returns_entity_key="cr_catalog_page_sk",
        entity_id_col="cp_catalog_page_id",
        entity_join_key="cp_catalog_page_sk",
        sales_price_col="cs_ext_sales_price",
        profit_col="cs_net_profit",
        return_amt_col="cr_return_amount",
        net_loss_col="cr_net_loss",
        start_date=start_date,
        end_date=end_date,
    )
    wsr = _channel_agg(
        web_sales,
        web_returns_with_site,
        date_dim,
        web_site,
        sales_date_key="ws_sold_date_sk",
        returns_date_key="wr_returned_date_sk",
        sales_entity_key="ws_web_site_sk",
        returns_entity_key="ws_web_site_sk",
        entity_id_col="web_site_id",
        entity_join_key="web_site_sk",
        sales_price_col="ws_ext_sales_price",
        profit_col="ws_net_profit",
        return_amt_col="wr_return_amt",
        net_loss_col="wr_net_loss",
        start_date=start_date,
        end_date=end_date,
    )

    store_channel = ssr.select(
        pl.lit("store channel").alias("channel"),
        (pl.lit("store") + pl.col("s_store_id")).alias("id"),
        pl.col("sales"),
        pl.col("returns1"),
        (pl.col("profit") - pl.col("profit_loss")).alias("profit"),
    )
    catalog_channel = csr.select(
        pl.lit("catalog channel").alias("channel"),
        (pl.lit("catalog_page") + pl.col("cp_catalog_page_id")).alias("id"),
        pl.col("sales"),
        pl.col("returns1"),
        (pl.col("profit") - pl.col("profit_loss")).alias("profit"),
    )
    web_channel = wsr.select(
        pl.lit("web channel").alias("channel"),
        (pl.lit("web_site") + pl.col("web_site_id")).alias("id"),
        pl.col("sales"),
        pl.col("returns1"),
        (pl.col("profit") - pl.col("profit_loss")).alias("profit"),
    )
    all_channels = pl.concat([store_channel, catalog_channel, web_channel])

    agg_exprs = [
        sql_sum("sales").alias("sales"),
        sql_sum("returns1").alias("returns1"),
        sql_sum("profit").alias("profit"),
    ]
    per_id = all_channels.group_by(["channel", "id"]).agg(agg_exprs)
    per_channel = (
        all_channels.group_by("channel")
        .agg(agg_exprs)
        .with_columns(pl.lit(None, dtype=pl.String).alias("id"))
        .select(["channel", "id", "sales", "returns1", "profit"])
    )
    grand_total = all_channels.select(
        pl.lit(None, dtype=pl.String).alias("channel"),
        pl.lit(None, dtype=pl.String).alias("id"),
        *agg_exprs,
    )

    return QueryResult(
        frame=(
            pl.concat([per_id, per_channel, grand_total])
            .sort(["channel", "id"], nulls_last=True)
            .limit(100)
        ),
        sort_by=[("channel", False), ("id", False)],
        limit=100,
    )
