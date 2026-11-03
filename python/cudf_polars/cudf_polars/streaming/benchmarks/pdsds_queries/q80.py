# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 80."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.streaming.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.streaming.benchmarks.utils import QueryResult, get_data

if TYPE_CHECKING:
    from cudf_polars.streaming.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 80."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=80,
        qualification=run_config.qualification,
    )
    sdate = params["sdate"]

    return f"""
    WITH ssr AS
    (
                    SELECT          s_store_id                                    AS store_id,
                                    Sum(ss_ext_sales_price)                       AS sales,
                                    Sum(COALESCE(sr_return_amt, 0))               AS returns1,
                                    Sum(ss_net_profit - COALESCE(sr_net_loss, 0)) AS profit
                    FROM            store_sales
                    LEFT OUTER JOIN store_returns
                    ON              (
                                                    ss_item_sk = sr_item_sk
                                    AND             ss_ticket_number = sr_ticket_number),
                                    date_dim,
                                    store,
                                    item,
                                    promotion
                    WHERE           ss_sold_date_sk = d_date_sk
                    AND             d_date BETWEEN Cast('{sdate}' AS DATE) AND             (
                                                    Cast('{sdate}' AS DATE) + INTERVAL '30' day)
                    AND             ss_store_sk = s_store_sk
                    AND             ss_item_sk = i_item_sk
                    AND             i_current_price > 50
                    AND             ss_promo_sk = p_promo_sk
                    AND             p_channel_tv = 'N'
                    GROUP BY        s_store_id) , csr AS
    (
                    SELECT          cp_catalog_page_id                            AS catalog_page_id,
                                    sum(cs_ext_sales_price)                       AS sales,
                                    sum(COALESCE(cr_return_amount, 0))            AS returns1,
                                    sum(cs_net_profit - COALESCE(cr_net_loss, 0)) AS profit
                    FROM            catalog_sales
                    LEFT OUTER JOIN catalog_returns
                    ON              (
                                                    cs_item_sk = cr_item_sk
                                    AND             cs_order_number = cr_order_number),
                                    date_dim,
                                    catalog_page,
                                    item,
                                    promotion
                    WHERE           cs_sold_date_sk = d_date_sk
                    AND             d_date BETWEEN cast('{sdate}' AS date) AND             (
                                                    cast('{sdate}' AS date) + INTERVAL '30' day)
                    AND             cs_catalog_page_sk = cp_catalog_page_sk
                    AND             cs_item_sk = i_item_sk
                    AND             i_current_price > 50
                    AND             cs_promo_sk = p_promo_sk
                    AND             p_channel_tv = 'N'
                    GROUP BY        cp_catalog_page_id) , wsr AS
    (
                    SELECT          web_site_id,
                                    sum(ws_ext_sales_price)                       AS sales,
                                    sum(COALESCE(wr_return_amt, 0))               AS returns1,
                                    sum(ws_net_profit - COALESCE(wr_net_loss, 0)) AS profit
                    FROM            web_sales
                    LEFT OUTER JOIN web_returns
                    ON              (
                                                    ws_item_sk = wr_item_sk
                                    AND             ws_order_number = wr_order_number),
                                    date_dim,
                                    web_site,
                                    item,
                                    promotion
                    WHERE           ws_sold_date_sk = d_date_sk
                    AND             d_date BETWEEN cast('{sdate}' AS date) AND             (
                                                    cast('{sdate}' AS date) + INTERVAL '30' day)
                    AND             ws_web_site_sk = web_site_sk
                    AND             ws_item_sk = i_item_sk
                    AND             i_current_price > 50
                    AND             ws_promo_sk = p_promo_sk
                    AND             p_channel_tv = 'N'
                    GROUP BY        web_site_id)
    SELECT
             channel ,
             id ,
             sum(sales)   AS sales ,
             sum(returns1) AS returns1 ,
             sum(profit)  AS profit
    FROM     (
                    SELECT 'store channel' AS channel ,
                           'store'
                                  || store_id AS id ,
                           sales ,
                           returns1 ,
                           profit
                    FROM   ssr
                    UNION ALL
                    SELECT 'catalog channel' AS channel ,
                           'catalog_page'
                                  || catalog_page_id AS id ,
                           sales ,
                           returns1 ,
                           profit
                    FROM   csr
                    UNION ALL
                    SELECT 'web channel' AS channel ,
                           'web_site'
                                  || web_site_id AS id ,
                           sales ,
                           returns1 ,
                           profit
                    FROM   wsr ) x
    GROUP BY rollup (channel, id)
    ORDER BY channel ,
             id
    LIMIT 100;
    """


def q80_segment(
    sales: pl.LazyFrame,
    returns: pl.LazyFrame,
    dates: pl.LazyFrame,
    id_dim: pl.LazyFrame,
    item: pl.LazyFrame,
    promotion: pl.LazyFrame,
    *,
    returns_join_left: list[str],
    returns_join_right: list[str],
    sold_date_key: str,
    id_left_key: str,
    id_right_key: str,
    id_out_col: str,
    item_left_key: str,
    promo_left_key: str,
    ext_sales_col: str,
    net_profit_col: str,
    ret_amt_col: str,
    ret_loss_col: str,
) -> pl.LazyFrame:
    """
    Builds one channel sub-aggregation (sales, returns, profit) grouped by an ID column.

    Join order: filter sales through selective dimension tables first
    (date range, item price > 50, promotion channel_tv = 'N'), then
    LEFT JOIN returns last so the expensive outer join operates on the
    already-reduced sales set.
    """
    filtered_item = item.filter(pl.col("i_current_price") > 50)
    filtered_promo = promotion.filter(pl.col("p_channel_tv") == "N")
    return (
        sales.join(dates, left_on=sold_date_key, right_on="d_date_sk")
        .join(filtered_item, left_on=item_left_key, right_on="i_item_sk")
        .join(filtered_promo, left_on=promo_left_key, right_on="p_promo_sk")
        .join(id_dim, left_on=id_left_key, right_on=id_right_key)
        .join(
            returns, left_on=returns_join_left, right_on=returns_join_right, how="left"
        )
        .group_by(id_out_col)
        .agg(
            [
                pl.col(ext_sales_col).sum().alias("sales"),
                pl.col(ret_amt_col).fill_null(0).sum().alias("returns1"),
                (pl.col(net_profit_col) - pl.col(ret_loss_col).fill_null(0))
                .sum()
                .alias("profit"),
            ]
        )
    )


def q80_channel_frame(
    label: str, lf: pl.LazyFrame, id_col: str, prefix: str
) -> pl.LazyFrame:
    """Shapes a per-channel frame with channel label and stringified ID."""
    return lf.select(
        [
            pl.lit(label).alias("channel"),
            (pl.lit(prefix) + pl.col(id_col).cast(pl.Utf8)).alias("id"),
            "sales",
            "returns1",
            "profit",
        ]
    )


def polars_impl(run_config: RunConfig) -> QueryResult:
    """Query 80."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=80,
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
    catalog_page = get_data(run_config.dataset_path, "catalog_page", run_config.suffix)
    web_site = get_data(run_config.dataset_path, "web_site", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)
    promotion = get_data(run_config.dataset_path, "promotion", run_config.suffix)

    start_date = (pl.date(year, month, day)).cast(pl.Datetime("us"))
    end_date = (start_date + pl.duration(days=30)).cast(pl.Datetime("us"))
    dates = date_dim.filter(
        (pl.col("d_date") >= start_date) & (pl.col("d_date") <= end_date)
    ).select("d_date_sk")

    ssr = q80_segment(
        store_sales,
        store_returns,
        dates,
        store,
        item,
        promotion,
        returns_join_left=["ss_item_sk", "ss_ticket_number"],
        returns_join_right=["sr_item_sk", "sr_ticket_number"],
        sold_date_key="ss_sold_date_sk",
        id_left_key="ss_store_sk",
        id_right_key="s_store_sk",
        id_out_col="s_store_id",
        item_left_key="ss_item_sk",
        promo_left_key="ss_promo_sk",
        ext_sales_col="ss_ext_sales_price",
        net_profit_col="ss_net_profit",
        ret_amt_col="sr_return_amt",
        ret_loss_col="sr_net_loss",
    ).select([pl.col("s_store_id").alias("store_id"), "sales", "returns1", "profit"])

    csr = q80_segment(
        catalog_sales,
        catalog_returns,
        dates,
        catalog_page,
        item,
        promotion,
        returns_join_left=["cs_item_sk", "cs_order_number"],
        returns_join_right=["cr_item_sk", "cr_order_number"],
        sold_date_key="cs_sold_date_sk",
        id_left_key="cs_catalog_page_sk",
        id_right_key="cp_catalog_page_sk",
        id_out_col="cp_catalog_page_id",
        item_left_key="cs_item_sk",
        promo_left_key="cs_promo_sk",
        ext_sales_col="cs_ext_sales_price",
        net_profit_col="cs_net_profit",
        ret_amt_col="cr_return_amount",
        ret_loss_col="cr_net_loss",
    ).select(
        [
            pl.col("cp_catalog_page_id").alias("catalog_page_id"),
            "sales",
            "returns1",
            "profit",
        ]
    )

    wsr = q80_segment(
        web_sales,
        web_returns,
        dates,
        web_site,
        item,
        promotion,
        returns_join_left=["ws_item_sk", "ws_order_number"],
        returns_join_right=["wr_item_sk", "wr_order_number"],
        sold_date_key="ws_sold_date_sk",
        id_left_key="ws_web_site_sk",
        id_right_key="web_site_sk",
        id_out_col="web_site_id",
        item_left_key="ws_item_sk",
        promo_left_key="ws_promo_sk",
        ext_sales_col="ws_ext_sales_price",
        net_profit_col="ws_net_profit",
        ret_amt_col="wr_return_amt",
        ret_loss_col="wr_net_loss",
    ).select(["web_site_id", "sales", "returns1", "profit"])

    combined = pl.concat(
        [
            q80_channel_frame("store channel", ssr, "store_id", "store"),
            q80_channel_frame(
                "catalog channel", csr, "catalog_page_id", "catalog_page"
            ),
            q80_channel_frame("web channel", wsr, "web_site_id", "web_site"),
        ]
    )

    level1 = combined.group_by(["channel", "id"]).agg(
        [
            pl.col("sales").sum().alias("sales"),
            pl.col("returns1").sum().alias("returns1"),
            pl.col("profit").sum().alias("profit"),
        ]
    )

    level2 = (
        combined.group_by("channel")
        .agg(
            [
                pl.col("sales").sum().alias("sales"),
                pl.col("returns1").sum().alias("returns1"),
                pl.col("profit").sum().alias("profit"),
            ]
        )
        .with_columns(pl.lit(None).cast(pl.Utf8).alias("id"))
    )

    sort_by = {"channel": False, "id": False}
    limit = 100
    return QueryResult(
        frame=(
            pl.concat([level1, level2], how="diagonal")
            .sort(sort_by.keys(), nulls_last=True)
            .limit(limit)
        ),
        sort_by=list(sort_by.items()),
        limit=limit,
    )
