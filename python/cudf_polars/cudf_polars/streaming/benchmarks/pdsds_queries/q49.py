# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Query 49."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.streaming.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.streaming.benchmarks.utils import QueryResult, get_data

if TYPE_CHECKING:
    from cudf_polars.streaming.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 49."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=49,
        qualification=run_config.qualification,
    )

    year = params["year"]
    month = params["month"]

    return f"""
    SELECT 'web' AS channel,
                   web.item,
                   web.return_ratio,
                   web.return_rank,
                   web.currency_rank
    FROM   (SELECT item,
                   return_ratio,
                   currency_ratio,
                   Rank()
                     OVER (
                       ORDER BY return_ratio)   AS return_rank,
                   Rank()
                     OVER (
                       ORDER BY currency_ratio) AS currency_rank
            FROM   (SELECT ws.ws_item_sk                                       AS
                           item,
                           ( Cast(Sum(COALESCE(wr.wr_return_quantity, 0)) AS DEC(15,
                                  4)) /
                             Cast(
                             Sum(COALESCE(ws.ws_quantity, 0)) AS DEC(15, 4)) ) AS
                           return_ratio,
                           ( Cast(Sum(COALESCE(wr.wr_return_amt, 0)) AS DEC(15, 4))
                             / Cast(
                             Sum(
                             COALESCE(ws.ws_net_paid, 0)) AS DEC(15,
                             4)) )                                             AS
                           currency_ratio
                    FROM   web_sales ws
                           LEFT OUTER JOIN web_returns wr
                                        ON ( ws.ws_order_number = wr.wr_order_number
                                             AND ws.ws_item_sk = wr.wr_item_sk ),
                           date_dim
                    WHERE  wr.wr_return_amt > 10000
                           AND ws.ws_net_profit > 1
                           AND ws.ws_net_paid > 0
                           AND ws.ws_quantity > 0
                           AND ws_sold_date_sk = d_date_sk
                           AND d_year = {year}
                           AND d_moy = {month}
                    GROUP  BY ws.ws_item_sk) in_web) web
    WHERE  ( web.return_rank <= 10
              OR web.currency_rank <= 10 )
    UNION
    SELECT 'catalog' AS channel,
           catalog.item,
           catalog.return_ratio,
           catalog.return_rank,
           catalog.currency_rank
    FROM   (SELECT item,
                   return_ratio,
                   currency_ratio,
                   Rank()
                     OVER (
                       ORDER BY return_ratio)   AS return_rank,
                   Rank()
                     OVER (
                       ORDER BY currency_ratio) AS currency_rank
            FROM   (SELECT cs.cs_item_sk                                       AS
                           item,
                           ( Cast(Sum(COALESCE(cr.cr_return_quantity, 0)) AS DEC(15,
                                  4)) /
                             Cast(
                             Sum(COALESCE(cs.cs_quantity, 0)) AS DEC(15, 4)) ) AS
                           return_ratio,
                           ( Cast(Sum(COALESCE(cr.cr_return_amount, 0)) AS DEC(15, 4
                                  )) /
                             Cast(Sum(
                             COALESCE(cs.cs_net_paid, 0)) AS DEC(
                             15, 4)) )                                         AS
                           currency_ratio
                    FROM   catalog_sales cs
                           LEFT OUTER JOIN catalog_returns cr
                                        ON ( cs.cs_order_number = cr.cr_order_number
                                             AND cs.cs_item_sk = cr.cr_item_sk ),
                           date_dim
                    WHERE  cr.cr_return_amount > 10000
                           AND cs.cs_net_profit > 1
                           AND cs.cs_net_paid > 0
                           AND cs.cs_quantity > 0
                           AND cs_sold_date_sk = d_date_sk
                           AND d_year = {year}
                           AND d_moy = {month}
                    GROUP  BY cs.cs_item_sk) in_cat) catalog
    WHERE  ( catalog.return_rank <= 10
              OR catalog.currency_rank <= 10 )
    UNION
    SELECT 'store' AS channel,
           store.item,
           store.return_ratio,
           store.return_rank,
           store.currency_rank
    FROM   (SELECT item,
                   return_ratio,
                   currency_ratio,
                   Rank()
                     OVER (
                       ORDER BY return_ratio)   AS return_rank,
                   Rank()
                     OVER (
                       ORDER BY currency_ratio) AS currency_rank
            FROM   (SELECT sts.ss_item_sk                                       AS
                           item,
                           ( Cast(Sum(COALESCE(sr.sr_return_quantity, 0)) AS DEC(15,
                                  4)) /
                             Cast(
                             Sum(COALESCE(sts.ss_quantity, 0)) AS DEC(15, 4)) ) AS
                           return_ratio,
                           ( Cast(Sum(COALESCE(sr.sr_return_amt, 0)) AS DEC(15, 4))
                             / Cast(
                             Sum(
                             COALESCE(sts.ss_net_paid, 0)) AS DEC(15, 4)) )     AS
                           currency_ratio
                    FROM   store_sales sts
                           LEFT OUTER JOIN store_returns sr
                                        ON ( sts.ss_ticket_number =
                                             sr.sr_ticket_number
                                             AND sts.ss_item_sk = sr.sr_item_sk ),
                           date_dim
                    WHERE  sr.sr_return_amt > 10000
                           AND sts.ss_net_profit > 1
                           AND sts.ss_net_paid > 0
                           AND sts.ss_quantity > 0
                           AND ss_sold_date_sk = d_date_sk
                           AND d_year = {year}
                           AND d_moy = {month}
                    GROUP  BY sts.ss_item_sk) in_store) store
    WHERE  ( store.return_rank <= 10
              OR store.currency_rank <= 10 )
    ORDER  BY 1,
              4,
              5
    LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> QueryResult:
    """Query 49."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=49,
        qualification=run_config.qualification,
    )

    year = params["year"]
    month = params["month"]

    # Load tables
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
    # Web channel data
    web_data = (
        web_sales.join(
            web_returns,
            left_on=["ws_order_number", "ws_item_sk"],
            right_on=["wr_order_number", "wr_item_sk"],
            how="left",
        )
        .join(date_dim, left_on="ws_sold_date_sk", right_on="d_date_sk")
        .filter(
            (pl.col("wr_return_amt").fill_null(0) > 10000)
            & (pl.col("ws_net_profit") > 1)
            & (pl.col("ws_net_paid") > 0)
            & (pl.col("ws_quantity") > 0)
            & (pl.col("d_year") == year)
            & (pl.col("d_moy") == month)
        )
        .group_by("ws_item_sk")
        .agg(
            [
                # Return ratio calculation
                (
                    pl.when(pl.col("ws_quantity").drop_nulls().count() > 0)
                    .then(
                        pl.col("wr_return_quantity").fill_null(0).sum().cast(pl.Float64)
                        / pl.col("ws_quantity").fill_null(0).sum().cast(pl.Float64)
                    )
                    .otherwise(None)
                ).alias("return_ratio"),
                # Currency ratio calculation
                # DuckDB SQL promotes Decimal/Decimal to Float64; Polars keeps it as
                # Decimal. Cast to Float64 to match DuckDB return type.
                (
                    pl.when(pl.col("ws_net_paid").drop_nulls().count() > 0)
                    .then(
                        pl.col("wr_return_amt")
                        .fill_null(0)
                        .sum()
                        .cast(pl.Decimal(15, 4))
                        .cast(pl.Float64)
                        / pl.col("ws_net_paid")
                        .fill_null(0)
                        .sum()
                        .cast(pl.Decimal(15, 4))
                        .cast(pl.Float64)
                    )
                    .otherwise(None)
                ).alias("currency_ratio"),
            ]
        )
        .with_columns(
            [
                pl.col("ws_item_sk").alias("item"),
                pl.col("return_ratio").rank(method="min").alias("return_rank"),
                pl.col("currency_ratio").rank(method="min").alias("currency_rank"),
            ]
        )
        .filter((pl.col("return_rank") <= 10) | (pl.col("currency_rank") <= 10))
        .with_columns([pl.lit("web").alias("channel")])
        .select(["channel", "item", "return_ratio", "return_rank", "currency_rank"])
    )
    # Catalog channel data
    catalog_data = (
        catalog_sales.join(
            catalog_returns,
            left_on=["cs_order_number", "cs_item_sk"],
            right_on=["cr_order_number", "cr_item_sk"],
            how="left",
        )
        .join(date_dim, left_on="cs_sold_date_sk", right_on="d_date_sk")
        .filter(
            (pl.col("cr_return_amount").fill_null(0) > 10000)
            & (pl.col("cs_net_profit") > 1)
            & (pl.col("cs_net_paid") > 0)
            & (pl.col("cs_quantity") > 0)
            & (pl.col("d_year") == year)
            & (pl.col("d_moy") == month)
        )
        .group_by("cs_item_sk")
        .agg(
            [
                # Return ratio calculation
                (
                    pl.when(pl.col("cs_quantity").drop_nulls().count() > 0)
                    .then(
                        pl.col("cr_return_quantity").fill_null(0).sum().cast(pl.Float64)
                        / pl.col("cs_quantity").fill_null(0).sum().cast(pl.Float64)
                    )
                    .otherwise(None)
                ).alias("return_ratio"),
                # Currency ratio calculation
                # DuckDB SQL promotes Decimal/Decimal to Float64; Polars keeps it as
                # Decimal. Cast to Float64 to match DuckDB return type.
                (
                    pl.when(pl.col("cs_net_paid").drop_nulls().count() > 0)
                    .then(
                        pl.col("cr_return_amount")
                        .fill_null(0)
                        .sum()
                        .cast(pl.Decimal(15, 4))
                        .cast(pl.Float64)
                        / pl.col("cs_net_paid")
                        .fill_null(0)
                        .sum()
                        .cast(pl.Decimal(15, 4))
                        .cast(pl.Float64)
                    )
                    .otherwise(None)
                ).alias("currency_ratio"),
            ]
        )
        .with_columns(
            [
                pl.col("cs_item_sk").alias("item"),
                pl.col("return_ratio").rank(method="min").alias("return_rank"),
                pl.col("currency_ratio").rank(method="min").alias("currency_rank"),
            ]
        )
        .filter((pl.col("return_rank") <= 10) | (pl.col("currency_rank") <= 10))
        .with_columns([pl.lit("catalog").alias("channel")])
        .select(["channel", "item", "return_ratio", "return_rank", "currency_rank"])
    )
    # Store channel data
    store_data = (
        store_sales.join(
            store_returns,
            left_on=["ss_ticket_number", "ss_item_sk"],
            right_on=["sr_ticket_number", "sr_item_sk"],
            how="left",
        )
        .join(date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk")
        .filter(
            (pl.col("sr_return_amt").fill_null(0) > 10000)
            & (pl.col("ss_net_profit") > 1)
            & (pl.col("ss_net_paid") > 0)
            & (pl.col("ss_quantity") > 0)
            & (pl.col("d_year") == year)
            & (pl.col("d_moy") == month)
        )
        .group_by("ss_item_sk")
        .agg(
            [
                # Return ratio calculation
                (
                    pl.when(pl.col("ss_quantity").drop_nulls().count() > 0)
                    .then(
                        pl.col("sr_return_quantity").fill_null(0).sum().cast(pl.Float64)
                        / pl.col("ss_quantity").fill_null(0).sum().cast(pl.Float64)
                    )
                    .otherwise(None)
                ).alias("return_ratio"),
                # Currency ratio calculation
                # DuckDB SQL promotes Decimal/Decimal to Float64; Polars keeps it as
                # Decimal. Cast to Float64 to match DuckDB return type.
                (
                    pl.when(pl.col("ss_net_paid").drop_nulls().count() > 0)
                    .then(
                        pl.col("sr_return_amt")
                        .fill_null(0)
                        .sum()
                        .cast(pl.Decimal(15, 4))
                        .cast(pl.Float64)
                        / pl.col("ss_net_paid")
                        .fill_null(0)
                        .sum()
                        .cast(pl.Decimal(15, 4))
                        .cast(pl.Float64)
                    )
                    .otherwise(None)
                ).alias("currency_ratio"),
            ]
        )
        .with_columns(
            [
                pl.col("ss_item_sk").alias("item"),
                pl.col("return_ratio").rank(method="min").alias("return_rank"),
                pl.col("currency_ratio").rank(method="min").alias("currency_rank"),
            ]
        )
        .filter((pl.col("return_rank") <= 10) | (pl.col("currency_rank") <= 10))
        .with_columns([pl.lit("store").alias("channel")])
        .select(["channel", "item", "return_ratio", "return_rank", "currency_rank"])
    )
    # Union all channels and apply final ordering
    sort_by = {"channel": False, "return_rank": False, "currency_rank": False}
    limit = 100
    return QueryResult(
        frame=(
            pl.concat([web_data, catalog_data, store_data])
            .select(["channel", "item", "return_ratio", "return_rank", "currency_rank"])
            .sort(sort_by.keys(), nulls_last=True)
            .limit(limit)
        ),
        sort_by=list(sort_by.items()),
        limit=limit,
    )
