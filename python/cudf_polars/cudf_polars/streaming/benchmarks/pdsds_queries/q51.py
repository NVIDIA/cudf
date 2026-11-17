# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Query 51."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.streaming.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.streaming.benchmarks.pdsds_queries import sql_sum
from cudf_polars.streaming.benchmarks.utils import QueryResult, get_data

if TYPE_CHECKING:
    from cudf_polars.streaming.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 51."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=51,
        qualification=run_config.qualification,
    )

    dms = params["dms"]

    return f"""
        WITH web_v1 AS
        (SELECT ws_item_sk item_sk,
                d_date,
                sum(sum(ws_sales_price)) OVER (PARTITION BY ws_item_sk
                                                ORDER BY d_date ROWS BETWEEN unbounded preceding AND CURRENT ROW) cume_sales
        FROM web_sales,
                date_dim
        WHERE ws_sold_date_sk=d_date_sk
            AND d_month_seq BETWEEN {dms} AND {dms}+11
            AND ws_item_sk IS NOT NULL
        GROUP BY ws_item_sk,
                    d_date),
            store_v1 AS
        (SELECT ss_item_sk item_sk,
                d_date,
                sum(sum(ss_sales_price)) OVER (PARTITION BY ss_item_sk
                                                ORDER BY d_date ROWS BETWEEN unbounded preceding AND CURRENT ROW) cume_sales
        FROM store_sales,
                date_dim
        WHERE ss_sold_date_sk=d_date_sk
            AND d_month_seq BETWEEN {dms} AND {dms}+11
            AND ss_item_sk IS NOT NULL
        GROUP BY ss_item_sk,
                    d_date)
        SELECT *
        FROM
        (SELECT item_sk,
                d_date,
                web_sales,
                store_sales,
                max(web_sales) OVER (PARTITION BY item_sk
                                    ORDER BY d_date ROWS BETWEEN unbounded preceding AND CURRENT ROW) web_cumulative,
                max(store_sales) OVER (PARTITION BY item_sk
                                        ORDER BY d_date ROWS BETWEEN unbounded preceding AND CURRENT ROW) store_cumulative
        FROM
            (SELECT CASE
                        WHEN web.item_sk IS NOT NULL THEN web.item_sk
                        ELSE store.item_sk
                    END item_sk,
                    CASE
                        WHEN web.d_date IS NOT NULL THEN web.d_date
                        ELSE store.d_date
                    END d_date,
                    web.cume_sales web_sales,
                    store.cume_sales store_sales
            FROM web_v1 web
            FULL OUTER JOIN store_v1 store ON (web.item_sk = store.item_sk
                                                AND web.d_date = store.d_date)) x) y
        WHERE web_cumulative > store_cumulative
        ORDER BY item_sk NULLS FIRST,
                d_date NULLS FIRST
        LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> QueryResult:
    """Query 51."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=51,
        qualification=run_config.qualification,
    )

    dms = params["dms"]

    web_sales = get_data(run_config.dataset_path, "web_sales", run_config.suffix)
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)

    def _cume_sales(partition_col: str) -> pl.Expr:
        return (
            pl.col("daily_sum")
            .cum_sum()
            .forward_fill()
            .over(partition_by=partition_col, order_by="d_date")
            .alias("cume_sales")
        )

    # web_v1: daily sums -> cumulative sum per (item, ordered by date)
    web_v1 = (
        web_sales.join(date_dim, left_on="ws_sold_date_sk", right_on="d_date_sk")
        .filter(
            pl.col("d_month_seq").is_between(dms, dms + 11)
            & pl.col("ws_item_sk").is_not_null()
        )
        .group_by(["ws_item_sk", "d_date"])
        .agg(sql_sum("ws_sales_price").alias("daily_sum"))
        .with_columns(_cume_sales("ws_item_sk"))
        .select(
            pl.col("ws_item_sk").alias("item_sk"),
            "d_date",
            pl.col("cume_sales").alias("web_sales"),
        )
    )

    # store_v1: daily sums -> cumulative sum per (item, ordered by date)
    store_v1 = (
        store_sales.join(date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk")
        .filter(
            pl.col("d_month_seq").is_between(dms, dms + 11)
            & pl.col("ss_item_sk").is_not_null()
        )
        .group_by(["ss_item_sk", "d_date"])
        .agg(sql_sum("ss_sales_price").alias("daily_sum"))
        .with_columns(_cume_sales("ss_item_sk"))
        .select(
            pl.col("ss_item_sk").alias("item_sk"),
            "d_date",
            pl.col("cume_sales").alias("store_sales"),
        )
    )

    # FULL OUTER JOIN on (item_sk, d_date), then fill_null keys like the CASE logic
    combined = (
        web_v1.join(
            store_v1,
            on=["item_sk", "d_date"],
            how="full",
            suffix="_store",
        )
        .with_columns(
            item_sk=pl.col("item_sk").fill_null(pl.col("item_sk_store")),
            d_date=pl.col("d_date").fill_null(pl.col("d_date_store")),
        )
        .select("item_sk", "d_date", "web_sales", "store_sales")
    )

    # Collapse any duplicate (item_sk, d_date) rows by taking the max of each cumulative side
    collapsed = combined.group_by(["item_sk", "d_date"], maintain_order=True).agg(
        pl.col("web_sales").max().alias("web_sales"),
        pl.col("store_sales").max().alias("store_sales"),
    )

    # Sort once; then forward-fill cumulative series per item by date
    ordered = collapsed.sort(["item_sk", "d_date"])

    with_ff = ordered.with_columns(
        web_cumulative=pl.col("web_sales")
        .forward_fill()
        .over(partition_by="item_sk", order_by="d_date"),
        store_cumulative=pl.col("store_sales")
        .forward_fill()
        .over(partition_by="item_sk", order_by="d_date"),
    )

    sort_by = {"item_sk": False, "d_date": False}
    limit = 100
    return QueryResult(
        frame=(
            with_ff.filter(pl.col("web_cumulative") > pl.col("store_cumulative"))
            .select(
                "item_sk",
                "d_date",
                "web_sales",
                "store_sales",
                "web_cumulative",
                "store_cumulative",
            )
            .sort(list(sort_by.keys()), nulls_last=False)
            .limit(limit)
        ),
        sort_by=list(sort_by.items()),
        limit=limit,
        nulls_last=False,
    )
