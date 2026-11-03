# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 58."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.streaming.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.streaming.benchmarks.utils import QueryResult, get_data

if TYPE_CHECKING:
    from cudf_polars.streaming.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 58."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=58,
        qualification=run_config.qualification,
    )

    sales_date = params["sales_date"]

    return f"""
        WITH ss_items
            AS (SELECT i_item_id               item_id,
                        Sum(ss_ext_sales_price) ss_item_rev
                FROM   store_sales,
                        item,
                        date_dim
                WHERE  ss_item_sk = i_item_sk
                        AND d_date IN (SELECT d_date
                                    FROM   date_dim
                                    WHERE  d_week_seq = (SELECT d_week_seq
                                                            FROM   date_dim
                                                            WHERE  d_date = '{sales_date}'
                                                        ))
                        AND ss_sold_date_sk = d_date_sk
                GROUP  BY i_item_id),
            cs_items
            AS (SELECT i_item_id               item_id,
                        Sum(cs_ext_sales_price) cs_item_rev
                FROM   catalog_sales,
                        item,
                        date_dim
                WHERE  cs_item_sk = i_item_sk
                        AND d_date IN (SELECT d_date
                                    FROM   date_dim
                                    WHERE  d_week_seq = (SELECT d_week_seq
                                                            FROM   date_dim
                                                            WHERE  d_date = '{sales_date}'
                                                        ))
                        AND cs_sold_date_sk = d_date_sk
                GROUP  BY i_item_id),
            ws_items
            AS (SELECT i_item_id               item_id,
                        Sum(ws_ext_sales_price) ws_item_rev
                FROM   web_sales,
                        item,
                        date_dim
                WHERE  ws_item_sk = i_item_sk
                        AND d_date IN (SELECT d_date
                                    FROM   date_dim
                                    WHERE  d_week_seq = (SELECT d_week_seq
                                                            FROM   date_dim
                                                            WHERE  d_date = '{sales_date}'
                                                        ))
                        AND ws_sold_date_sk = d_date_sk
                GROUP  BY i_item_id)
        SELECT ss_items.item_id,
                    ss_item_rev,
                    ss_item_rev / ( ss_item_rev + cs_item_rev + ws_item_rev ) / 3 *
                    100 ss_dev,
                    cs_item_rev,
                    cs_item_rev / ( ss_item_rev + cs_item_rev + ws_item_rev ) / 3 *
                    100 cs_dev,
                    ws_item_rev,
                    ws_item_rev / ( ss_item_rev + cs_item_rev + ws_item_rev ) / 3 *
                    100 ws_dev,
                    ( ss_item_rev + cs_item_rev + ws_item_rev ) / 3
                    average
        FROM   ss_items,
            cs_items,
            ws_items
        WHERE  ss_items.item_id = cs_items.item_id
            AND ss_items.item_id = ws_items.item_id
            AND ss_item_rev BETWEEN 0.9 * cs_item_rev AND 1.1 * cs_item_rev
            AND ss_item_rev BETWEEN 0.9 * ws_item_rev AND 1.1 * ws_item_rev
            AND cs_item_rev BETWEEN 0.9 * ss_item_rev AND 1.1 * ss_item_rev
            AND cs_item_rev BETWEEN 0.9 * ws_item_rev AND 1.1 * ws_item_rev
            AND ws_item_rev BETWEEN 0.9 * ss_item_rev AND 1.1 * ss_item_rev
            AND ws_item_rev BETWEEN 0.9 * cs_item_rev AND 1.1 * cs_item_rev
        ORDER  BY ss_items.item_id,
                ss_item_rev
        LIMIT 100;
    """


def _build_sales_agg(
    sales: pl.LazyFrame,
    item_lf: pl.LazyFrame,
    week_dates: pl.LazyFrame,
    sales_item_sk: str,
    sales_date_sk: str,
    price_col: str,
    alias: str,
) -> pl.LazyFrame:
    return (
        sales.join(item_lf, left_on=sales_item_sk, right_on="i_item_sk")
        .join(week_dates, left_on=sales_date_sk, right_on="d_date_sk")
        .group_by("i_item_id")
        .agg(pl.col(price_col).sum().alias(alias))
        .select(pl.col("i_item_id").alias("item_id"), pl.col(alias))
    )


def polars_impl(run_config: RunConfig) -> QueryResult:
    """Query 58."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=58,
        qualification=run_config.qualification,
    )

    sales_date = params["sales_date"]

    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    catalog_sales = get_data(
        run_config.dataset_path, "catalog_sales", run_config.suffix
    )
    web_sales = get_data(run_config.dataset_path, "web_sales", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)

    # Parse sales_date (format: "YYYY-MM-DD")
    year, month, day = map(int, sales_date.split("-"))
    target_week = date_dim.filter(pl.col("d_date") == pl.date(year, month, day)).select(
        "d_week_seq"
    )
    week_dates = date_dim.join(target_week, on="d_week_seq", how="inner").select(
        ["d_date_sk"]
    )

    ss_items = _build_sales_agg(
        store_sales,
        item,
        week_dates,
        "ss_item_sk",
        "ss_sold_date_sk",
        "ss_ext_sales_price",
        "ss_item_rev",
    )
    cs_items = _build_sales_agg(
        catalog_sales,
        item,
        week_dates,
        "cs_item_sk",
        "cs_sold_date_sk",
        "cs_ext_sales_price",
        "cs_item_rev",
    )
    ws_items = _build_sales_agg(
        web_sales,
        item,
        week_dates,
        "ws_item_sk",
        "ws_sold_date_sk",
        "ws_ext_sales_price",
        "ws_item_rev",
    )

    sort_by = {"item_id": False, "ss_item_rev": False}
    limit = 100
    return QueryResult(
        frame=(
            ss_items.join(cs_items, on="item_id", how="inner")
            .join(ws_items, on="item_id", how="inner")
            .filter(
                (
                    pl.col("ss_item_rev").is_between(
                        0.9 * pl.col("cs_item_rev"), 1.1 * pl.col("cs_item_rev")
                    )
                )
                & (
                    pl.col("ss_item_rev").is_between(
                        0.9 * pl.col("ws_item_rev"), 1.1 * pl.col("ws_item_rev")
                    )
                )
                & (
                    pl.col("cs_item_rev").is_between(
                        0.9 * pl.col("ss_item_rev"), 1.1 * pl.col("ss_item_rev")
                    )
                )
                & (
                    pl.col("cs_item_rev").is_between(
                        0.9 * pl.col("ws_item_rev"), 1.1 * pl.col("ws_item_rev")
                    )
                )
                & (
                    pl.col("ws_item_rev").is_between(
                        0.9 * pl.col("ss_item_rev"), 1.1 * pl.col("ss_item_rev")
                    )
                )
                & (
                    pl.col("ws_item_rev").is_between(
                        0.9 * pl.col("cs_item_rev"), 1.1 * pl.col("cs_item_rev")
                    )
                )
            )
            .with_columns(
                [
                    (
                        pl.col("ss_item_rev")
                        + pl.col("cs_item_rev")
                        + pl.col("ws_item_rev")
                    ).alias("total_rev"),
                    (
                        (
                            pl.col("ss_item_rev")
                            + pl.col("cs_item_rev")
                            + pl.col("ws_item_rev")
                        )
                        / 3
                    ).alias("average"),
                ]
            )
            .with_columns(
                [
                    (pl.col("ss_item_rev") / pl.col("total_rev") / 3 * 100).alias(
                        "ss_dev"
                    ),
                    (pl.col("cs_item_rev") / pl.col("total_rev") / 3 * 100).alias(
                        "cs_dev"
                    ),
                    (pl.col("ws_item_rev") / pl.col("total_rev") / 3 * 100).alias(
                        "ws_dev"
                    ),
                ]
            )
            .select(
                [
                    "item_id",
                    "ss_item_rev",
                    "ss_dev",
                    "cs_item_rev",
                    "cs_dev",
                    "ws_item_rev",
                    "ws_dev",
                    "average",
                ]
            )
            .sort(sort_by.keys(), nulls_last=True)
            .limit(limit)
        ),
        sort_by=list(sort_by.items()),
        limit=limit,
    )
