# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 83."""

from __future__ import annotations

from datetime import date as _date
from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.streaming.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.streaming.benchmarks.utils import QueryResult, get_data

if TYPE_CHECKING:
    from cudf_polars.streaming.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 83."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=83,
        qualification=run_config.qualification,
    )
    dates = params["dates"]

    dates_str = ", ".join(f"'{d}'" for d in dates)

    return f"""
    WITH sr_items
         AS (SELECT i_item_id               item_id,
                    Sum(sr_return_quantity) sr_item_qty
             FROM   store_returns,
                    item,
                    date_dim
             WHERE  sr_item_sk = i_item_sk
                    AND d_date IN (SELECT d_date
                                   FROM   date_dim
                                   WHERE  d_week_seq IN (SELECT d_week_seq
                                                         FROM   date_dim
                                                         WHERE
                                          d_date IN ( {dates_str} )))
                    AND sr_returned_date_sk = d_date_sk
             GROUP  BY i_item_id),
         cr_items
         AS (SELECT i_item_id               item_id,
                    Sum(cr_return_quantity) cr_item_qty
             FROM   catalog_returns,
                    item,
                    date_dim
             WHERE  cr_item_sk = i_item_sk
                    AND d_date IN (SELECT d_date
                                   FROM   date_dim
                                   WHERE  d_week_seq IN (SELECT d_week_seq
                                                         FROM   date_dim
                                                         WHERE
                                          d_date IN ( {dates_str} )))
                    AND cr_returned_date_sk = d_date_sk
             GROUP  BY i_item_id),
         wr_items
         AS (SELECT i_item_id               item_id,
                    Sum(wr_return_quantity) wr_item_qty
             FROM   web_returns,
                    item,
                    date_dim
             WHERE  wr_item_sk = i_item_sk
                    AND d_date IN (SELECT d_date
                                   FROM   date_dim
                                   WHERE  d_week_seq IN (SELECT d_week_seq
                                                         FROM   date_dim
                                                         WHERE
                                          d_date IN ( {dates_str} )))
                    AND wr_returned_date_sk = d_date_sk
             GROUP  BY i_item_id)
    SELECT sr_items.item_id,
                   sr_item_qty,
                   sr_item_qty / ( sr_item_qty + cr_item_qty + wr_item_qty ) / 3.0 *
                   100 sr_dev,
                   cr_item_qty,
                   cr_item_qty / ( sr_item_qty + cr_item_qty + wr_item_qty ) / 3.0 *
                   100 cr_dev,
                   wr_item_qty,
                   wr_item_qty / ( sr_item_qty + cr_item_qty + wr_item_qty ) / 3.0 *
                   100 wr_dev,
                   ( sr_item_qty + cr_item_qty + wr_item_qty ) / 3.0
                   average
    FROM   sr_items,
           cr_items,
           wr_items
    WHERE  sr_items.item_id = cr_items.item_id
           AND sr_items.item_id = wr_items.item_id
    ORDER  BY sr_items.item_id,
              sr_item_qty
    LIMIT 100;
    """


def q83_segment(
    returns: pl.LazyFrame,
    item: pl.LazyFrame,
    dates: pl.LazyFrame,
    *,
    item_key: str,
    returned_date_key: str,
    qty_col: str,
    out_qty_name: str,
) -> pl.LazyFrame:
    """Aggregate a returns table to per-item quantities over selected dates."""
    return (
        returns.join(
            item.select(["i_item_sk", "i_item_id"]),
            left_on=item_key,
            right_on="i_item_sk",
        )
        .join(dates, left_on=returned_date_key, right_on="d_date_sk")
        .group_by("i_item_id")
        .agg(
            [
                pl.col(qty_col).count().alias(f"{out_qty_name}_count"),
                pl.col(qty_col).sum().alias(f"{out_qty_name}_sum"),
            ]
        )
        .with_columns(
            pl.when(pl.col(f"{out_qty_name}_count") > 0)
            .then(pl.col(f"{out_qty_name}_sum"))
            .otherwise(None)
            .alias(out_qty_name)
        )
        .select(["i_item_id", out_qty_name])
    )


def polars_impl(run_config: RunConfig) -> QueryResult:
    """Query 83."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=83,
        qualification=run_config.qualification,
    )

    dates_list = params["dates"]
    # Parse dates like '2000-06-30' to date objects
    anchor_dates = []
    for date_str in dates_list:
        year, month, day = map(int, date_str.split("-"))
        anchor_dates.append(_date(year, month, day))

    store_returns = get_data(
        run_config.dataset_path, "store_returns", run_config.suffix
    )
    catalog_returns = get_data(
        run_config.dataset_path, "catalog_returns", run_config.suffix
    )
    web_returns = get_data(run_config.dataset_path, "web_returns", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    weeks = (
        date_dim.filter(pl.col("d_date").is_in(anchor_dates))
        .select("d_week_seq")
        .unique()
    )
    dates = date_dim.join(weeks, on="d_week_seq").select("d_date_sk").unique()

    sr_items = q83_segment(
        store_returns,
        item,
        dates,
        item_key="sr_item_sk",
        returned_date_key="sr_returned_date_sk",
        qty_col="sr_return_quantity",
        out_qty_name="sr_item_qty",
    )
    cr_items = q83_segment(
        catalog_returns,
        item,
        dates,
        item_key="cr_item_sk",
        returned_date_key="cr_returned_date_sk",
        qty_col="cr_return_quantity",
        out_qty_name="cr_item_qty",
    )
    wr_items = q83_segment(
        web_returns,
        item,
        dates,
        item_key="wr_item_sk",
        returned_date_key="wr_returned_date_sk",
        qty_col="wr_return_quantity",
        out_qty_name="wr_item_qty",
    )

    sort_by = {"item_id": False, "sr_item_qty": False}
    limit = 100
    return QueryResult(
        frame=(
            sr_items.join(cr_items, on="i_item_id")
            .join(wr_items, on="i_item_id")
            .with_columns(
                (
                    pl.col("sr_item_qty")
                    + pl.col("cr_item_qty")
                    + pl.col("wr_item_qty")
                ).alias("total_qty")
            )
            .with_columns(
                [
                    (pl.col("total_qty") / 3.0).cast(pl.Float64).alias("average"),
                    (pl.col("sr_item_qty") / pl.col("total_qty") / 3.0 * 100)
                    .cast(pl.Float64)
                    .alias("sr_dev"),
                    (pl.col("cr_item_qty") / pl.col("total_qty") / 3.0 * 100)
                    .cast(pl.Float64)
                    .alias("cr_dev"),
                    (pl.col("wr_item_qty") / pl.col("total_qty") / 3.0 * 100)
                    .cast(pl.Float64)
                    .alias("wr_dev"),
                ]
            )
            .select(
                [
                    pl.col("i_item_id").alias("item_id"),
                    "sr_item_qty",
                    "sr_dev",
                    "cr_item_qty",
                    "cr_dev",
                    "wr_item_qty",
                    "wr_dev",
                    "average",
                ]
            )
            .sort(["item_id", "sr_item_qty"], nulls_last=True)
            .limit(limit)
        ),
        sort_by=list(sort_by.items()),
        limit=limit,
    )
