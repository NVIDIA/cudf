# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 82."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.streaming.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.streaming.benchmarks.utils import QueryResult, get_data

if TYPE_CHECKING:
    from cudf_polars.streaming.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 82."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=82,
        qualification=run_config.qualification,
    )
    price = params["price"]
    sdate = params["sdate"]
    manufact = params["manufact"]
    inv_min = params["inv_min"]
    inv_max = params["inv_max"]

    manufact_str = ", ".join(str(m) for m in manufact)

    return f"""
    -- start query 82 in stream 0 using template query82.tpl
    SELECT
             i_item_id ,
             i_item_desc ,
             i_current_price
    FROM     item,
             inventory,
             date_dim,
             store_sales
    WHERE    i_current_price BETWEEN {price} AND      {price}+30
    AND      inv_item_sk = i_item_sk
    AND      d_date_sk=inv_date_sk
    AND      d_date BETWEEN Cast('{sdate}' AS DATE) AND      (
                      Cast('{sdate}' AS DATE) + INTERVAL '60' day)
    AND      i_manufact_id IN ({manufact_str})
    AND      inv_quantity_on_hand BETWEEN {inv_min} AND      {inv_max}
    AND      ss_item_sk = i_item_sk
    GROUP BY i_item_id,
             i_item_desc,
             i_current_price
    ORDER BY i_item_id
    LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> QueryResult:
    """Query 82."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=82,
        qualification=run_config.qualification,
    )

    price = params["price"]
    sdate = params["sdate"]
    manufact = params["manufact"]
    inv_min = params["inv_min"]
    inv_max = params["inv_max"]

    year, month, day = map(int, sdate.split("-"))
    start_date = pl.date(year, month, day)
    # Add 60 days to start_date
    from datetime import date, timedelta

    end_date_obj = date(year, month, day) + timedelta(days=60)
    end_date = pl.date(end_date_obj.year, end_date_obj.month, end_date_obj.day)

    item = get_data(run_config.dataset_path, "item", run_config.suffix)
    inventory = get_data(run_config.dataset_path, "inventory", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    sort_by = {"i_item_id": False}
    limit = 100
    return QueryResult(
        frame=(
            item.join(inventory, left_on="i_item_sk", right_on="inv_item_sk")
            .join(date_dim, left_on="inv_date_sk", right_on="d_date_sk")
            .join(store_sales, left_on="i_item_sk", right_on="ss_item_sk")
            .filter(
                (pl.col("i_current_price") >= price)
                & (pl.col("i_current_price") <= price + 30)
                & (pl.col("d_date") >= start_date)
                & (pl.col("d_date") <= end_date)
                & pl.col("i_manufact_id").is_in(manufact)
                & (pl.col("inv_quantity_on_hand") >= inv_min)
                & (pl.col("inv_quantity_on_hand") <= inv_max)
            )
            .group_by(["i_item_id", "i_item_desc", "i_current_price"])
            .agg([])
            .select(["i_item_id", "i_item_desc", "i_current_price"])
            .sort("i_item_id", nulls_last=True)
            .limit(limit)
        ),
        sort_by=list(sort_by.items()),
        limit=limit,
    )
