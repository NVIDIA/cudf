# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 37."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.streaming.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.streaming.benchmarks.utils import QueryResult, get_data

if TYPE_CHECKING:
    from cudf_polars.streaming.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 37."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=37,
        qualification=run_config.qualification,
    )

    price = params["price"]
    manufact = params["manufact"]
    invdate = params["invdate"]

    # Format manufacturer list for SQL IN clause
    manufact_list = ", ".join(str(m) for m in manufact)

    return f"""
    SELECT
             i_item_id ,
             i_item_desc ,
             i_current_price
    FROM     item,
             inventory,
             date_dim,
             catalog_sales
    WHERE    i_current_price BETWEEN {price} AND      {price} + 30
    AND      inv_item_sk = i_item_sk
    AND      d_date_sk=inv_date_sk
    AND      d_date BETWEEN Cast('{invdate}' AS DATE) AND      (
                      Cast('{invdate}' AS DATE) + INTERVAL '60' day)
    AND      i_manufact_id IN ({manufact_list})
    AND      inv_quantity_on_hand BETWEEN 100 AND      500
    AND      cs_item_sk = i_item_sk
    GROUP BY i_item_id,
             i_item_desc,
             i_current_price
    ORDER BY i_item_id
    LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> QueryResult:
    """Query 37."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=37,
        qualification=run_config.qualification,
    )

    price = params["price"]
    manufact = params["manufact"]
    invdate = params["invdate"]

    # Calculate end date (invdate + 60 days)
    from datetime import datetime, timedelta

    start_date_obj = datetime.strptime(invdate, "%Y-%m-%d")
    end_date_obj = start_date_obj + timedelta(days=60)

    start_date = pl.date(start_date_obj.year, start_date_obj.month, start_date_obj.day)
    end_date = pl.date(end_date_obj.year, end_date_obj.month, end_date_obj.day)

    # Load tables
    item = get_data(run_config.dataset_path, "item", run_config.suffix)
    inventory = get_data(run_config.dataset_path, "inventory", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    catalog_sales = get_data(
        run_config.dataset_path, "catalog_sales", run_config.suffix
    )
    sort_by = {"i_item_id": False}
    limit = 100
    return QueryResult(
        frame=(
            item.join(inventory, left_on="i_item_sk", right_on="inv_item_sk")
            .join(date_dim, left_on="inv_date_sk", right_on="d_date_sk")
            .join(catalog_sales, left_on="i_item_sk", right_on="cs_item_sk")
            .filter(
                (pl.col("i_current_price").is_between(price, price + 30))
                & (pl.col("i_manufact_id").is_in(manufact))
                & (pl.col("inv_quantity_on_hand").is_between(100, 500))
                & (pl.col("d_date").is_between(start_date, end_date))
            )
            .group_by(["i_item_id", "i_item_desc", "i_current_price"])
            .agg([])
            .sort(sort_by.keys(), nulls_last=True)
            .limit(limit)
        ),
        sort_by=list(sort_by.items()),
        limit=limit,
    )
