# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 25."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.streaming.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.streaming.benchmarks.utils import QueryResult, get_data

if TYPE_CHECKING:
    from cudf_polars.streaming.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 25."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=25,
        qualification=run_config.qualification,
    )

    year = params["year"]
    agg = params["agg"]

    return f"""
    SELECT i_item_id,
                   i_item_desc,
                   s_store_id,
                   s_store_name,
                   {agg}(ss_net_profit) AS store_sales_profit,
                   {agg}(sr_net_loss)   AS store_returns_loss,
                   {agg}(cs_net_profit) AS catalog_sales_profit
    FROM   store_sales,
           store_returns,
           catalog_sales,
           date_dim d1,
           date_dim d2,
           date_dim d3,
           store,
           item
    WHERE  d1.d_moy = 4
           AND d1.d_year = {year}
           AND d1.d_date_sk = ss_sold_date_sk
           AND i_item_sk = ss_item_sk
           AND s_store_sk = ss_store_sk
           AND ss_customer_sk = sr_customer_sk
           AND ss_item_sk = sr_item_sk
           AND ss_ticket_number = sr_ticket_number
           AND sr_returned_date_sk = d2.d_date_sk
           AND d2.d_moy BETWEEN 4 AND 10
           AND d2.d_year = {year}
           AND sr_customer_sk = cs_bill_customer_sk
           AND sr_item_sk = cs_item_sk
           AND cs_sold_date_sk = d3.d_date_sk
           AND d3.d_moy BETWEEN 4 AND 10
           AND d3.d_year = {year}
    GROUP  BY i_item_id,
              i_item_desc,
              s_store_id,
              s_store_name
    ORDER  BY i_item_id,
              i_item_desc,
              s_store_id,
              s_store_name
    LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> QueryResult:
    """Query 25."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=25,
        qualification=run_config.qualification,
    )

    year = params["year"]
    agg = params["agg"]

    # Map SQL aggregation functions to Polars method names
    polars_agg = "mean" if agg == "avg" else "std" if agg == "stddev_samp" else agg

    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    store_returns = get_data(
        run_config.dataset_path, "store_returns", run_config.suffix
    )
    catalog_sales = get_data(
        run_config.dataset_path, "catalog_sales", run_config.suffix
    )
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    store = get_data(run_config.dataset_path, "store", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)

    sort_by = {
        "i_item_id": False,
        "i_item_desc": False,
        "s_store_id": False,
        "s_store_name": False,
    }
    limit = 100

    # d1: only April of the target year — very selective (~1/60 of date_dim rows).
    # d2/d3: from April to October of the target year: same condition, one pre-filtered frame.
    d1_dates = date_dim.filter(
        (pl.col("d_moy") == 4) & (pl.col("d_year") == year)
    ).select("d_date_sk")
    d2_d3_dates = date_dim.filter(
        pl.col("d_moy").is_between(4, 10) & (pl.col("d_year") == year)
    ).select("d_date_sk")

    # store_returns [6] ≤ broadcast limit: filter to qualifying return dates first,
    # then extract (customer, item) pairs to pre-filter ss and cs before shuffle joins.
    store_returns_filtered = store_returns.join(
        d2_d3_dates, left_on="sr_returned_date_sk", right_on="d_date_sk", how="semi"
    ).select(["sr_customer_sk", "sr_item_sk", "sr_ticket_number", "sr_net_loss"])
    sr_customer_item = store_returns_filtered.select(["sr_customer_sk", "sr_item_sk"])

    store_sales_filtered = (
        store_sales.join(
            d1_dates, left_on="ss_sold_date_sk", right_on="d_date_sk", how="semi"
        )
        .join(
            sr_customer_item,
            left_on=["ss_customer_sk", "ss_item_sk"],
            right_on=["sr_customer_sk", "sr_item_sk"],
            how="semi",
        )
        .select(
            [
                "ss_customer_sk",
                "ss_item_sk",
                "ss_store_sk",
                "ss_ticket_number",
                "ss_net_profit",
            ]
        )
        .join(
            item.select(["i_item_sk", "i_item_id", "i_item_desc"]),
            left_on="ss_item_sk",
            right_on="i_item_sk",
        )
        .join(
            store.select(["s_store_sk", "s_store_id", "s_store_name"]),
            left_on="ss_store_sk",
            right_on="s_store_sk",
        )
        .select(
            [
                "ss_customer_sk",
                "ss_item_sk",
                "ss_ticket_number",
                "ss_net_profit",
                "i_item_id",
                "i_item_desc",
                "s_store_id",
                "s_store_name",
            ]
        )
    )

    catalog_sales_filtered = (
        catalog_sales.join(
            d2_d3_dates, left_on="cs_sold_date_sk", right_on="d_date_sk", how="semi"
        )
        .join(
            sr_customer_item,
            left_on=["cs_bill_customer_sk", "cs_item_sk"],
            right_on=["sr_customer_sk", "sr_item_sk"],
            how="semi",
        )
        .select(["cs_bill_customer_sk", "cs_item_sk", "cs_net_profit"])
    )

    return QueryResult(
        frame=(
            store_sales_filtered.join(
                store_returns_filtered,
                left_on=["ss_customer_sk", "ss_item_sk", "ss_ticket_number"],
                right_on=["sr_customer_sk", "sr_item_sk", "sr_ticket_number"],
            )
            .select(
                [
                    "ss_customer_sk",
                    "ss_item_sk",
                    "ss_net_profit",
                    "sr_net_loss",
                    "i_item_id",
                    "i_item_desc",
                    "s_store_id",
                    "s_store_name",
                ]
            )
            .join(
                catalog_sales_filtered,
                left_on=["ss_customer_sk", "ss_item_sk"],
                right_on=["cs_bill_customer_sk", "cs_item_sk"],
            )
            .group_by(["i_item_id", "i_item_desc", "s_store_id", "s_store_name"])
            .agg(
                [
                    getattr(pl.col("ss_net_profit"), polars_agg)().alias(
                        "store_sales_profit"
                    ),
                    getattr(pl.col("sr_net_loss"), polars_agg)().alias(
                        "store_returns_loss"
                    ),
                    getattr(pl.col("cs_net_profit"), polars_agg)().alias(
                        "catalog_sales_profit"
                    ),
                ]
            )
            .sort(sort_by.keys(), nulls_last=True)
            .limit(limit)
        ),
        sort_by=list(sort_by.items()),
        limit=limit,
    )
