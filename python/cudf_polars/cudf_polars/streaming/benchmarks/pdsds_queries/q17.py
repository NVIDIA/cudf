# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 17."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.streaming.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.streaming.benchmarks.utils import QueryResult, get_data

if TYPE_CHECKING:
    from cudf_polars.streaming.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 17."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=17,
        qualification=run_config.qualification,
    )

    year = params["year"]

    return f"""
    SELECT i_item_id,
                   i_item_desc,
                   s_state,
                   Count(ss_quantity)                                        AS
                   store_sales_quantitycount,
                   Avg(ss_quantity)                                          AS
                   store_sales_quantityave,
                   Stddev_samp(ss_quantity)                                  AS
                   store_sales_quantitystdev,
                   Stddev_samp(ss_quantity) / Avg(ss_quantity)               AS
                   store_sales_quantitycov,
                   Count(sr_return_quantity)                                 AS
                   store_returns_quantitycount,
                   Avg(sr_return_quantity)                                   AS
                   store_returns_quantityave,
                   Stddev_samp(sr_return_quantity)                           AS
                   store_returns_quantitystdev,
                   Stddev_samp(sr_return_quantity) / Avg(sr_return_quantity) AS
                   store_returns_quantitycov,
                   Count(cs_quantity)                                        AS
                   catalog_sales_quantitycount,
                   Avg(cs_quantity)                                          AS
                   catalog_sales_quantityave,
                   Stddev_samp(cs_quantity) / Avg(cs_quantity)               AS
                   catalog_sales_quantitystdev,
                   Stddev_samp(cs_quantity) / Avg(cs_quantity)               AS
                   catalog_sales_quantitycov
    FROM   store_sales,
           store_returns,
           catalog_sales,
           date_dim d1,
           date_dim d2,
           date_dim d3,
           store,
           item
    WHERE  d1.d_quarter_name = '{year}Q1'
           AND d1.d_date_sk = ss_sold_date_sk
           AND i_item_sk = ss_item_sk
           AND s_store_sk = ss_store_sk
           AND ss_customer_sk = sr_customer_sk
           AND ss_item_sk = sr_item_sk
           AND ss_ticket_number = sr_ticket_number
           AND sr_returned_date_sk = d2.d_date_sk
           AND d2.d_quarter_name IN ( '{year}Q1', '{year}Q2', '{year}Q3' )
           AND sr_customer_sk = cs_bill_customer_sk
           AND sr_item_sk = cs_item_sk
           AND cs_sold_date_sk = d3.d_date_sk
           AND d3.d_quarter_name IN ( '{year}Q1', '{year}Q2', '{year}Q3' )
    GROUP  BY i_item_id,
              i_item_desc,
              s_state
    ORDER  BY i_item_id,
              i_item_desc,
              s_state
    LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> QueryResult:
    """Query 17."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=17,
        qualification=run_config.qualification,
    )

    year = params["year"]

    # Load tables
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

    sort_by = {"i_item_id": False, "i_item_desc": False, "s_state": False}
    limit = 100

    q1 = f"{year}Q1"
    q1_q3 = [f"{year}Q1", f"{year}Q2", f"{year}Q3"]

    # Pre-filter date_dim to only qualifying d_date_sk values.
    d1_dates = date_dim.filter(pl.col("d_quarter_name") == q1).select("d_date_sk")
    d_q3_dates = date_dim.filter(pl.col("d_quarter_name").is_in(q1_q3)).select(
        "d_date_sk"
    )

    # store_returns has [6] partitions — at the broadcast limit. Filter it to Q1-Q3 dates
    # first, then use the (customer, item) pairs it contains to pre-filter both store_sales
    # and catalog_sales before those larger tables enter the expensive shuffle joins.
    store_returns_filtered = store_returns.join(
        d_q3_dates, left_on="sr_returned_date_sk", right_on="d_date_sk", how="semi"
    ).select(["sr_customer_sk", "sr_item_sk", "sr_ticket_number", "sr_return_quantity"])

    # (customer, item) pairs present in any qualifying store return; stays at [6] partitions
    # so broadcast is free. Polars will CACHE this shared subplan.
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
                "ss_quantity",
            ]
        )
        .join(
            item.select(["i_item_sk", "i_item_id", "i_item_desc"]),
            left_on="ss_item_sk",
            right_on="i_item_sk",
        )
        .join(
            store.select(["s_store_sk", "s_state"]),
            left_on="ss_store_sk",
            right_on="s_store_sk",
        )
        .select(
            [
                "ss_customer_sk",
                "ss_item_sk",
                "ss_ticket_number",
                "ss_quantity",
                "i_item_id",
                "i_item_desc",
                "s_state",
            ]
        )
    )

    catalog_sales_filtered = (
        catalog_sales.join(
            d_q3_dates, left_on="cs_sold_date_sk", right_on="d_date_sk", how="semi"
        )
        .join(
            sr_customer_item,
            left_on=["cs_bill_customer_sk", "cs_item_sk"],
            right_on=["sr_customer_sk", "sr_item_sk"],
            how="semi",
        )
        .select(["cs_bill_customer_sk", "cs_item_sk", "cs_quantity"])
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
                    "ss_quantity",
                    "sr_return_quantity",
                    "i_item_id",
                    "i_item_desc",
                    "s_state",
                ]
            )
            .join(
                catalog_sales_filtered,
                left_on=["ss_customer_sk", "ss_item_sk"],
                right_on=["cs_bill_customer_sk", "cs_item_sk"],
            )
            .group_by(["i_item_id", "i_item_desc", "s_state"])
            .agg(
                [
                    pl.col("ss_quantity").count().alias("store_sales_quantitycount"),
                    pl.col("ss_quantity").mean().alias("store_sales_quantityave"),
                    pl.col("ss_quantity").std().alias("store_sales_quantitystdev"),
                    (pl.col("ss_quantity").std() / pl.col("ss_quantity").mean()).alias(
                        "store_sales_quantitycov"
                    ),
                    pl.col("sr_return_quantity")
                    .count()
                    .alias("store_returns_quantitycount"),
                    pl.col("sr_return_quantity")
                    .mean()
                    .alias("store_returns_quantityave"),
                    pl.col("sr_return_quantity")
                    .std()
                    .alias("store_returns_quantitystdev"),
                    (
                        pl.col("sr_return_quantity").std()
                        / pl.col("sr_return_quantity").mean()
                    ).alias("store_returns_quantitycov"),
                    pl.col("cs_quantity").count().alias("catalog_sales_quantitycount"),
                    pl.col("cs_quantity").mean().alias("catalog_sales_quantityave"),
                    pl.col("cs_quantity").std().alias("catalog_sales_quantitystdev"),
                    (pl.col("cs_quantity").std() / pl.col("cs_quantity").mean()).alias(
                        "catalog_sales_quantitycov"
                    ),
                ]
            )
            .sort(sort_by.keys(), nulls_last=True)
            .limit(limit)
        ),
        sort_by=list(sort_by.items()),
        limit=limit,
    )
