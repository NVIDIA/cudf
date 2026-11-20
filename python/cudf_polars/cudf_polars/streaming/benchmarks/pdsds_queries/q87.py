# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Query 87."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.streaming.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.streaming.benchmarks.utils import QueryResult, get_data

if TYPE_CHECKING:
    from cudf_polars.streaming.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 87."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=87,
        qualification=run_config.qualification,
    )

    d_month_seq = params["d_month_seq"]

    return f"""
    select count(*)
    from ((select distinct c_last_name, c_first_name, d_date
           from store_sales, date_dim, customer
           where store_sales.ss_sold_date_sk = date_dim.d_date_sk
             and store_sales.ss_customer_sk = customer.c_customer_sk
             and d_month_seq between {d_month_seq} and {d_month_seq}+11)
           except
          (select distinct c_last_name, c_first_name, d_date
           from catalog_sales, date_dim, customer
           where catalog_sales.cs_sold_date_sk = date_dim.d_date_sk
             and catalog_sales.cs_bill_customer_sk = customer.c_customer_sk
             and d_month_seq between {d_month_seq} and {d_month_seq}+11)
           except
          (select distinct c_last_name, c_first_name, d_date
           from web_sales, date_dim, customer
           where web_sales.ws_sold_date_sk = date_dim.d_date_sk
             and web_sales.ws_bill_customer_sk = customer.c_customer_sk
             and d_month_seq between {d_month_seq} and {d_month_seq}+11)
    ) cool_cust;
    """


def polars_impl(run_config: RunConfig) -> QueryResult:
    """Query 87."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=87,
        qualification=run_config.qualification,
    )

    d_month_seq = params["d_month_seq"]

    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    catalog_sales = get_data(
        run_config.dataset_path, "catalog_sales", run_config.suffix
    )
    web_sales = get_data(run_config.dataset_path, "web_sales", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    customer = get_data(run_config.dataset_path, "customer", run_config.suffix)
    store_customers = (
        store_sales.filter(
            pl.col("ss_sold_date_sk").is_not_null()
            & pl.col("ss_customer_sk").is_not_null()
        )
        .join(
            date_dim.filter(
                (pl.col("d_month_seq") >= d_month_seq)
                & (pl.col("d_month_seq") <= d_month_seq + 11)
            ),
            left_on="ss_sold_date_sk",
            right_on="d_date_sk",
            how="inner",
        )
        .join(customer, left_on="ss_customer_sk", right_on="c_customer_sk", how="inner")
        .select(["c_last_name", "c_first_name", "d_date"])
        .unique()
    )
    catalog_customers = (
        catalog_sales.filter(
            pl.col("cs_sold_date_sk").is_not_null()
            & pl.col("cs_bill_customer_sk").is_not_null()
        )
        .join(
            date_dim.filter(
                (pl.col("d_month_seq") >= d_month_seq)
                & (pl.col("d_month_seq") <= d_month_seq + 11)
            ),
            left_on="cs_sold_date_sk",
            right_on="d_date_sk",
            how="inner",
        )
        .join(
            customer,
            left_on="cs_bill_customer_sk",
            right_on="c_customer_sk",
            how="inner",
        )
        .select(["c_last_name", "c_first_name", "d_date"])
        .unique()
    )
    web_customers = (
        web_sales.filter(
            pl.col("ws_sold_date_sk").is_not_null()
            & pl.col("ws_bill_customer_sk").is_not_null()
        )
        .join(
            date_dim.filter(
                (pl.col("d_month_seq") >= d_month_seq)
                & (pl.col("d_month_seq") <= d_month_seq + 11)
            ),
            left_on="ws_sold_date_sk",
            right_on="d_date_sk",
            how="inner",
        )
        .join(
            customer,
            left_on="ws_bill_customer_sk",
            right_on="c_customer_sk",
            how="inner",
        )
        .select(["c_last_name", "c_first_name", "d_date"])
        .unique()
    )
    join_keys = ["c_last_name", "c_first_name", "d_date"]
    result = store_customers.join(
        catalog_customers, on=join_keys, how="anti", nulls_equal=True
    ).join(web_customers, on=join_keys, how="anti", nulls_equal=True)
    return QueryResult(
        # Use pl.col("d_date").len() instead of pl.len() to avoid the zero-column
        # streaming chunk bug (https://github.com/rapidsai/cudf/issues/21428)
        frame=result.select(
            [pl.col("d_date").len().cast(pl.Int64).alias("count_star()")]
        ),
        sort_by=[],
        limit=None,
    )
