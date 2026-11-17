# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Query 38."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.streaming.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.streaming.benchmarks.utils import QueryResult, get_data

if TYPE_CHECKING:
    from cudf_polars.streaming.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 38."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=38,
        qualification=run_config.qualification,
    )

    dms = params["dms"]

    return f"""
    SELECT Count(*)
    FROM   (SELECT DISTINCT c_last_name,
                            c_first_name,
                            d_date
            FROM   store_sales,
                   date_dim,
                   customer
            WHERE  store_sales.ss_sold_date_sk = date_dim.d_date_sk
                   AND store_sales.ss_customer_sk = customer.c_customer_sk
                   AND d_month_seq BETWEEN {dms} AND {dms} + 11
            INTERSECT
            SELECT DISTINCT c_last_name,
                            c_first_name,
                            d_date
            FROM   catalog_sales,
                   date_dim,
                   customer
            WHERE  catalog_sales.cs_sold_date_sk = date_dim.d_date_sk
                   AND catalog_sales.cs_bill_customer_sk = customer.c_customer_sk
                   AND d_month_seq BETWEEN {dms} AND {dms} + 11
            INTERSECT
            SELECT DISTINCT c_last_name,
                            c_first_name,
                            d_date
            FROM   web_sales,
                   date_dim,
                   customer
            WHERE  web_sales.ws_sold_date_sk = date_dim.d_date_sk
                   AND web_sales.ws_bill_customer_sk = customer.c_customer_sk
                   AND d_month_seq BETWEEN {dms} AND {dms} + 11) hot_cust
    LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> QueryResult:
    """Query 38."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=38,
        qualification=run_config.qualification,
    )

    dms = params["dms"]

    # Load tables
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    catalog_sales = get_data(
        run_config.dataset_path, "catalog_sales", run_config.suffix
    )
    web_sales = get_data(run_config.dataset_path, "web_sales", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    customer = get_data(run_config.dataset_path, "customer", run_config.suffix)
    # Filter date_dim for the specified month sequence range
    date_filter = date_dim.filter(pl.col("d_month_seq").is_between(dms, dms + 11))
    # Store sales customers with names and dates
    store_customers = (
        store_sales.join(date_filter, left_on="ss_sold_date_sk", right_on="d_date_sk")
        .join(customer, left_on="ss_customer_sk", right_on="c_customer_sk")
        .select(["c_last_name", "c_first_name", "d_date"])
        .unique()
    )
    # Catalog sales customers with names and dates
    catalog_customers = (
        catalog_sales.join(date_filter, left_on="cs_sold_date_sk", right_on="d_date_sk")
        .join(customer, left_on="cs_bill_customer_sk", right_on="c_customer_sk")
        .select(["c_last_name", "c_first_name", "d_date"])
        .unique()
    )
    # Web sales customers with names and dates
    web_customers = (
        web_sales.join(date_filter, left_on="ws_sold_date_sk", right_on="d_date_sk")
        .join(customer, left_on="ws_bill_customer_sk", right_on="c_customer_sk")
        .select(["c_last_name", "c_first_name", "d_date"])
        .unique()
    )
    # Implement INTERSECT via semi-joins: keep only store rows that also appear
    # in catalog and web (nulls_equal so NULL keys match, as SQL INTERSECT does).
    intersect_final = store_customers.join(
        catalog_customers,
        on=["c_last_name", "c_first_name", "d_date"],
        how="semi",
        nulls_equal=True,
    ).join(
        web_customers,
        on=["c_last_name", "c_first_name", "d_date"],
        how="semi",
        nulls_equal=True,
    )
    limit = 100
    # Count the final result.
    # Use pl.col("d_date").len() instead of pl.len() to avoid the zero-column
    # streaming chunk bug (https://github.com/rapidsai/cudf/issues/21428).
    return QueryResult(
        frame=(
            intersect_final.select(
                [pl.col("d_date").len().cast(pl.Int64).alias("count_star()")]
            ).limit(limit)
        ),
        sort_by=[],
        limit=limit,
    )
