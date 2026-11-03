# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 30."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.streaming.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.streaming.benchmarks.utils import QueryResult, get_data

if TYPE_CHECKING:
    from cudf_polars.streaming.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 30."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=30,
        qualification=run_config.qualification,
    )

    year = params["year"]
    state = params["state"]

    return f"""
    WITH customer_total_return
         AS (SELECT wr_returning_customer_sk AS ctr_customer_sk,
                    ca_state                 AS ctr_state,
                    Sum(wr_return_amt)       AS ctr_total_return
             FROM   web_returns,
                    date_dim,
                    customer_address
             WHERE  wr_returned_date_sk = d_date_sk
                    AND d_year = {year}
                    AND wr_returning_addr_sk = ca_address_sk
             GROUP  BY wr_returning_customer_sk,
                       ca_state)
    SELECT c_customer_id,
                   c_salutation,
                   c_first_name,
                   c_last_name,
                   c_preferred_cust_flag,
                   c_birth_day,
                   c_birth_month,
                   c_birth_year,
                   c_birth_country,
                   c_login,
                   c_email_address,
                   c_last_review_date_sk,
                   ctr_total_return
    FROM   customer_total_return ctr1,
           customer_address,
           customer
    WHERE  ctr1.ctr_total_return > (SELECT Avg(ctr_total_return) * 1.2
                                    FROM   customer_total_return ctr2
                                    WHERE  ctr1.ctr_state = ctr2.ctr_state)
           AND ca_address_sk = c_current_addr_sk
           AND ca_state = '{state}'
           AND ctr1.ctr_customer_sk = c_customer_sk
    ORDER  BY c_customer_id,
              c_salutation,
              c_first_name,
              c_last_name,
              c_preferred_cust_flag,
              c_birth_day,
              c_birth_month,
              c_birth_year,
              c_birth_country,
              c_login,
              c_email_address,
              c_last_review_date_sk,
              ctr_total_return
    LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> QueryResult:
    """Query 30."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=30,
        qualification=run_config.qualification,
    )

    year = params["year"]
    state = params["state"]

    # Load tables
    web_returns = get_data(run_config.dataset_path, "web_returns", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    customer_address = get_data(
        run_config.dataset_path, "customer_address", run_config.suffix
    )
    customer = get_data(run_config.dataset_path, "customer", run_config.suffix)
    # CTE: customer_total_return
    customer_total_return = (
        web_returns.join(date_dim, left_on="wr_returned_date_sk", right_on="d_date_sk")
        .join(
            customer_address, left_on="wr_returning_addr_sk", right_on="ca_address_sk"
        )
        .filter(pl.col("d_year") == year)
        .group_by(
            [
                pl.col("wr_returning_customer_sk").alias("ctr_customer_sk"),
                pl.col("ca_state").alias("ctr_state"),
            ]
        )
        .agg([pl.col("wr_return_amt").sum().alias("ctr_total_return")])
    )
    # Calculate state averages for the correlated subquery
    state_averages = (
        customer_total_return.group_by("ctr_state")
        .agg([pl.col("ctr_total_return").mean().alias("avg_return")])
        .with_columns([pl.col("avg_return") * 1.2])
    )
    # Join customer_total_return with state averages to implement correlated subquery
    qualified_customers = customer_total_return.join(
        state_averages, left_on="ctr_state", right_on="ctr_state"
    ).filter(pl.col("ctr_total_return") > pl.col("avg_return"))
    # Join customer with customer_address first (avoiding cartesian product)
    # Then join with qualified customers
    sort_by = {
        "c_customer_id": False,
        "c_salutation": False,
        "c_first_name": False,
        "c_last_name": False,
        "c_preferred_cust_flag": False,
        "c_birth_day": False,
        "c_birth_month": False,
        "c_birth_year": False,
        "c_birth_country": False,
        "c_login": False,
        "c_email_address": False,
        "c_last_review_date_sk": False,
        "ctr_total_return": False,
    }
    limit = 100
    return QueryResult(
        frame=(
            customer.join(
                customer_address, left_on="c_current_addr_sk", right_on="ca_address_sk"
            )
            .filter(pl.col("ca_state") == state)
            .join(
                qualified_customers, left_on="c_customer_sk", right_on="ctr_customer_sk"
            )
            .select(
                [
                    "c_customer_id",
                    "c_salutation",
                    "c_first_name",
                    "c_last_name",
                    "c_preferred_cust_flag",
                    "c_birth_day",
                    "c_birth_month",
                    "c_birth_year",
                    "c_birth_country",
                    "c_login",
                    "c_email_address",
                    "c_last_review_date_sk",
                    "ctr_total_return",
                ]
            )
            .sort(sort_by.keys(), nulls_last=True)
            .limit(limit)
        ),
        sort_by=list(sort_by.items()),
        limit=limit,
    )
