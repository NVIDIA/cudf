# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 81."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.streaming.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.streaming.benchmarks.utils import QueryResult, get_data

if TYPE_CHECKING:
    from cudf_polars.streaming.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 81."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=81,
        qualification=run_config.qualification,
    )
    state = params["state"]

    return f"""
    -- start query 81 in stream 0 using template query81.tpl
    WITH customer_total_return
         AS (SELECT cr_returning_customer_sk   AS ctr_customer_sk,
                    ca_state                   AS ctr_state,
                    Sum(cr_return_amt_inc_tax) AS ctr_total_return
             FROM   catalog_returns,
                    date_dim,
                    customer_address
             WHERE  cr_returned_date_sk = d_date_sk
                    AND d_year = 1999
                    AND cr_returning_addr_sk = ca_address_sk
             GROUP  BY cr_returning_customer_sk,
                       ca_state)
    SELECT c_customer_id,
                   c_salutation,
                   c_first_name,
                   c_last_name,
                   ca_street_number,
                   ca_street_name,
                   ca_street_type,
                   ca_suite_number,
                   ca_city,
                   ca_county,
                   ca_state,
                   ca_zip,
                   ca_country,
                   ca_gmt_offset,
                   ca_location_type,
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
              ca_street_number,
              ca_street_name,
              ca_street_type,
              ca_suite_number,
              ca_city,
              ca_county,
              ca_state,
              ca_zip,
              ca_country,
              ca_gmt_offset,
              ca_location_type,
              ctr_total_return
    LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> QueryResult:
    """Query 81."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=81,
        qualification=run_config.qualification,
    )

    state = params["state"]

    catalog_returns = get_data(
        run_config.dataset_path, "catalog_returns", run_config.suffix
    )
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    customer_address = get_data(
        run_config.dataset_path, "customer_address", run_config.suffix
    )
    customer = get_data(run_config.dataset_path, "customer", run_config.suffix)
    customer_total_return = (
        catalog_returns.join(
            date_dim, left_on="cr_returned_date_sk", right_on="d_date_sk"
        )
        .join(
            customer_address, left_on="cr_returning_addr_sk", right_on="ca_address_sk"
        )
        .filter(pl.col("d_year") == 1999)
        .group_by(["cr_returning_customer_sk", "ca_state"])
        .agg(
            [
                pl.col("cr_return_amt_inc_tax").count().alias("ctr_total_return_count"),
                pl.col("cr_return_amt_inc_tax").sum().alias("ctr_total_return_sum"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("ctr_total_return_count") == 0)
                .then(None)
                .otherwise(pl.col("ctr_total_return_sum"))
                .alias("ctr_total_return"),
                pl.col("cr_returning_customer_sk").alias("ctr_customer_sk"),
                pl.col("ca_state").alias("ctr_state"),
            ]
        )
        .select(["ctr_customer_sk", "ctr_state", "ctr_total_return"])
    )
    state_averages = (
        customer_total_return.group_by("ctr_state")
        .agg([pl.col("ctr_total_return").mean().alias("avg_return")])
        .with_columns([(pl.col("avg_return") * 1.2).alias("threshold")])
    )
    ctr1_with_threshold = customer_total_return.join(
        state_averages, left_on="ctr_state", right_on="ctr_state"
    ).filter(pl.col("ctr_total_return") > pl.col("threshold"))
    state_addresses = customer_address.filter(pl.col("ca_state") == state)
    sort_by = {
        "c_customer_id": False,
        "c_salutation": False,
        "c_first_name": False,
        "c_last_name": False,
        "ca_street_number": False,
        "ca_street_name": False,
        "ca_street_type": False,
        "ca_suite_number": False,
        "ca_city": False,
        "ca_county": False,
        "ca_state": False,
        "ca_zip": False,
        "ca_country": False,
        "ca_gmt_offset": False,
        "ca_location_type": False,
        "ctr_total_return": False,
    }
    limit = 100
    return QueryResult(
        frame=(
            ctr1_with_threshold.join(
                customer, left_on="ctr_customer_sk", right_on="c_customer_sk"
            )
            .join(
                state_addresses, left_on="c_current_addr_sk", right_on="ca_address_sk"
            )
            .select(
                [
                    "c_customer_id",
                    "c_salutation",
                    "c_first_name",
                    "c_last_name",
                    "ca_street_number",
                    "ca_street_name",
                    "ca_street_type",
                    "ca_suite_number",
                    "ca_city",
                    "ca_county",
                    "ca_state",
                    "ca_zip",
                    "ca_country",
                    "ca_gmt_offset",
                    "ca_location_type",
                    "ctr_total_return",
                ]
            )
            .sort(list(sort_by.keys()), nulls_last=True)
            .limit(limit)
        ),
        sort_by=list(sort_by.items()),
        limit=limit,
    )
