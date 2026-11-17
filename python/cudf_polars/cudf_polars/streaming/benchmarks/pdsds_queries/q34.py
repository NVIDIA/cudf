# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 34."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.streaming.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.streaming.benchmarks.utils import QueryResult, get_data

if TYPE_CHECKING:
    from cudf_polars.streaming.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 34."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=34,
        qualification=run_config.qualification,
    )

    year = params["year"]
    bpone = params["bpone"]
    bptwo = params["bptwo"]
    county = params["county"]

    # Format county list for SQL IN clause
    county_list = ", ".join(f"'{c}'" for c in county)

    return f"""
    SELECT c_last_name,
           c_first_name,
           c_salutation,
           c_preferred_cust_flag,
           ss_ticket_number,
           cnt
    FROM   (SELECT ss_ticket_number,
                   ss_customer_sk,
                   Count(*) cnt
            FROM   store_sales,
                   date_dim,
                   store,
                   household_demographics
            WHERE  store_sales.ss_sold_date_sk = date_dim.d_date_sk
                   AND store_sales.ss_store_sk = store.s_store_sk
                   AND store_sales.ss_hdemo_sk = household_demographics.hd_demo_sk
                   AND ( date_dim.d_dom BETWEEN 1 AND 3
                          OR date_dim.d_dom BETWEEN 25 AND 28 )
                   AND ( household_demographics.hd_buy_potential = '{bpone}'
                          OR household_demographics.hd_buy_potential = '{bptwo}' )
                   AND household_demographics.hd_vehicle_count > 0
                   AND ( CASE
                           WHEN household_demographics.hd_vehicle_count > 0 THEN
                           household_demographics.hd_dep_count /
                           household_demographics.hd_vehicle_count
                           ELSE NULL
                         END ) > 1.2
                   AND date_dim.d_year IN ( {year}, {year} + 1, {year} + 2 )
                   AND store.s_county IN ( {county_list} )
            GROUP  BY ss_ticket_number,
                      ss_customer_sk) dn,
           customer
    WHERE  ss_customer_sk = c_customer_sk
           AND cnt BETWEEN 15 AND 20
    ORDER  BY c_last_name,
              c_first_name,
              c_salutation,
              c_preferred_cust_flag DESC;
    """


def polars_impl(run_config: RunConfig) -> QueryResult:
    """Query 34."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=34,
        qualification=run_config.qualification,
    )

    year = params["year"]
    bpone = params["bpone"]
    bptwo = params["bptwo"]
    county = params["county"]

    # Load tables
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    store = get_data(run_config.dataset_path, "store", run_config.suffix)
    household_demographics = get_data(
        run_config.dataset_path, "household_demographics", run_config.suffix
    )
    customer = get_data(run_config.dataset_path, "customer", run_config.suffix)
    dn = (
        store_sales.join(date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk")
        .join(store, left_on="ss_store_sk", right_on="s_store_sk")
        .join(household_demographics, left_on="ss_hdemo_sk", right_on="hd_demo_sk")
        .filter(
            ((pl.col("d_dom").is_between(1, 3)) | (pl.col("d_dom").is_between(25, 28)))
            & (
                (pl.col("hd_buy_potential") == bpone)
                | (pl.col("hd_buy_potential") == bptwo)
            )
            & (pl.col("hd_vehicle_count") > 0)
            & (
                pl.when(pl.col("hd_vehicle_count") > 0)
                .then(pl.col("hd_dep_count") / pl.col("hd_vehicle_count"))
                .otherwise(None)
                > 1.2
            )
            & (pl.col("d_year").is_in([year, year + 1, year + 2]))
            & (pl.col("s_county").is_in(county))
        )
        .group_by(["ss_ticket_number", "ss_customer_sk"])
        .agg([pl.len().alias("cnt")])
    )
    sort_by = {
        "c_last_name": False,
        "c_first_name": False,
        "c_salutation": False,
        "c_preferred_cust_flag": True,
    }
    return QueryResult(
        frame=(
            dn.join(customer, left_on="ss_customer_sk", right_on="c_customer_sk")
            .filter(pl.col("cnt").is_between(15, 20))
            .select(
                [
                    "c_last_name",
                    "c_first_name",
                    "c_salutation",
                    "c_preferred_cust_flag",
                    "ss_ticket_number",
                    "cnt",
                ]
            )
            .sort(
                list(sort_by.keys()),
                descending=list(sort_by.values()),
                nulls_last=True,
            )
        ),
        sort_by=list(sort_by.items()),
        limit=None,
    )
