# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Query 13."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.streaming.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.streaming.benchmarks.pdsds_queries import sql_sum
from cudf_polars.streaming.benchmarks.utils import QueryResult, get_data

if TYPE_CHECKING:
    from cudf_polars.streaming.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 13."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=13,
        qualification=run_config.qualification,
    )

    ms = params["ms"]  # 3 marital statuses
    es = params["es"]  # 3 education statuses
    state = params["state"]  # 9 states (3 groups of 3)

    return f"""
    SELECT Avg(ss_quantity),
           Avg(ss_ext_sales_price),
           Avg(ss_ext_wholesale_cost),
           Sum(ss_ext_wholesale_cost)
    FROM   store_sales,
           store,
           customer_demographics,
           household_demographics,
           customer_address,
           date_dim
    WHERE  s_store_sk = ss_store_sk
           AND ss_sold_date_sk = d_date_sk
           AND d_year = 2001
           AND ( ( ss_hdemo_sk = hd_demo_sk
                   AND cd_demo_sk = ss_cdemo_sk
                   AND cd_marital_status = '{ms[0]}'
                   AND cd_education_status = '{es[0]}'
                   AND ss_sales_price BETWEEN 100.00 AND 150.00
                   AND hd_dep_count = 3 )
                  OR ( ss_hdemo_sk = hd_demo_sk
                       AND cd_demo_sk = ss_cdemo_sk
                       AND cd_marital_status = '{ms[1]}'
                       AND cd_education_status = '{es[1]}'
                       AND ss_sales_price BETWEEN 50.00 AND 100.00
                       AND hd_dep_count = 1 )
                  OR ( ss_hdemo_sk = hd_demo_sk
                       AND cd_demo_sk = ss_cdemo_sk
                       AND cd_marital_status = '{ms[2]}'
                       AND cd_education_status = '{es[2]}'
                       AND ss_sales_price BETWEEN 150.00 AND 200.00
                       AND hd_dep_count = 1 ) )
           AND ( ( ss_addr_sk = ca_address_sk
                   AND ca_country = 'United States'
                   AND ca_state IN ( '{state[0]}', '{state[1]}', '{state[2]}' )
                   AND ss_net_profit BETWEEN 100 AND 200 )
                  OR ( ss_addr_sk = ca_address_sk
                       AND ca_country = 'United States'
                       AND ca_state IN ( '{state[3]}', '{state[4]}', '{state[5]}' )
                       AND ss_net_profit BETWEEN 150 AND 300 )
                  OR ( ss_addr_sk = ca_address_sk
                       AND ca_country = 'United States'
                       AND ca_state IN ( '{state[6]}', '{state[7]}', '{state[8]}' )
                       AND ss_net_profit BETWEEN 50 AND 250 ) );
    """


def polars_impl(run_config: RunConfig) -> QueryResult:
    """Query 13."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=13,
        qualification=run_config.qualification,
    )

    ms = params["ms"]  # 3 marital statuses
    es = params["es"]  # 3 education statuses
    state = params["state"]  # 9 states (3 groups of 3)

    # Load tables
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    store = get_data(run_config.dataset_path, "store", run_config.suffix)
    customer_demographics = get_data(
        run_config.dataset_path, "customer_demographics", run_config.suffix
    )
    household_demographics = get_data(
        run_config.dataset_path, "household_demographics", run_config.suffix
    )
    customer_address = get_data(
        run_config.dataset_path, "customer_address", run_config.suffix
    )
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    return QueryResult(
        frame=(
            store_sales.join(store, left_on="ss_store_sk", right_on="s_store_sk")
            .join(date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk")
            .join(household_demographics, left_on="ss_hdemo_sk", right_on="hd_demo_sk")
            .join(customer_demographics, left_on="ss_cdemo_sk", right_on="cd_demo_sk")
            .join(customer_address, left_on="ss_addr_sk", right_on="ca_address_sk")
            .filter(
                (pl.col("d_year") == 2001)
                & (pl.col("ca_country") == "United States")
                &
                # Demographic conditions (any one of these)
                (
                    (
                        (pl.col("cd_marital_status") == ms[0])
                        & (pl.col("cd_education_status") == es[0])
                        & pl.col("ss_sales_price").is_between(100.00, 150.00)
                        & (pl.col("hd_dep_count") == 3)
                    )
                    | (
                        (pl.col("cd_marital_status") == ms[1])
                        & (pl.col("cd_education_status") == es[1])
                        & pl.col("ss_sales_price").is_between(50.00, 100.00)
                        & (pl.col("hd_dep_count") == 1)
                    )
                    | (
                        (pl.col("cd_marital_status") == ms[2])
                        & (pl.col("cd_education_status") == es[2])
                        & pl.col("ss_sales_price").is_between(150.00, 200.00)
                        & (pl.col("hd_dep_count") == 1)
                    )
                )
                &
                # Address conditions (any one of these)
                (
                    (
                        pl.col("ca_state").is_in(state[0:3])
                        & pl.col("ss_net_profit").is_between(100, 200)
                    )
                    | (
                        pl.col("ca_state").is_in(state[3:6])
                        & pl.col("ss_net_profit").is_between(150, 300)
                    )
                    | (
                        pl.col("ca_state").is_in(state[6:9])
                        & pl.col("ss_net_profit").is_between(50, 250)
                    )
                )
            )
            .select(
                [
                    pl.col("ss_quantity").mean().alias("avg(ss_quantity)"),
                    pl.col("ss_ext_sales_price")
                    .mean()
                    .alias("avg(ss_ext_sales_price)"),
                    pl.col("ss_ext_wholesale_cost")
                    .mean()
                    .alias("avg(ss_ext_wholesale_cost)"),
                    sql_sum("ss_ext_wholesale_cost").alias(
                        "sum(ss_ext_wholesale_cost)"
                    ),
                ]
            )
        ),
        sort_by=[],
        limit=None,
    )
