# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Query 48."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.streaming.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.streaming.benchmarks.pdsds_queries import sql_sum
from cudf_polars.streaming.benchmarks.utils import QueryResult, get_data

if TYPE_CHECKING:
    from cudf_polars.streaming.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 48."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=48,
        qualification=run_config.qualification,
    )

    year = params["year"]
    demographics = params["demographics"]
    geography = params["geography"]

    # Build demographics conditions
    demo_conditions = [
        f"( cd_demo_sk = ss_cdemo_sk\n"
        f"                   AND cd_marital_status = '{d['marital_status']}'\n"
        f"                   AND cd_education_status = '{d['education_status']}'\n"
        f"                   AND ss_sales_price BETWEEN {d['price_min']} AND {d['price_max']} )"
        for d in demographics
    ]
    demo_sql = "\n                  OR ".join(demo_conditions)

    # Build geography conditions
    geo_conditions = []
    for g in geography:
        states_str = ", ".join(f"'{s}'" for s in g["states"])
        geo_conditions.append(
            f"( ss_addr_sk = ca_address_sk\n"
            f"                   AND ca_country = 'United States'\n"
            f"                   AND ca_state IN ( {states_str} )\n"
            f"                   AND ss_net_profit BETWEEN {g['profit_min']} AND {g['profit_max']} )"
        )
    geo_sql = "\n                  OR ".join(geo_conditions)

    return f"""
    SELECT Sum (ss_quantity)
    FROM   store_sales,
           store,
           customer_demographics,
           customer_address,
           date_dim
    WHERE  s_store_sk = ss_store_sk
           AND ss_sold_date_sk = d_date_sk
           AND d_year = {year}
           AND ( {demo_sql} )
           AND ( {geo_sql} );
    """


def polars_impl(run_config: RunConfig) -> QueryResult:
    """Query 48."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=48,
        qualification=run_config.qualification,
    )

    year = params["year"]
    demographics = params["demographics"]
    geography = params["geography"]

    # Load tables
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    store = get_data(run_config.dataset_path, "store", run_config.suffix)
    customer_demographics = get_data(
        run_config.dataset_path, "customer_demographics", run_config.suffix
    )
    customer_address = get_data(
        run_config.dataset_path, "customer_address", run_config.suffix
    )
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)

    # Build demographics conditions
    demo_conditions = [
        (
            (pl.col("cd_marital_status") == d["marital_status"])
            & (pl.col("cd_education_status") == d["education_status"])
            & (pl.col("ss_sales_price").is_between(d["price_min"], d["price_max"]))
        )
        for d in demographics
    ]
    demo_filter = demo_conditions[0]
    for cond in demo_conditions[1:]:
        demo_filter = demo_filter | cond

    # Build geography conditions
    geo_conditions = [
        (
            (pl.col("ca_country") == "United States")
            & (pl.col("ca_state").is_in(g["states"]))
            & (pl.col("ss_net_profit").is_between(g["profit_min"], g["profit_max"]))
        )
        for g in geography
    ]
    geo_filter = geo_conditions[0]
    for cond in geo_conditions[1:]:
        geo_filter = geo_filter | cond

    return QueryResult(
        frame=(
            store_sales.join(store, left_on="ss_store_sk", right_on="s_store_sk")
            .join(customer_demographics, left_on="ss_cdemo_sk", right_on="cd_demo_sk")
            .join(customer_address, left_on="ss_addr_sk", right_on="ca_address_sk")
            .join(date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk")
            .filter((pl.col("d_year") == year) & demo_filter & geo_filter)
            .select([sql_sum("ss_quantity").alias("sum(ss_quantity)")])
        ),
        sort_by=[],
        limit=None,
    )
