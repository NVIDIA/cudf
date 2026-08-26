# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Query 1."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.streaming.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.streaming.benchmarks.pdsds_queries import sql_sum
from cudf_polars.streaming.benchmarks.utils import QueryResult, get_data

if TYPE_CHECKING:
    from cudf_polars.streaming.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 1."""
    params = load_parameters(
        int(run_config.scale_factor), query_id=1, qualification=run_config.qualification
    )

    year = params["year"]
    state = params["state"]

    return f"""
    WITH customer_total_return
        AS (SELECT sr_customer_sk     AS ctr_customer_sk,
                    sr_store_sk        AS ctr_store_sk,
                    Sum(sr_return_amt) AS ctr_total_return
            FROM   store_returns,
                    date_dim
            WHERE  sr_returned_date_sk = d_date_sk
                    AND d_year = {year}
            GROUP  BY sr_customer_sk,
                    sr_store_sk)
    SELECT c_customer_id
    FROM   customer_total_return ctr1,
        store,
        customer
    WHERE  ctr1.ctr_total_return > (SELECT Avg(ctr_total_return) * 1.2
                                    FROM   customer_total_return ctr2
                                    WHERE  ctr1.ctr_store_sk = ctr2.ctr_store_sk)
        AND s_store_sk = ctr1.ctr_store_sk
        AND s_state = '{state}'
        AND ctr1.ctr_customer_sk = c_customer_sk
    ORDER  BY c_customer_id
    LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> QueryResult:
    """Query 1."""
    params = load_parameters(
        int(run_config.scale_factor), query_id=1, qualification=run_config.qualification
    )

    year = params["year"]
    state = params["state"]

    store_returns = get_data(
        run_config.dataset_path, "store_returns", run_config.suffix
    )
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    store = get_data(run_config.dataset_path, "store", run_config.suffix)
    customer = get_data(run_config.dataset_path, "customer", run_config.suffix)

    # Step 1: Create customer_total_return CTE equivalent
    customer_total_return = (
        store_returns.join(
            date_dim, left_on="sr_returned_date_sk", right_on="d_date_sk"
        )
        .filter(pl.col("d_year") == year)
        .group_by(["sr_customer_sk", "sr_store_sk"])
        .agg(sql_sum("sr_return_amt").alias("ctr_total_return"))
        .rename(
            {
                "sr_customer_sk": "ctr_customer_sk",
                "sr_store_sk": "ctr_store_sk",
            }
        )
    )

    # Step 2: Calculate average return per store for the subquery
    store_avg_returns = customer_total_return.group_by("ctr_store_sk").agg(
        [(pl.col("ctr_total_return").mean() * 1.2).alias("avg_return_threshold")]
    )

    # Step 3: Join everything together and apply filters
    return QueryResult(
        frame=(
            customer_total_return.join(
                store_avg_returns, left_on="ctr_store_sk", right_on="ctr_store_sk"
            )
            .filter(pl.col("ctr_total_return") > pl.col("avg_return_threshold"))
            .join(store, left_on="ctr_store_sk", right_on="s_store_sk")
            .filter(pl.col("s_state") == state)
            .join(customer, left_on="ctr_customer_sk", right_on="c_customer_sk")
            .select(["c_customer_id"])
            .sort("c_customer_id")
            .limit(100)
        ),
        sort_by=[("c_customer_id", False)],
        limit=100,
    )
