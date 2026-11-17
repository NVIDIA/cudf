# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 96."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.streaming.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.streaming.benchmarks.utils import QueryResult, get_data

if TYPE_CHECKING:
    from cudf_polars.streaming.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 96."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=96,
        qualification=run_config.qualification,
    )

    t_hour = params["t_hour"]
    t_minute = params["t_minute"]
    hd_dep_count = params["hd_dep_count"]
    s_store_name = params["s_store_name"]

    return f"""
    SELECT Count(*)
    FROM   store_sales,
           household_demographics,
           time_dim,
           store
    WHERE  ss_sold_time_sk = time_dim.t_time_sk
           AND ss_hdemo_sk = household_demographics.hd_demo_sk
           AND ss_store_sk = s_store_sk
           AND time_dim.t_hour = {t_hour}
           AND time_dim.t_minute >= {t_minute}
           AND household_demographics.hd_dep_count = {hd_dep_count}
           AND store.s_store_name = '{s_store_name}'
    ORDER  BY Count(*)
    LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> QueryResult:
    """Query 96."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=96,
        qualification=run_config.qualification,
    )

    t_hour = params["t_hour"]
    t_minute = params["t_minute"]
    hd_dep_count = params["hd_dep_count"]
    s_store_name = params["s_store_name"]
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    household_demographics = get_data(
        run_config.dataset_path, "household_demographics", run_config.suffix
    )
    time_dim = get_data(run_config.dataset_path, "time_dim", run_config.suffix)
    store = get_data(run_config.dataset_path, "store", run_config.suffix)
    return QueryResult(
        frame=(
            store_sales.join(
                time_dim, left_on="ss_sold_time_sk", right_on="t_time_sk", how="inner"
            )
            .join(
                household_demographics,
                left_on="ss_hdemo_sk",
                right_on="hd_demo_sk",
                how="inner",
            )
            .join(store, left_on="ss_store_sk", right_on="s_store_sk", how="inner")
            .filter(
                (pl.col("t_hour") == t_hour)
                & (pl.col("t_minute") >= t_minute)
                & (pl.col("hd_dep_count") == hd_dep_count)
                & (pl.col("s_store_name") == s_store_name)
            )
            .select([pl.len().alias("count_star()")])
            .sort("count_star()", nulls_last=True)
            .limit(100)
        ),
        sort_by=[("count_star()", False)],
        limit=100,
    )
