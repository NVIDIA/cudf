# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 90."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.streaming.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.streaming.benchmarks.utils import QueryResult, get_data

if TYPE_CHECKING:
    from cudf_polars.streaming.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 90."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=90,
        qualification=run_config.qualification,
    )

    am_hour = params["am_hour"]
    pm_hour = params["pm_hour"]
    hd_dep_count = params["hd_dep_count"]
    wp_char_count_min = params["wp_char_count_min"]
    wp_char_count_max = params["wp_char_count_max"]

    return f"""
    -- start query 90 in stream 0 using template query90.tpl
    SELECT Cast(amc AS DECIMAL(15, 4)) / Cast(pmc AS DECIMAL(15, 4))
                   am_pm_ratio
    FROM   (SELECT Count(*) amc
            FROM   web_sales,
                   household_demographics,
                   time_dim,
                   web_page
            WHERE  ws_sold_time_sk = time_dim.t_time_sk
                   AND ws_ship_hdemo_sk = household_demographics.hd_demo_sk
                   AND ws_web_page_sk = web_page.wp_web_page_sk
                   AND time_dim.t_hour BETWEEN {am_hour} AND {am_hour} + 1
                   AND household_demographics.hd_dep_count = {hd_dep_count}
                   AND web_page.wp_char_count BETWEEN {wp_char_count_min} AND {wp_char_count_max}) at1,
           (SELECT Count(*) pmc
            FROM   web_sales,
                   household_demographics,
                   time_dim,
                   web_page
            WHERE  ws_sold_time_sk = time_dim.t_time_sk
                   AND ws_ship_hdemo_sk = household_demographics.hd_demo_sk
                   AND ws_web_page_sk = web_page.wp_web_page_sk
                   AND time_dim.t_hour BETWEEN {pm_hour} AND {pm_hour} + 1
                   AND household_demographics.hd_dep_count = {hd_dep_count}
                   AND web_page.wp_char_count BETWEEN {wp_char_count_min} AND {wp_char_count_max}) pt
    ORDER  BY am_pm_ratio
    LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> QueryResult:
    """Query 90."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=90,
        qualification=run_config.qualification,
    )

    am_hour = params["am_hour"]
    pm_hour = params["pm_hour"]
    hd_dep_count = params["hd_dep_count"]
    wp_char_count_min = params["wp_char_count_min"]
    wp_char_count_max = params["wp_char_count_max"]

    web_sales = get_data(run_config.dataset_path, "web_sales", run_config.suffix)
    household_demographics = get_data(
        run_config.dataset_path, "household_demographics", run_config.suffix
    )
    time_dim = get_data(run_config.dataset_path, "time_dim", run_config.suffix)
    web_page = get_data(run_config.dataset_path, "web_page", run_config.suffix)
    base_query = (
        web_sales.join(
            time_dim, left_on="ws_sold_time_sk", right_on="t_time_sk", how="inner"
        )
        .join(
            household_demographics,
            left_on="ws_ship_hdemo_sk",
            right_on="hd_demo_sk",
            how="inner",
        )
        .join(
            web_page, left_on="ws_web_page_sk", right_on="wp_web_page_sk", how="inner"
        )
        .filter(
            (
                pl.col("hd_dep_count").is_not_null()
                & (pl.col("hd_dep_count") == hd_dep_count)
            )
            & (
                pl.col("wp_char_count").is_not_null()
                & (
                    pl.col("wp_char_count").is_between(
                        wp_char_count_min, wp_char_count_max
                    )
                )
            )
        )
    )
    return QueryResult(
        frame=(
            base_query.select(
                [
                    pl.when(pl.col("t_hour").is_between(am_hour, am_hour + 1))
                    .then(1)
                    .otherwise(0)
                    .sum()
                    .alias("amc"),
                    pl.when(pl.col("t_hour").is_between(pm_hour, pm_hour + 1))
                    .then(1)
                    .otherwise(0)
                    .sum()
                    .alias("pmc"),
                ]
            )
            .with_columns(
                [
                    pl.when(pl.col("pmc") != 0)
                    .then(
                        pl.col("amc").cast(pl.Float64) / pl.col("pmc").cast(pl.Float64)
                    )
                    .otherwise(None)
                    .alias("am_pm_ratio")
                ]
            )
            .select("am_pm_ratio")
            .sort("am_pm_ratio", nulls_last=True)
            .limit(100)
        ),
        sort_by=[("am_pm_ratio", False)],
        limit=100,
    )
