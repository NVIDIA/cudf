# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 88."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.streaming.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.streaming.benchmarks.utils import QueryResult, get_data

if TYPE_CHECKING:
    from cudf_polars.streaming.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 88."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=88,
        qualification=run_config.qualification,
    )

    s_store_name = params["s_store_name"]
    hd1 = params["hd_dep_count1"]
    hd2 = params["hd_dep_count2"]
    hd3 = params["hd_dep_count3"]

    return f"""
    select  *
    from
     (select count(*) h8_30_to_9
     from store_sales, household_demographics , time_dim, store
     where ss_sold_time_sk = time_dim.t_time_sk
         and ss_hdemo_sk = household_demographics.hd_demo_sk
         and ss_store_sk = s_store_sk
         and time_dim.t_hour = 8
         and time_dim.t_minute >= 30
         and ((household_demographics.hd_dep_count = {hd1} and household_demographics.hd_vehicle_count<={hd1}+2) or
              (household_demographics.hd_dep_count = {hd2} and household_demographics.hd_vehicle_count<={hd2}+2) or
              (household_demographics.hd_dep_count = {hd3} and household_demographics.hd_vehicle_count<={hd3}+2))
         and store.s_store_name = '{s_store_name}') s1,
     (select count(*) h9_to_9_30
     from store_sales, household_demographics , time_dim, store
     where ss_sold_time_sk = time_dim.t_time_sk
         and ss_hdemo_sk = household_demographics.hd_demo_sk
         and ss_store_sk = s_store_sk
         and time_dim.t_hour = 9
         and time_dim.t_minute < 30
         and ((household_demographics.hd_dep_count = {hd1} and household_demographics.hd_vehicle_count<={hd1}+2) or
              (household_demographics.hd_dep_count = {hd2} and household_demographics.hd_vehicle_count<={hd2}+2) or
              (household_demographics.hd_dep_count = {hd3} and household_demographics.hd_vehicle_count<={hd3}+2))
         and store.s_store_name = '{s_store_name}') s2,
     (select count(*) h9_30_to_10
     from store_sales, household_demographics , time_dim, store
     where ss_sold_time_sk = time_dim.t_time_sk
         and ss_hdemo_sk = household_demographics.hd_demo_sk
         and ss_store_sk = s_store_sk
         and time_dim.t_hour = 9
         and time_dim.t_minute >= 30
         and ((household_demographics.hd_dep_count = {hd1} and household_demographics.hd_vehicle_count<={hd1}+2) or
              (household_demographics.hd_dep_count = {hd2} and household_demographics.hd_vehicle_count<={hd2}+2) or
              (household_demographics.hd_dep_count = {hd3} and household_demographics.hd_vehicle_count<={hd3}+2))
         and store.s_store_name = '{s_store_name}') s3,
     (select count(*) h10_to_10_30
     from store_sales, household_demographics , time_dim, store
     where ss_sold_time_sk = time_dim.t_time_sk
         and ss_hdemo_sk = household_demographics.hd_demo_sk
         and ss_store_sk = s_store_sk
         and time_dim.t_hour = 10
         and time_dim.t_minute < 30
         and ((household_demographics.hd_dep_count = {hd1} and household_demographics.hd_vehicle_count<={hd1}+2) or
              (household_demographics.hd_dep_count = {hd2} and household_demographics.hd_vehicle_count<={hd2}+2) or
              (household_demographics.hd_dep_count = {hd3} and household_demographics.hd_vehicle_count<={hd3}+2))
         and store.s_store_name = '{s_store_name}') s4,
     (select count(*) h10_30_to_11
     from store_sales, household_demographics , time_dim, store
     where ss_sold_time_sk = time_dim.t_time_sk
         and ss_hdemo_sk = household_demographics.hd_demo_sk
         and ss_store_sk = s_store_sk
         and time_dim.t_hour = 10
         and time_dim.t_minute >= 30
         and ((household_demographics.hd_dep_count = {hd1} and household_demographics.hd_vehicle_count<={hd1}+2) or
              (household_demographics.hd_dep_count = {hd2} and household_demographics.hd_vehicle_count<={hd2}+2) or
              (household_demographics.hd_dep_count = {hd3} and household_demographics.hd_vehicle_count<={hd3}+2))
         and store.s_store_name = '{s_store_name}') s5,
     (select count(*) h11_to_11_30
     from store_sales, household_demographics , time_dim, store
     where ss_sold_time_sk = time_dim.t_time_sk
         and ss_hdemo_sk = household_demographics.hd_demo_sk
         and ss_store_sk = s_store_sk
         and time_dim.t_hour = 11
         and time_dim.t_minute < 30
         and ((household_demographics.hd_dep_count = {hd1} and household_demographics.hd_vehicle_count<={hd1}+2) or
              (household_demographics.hd_dep_count = {hd2} and household_demographics.hd_vehicle_count<={hd2}+2) or
              (household_demographics.hd_dep_count = {hd3} and household_demographics.hd_vehicle_count<={hd3}+2))
         and store.s_store_name = '{s_store_name}') s6,
     (select count(*) h11_30_to_12
     from store_sales, household_demographics , time_dim, store
     where ss_sold_time_sk = time_dim.t_time_sk
         and ss_hdemo_sk = household_demographics.hd_demo_sk
         and ss_store_sk = s_store_sk
         and time_dim.t_hour = 11
         and time_dim.t_minute >= 30
         and ((household_demographics.hd_dep_count = {hd1} and household_demographics.hd_vehicle_count<={hd1}+2) or
              (household_demographics.hd_dep_count = {hd2} and household_demographics.hd_vehicle_count<={hd2}+2) or
              (household_demographics.hd_dep_count = {hd3} and household_demographics.hd_vehicle_count<={hd3}+2))
         and store.s_store_name = '{s_store_name}') s7,
     (select count(*) h12_to_12_30
     from store_sales, household_demographics , time_dim, store
     where ss_sold_time_sk = time_dim.t_time_sk
         and ss_hdemo_sk = household_demographics.hd_demo_sk
         and ss_store_sk = s_store_sk
         and time_dim.t_hour = 12
         and time_dim.t_minute < 30
         and ((household_demographics.hd_dep_count = {hd1} and household_demographics.hd_vehicle_count<={hd1}+2) or
              (household_demographics.hd_dep_count = {hd2} and household_demographics.hd_vehicle_count<={hd2}+2) or
              (household_demographics.hd_dep_count = {hd3} and household_demographics.hd_vehicle_count<={hd3}+2))
         and store.s_store_name = '{s_store_name}') s8;
    """


def polars_impl(run_config: RunConfig) -> QueryResult:
    """Query 88."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=88,
        qualification=run_config.qualification,
    )

    s_store_name = params["s_store_name"]
    hd1 = params["hd_dep_count1"]
    hd2 = params["hd_dep_count2"]
    hd3 = params["hd_dep_count3"]

    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    household_demographics = get_data(
        run_config.dataset_path, "household_demographics", run_config.suffix
    )
    time_dim = get_data(run_config.dataset_path, "time_dim", run_config.suffix)
    store = get_data(run_config.dataset_path, "store", run_config.suffix)

    # Pre-filter each small table before joining against store_sales [58 partitions].
    filtered_hdemo = household_demographics.filter(
        ((pl.col("hd_dep_count") == hd1) & (pl.col("hd_vehicle_count") <= hd1 + 2))
        | ((pl.col("hd_dep_count") == hd2) & (pl.col("hd_vehicle_count") <= hd2 + 2))
        | ((pl.col("hd_dep_count") == hd3) & (pl.col("hd_vehicle_count") <= hd3 + 2))
    ).select("hd_demo_sk")
    filtered_store = store.filter(pl.col("s_store_name") == s_store_name).select(
        "s_store_sk"
    )
    # Restrict time_dim to the union of all 8 slot conditions; every surviving row maps
    # to exactly one bucket, so the downstream pl.when chain is exhaustive.
    filtered_time = time_dim.filter(
        ((pl.col("t_hour") == 8) & (pl.col("t_minute") >= 30))
        | pl.col("t_hour").is_in([9, 10, 11])
        | ((pl.col("t_hour") == 12) & (pl.col("t_minute") < 30))
    ).select(["t_time_sk", "t_hour", "t_minute"])

    bucket_names = [
        "h8_30_to_9",
        "h9_to_9_30",
        "h9_30_to_10",
        "h10_to_10_30",
        "h10_30_to_11",
        "h11_to_11_30",
        "h11_30_to_12",
        "h12_to_12_30",
    ]

    # Collapse the 58-partition store_sales pipeline to an 8-row bucket-count table first.
    # The 8 conditional sums in the final select then operate on [1] partition, so even if
    # the streaming executor creates separate sub-plans for each sum, each reads only the
    # tiny CACHE'd group_by output rather than re-scanning store_sales.
    counts_lf = (
        store_sales.select(["ss_sold_time_sk", "ss_hdemo_sk", "ss_store_sk"])
        .join(filtered_time, left_on="ss_sold_time_sk", right_on="t_time_sk")
        .join(filtered_hdemo, left_on="ss_hdemo_sk", right_on="hd_demo_sk", how="semi")
        .join(filtered_store, left_on="ss_store_sk", right_on="s_store_sk", how="semi")
        .select(
            pl.when((pl.col("t_hour") == 8) & (pl.col("t_minute") >= 30))
            .then(pl.lit(0))
            .when((pl.col("t_hour") == 9) & (pl.col("t_minute") < 30))
            .then(pl.lit(1))
            .when((pl.col("t_hour") == 9) & (pl.col("t_minute") >= 30))
            .then(pl.lit(2))
            .when((pl.col("t_hour") == 10) & (pl.col("t_minute") < 30))
            .then(pl.lit(3))
            .when((pl.col("t_hour") == 10) & (pl.col("t_minute") >= 30))
            .then(pl.lit(4))
            .when((pl.col("t_hour") == 11) & (pl.col("t_minute") < 30))
            .then(pl.lit(5))
            .when((pl.col("t_hour") == 11) & (pl.col("t_minute") >= 30))
            .then(pl.lit(6))
            .when((pl.col("t_hour") == 12) & (pl.col("t_minute") < 30))
            .then(pl.lit(7))
            .alias("bucket")
        )
        .group_by("bucket")
        .agg(pl.len().cast(pl.Int64).alias("cnt"))
    )

    return QueryResult(
        frame=counts_lf.select(
            [
                pl.when(pl.col("bucket") == i)
                .then(pl.col("cnt"))
                .otherwise(pl.lit(0).cast(pl.Int64))
                .sum()
                .alias(name)
                for i, name in enumerate(bucket_names)
            ]
        ),
        sort_by=[],
        limit=None,
    )
