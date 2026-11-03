# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 43."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.streaming.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.streaming.benchmarks.utils import QueryResult, get_data

if TYPE_CHECKING:
    from cudf_polars.streaming.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 43."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=43,
        qualification=run_config.qualification,
    )

    year = params["year"]
    gmt = params["gmt"]

    return f"""
    SELECT s_store_name,
                   s_store_id,
                   Sum(CASE
                         WHEN ( d_day_name = 'Sunday' ) THEN ss_sales_price
                         ELSE NULL
                       END) sun_sales,
                   Sum(CASE
                         WHEN ( d_day_name = 'Monday' ) THEN ss_sales_price
                         ELSE NULL
                       END) mon_sales,
                   Sum(CASE
                         WHEN ( d_day_name = 'Tuesday' ) THEN ss_sales_price
                         ELSE NULL
                       END) tue_sales,
                   Sum(CASE
                         WHEN ( d_day_name = 'Wednesday' ) THEN ss_sales_price
                         ELSE NULL
                       END) wed_sales,
                   Sum(CASE
                         WHEN ( d_day_name = 'Thursday' ) THEN ss_sales_price
                         ELSE NULL
                       END) thu_sales,
                   Sum(CASE
                         WHEN ( d_day_name = 'Friday' ) THEN ss_sales_price
                         ELSE NULL
                       END) fri_sales,
                   Sum(CASE
                         WHEN ( d_day_name = 'Saturday' ) THEN ss_sales_price
                         ELSE NULL
                       END) sat_sales
    FROM   date_dim,
           store_sales,
           store
    WHERE  d_date_sk = ss_sold_date_sk
           AND s_store_sk = ss_store_sk
           AND s_gmt_offset = {gmt}
           AND d_year = {year}
    GROUP  BY s_store_name,
              s_store_id
    ORDER  BY s_store_name,
              s_store_id,
              sun_sales,
              mon_sales,
              tue_sales,
              wed_sales,
              thu_sales,
              fri_sales,
              sat_sales
    LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> QueryResult:
    """Query 43."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=43,
        qualification=run_config.qualification,
    )

    year = params["year"]
    gmt = params["gmt"]

    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    store = get_data(run_config.dataset_path, "store", run_config.suffix)

    # Pre-filter lookup tables before joining against store_sales [58 partitions].
    # d_year not needed after filter; d_day_name drives the conditional agg columns.
    filtered_dates = date_dim.filter(pl.col("d_year") == year).select(
        ["d_date_sk", "d_day_name"]
    )
    # s_gmt_offset not needed after filter; keep group-by output columns.
    filtered_store = store.filter(pl.col("s_gmt_offset") == gmt).select(
        ["s_store_sk", "s_store_name", "s_store_id"]
    )

    sort_by = {
        "s_store_name": False,
        "s_store_id": False,
        "sun_sales": False,
        "mon_sales": False,
        "tue_sales": False,
        "wed_sales": False,
        "thu_sales": False,
        "fri_sales": False,
        "sat_sales": False,
    }
    limit = 100
    return QueryResult(
        frame=(
            store_sales.select(["ss_sold_date_sk", "ss_store_sk", "ss_sales_price"])
            .join(filtered_dates, left_on="ss_sold_date_sk", right_on="d_date_sk")
            .join(filtered_store, left_on="ss_store_sk", right_on="s_store_sk")
            .with_columns(
                [
                    pl.when(pl.col("d_day_name") == "Sunday")
                    .then(pl.col("ss_sales_price"))
                    .otherwise(0)
                    .alias("sun_sales_amount"),
                    pl.when(pl.col("d_day_name") == "Monday")
                    .then(pl.col("ss_sales_price"))
                    .otherwise(0)
                    .alias("mon_sales_amount"),
                    pl.when(pl.col("d_day_name") == "Tuesday")
                    .then(pl.col("ss_sales_price"))
                    .otherwise(0)
                    .alias("tue_sales_amount"),
                    pl.when(pl.col("d_day_name") == "Wednesday")
                    .then(pl.col("ss_sales_price"))
                    .otherwise(0)
                    .alias("wed_sales_amount"),
                    pl.when(pl.col("d_day_name") == "Thursday")
                    .then(pl.col("ss_sales_price"))
                    .otherwise(0)
                    .alias("thu_sales_amount"),
                    pl.when(pl.col("d_day_name") == "Friday")
                    .then(pl.col("ss_sales_price"))
                    .otherwise(0)
                    .alias("fri_sales_amount"),
                    pl.when(pl.col("d_day_name") == "Saturday")
                    .then(pl.col("ss_sales_price"))
                    .otherwise(0)
                    .alias("sat_sales_amount"),
                ]
            )
            .group_by(["s_store_name", "s_store_id"])
            .agg(
                [
                    pl.col("sun_sales_amount").sum().alias("sun_sales"),
                    pl.col("mon_sales_amount").sum().alias("mon_sales"),
                    pl.col("tue_sales_amount").sum().alias("tue_sales"),
                    pl.col("wed_sales_amount").sum().alias("wed_sales"),
                    pl.col("thu_sales_amount").sum().alias("thu_sales"),
                    pl.col("fri_sales_amount").sum().alias("fri_sales"),
                    pl.col("sat_sales_amount").sum().alias("sat_sales"),
                ]
            )
            .select(
                [
                    "s_store_name",
                    "s_store_id",
                    "sun_sales",
                    "mon_sales",
                    "tue_sales",
                    "wed_sales",
                    "thu_sales",
                    "fri_sales",
                    "sat_sales",
                ]
            )
            .sort(sort_by.keys(), nulls_last=True)
            .limit(limit)
        ),
        sort_by=list(sort_by.items()),
        limit=limit,
    )
