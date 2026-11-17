# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Query 2."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.streaming.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.streaming.benchmarks.pdsds_queries import sql_sum
from cudf_polars.streaming.benchmarks.utils import QueryResult, get_data

if TYPE_CHECKING:
    from cudf_polars.streaming.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 2."""
    params = load_parameters(
        int(run_config.scale_factor), query_id=2, qualification=run_config.qualification
    )

    year = params["year"]

    return f"""
    WITH wscs
         AS (SELECT sold_date_sk,
                    sales_price
             FROM   (SELECT ws_sold_date_sk    sold_date_sk,
                            ws_ext_sales_price sales_price
                     FROM   web_sales)
             UNION ALL
             (SELECT cs_sold_date_sk    sold_date_sk,
                     cs_ext_sales_price sales_price
              FROM   catalog_sales)),
         wswscs
         AS (SELECT d_week_seq,
                    Sum(CASE
                          WHEN ( d_day_name = 'Sunday' ) THEN sales_price
                          ELSE NULL
                        END) sun_sales,
                    Sum(CASE
                          WHEN ( d_day_name = 'Monday' ) THEN sales_price
                          ELSE NULL
                        END) mon_sales,
                    Sum(CASE
                          WHEN ( d_day_name = 'Tuesday' ) THEN sales_price
                          ELSE NULL
                        END) tue_sales,
                    Sum(CASE
                          WHEN ( d_day_name = 'Wednesday' ) THEN sales_price
                          ELSE NULL
                        END) wed_sales,
                    Sum(CASE
                          WHEN ( d_day_name = 'Thursday' ) THEN sales_price
                          ELSE NULL
                        END) thu_sales,
                    Sum(CASE
                          WHEN ( d_day_name = 'Friday' ) THEN sales_price
                          ELSE NULL
                        END) fri_sales,
                    Sum(CASE
                          WHEN ( d_day_name = 'Saturday' ) THEN sales_price
                          ELSE NULL
                        END) sat_sales
             FROM   wscs,
                    date_dim
             WHERE  d_date_sk = sold_date_sk
             GROUP  BY d_week_seq)
    SELECT d_week_seq1,
           Round(sun_sales1 / sun_sales2, 2),
           Round(mon_sales1 / mon_sales2, 2),
           Round(tue_sales1 / tue_sales2, 2),
           Round(wed_sales1 / wed_sales2, 2),
           Round(thu_sales1 / thu_sales2, 2),
           Round(fri_sales1 / fri_sales2, 2),
           Round(sat_sales1 / sat_sales2, 2)
    FROM   (SELECT wswscs.d_week_seq d_week_seq1,
                   sun_sales         sun_sales1,
                   mon_sales         mon_sales1,
                   tue_sales         tue_sales1,
                   wed_sales         wed_sales1,
                   thu_sales         thu_sales1,
                   fri_sales         fri_sales1,
                   sat_sales         sat_sales1
            FROM   wswscs,
                   date_dim
            WHERE  date_dim.d_week_seq = wswscs.d_week_seq
                   AND d_year = {year}) y,
           (SELECT wswscs.d_week_seq d_week_seq2,
                   sun_sales         sun_sales2,
                   mon_sales         mon_sales2,
                   tue_sales         tue_sales2,
                   wed_sales         wed_sales2,
                   thu_sales         thu_sales2,
                   fri_sales         fri_sales2,
                   sat_sales         sat_sales2
            FROM   wswscs,
                   date_dim
            WHERE  date_dim.d_week_seq = wswscs.d_week_seq
                   AND d_year = {year + 1}) z
    WHERE  d_week_seq1 = d_week_seq2 - 53
    ORDER  BY d_week_seq1;
    """


def polars_impl(run_config: RunConfig) -> QueryResult:
    """Query 2."""
    params = load_parameters(
        int(run_config.scale_factor), query_id=2, qualification=run_config.qualification
    )

    year = params["year"]

    web_sales = get_data(run_config.dataset_path, "web_sales", run_config.suffix)
    catalog_sales = get_data(
        run_config.dataset_path, "catalog_sales", run_config.suffix
    )
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    # Step 1: Create wscs CTE equivalent (union of web and catalog sales)
    wscs = pl.concat(
        [
            web_sales.select(
                [
                    pl.col("ws_sold_date_sk").alias("sold_date_sk"),
                    pl.col("ws_ext_sales_price").alias("sales_price"),
                ]
            ),
            catalog_sales.select(
                [
                    pl.col("cs_sold_date_sk").alias("sold_date_sk"),
                    pl.col("cs_ext_sales_price").alias("sales_price"),
                ]
            ),
        ]
    )
    days = (
        "Sunday",
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
    )
    day_cols = (
        "sun_sales",
        "mon_sales",
        "tue_sales",
        "wed_sales",
        "thu_sales",
        "fri_sales",
        "sat_sales",
    )
    # Pre-filter date_dim to 4 years ([year-1, year, year+1, year+2]) to capture
    # boundary weeks that span year transitions (e.g. from Dec 28 to Jan 3). Filtering to
    # only [year, year+1] incorrectly excludes Dec days whose d_week_seq also
    # appears in year's date_dim, producing null sales for those boundary weeks.
    date_dim_prefilter = date_dim.filter(
        pl.col("d_year").is_in([year - 1, year, year + 1, year + 2])
    ).select(["d_date_sk", "d_week_seq", "d_day_name"])
    wswscs = (
        wscs.join(date_dim_prefilter, left_on="sold_date_sk", right_on="d_date_sk")
        .group_by("d_week_seq")
        .agg(
            [
                sql_sum(
                    pl.when(pl.col("d_day_name") == day)
                    .then(pl.col("sales_price"))
                    .otherwise(None)
                ).alias(name)
                for day, name in zip(days, day_cols, strict=True)
            ]
        )
    )

    # Step 3: Create year data (y subquery equivalent)
    y_year = (
        wswscs.join(date_dim, left_on="d_week_seq", right_on="d_week_seq")
        .filter(pl.col("d_year") == year)
        .select(
            [
                pl.col("d_week_seq").alias("d_week_seq1"),
                pl.col("sun_sales").alias("sun_sales1"),
                pl.col("mon_sales").alias("mon_sales1"),
                pl.col("tue_sales").alias("tue_sales1"),
                pl.col("wed_sales").alias("wed_sales1"),
                pl.col("thu_sales").alias("thu_sales1"),
                pl.col("fri_sales").alias("fri_sales1"),
                pl.col("sat_sales").alias("sat_sales1"),
            ]
        )
    )
    # Step 4: Create year+1 data (z subquery equivalent)
    z_year_plus_1 = (
        wswscs.join(date_dim, left_on="d_week_seq", right_on="d_week_seq")
        .filter(pl.col("d_year") == year + 1)
        .select(
            [
                pl.col("d_week_seq").alias("d_week_seq2"),
                pl.col("sun_sales").alias("sun_sales2"),
                pl.col("mon_sales").alias("mon_sales2"),
                pl.col("tue_sales").alias("tue_sales2"),
                pl.col("wed_sales").alias("wed_sales2"),
                pl.col("thu_sales").alias("thu_sales2"),
                pl.col("fri_sales").alias("fri_sales2"),
                pl.col("sat_sales").alias("sat_sales2"),
            ]
        )
    )
    # Step 5: Join the two years and calculate ratios
    return QueryResult(
        frame=(
            y_year.join(
                z_year_plus_1,
                left_on="d_week_seq1",
                right_on=pl.col("d_week_seq2") - 53,
            )
            .select(
                [
                    pl.col("d_week_seq1"),
                    (pl.col("sun_sales1") / pl.col("sun_sales2"))
                    .round(2)
                    .alias("round((sun_sales1 / sun_sales2), 2)"),
                    (pl.col("mon_sales1") / pl.col("mon_sales2"))
                    .round(2)
                    .alias("round((mon_sales1 / mon_sales2), 2)"),
                    (pl.col("tue_sales1") / pl.col("tue_sales2"))
                    .round(2)
                    .alias("round((tue_sales1 / tue_sales2), 2)"),
                    (pl.col("wed_sales1") / pl.col("wed_sales2"))
                    .round(2)
                    .alias("round((wed_sales1 / wed_sales2), 2)"),
                    (pl.col("thu_sales1") / pl.col("thu_sales2"))
                    .round(2)
                    .alias("round((thu_sales1 / thu_sales2), 2)"),
                    (pl.col("fri_sales1") / pl.col("fri_sales2"))
                    .round(2)
                    .alias("round((fri_sales1 / fri_sales2), 2)"),
                    (pl.col("sat_sales1") / pl.col("sat_sales2"))
                    .round(2)
                    .alias("round((sat_sales1 / sat_sales2), 2)"),
                ]
            )
            .sort("d_week_seq1")
        ),
        sort_by=[("d_week_seq1", False)],
        limit=None,
    )
