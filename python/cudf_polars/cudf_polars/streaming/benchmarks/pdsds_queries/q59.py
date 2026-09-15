# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Query 59."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.streaming.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.streaming.benchmarks.pdsds_queries import sql_sum
from cudf_polars.streaming.benchmarks.utils import QueryResult, get_data

if TYPE_CHECKING:
    from cudf_polars.streaming.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 59."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=59,
        qualification=run_config.qualification,
    )

    dms = params["dms"]

    return f"""
    WITH wss
         AS (SELECT d_week_seq,
                    ss_store_sk,
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
             FROM   store_sales,
                    date_dim
             WHERE  d_date_sk = ss_sold_date_sk
             GROUP  BY d_week_seq,
                       ss_store_sk)
    SELECT s_store_name1,
                   s_store_id1,
                   d_week_seq1,
                   sun_sales1 / sun_sales2,
                   mon_sales1 / mon_sales2,
                   tue_sales1 / tue_sales2,
                   wed_sales1 / wed_sales2,
                   thu_sales1 / thu_sales2,
                   fri_sales1 / fri_sales2,
                   sat_sales1 / sat_sales2
    FROM   (SELECT s_store_name   s_store_name1,
                   wss.d_week_seq d_week_seq1,
                   s_store_id     s_store_id1,
                   sun_sales      sun_sales1,
                   mon_sales      mon_sales1,
                   tue_sales      tue_sales1,
                   wed_sales      wed_sales1,
                   thu_sales      thu_sales1,
                   fri_sales      fri_sales1,
                   sat_sales      sat_sales1
            FROM   wss,
                   store,
                   date_dim d
            WHERE  d.d_week_seq = wss.d_week_seq
                   AND ss_store_sk = s_store_sk
                   AND d_month_seq BETWEEN {dms} AND {dms} + 11) y,
           (SELECT s_store_name   s_store_name2,
                   wss.d_week_seq d_week_seq2,
                   s_store_id     s_store_id2,
                   sun_sales      sun_sales2,
                   mon_sales      mon_sales2,
                   tue_sales      tue_sales2,
                   wed_sales      wed_sales2,
                   thu_sales      thu_sales2,
                   fri_sales      fri_sales2,
                   sat_sales      sat_sales2
            FROM   wss,
                   store,
                   date_dim d
            WHERE  d.d_week_seq = wss.d_week_seq
                   AND ss_store_sk = s_store_sk
                   AND d_month_seq BETWEEN {dms} + 12 AND {dms} + 23) x
    WHERE  s_store_id1 = s_store_id2
           AND d_week_seq1 = d_week_seq2 - 52
    ORDER  BY s_store_name1,
              s_store_id1,
              d_week_seq1
    LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> QueryResult:
    """Query 59."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=59,
        qualification=run_config.qualification,
    )

    dms = params["dms"]

    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    store = get_data(run_config.dataset_path, "store", run_config.suffix)

    base = store_sales.join(
        date_dim.select(["d_date_sk", "d_week_seq", "d_day_name"]),
        left_on="ss_sold_date_sk",
        right_on="d_date_sk",
    ).select(["d_week_seq", "ss_store_sk", "d_day_name", "ss_sales_price"])
    wss = base.group_by(["d_week_seq", "ss_store_sk"]).agg(
        [
            sql_sum(
                pl.when(pl.col("d_day_name") == day)
                .then(pl.col("ss_sales_price"))
                .otherwise(None)
            ).alias(alias)
            for day, alias in [
                ("Sunday", "sun_sales"),
                ("Monday", "mon_sales"),
                ("Tuesday", "tue_sales"),
                ("Wednesday", "wed_sales"),
                ("Thursday", "thu_sales"),
                ("Friday", "fri_sales"),
                ("Saturday", "sat_sales"),
            ]
        ]
    )
    wss_enriched = wss.join(
        store.select(["s_store_sk", "s_store_id", "s_store_name"]),
        left_on="ss_store_sk",
        right_on="s_store_sk",
    ).join(date_dim.select(["d_week_seq", "d_month_seq"]), on="d_week_seq")
    y = wss_enriched.filter(pl.col("d_month_seq").is_between(dms, dms + 11)).select(
        [
            pl.col("s_store_name").alias("s_store_name1"),
            pl.col("s_store_id").alias("s_store_id1"),
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
    x = wss_enriched.filter(
        pl.col("d_month_seq").is_between(dms + 12, dms + 23)
    ).select(
        [
            pl.col("s_store_id").alias("s_store_id2"),
            (pl.col("d_week_seq") - 52).alias("d_week_seq_join"),
            pl.col("sun_sales").alias("sun_sales2"),
            pl.col("mon_sales").alias("mon_sales2"),
            pl.col("tue_sales").alias("tue_sales2"),
            pl.col("wed_sales").alias("wed_sales2"),
            pl.col("thu_sales").alias("thu_sales2"),
            pl.col("fri_sales").alias("fri_sales2"),
            pl.col("sat_sales").alias("sat_sales2"),
        ]
    )
    joined = y.join(
        x,
        left_on=["s_store_id1", "d_week_seq1"],
        right_on=["s_store_id2", "d_week_seq_join"],
        how="inner",
    )
    projected = joined.select(
        [
            "s_store_name1",
            "s_store_id1",
            "d_week_seq1",
            (pl.col("sun_sales1") / pl.col("sun_sales2")).alias(
                "(sun_sales1 / sun_sales2)"
            ),
            (pl.col("mon_sales1") / pl.col("mon_sales2")).alias(
                "(mon_sales1 / mon_sales2)"
            ),
            (pl.col("tue_sales1") / pl.col("tue_sales2")).alias(
                "(tue_sales1 / tue_sales2)"
            ),
            (pl.col("wed_sales1") / pl.col("wed_sales2")).alias(
                "(wed_sales1 / wed_sales2)"
            ),
            (pl.col("thu_sales1") / pl.col("thu_sales2")).alias(
                "(thu_sales1 / thu_sales2)"
            ),
            (pl.col("fri_sales1") / pl.col("fri_sales2")).alias(
                "(fri_sales1 / fri_sales2)"
            ),
            (pl.col("sat_sales1") / pl.col("sat_sales2")).alias(
                "(sat_sales1 / sat_sales2)"
            ),
        ]
    )
    sort_by = {"s_store_name1": False, "s_store_id1": False, "d_week_seq1": False}
    limit = 100
    return QueryResult(
        frame=projected.sort(sort_by.keys(), nulls_last=True).limit(limit),
        sort_by=list(sort_by.items()),
        limit=limit,
    )
