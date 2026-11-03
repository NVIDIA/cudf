# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 9."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.streaming.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.streaming.benchmarks.utils import QueryResult, get_data

if TYPE_CHECKING:
    from cudf_polars.streaming.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 9."""
    params = load_parameters(
        int(run_config.scale_factor), query_id=9, qualification=run_config.qualification
    )

    aggcthen = params["aggcthen"]
    aggcelse = params["aggcelse"]
    rc = params["rc"]

    return f"""
    SELECT CASE
             WHEN (SELECT Count(*)
                   FROM   store_sales
                   WHERE  ss_quantity BETWEEN 1 AND 20) > {rc[0]} THEN
             (SELECT Avg({aggcthen})
              FROM   store_sales
              WHERE
             ss_quantity BETWEEN 1 AND 20)
             ELSE (SELECT Avg({aggcelse})
                   FROM   store_sales
                   WHERE  ss_quantity BETWEEN 1 AND 20)
           END bucket1,
           CASE
             WHEN (SELECT Count(*)
                   FROM   store_sales
                   WHERE  ss_quantity BETWEEN 21 AND 40) > {rc[1]} THEN
             (SELECT Avg({aggcthen})
              FROM   store_sales
              WHERE
             ss_quantity BETWEEN 21 AND 40)
             ELSE (SELECT Avg({aggcelse})
                   FROM   store_sales
                   WHERE  ss_quantity BETWEEN 21 AND 40)
           END bucket2,
           CASE
             WHEN (SELECT Count(*)
                   FROM   store_sales
                   WHERE  ss_quantity BETWEEN 41 AND 60) > {rc[2]} THEN
             (SELECT Avg({aggcthen})
              FROM   store_sales
              WHERE
             ss_quantity BETWEEN 41 AND 60)
             ELSE (SELECT Avg({aggcelse})
                   FROM   store_sales
                   WHERE  ss_quantity BETWEEN 41 AND 60)
           END bucket3,
           CASE
             WHEN (SELECT Count(*)
                   FROM   store_sales
                   WHERE  ss_quantity BETWEEN 61 AND 80) > {rc[3]} THEN
             (SELECT Avg({aggcthen})
              FROM   store_sales
              WHERE
             ss_quantity BETWEEN 61 AND 80)
             ELSE (SELECT Avg({aggcelse})
                   FROM   store_sales
                   WHERE  ss_quantity BETWEEN 61 AND 80)
           END bucket4,
           CASE
             WHEN (SELECT Count(*)
                   FROM   store_sales
                   WHERE  ss_quantity BETWEEN 81 AND 100) > {rc[4]} THEN
             (SELECT Avg({aggcthen})
              FROM   store_sales
              WHERE
             ss_quantity BETWEEN 81 AND 100)
             ELSE (SELECT Avg({aggcelse})
                   FROM   store_sales
                   WHERE  ss_quantity BETWEEN 81 AND 100)
           END bucket5
    FROM   reason
    WHERE  r_reason_sk = 1;
    """


def polars_impl(run_config: RunConfig) -> QueryResult:
    """Query 9."""
    params = load_parameters(
        int(run_config.scale_factor), query_id=9, qualification=run_config.qualification
    )

    aggcthen = params["aggcthen"]
    aggcelse = params["aggcelse"]
    rc = params["rc"]

    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    reason = get_data(run_config.dataset_path, "reason", run_config.suffix)

    thresholds = pl.LazyFrame({"bucket": [1, 2, 3, 4, 5], "threshold": list(rc)})

    # Single scan: the 5 ss_quantity ranges are non-overlapping, so a group_by
    # computes all counts and averages in one pass over store_sales.
    stats = (
        store_sales.with_columns(
            pl.when(pl.col("ss_quantity").is_between(1, 20))
            .then(pl.lit(1))
            .when(pl.col("ss_quantity").is_between(21, 40))
            .then(pl.lit(2))
            .when(pl.col("ss_quantity").is_between(41, 60))
            .then(pl.lit(3))
            .when(pl.col("ss_quantity").is_between(61, 80))
            .then(pl.lit(4))
            .when(pl.col("ss_quantity").is_between(81, 100))
            .then(pl.lit(5))
            .alias("bucket")
        )
        .filter(pl.col("bucket").is_not_null())
        .group_by("bucket")
        .agg(
            pl.len().alias("count"),
            pl.col(aggcthen).mean().alias("avg_then"),
            pl.col(aggcelse).mean().alias("avg_else"),
        )
        .join(thresholds, on="bucket")
        .select(
            pl.col("bucket"),
            pl.when(pl.col("count") > pl.col("threshold"))
            .then(pl.col("avg_then"))
            .otherwise(pl.col("avg_else"))
            .alias("value"),
        )
        .sort("bucket")
    )

    # Pivot 5 rows → 1 row with 5 named columns (operates on 5 rows, trivially fast)
    wide = stats.select(
        pl.col("value").filter(pl.col("bucket") == i).first().alias(f"bucket{i}")
        for i in range(1, 6)
    )

    return QueryResult(
        frame=(
            reason.filter(pl.col("r_reason_sk") == 1)
            .join(wide, how="cross")
            .select([f"bucket{i}" for i in range(1, 6)])
            .limit(1)
        ),
        sort_by=[],
        limit=1,
    )
