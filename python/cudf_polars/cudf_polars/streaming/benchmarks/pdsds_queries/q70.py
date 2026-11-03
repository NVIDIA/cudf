# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 70."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.streaming.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.streaming.benchmarks.utils import QueryResult, get_data

if TYPE_CHECKING:
    from cudf_polars.streaming.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 70."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=70,
        qualification=run_config.qualification,
    )

    dms = params["dms"]

    return f"""
    SELECT Sum(ss_net_profit)                     AS total_sum,
                   s_state,
                   s_county,
                   Grouping(s_state) + Grouping(s_county) AS lochierarchy,
                   Rank()
                     OVER (
                       partition BY Grouping(s_state)+Grouping(s_county), CASE WHEN
                     Grouping(
                     s_county) = 0 THEN s_state END
                       ORDER BY Sum(ss_net_profit) DESC)  AS rank_within_parent
    FROM   store_sales,
           date_dim d1,
           store
    WHERE  d1.d_month_seq BETWEEN {dms} AND {dms} + 11
           AND d1.d_date_sk = ss_sold_date_sk
           AND s_store_sk = ss_store_sk
           AND s_state IN (SELECT s_state
                           FROM   (SELECT s_state                               AS
                                          s_state,
                                          Rank()
                                            OVER (
                                              partition BY s_state
                                              ORDER BY Sum(ss_net_profit) DESC) AS
                                          ranking
                                   FROM   store_sales,
                                          store,
                                          date_dim
                                   WHERE  d_month_seq BETWEEN {dms} AND {dms} + 11
                                          AND d_date_sk = ss_sold_date_sk
                                          AND s_store_sk = ss_store_sk
                                   GROUP  BY s_state) tmp1
                           WHERE  ranking <= 5)
    GROUP  BY rollup( s_state, s_county )
    ORDER  BY lochierarchy DESC,
              CASE
                WHEN lochierarchy = 0 THEN s_state
              END,
              rank_within_parent
    LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> QueryResult:
    """Query 70."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=70,
        qualification=run_config.qualification,
    )

    dms = params["dms"]

    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    store = get_data(run_config.dataset_path, "store", run_config.suffix)

    top_states = (
        store_sales.join(date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk")
        .join(store, left_on="ss_store_sk", right_on="s_store_sk")
        .filter(pl.col("d_month_seq").is_between(dms, dms + 11))
        .select("s_state")
        .unique()
    )

    base = (
        store_sales.join(date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk")
        .join(store, left_on="ss_store_sk", right_on="s_store_sk")
        .filter(pl.col("d_month_seq").is_between(dms, dms + 11))
        .join(top_states, on="s_state", how="semi")
    )

    detail = (
        base.group_by(["s_state", "s_county"])
        .agg(pl.col("ss_net_profit").sum().alias("total_sum"))
        .with_columns(pl.lit(0, dtype=pl.Int64).alias("lochierarchy"))
    )

    by_state = (
        detail.group_by("s_state")
        .agg(pl.col("total_sum").sum())
        .with_columns(
            [
                pl.lit(None, dtype=pl.Utf8).alias("s_county"),
                pl.lit(1, dtype=pl.Int64).alias("lochierarchy"),
            ]
        )
        .select(["s_state", "s_county", "total_sum", "lochierarchy"])
    )

    total = (
        detail.select(pl.col("total_sum").sum())
        .with_columns(
            [
                pl.lit(None, dtype=pl.Utf8).alias("s_state"),
                pl.lit(None, dtype=pl.Utf8).alias("s_county"),
                pl.lit(2, dtype=pl.Int64).alias("lochierarchy"),
            ]
        )
        .select(["s_state", "s_county", "total_sum", "lochierarchy"])
    )

    return QueryResult(
        frame=(
            pl.concat([detail, by_state, total])
            .with_columns(
                pl.when(pl.col("lochierarchy") == 0)
                .then(pl.col("s_state"))
                .otherwise(None)
                .alias("partition_key")
            )
            .with_columns(
                pl.col("total_sum")
                .rank(method="min", descending=True)
                .over(["lochierarchy", "partition_key"])
                .alias("rank_within_parent")
            )
            .select(
                [
                    "total_sum",
                    "s_state",
                    "s_county",
                    "lochierarchy",
                    "rank_within_parent",
                ]
            )
            .sort(
                [
                    pl.col("lochierarchy"),
                    pl.when(pl.col("lochierarchy") == 0)
                    .then(pl.col("s_state"))
                    .otherwise(None),
                    pl.col("rank_within_parent"),
                ],
                descending=[True, False, False],
                nulls_last=True,
            )
            .limit(100)
        ),
        sort_by=[("lochierarchy", True), ("rank_within_parent", False)],
        limit=100,
        sort_keys=[
            (pl.col("lochierarchy"), True),
            (
                pl.when(pl.col("lochierarchy") == 0)
                .then(pl.col("s_state"))
                .otherwise(None),
                False,
            ),
            (pl.col("rank_within_parent"), False),
        ],
    )
