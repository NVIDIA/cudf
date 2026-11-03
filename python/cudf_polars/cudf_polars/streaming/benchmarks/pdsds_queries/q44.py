# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 44."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.streaming.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.streaming.benchmarks.utils import QueryResult, get_data

if TYPE_CHECKING:
    from cudf_polars.streaming.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 44."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=44,
        qualification=run_config.qualification,
    )

    store_sk = params["store_sk"]

    return f"""
    SELECT ascending.rnk,
                   i1.i_product_name best_performing,
                   i2.i_product_name worst_performing
    FROM  (SELECT *
           FROM   (SELECT item_sk,
                          Rank()
                            OVER (
                              ORDER BY rank_col ASC) rnk
                   FROM   (SELECT ss_item_sk         item_sk,
                                  Avg(ss_net_profit) rank_col
                           FROM   store_sales ss1
                           WHERE  ss_store_sk = {store_sk}
                           GROUP  BY ss_item_sk
                           HAVING Avg(ss_net_profit) > 0.9 *
                                  (SELECT Avg(ss_net_profit)
                                          rank_col
                                   FROM   store_sales
                                   WHERE  ss_store_sk = {store_sk}
                                          AND ss_cdemo_sk IS
                                              NULL
                                   GROUP  BY ss_store_sk))V1)
                  V11
           WHERE  rnk < 11) ascending,
          (SELECT *
           FROM   (SELECT item_sk,
                          Rank()
                            OVER (
                              ORDER BY rank_col DESC) rnk
                   FROM   (SELECT ss_item_sk         item_sk,
                                  Avg(ss_net_profit) rank_col
                           FROM   store_sales ss1
                           WHERE  ss_store_sk = {store_sk}
                           GROUP  BY ss_item_sk
                           HAVING Avg(ss_net_profit) > 0.9 *
                                  (SELECT Avg(ss_net_profit)
                                          rank_col
                                   FROM   store_sales
                                   WHERE  ss_store_sk = {store_sk}
                                          AND ss_cdemo_sk IS
                                              NULL
                                   GROUP  BY ss_store_sk))V2)
                  V21
           WHERE  rnk < 11) descending,
          item i1,
          item i2
    WHERE  ascending.rnk = descending.rnk
           AND i1.i_item_sk = ascending.item_sk
           AND i2.i_item_sk = descending.item_sk
    ORDER  BY ascending.rnk
    LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> QueryResult:
    """Query 44."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=44,
        qualification=run_config.qualification,
    )

    store_sk = params["store_sk"]

    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)

    # Benchmark: global mean profit for the store with null demographics — single row.
    # Use a constant-key equi-join instead of how="cross" so the streaming executor
    # treats it as a broadcast join (1 row ≤ broadcast_join_limit) rather than a
    # ConditionalJoin that falls back from multi-partition mode.
    benchmark = (
        store_sales.filter(
            (pl.col("ss_store_sk") == store_sk) & pl.col("ss_cdemo_sk").is_null()
        )
        .select(pl.col("ss_net_profit").mean().alias("benchmark_profit"))
        .with_columns(pl.lit(1, dtype=pl.Int32).alias("_key"))
    )

    # Item-level average profits, broadcast-joined with the 1-row benchmark.
    item_profits = (
        store_sales.filter(pl.col("ss_store_sk") == store_sk)
        .group_by("ss_item_sk")
        .agg(pl.col("ss_net_profit").mean().alias("avg_profit"))
        .with_columns(pl.lit(1, dtype=pl.Int32).alias("_key"))
        .join(benchmark, on="_key")
        .filter(pl.col("avg_profit") > 0.9 * pl.col("benchmark_profit"))
        .select(["ss_item_sk", "avg_profit"])
    )

    ascending_rank = (
        item_profits.with_columns(pl.col("avg_profit").rank(method="min").alias("rnk"))
        .filter(pl.col("rnk") < 11)
        .select(["ss_item_sk", "rnk"])
    )

    descending_rank = (
        item_profits.with_columns(
            pl.col("avg_profit").rank(method="min", descending=True).alias("rnk")
        )
        .filter(pl.col("rnk") < 11)
        .select(["ss_item_sk", "rnk"])
    )

    item_cols = item.select(["i_item_sk", "i_product_name"])
    sort_by = {"rnk": False}
    limit = 100
    return QueryResult(
        frame=(
            ascending_rank.join(descending_rank, on="rnk", how="inner", suffix="_desc")
            .join(item_cols, left_on="ss_item_sk", right_on="i_item_sk", how="inner")
            .join(
                item_cols,
                left_on="ss_item_sk_desc",
                right_on="i_item_sk",
                how="inner",
                suffix="_worst",
            )
            .select(
                [
                    pl.col("rnk"),
                    pl.col("i_product_name").alias("best_performing"),
                    pl.col("i_product_name_worst").alias("worst_performing"),
                ]
            )
            .sort(sort_by.keys(), nulls_last=True)
            .limit(limit)
        ),
        sort_by=list(sort_by.items()),
        limit=limit,
    )
