# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 53."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.streaming.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.streaming.benchmarks.utils import QueryResult, get_data

if TYPE_CHECKING:
    from cudf_polars.streaming.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 53."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=53,
        qualification=run_config.qualification,
    )

    dms = params["dms"]
    categories1 = params["categories1"]
    classes1 = params["classes1"]
    brands1 = params["brands1"]
    categories2 = params["categories2"]
    classes2 = params["classes2"]
    brands2 = params["brands2"]

    # Build lists for SQL
    cat1_str = ", ".join(f"'{c}'" for c in categories1)
    class1_str = ", ".join(f"'{c}'" for c in classes1)
    brand1_str = ", ".join(f"'{b}'" for b in brands1)
    cat2_str = ", ".join(f"'{c}'" for c in categories2)
    class2_str = ", ".join(f"'{c}'" for c in classes2)
    brand2_str = ", ".join(f"'{b}'" for b in brands2)

    return f"""
    SELECT *
    FROM   (SELECT i_manufact_id,
                   Sum(ss_sales_price)             sum_sales,
                   Avg(Sum(ss_sales_price))
                     OVER (
                       partition BY i_manufact_id) avg_quarterly_sales
            FROM   item,
                   store_sales,
                   date_dim,
                   store
            WHERE  ss_item_sk = i_item_sk
                   AND ss_sold_date_sk = d_date_sk
                   AND ss_store_sk = s_store_sk
                   AND d_month_seq IN ( {dms}, {dms} + 1, {dms} + 2, {dms} + 3,
                                        {dms} + 4, {dms} + 5, {dms} + 6, {dms} + 7,
                                        {dms} + 8, {dms} + 9, {dms} + 10, {dms} + 11 )
                   AND ( ( i_category IN ( {cat1_str} )
                           AND i_class IN ( {class1_str} )
                           AND i_brand IN ( {brand1_str} )
                         )
                          OR ( i_category IN ( {cat2_str} )
                               AND i_class IN ( {class2_str} )
                               AND i_brand IN ( {brand2_str} ) ) )
            GROUP  BY i_manufact_id,
                      d_qoy) tmp1
    WHERE  CASE
             WHEN avg_quarterly_sales > 0 THEN Abs (sum_sales - avg_quarterly_sales)
                                               /
                                               avg_quarterly_sales
             ELSE NULL
           END > 0.1
    ORDER  BY avg_quarterly_sales,
              sum_sales,
              i_manufact_id
    LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> QueryResult:
    """Query 53."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=53,
        qualification=run_config.qualification,
    )

    dms = params["dms"]
    categories1 = params["categories1"]
    classes1 = params["classes1"]
    brands1 = params["brands1"]
    categories2 = params["categories2"]
    classes2 = params["classes2"]
    brands2 = params["brands2"]

    # Load tables
    item = get_data(run_config.dataset_path, "item", run_config.suffix)
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    store = get_data(run_config.dataset_path, "store", run_config.suffix)
    month_seq_list = list(range(dms, dms + 12))

    # Pre-filter lookup tables before joining against store_sales [87 partitions].
    # date_dim: keep d_qoy because it appears in the GROUP BY.
    filtered_dates = date_dim.filter(
        pl.col("d_month_seq").is_in(month_seq_list)
    ).select(["d_date_sk", "d_qoy"])
    # item: apply both OR'd rule groups up front; only i_manufact_id needed after.
    filtered_item = item.filter(
        (
            pl.col("i_category").is_in(categories1)
            & pl.col("i_class").is_in(classes1)
            & pl.col("i_brand").is_in(brands1)
        )
        | (
            pl.col("i_category").is_in(categories2)
            & pl.col("i_class").is_in(classes2)
            & pl.col("i_brand").is_in(brands2)
        )
    ).select(["i_item_sk", "i_manufact_id"])

    grouped_data = (
        store_sales.select(
            ["ss_sold_date_sk", "ss_item_sk", "ss_store_sk", "ss_sales_price"]
        )
        .join(filtered_item, left_on="ss_item_sk", right_on="i_item_sk")
        .join(filtered_dates, left_on="ss_sold_date_sk", right_on="d_date_sk")
        .join(
            store.select("s_store_sk"),
            left_on="ss_store_sk",
            right_on="s_store_sk",
            how="semi",
        )
        .group_by(["i_manufact_id", "d_qoy"])
        .agg([pl.col("ss_sales_price").sum().alias("sum_sales_raw")])
        .with_columns(
            [
                pl.when(pl.col("sum_sales_raw").is_not_null())
                .then(pl.col("sum_sales_raw"))
                .otherwise(None)
                .alias("sum(ss_sales_price)")
            ]
        )
    )
    non_null_data = grouped_data.filter(pl.col("i_manufact_id").is_not_null())
    null_data = grouped_data.filter(pl.col("i_manufact_id").is_null())
    manufacturer_averages = non_null_data.group_by("i_manufact_id").agg(
        [pl.col("sum(ss_sales_price)").mean().alias("avg_quarterly_sales")]
    )
    non_null_result = non_null_data.join(
        manufacturer_averages, on="i_manufact_id", how="left"
    )
    null_result = null_data.with_columns(
        [pl.col("sum(ss_sales_price)").mean().alias("avg_quarterly_sales")]
    )
    inner_query = pl.concat([non_null_result, null_result]).select(
        [
            "i_manufact_id",
            pl.col("sum(ss_sales_price)").alias("sum_sales"),
            "avg_quarterly_sales",
        ]
    )
    sort_by = {"avg_quarterly_sales": False, "sum_sales": False, "i_manufact_id": False}
    limit = 100
    return QueryResult(
        frame=(
            inner_query.filter(
                # Percentage deviation > 10%
                pl.when(pl.col("avg_quarterly_sales") > 0)
                .then(
                    (pl.col("sum_sales") - pl.col("avg_quarterly_sales")).abs()
                    / pl.col("avg_quarterly_sales")
                )
                .otherwise(None)
                > 0.1
            )
            .select(["i_manufact_id", "sum_sales", "avg_quarterly_sales"])
            .sort(sort_by.keys(), nulls_last=True)
            .limit(limit)
        ),
        sort_by=list(sort_by.items()),
        limit=limit,
    )
