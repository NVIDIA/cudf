# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 63."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.streaming.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.streaming.benchmarks.utils import QueryResult, get_data

if TYPE_CHECKING:
    from cudf_polars.streaming.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 63."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=63,
        qualification=run_config.qualification,
    )

    dms = params["dms"]

    return f"""
    SELECT *
    FROM   (SELECT i_manager_id,
                   Sum(ss_sales_price)            sum_sales,
                   Avg(Sum(ss_sales_price))
                     OVER (
                       partition BY i_manager_id) avg_monthly_sales
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
                   AND ( ( i_category IN ( 'Books', 'Children', 'Electronics' )
                           AND i_class IN ( 'personal', 'portable', 'reference',
                                            'self-help' )
                           AND i_brand IN ( 'scholaramalgamalg #14',
                                            'scholaramalgamalg #7'
                                            ,
                                            'exportiunivamalg #9',
                                                           'scholaramalgamalg #9' )
                         )
                          OR ( i_category IN ( 'Women', 'Music', 'Men' )
                               AND i_class IN ( 'accessories', 'classical',
                                                'fragrances',
                                                'pants' )
                               AND i_brand IN ( 'amalgimporto #1',
                                                'edu packscholar #1',
                                                'exportiimporto #1',
                                                    'importoamalg #1' ) ) )
            GROUP  BY i_manager_id,
                      d_moy) tmp1
    WHERE  CASE
             WHEN avg_monthly_sales > 0 THEN Abs (sum_sales - avg_monthly_sales) /
                                             avg_monthly_sales
             ELSE NULL
           END > 0.1
    ORDER  BY i_manager_id,
              avg_monthly_sales,
              sum_sales
    LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> QueryResult:
    """Query 63."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=63,
        qualification=run_config.qualification,
    )

    dms = params["dms"]

    item = get_data(run_config.dataset_path, "item", run_config.suffix)
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    store = get_data(run_config.dataset_path, "store", run_config.suffix)

    # Pre-filter both lookup tables before joining against store_sales [58 partitions].
    # item: apply both OR'd rule groups up front; only i_manager_id needed after.
    # date_dim: keep d_moy because it appears in the GROUP BY.
    filtered_item = item.filter(
        (
            pl.col("i_category").is_in(["Books", "Children", "Electronics"])
            & pl.col("i_class").is_in(
                ["personal", "portable", "reference", "self-help"]
            )
            & pl.col("i_brand").is_in(
                [
                    "scholaramalgamalg #14",
                    "scholaramalgamalg #7",
                    "exportiunivamalg #9",
                    "scholaramalgamalg #9",
                ]
            )
        )
        | (
            pl.col("i_category").is_in(["Women", "Music", "Men"])
            & pl.col("i_class").is_in(
                ["accessories", "classical", "fragrances", "pants"]
            )
            & pl.col("i_brand").is_in(
                [
                    "amalgimporto #1",
                    "edu packscholar #1",
                    "exportiimporto #1",
                    "importoamalg #1",
                ]
            )
        )
    ).select(["i_item_sk", "i_manager_id"])
    filtered_dates = date_dim.filter(
        pl.col("d_month_seq").is_in([dms + i for i in range(12)])
    ).select(["d_date_sk", "d_moy"])

    inner_query = (
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
        .group_by(["i_manager_id", "d_moy"])
        .agg([pl.col("ss_sales_price").sum().alias("sum_sales")])
        .with_columns(
            pl.col("sum_sales").mean().over("i_manager_id").alias("avg_monthly_sales")
        )
    )

    return QueryResult(
        frame=(
            inner_query.with_columns(
                pl.when(pl.col("avg_monthly_sales") > 0)
                .then(
                    (pl.col("sum_sales") - pl.col("avg_monthly_sales")).abs()
                    / pl.col("avg_monthly_sales")
                )
                .otherwise(None)
                .alias("deviation")
            )
            .filter(pl.col("deviation") > 0.1)
            .select(["i_manager_id", "sum_sales", "avg_monthly_sales"])
            .sort(
                ["i_manager_id", "avg_monthly_sales", "sum_sales"],
                nulls_last=True,
                descending=[False, False, False],
            )
            .limit(100)
        ),
        sort_by=[
            ("i_manager_id", False),
            ("avg_monthly_sales", False),
            ("sum_sales", False),
        ],
        limit=100,
    )
