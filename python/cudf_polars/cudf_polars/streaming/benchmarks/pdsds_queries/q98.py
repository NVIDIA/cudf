# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 98."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.streaming.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.streaming.benchmarks.utils import QueryResult, get_data

if TYPE_CHECKING:
    from cudf_polars.streaming.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 98."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=98,
        qualification=run_config.qualification,
    )

    date = params["date"]
    categories_str = ", ".join([f"'{c}'" for c in params["categories"]])

    return f"""
    -- start query 98 in stream 0 using template query98.tpl
    SELECT i_item_id,
           i_item_desc,
           i_category,
           i_class,
           i_current_price,
           Sum(ss_ext_sales_price)                                   AS itemrevenue,
           Sum(ss_ext_sales_price) * 100 / Sum(Sum(ss_ext_sales_price))
                                             OVER (
                                               PARTITION BY i_class) AS revenueratio
    FROM   store_sales,
           item,
           date_dim
    WHERE  ss_item_sk = i_item_sk
           AND i_category IN ( {categories_str} )
           AND ss_sold_date_sk = d_date_sk
           AND d_date BETWEEN CAST('{date}' AS DATE) AND (
                              CAST('{date}' AS DATE) + INTERVAL '30' DAY )
    GROUP  BY i_item_id,
              i_item_desc,
              i_category,
              i_class,
              i_current_price
    ORDER  BY i_category,
              i_class,
              i_item_id,
              i_item_desc,
              revenueratio;
    """


def polars_impl(run_config: RunConfig) -> QueryResult:
    """Query 98."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=98,
        qualification=run_config.qualification,
    )

    date = params["date"]
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    start_date_py = datetime.strptime(date, "%Y-%m-%d").date()
    end_date_py = start_date_py + timedelta(days=30)
    start_date = pl.date(start_date_py.year, start_date_py.month, start_date_py.day)
    end_date = pl.date(end_date_py.year, end_date_py.month, end_date_py.day)

    # Pre-filter item to matching categories before joining against store_sales [58 partitions].
    filtered_item = item.filter(
        pl.col("i_category").is_in(params["categories"])
    ).select(
        [
            "i_item_sk",
            "i_item_id",
            "i_item_desc",
            "i_current_price",
            "i_class",
            "i_category",
        ]
    )
    # Pre-filter date_dim to the 30-day window; d_date not needed after filter — semi-join.
    filtered_dates = date_dim.filter(
        pl.col("d_date").is_between(start_date, end_date, closed="both")
    ).select("d_date_sk")

    return QueryResult(
        frame=(
            store_sales.select(["ss_sold_date_sk", "ss_item_sk", "ss_ext_sales_price"])
            .join(filtered_item, left_on="ss_item_sk", right_on="i_item_sk")
            .join(
                filtered_dates,
                left_on="ss_sold_date_sk",
                right_on="d_date_sk",
                how="semi",
            )
            .group_by(
                [
                    "i_item_id",
                    "i_item_desc",
                    "i_category",
                    "i_class",
                    "i_current_price",
                ]
            )
            .agg(
                [
                    pl.col("ss_ext_sales_price").count().alias("itemrevenue_count"),
                    pl.col("ss_ext_sales_price").sum().alias("itemrevenue_sum"),
                ]
            )
            .with_columns(
                [
                    pl.when(pl.col("itemrevenue_count") == 0)
                    .then(None)
                    .otherwise(pl.col("itemrevenue_sum"))
                    .alias("itemrevenue")
                ]
            )
            .with_columns(
                [
                    (
                        pl.col("itemrevenue")
                        * 100.0
                        / pl.col("itemrevenue").sum().over("i_class")
                    ).alias("revenueratio")
                ]
            )
            .select(
                [
                    "i_item_id",
                    "i_item_desc",
                    "i_category",
                    "i_class",
                    "i_current_price",
                    "itemrevenue",
                    "revenueratio",
                ]
            )
            .sort(
                ["i_category", "i_class", "i_item_id", "i_item_desc", "revenueratio"],
                nulls_last=True,
            )
        ),
        sort_by=[
            ("i_category", False),
            ("i_class", False),
            ("i_item_id", False),
            ("i_item_desc", False),
            ("revenueratio", False),
        ],
        limit=None,
    )
