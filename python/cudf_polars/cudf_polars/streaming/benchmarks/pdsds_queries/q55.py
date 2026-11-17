# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 55."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.streaming.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.streaming.benchmarks.utils import QueryResult, get_data

if TYPE_CHECKING:
    from cudf_polars.streaming.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 55."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=55,
        qualification=run_config.qualification,
    )

    year = params["year"]
    month = params["month"]
    manager_id = params["manager_id"]

    return f"""
    SELECT i_brand_id              brand_id,
                   i_brand                 brand,
                   Sum(ss_ext_sales_price) ext_price
    FROM   date_dim,
           store_sales,
           item
    WHERE  d_date_sk = ss_sold_date_sk
           AND ss_item_sk = i_item_sk
           AND i_manager_id = {manager_id}
           AND d_moy = {month}
           AND d_year = {year}
    GROUP  BY i_brand,
              i_brand_id
    ORDER  BY ext_price DESC,
              i_brand_id
    LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> QueryResult:
    """Query 55."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=55,
        qualification=run_config.qualification,
    )

    year = params["year"]
    month = params["month"]
    manager_id = params["manager_id"]

    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)
    sort_by = {"ext_price": True, "brand_id": False}
    limit = 100

    # d_year not in GROUP BY so date filter can be a semi-join (no date columns needed).
    filtered_dates = date_dim.filter(
        (pl.col("d_moy") == month) & (pl.col("d_year") == year)
    ).select("d_date_sk")
    filtered_item = item.filter(pl.col("i_manager_id") == manager_id).select(
        ["i_item_sk", "i_brand", "i_brand_id"]
    )

    return QueryResult(
        frame=(
            store_sales.select(["ss_sold_date_sk", "ss_item_sk", "ss_ext_sales_price"])
            .join(
                filtered_dates,
                left_on="ss_sold_date_sk",
                right_on="d_date_sk",
                how="semi",
            )
            .join(filtered_item, left_on="ss_item_sk", right_on="i_item_sk")
            .group_by(["i_brand", "i_brand_id"])
            .agg(pl.col("ss_ext_sales_price").sum().alias("ext_price"))
            .select(
                [
                    pl.col("i_brand_id").alias("brand_id"),
                    pl.col("i_brand").alias("brand"),
                    pl.col("ext_price"),
                ]
            )
            .sort(
                list(sort_by.keys()),
                descending=list(sort_by.values()),
                nulls_last=True,
            )
            .limit(limit)
        ),
        sort_by=list(sort_by.items()),
        limit=limit,
    )
