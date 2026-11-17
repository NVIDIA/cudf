# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 52."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.streaming.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.streaming.benchmarks.utils import QueryResult, get_data

if TYPE_CHECKING:
    from cudf_polars.streaming.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 52."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=52,
        qualification=run_config.qualification,
    )

    year = params["year"]
    month = params["month"]
    manager_id = params["manager_id"]

    return f"""
        SELECT dt.d_year,
                    item.i_brand_id         brand_id,
                    item.i_brand            brand,
                    Sum(ss_ext_sales_price) ext_price
        FROM   date_dim dt,
            store_sales,
            item
        WHERE  dt.d_date_sk = store_sales.ss_sold_date_sk
            AND store_sales.ss_item_sk = item.i_item_sk
            AND item.i_manager_id = {manager_id}
            AND dt.d_moy = {month}
            AND dt.d_year = {year}
        GROUP  BY dt.d_year,
                item.i_brand,
                item.i_brand_id
        ORDER  BY dt.d_year,
                ext_price DESC,
                brand_id
        LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> QueryResult:
    """Query 52."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=52,
        qualification=run_config.qualification,
    )

    year = params["year"]
    month = params["month"]
    manager_id = params["manager_id"]

    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)

    sort_by = {"d_year": False, "ext_price": True, "brand_id": False}
    limit = 100

    # Pre-filter both lookup tables before joining against store_sales [87 partitions].
    # date_dim keeps d_year because it appears in the GROUP BY.
    filtered_dates = date_dim.filter(
        (pl.col("d_moy") == month) & (pl.col("d_year") == year)
    ).select(["d_date_sk", "d_year"])
    filtered_item = item.filter(pl.col("i_manager_id") == manager_id).select(
        ["i_item_sk", "i_brand", "i_brand_id"]
    )

    return QueryResult(
        frame=(
            store_sales.select(["ss_sold_date_sk", "ss_item_sk", "ss_ext_sales_price"])
            .join(filtered_dates, left_on="ss_sold_date_sk", right_on="d_date_sk")
            .join(filtered_item, left_on="ss_item_sk", right_on="i_item_sk")
            .group_by(["d_year", "i_brand", "i_brand_id"])
            .agg(pl.col("ss_ext_sales_price").sum().alias("ext_price"))
            .select(
                [
                    pl.col("d_year"),
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
