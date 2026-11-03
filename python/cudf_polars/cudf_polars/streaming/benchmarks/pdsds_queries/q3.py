# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 3."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.streaming.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.streaming.benchmarks.utils import QueryResult, get_data

if TYPE_CHECKING:
    from cudf_polars.streaming.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 3."""
    params = load_parameters(
        int(run_config.scale_factor), query_id=3, qualification=run_config.qualification
    )

    aggc = params["aggc"]
    month = params["month"]
    manufact = params["manufact"]

    return f"""
    SELECT dt.d_year,
                   item.i_brand_id          brand_id,
                   item.i_brand             brand,
                   Sum({aggc})              sum_agg
    FROM   date_dim dt,
           store_sales,
           item
    WHERE  dt.d_date_sk = store_sales.ss_sold_date_sk
           AND store_sales.ss_item_sk = item.i_item_sk
           AND item.i_manufact_id = {manufact}
           AND dt.d_moy = {month}
    GROUP  BY dt.d_year,
              item.i_brand,
              item.i_brand_id
    ORDER  BY dt.d_year,
              sum_agg DESC,
              brand_id
    LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> QueryResult:
    """Query 3."""
    params = load_parameters(
        int(run_config.scale_factor), query_id=3, qualification=run_config.qualification
    )

    aggc = params["aggc"]
    month = params["month"]
    manufact = params["manufact"]

    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)
    return QueryResult(
        frame=(
            date_dim.join(store_sales, left_on="d_date_sk", right_on="ss_sold_date_sk")
            .join(item, left_on="ss_item_sk", right_on="i_item_sk")
            .filter((pl.col("i_manufact_id") == manufact) & (pl.col("d_moy") == month))
            .group_by(["d_year", "i_brand", "i_brand_id"])
            .agg([pl.col(aggc).sum().alias("sum_agg")])
            .select(
                [
                    pl.col("d_year"),
                    pl.col("i_brand_id").alias("brand_id"),
                    pl.col("i_brand").alias("brand"),
                    pl.col("sum_agg"),
                ]
            )
            .sort(["d_year", "sum_agg", "brand_id"], descending=[False, True, False])
            .limit(100)
        ),
        sort_by=[("d_year", False), ("sum_agg", True), ("brand_id", False)],
        limit=100,
    )
