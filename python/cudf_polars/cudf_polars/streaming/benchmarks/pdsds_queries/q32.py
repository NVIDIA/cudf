# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Query 32."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.streaming.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.streaming.benchmarks.pdsds_queries import sql_sum
from cudf_polars.streaming.benchmarks.utils import QueryResult, get_data

if TYPE_CHECKING:
    from cudf_polars.streaming.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 32."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=32,
        qualification=run_config.qualification,
    )

    imid = params["imid"]
    csdate = params["csdate"]

    return f"""
    SELECT
           Sum(cs_ext_discount_amt) AS 'excess discount amount'
    FROM   catalog_sales ,
           item ,
           date_dim
    WHERE  i_manufact_id = {imid}
    AND    i_item_sk = cs_item_sk
    AND    d_date BETWEEN '{csdate}' AND    (
                  Cast('{csdate}' AS DATE) + INTERVAL '90' day)
    AND    d_date_sk = cs_sold_date_sk
    AND    cs_ext_discount_amt >
           (
                  SELECT 1.3 * avg(cs_ext_discount_amt)
                  FROM   catalog_sales ,
                         date_dim
                  WHERE  cs_item_sk = i_item_sk
                  AND    d_date BETWEEN '{csdate}' AND    (
                                cast('{csdate}' AS date) + INTERVAL '90' day)
                  AND    d_date_sk = cs_sold_date_sk )
    LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> QueryResult:
    """Query 32."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=32,
        qualification=run_config.qualification,
    )

    imid = params["imid"]
    csdate = params["csdate"]

    # Load tables
    catalog_sales = get_data(
        run_config.dataset_path, "catalog_sales", run_config.suffix
    )
    item = get_data(run_config.dataset_path, "item", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)

    start_date_obj = datetime.strptime(csdate, "%Y-%m-%d")
    end_date_obj = start_date_obj + timedelta(days=90)

    start_date = pl.date(start_date_obj.year, start_date_obj.month, start_date_obj.day)
    end_date = pl.date(end_date_obj.year, end_date_obj.month, end_date_obj.day)

    # First, calculate the average discount amount for each item in the date range
    item_avg_discounts = (
        catalog_sales.join(date_dim, left_on="cs_sold_date_sk", right_on="d_date_sk")
        .filter(pl.col("d_date").is_between(start_date, end_date))
        .group_by("cs_item_sk")
        .agg([(pl.col("cs_ext_discount_amt").mean() * 1.3).alias("threshold_discount")])
    )
    # Main query: find items with specified manufacturer and high discount amounts
    return QueryResult(
        frame=(
            catalog_sales.join(item, left_on="cs_item_sk", right_on="i_item_sk")
            .join(date_dim, left_on="cs_sold_date_sk", right_on="d_date_sk")
            .join(item_avg_discounts, on="cs_item_sk")
            .filter(
                (pl.col("i_manufact_id") == imid)
                & (pl.col("d_date").is_between(start_date, end_date))
                & (pl.col("cs_ext_discount_amt") > pl.col("threshold_discount"))
            )
            .select([sql_sum("cs_ext_discount_amt").alias("excess discount amount")])
            .limit(100)
        ),
        sort_by=[],
        limit=None,
    )
