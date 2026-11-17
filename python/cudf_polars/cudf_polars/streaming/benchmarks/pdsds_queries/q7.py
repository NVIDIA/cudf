# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 7."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.streaming.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.streaming.benchmarks.utils import QueryResult, get_data

if TYPE_CHECKING:
    from cudf_polars.streaming.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 7."""
    params = load_parameters(
        int(run_config.scale_factor), query_id=7, qualification=run_config.qualification
    )

    year = params["year"]
    gender = params["gender"]
    marital_status = params["marital_status"]
    education_status = params["education_status"]
    promo_channel = params["promo_channel"]

    return f"""
    SELECT i_item_id,
                   Avg(ss_quantity)    agg1,
                   Avg(ss_list_price)  agg2,
                   Avg(ss_coupon_amt)  agg3,
                   Avg(ss_sales_price) agg4
    FROM   store_sales,
           customer_demographics,
           date_dim,
           item,
           promotion
    WHERE  ss_sold_date_sk = d_date_sk
           AND ss_item_sk = i_item_sk
           AND ss_cdemo_sk = cd_demo_sk
           AND ss_promo_sk = p_promo_sk
           AND cd_gender = '{gender}'
           AND cd_marital_status = '{marital_status}'
           AND cd_education_status = '{education_status}'
           AND ( p_channel_email = '{promo_channel}'
                  OR p_channel_event = '{promo_channel}' )
           AND d_year = {year}
    GROUP  BY i_item_id
    ORDER  BY i_item_id
    LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> QueryResult:
    """Query 7."""
    params = load_parameters(
        int(run_config.scale_factor), query_id=7, qualification=run_config.qualification
    )

    year = params["year"]
    gender = params["gender"]
    marital_status = params["marital_status"]
    education_status = params["education_status"]
    promo_channel = params["promo_channel"]

    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    customer_demographics = get_data(
        run_config.dataset_path, "customer_demographics", run_config.suffix
    )
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)
    promotion = get_data(run_config.dataset_path, "promotion", run_config.suffix)

    return QueryResult(
        frame=(
            store_sales.join(date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk")
            .join(item, left_on="ss_item_sk", right_on="i_item_sk")
            .join(customer_demographics, left_on="ss_cdemo_sk", right_on="cd_demo_sk")
            .join(promotion, left_on="ss_promo_sk", right_on="p_promo_sk")
            .filter(pl.col("cd_gender") == gender)
            .filter(pl.col("cd_marital_status") == marital_status)
            .filter(pl.col("cd_education_status") == education_status)
            .filter(
                (pl.col("p_channel_email") == promo_channel)
                | (pl.col("p_channel_event") == promo_channel)
            )
            .filter(pl.col("d_year") == year)
            .group_by("i_item_id")
            .agg(
                [
                    pl.col("ss_quantity").mean().alias("agg1"),
                    pl.col("ss_list_price").mean().alias("agg2"),
                    pl.col("ss_coupon_amt").mean().alias("agg3"),
                    pl.col("ss_sales_price").mean().alias("agg4"),
                ]
            )
            .sort("i_item_id", nulls_last=True)
            .limit(100)
        ),
        sort_by=[("i_item_id", False)],
        limit=100,
    )
