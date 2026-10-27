# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Query 61."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.streaming.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.streaming.benchmarks.utils import QueryResult, get_data

if TYPE_CHECKING:
    from cudf_polars.streaming.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 61."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=61,
        qualification=run_config.qualification,
    )

    year = params["year"]
    month = params["month"]
    gmt_offset = params["gmt_offset"]
    category = params["category"]

    return f"""
    SELECT promotions,
                   total,
                   Cast(promotions AS DECIMAL(15, 4)) /
                   Cast(total AS DECIMAL(15, 4)) * 100
    FROM   (SELECT Sum(ss_ext_sales_price) promotions
            FROM   store_sales,
                   store,
                   promotion,
                   date_dim,
                   customer,
                   customer_address,
                   item
            WHERE  ss_sold_date_sk = d_date_sk
                   AND ss_store_sk = s_store_sk
                   AND ss_promo_sk = p_promo_sk
                   AND ss_customer_sk = c_customer_sk
                   AND ca_address_sk = c_current_addr_sk
                   AND ss_item_sk = i_item_sk
                   AND ca_gmt_offset = {gmt_offset}
                   AND i_category = '{category}'
                   AND ( p_channel_dmail = 'Y'
                          OR p_channel_email = 'Y'
                          OR p_channel_tv = 'Y' )
                   AND s_gmt_offset = {gmt_offset}
                   AND d_year = {year}
                   AND d_moy = {month}) promotional_sales,
           (SELECT Sum(ss_ext_sales_price) total
            FROM   store_sales,
                   store,
                   date_dim,
                   customer,
                   customer_address,
                   item
            WHERE  ss_sold_date_sk = d_date_sk
                   AND ss_store_sk = s_store_sk
                   AND ss_customer_sk = c_customer_sk
                   AND ca_address_sk = c_current_addr_sk
                   AND ss_item_sk = i_item_sk
                   AND ca_gmt_offset = {gmt_offset}
                   AND i_category = '{category}'
                   AND s_gmt_offset = {gmt_offset}
                   AND d_year = {year}
                   AND d_moy = {month}) all_sales
    ORDER  BY promotions,
              total
    LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> QueryResult:
    """Query 61."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=61,
        qualification=run_config.qualification,
    )

    year = params["year"]
    month = params["month"]
    gmt_offset = params["gmt_offset"]
    category = params["category"]

    # Load tables
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    store = get_data(run_config.dataset_path, "store", run_config.suffix)
    promotion = get_data(run_config.dataset_path, "promotion", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    customer = get_data(run_config.dataset_path, "customer", run_config.suffix)
    customer_address = get_data(
        run_config.dataset_path, "customer_address", run_config.suffix
    )
    item = get_data(run_config.dataset_path, "item", run_config.suffix)

    # Promotional sales (with promotion filters)
    promotional_sales = (
        store_sales.join(store, left_on="ss_store_sk", right_on="s_store_sk")
        .join(promotion, left_on="ss_promo_sk", right_on="p_promo_sk")
        .join(date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk")
        .join(customer, left_on="ss_customer_sk", right_on="c_customer_sk")
        .join(customer_address, left_on="c_current_addr_sk", right_on="ca_address_sk")
        .join(item, left_on="ss_item_sk", right_on="i_item_sk")
        .filter(
            (pl.col("ca_gmt_offset") == gmt_offset)
            & (pl.col("i_category") == category)
            & (
                (pl.col("p_channel_dmail") == "Y")
                | (pl.col("p_channel_email") == "Y")
                | (pl.col("p_channel_tv") == "Y")
            )
            & (pl.col("s_gmt_offset") == gmt_offset)
            & (pl.col("d_year") == year)
            & (pl.col("d_moy") == month)
        )
        .select(
            [
                pl.when(pl.len() > 0)
                .then(pl.col("ss_ext_sales_price").sum())
                .otherwise(None)
                .alias("promotions")
            ]
        )
    )

    # All sales (no promotion filters)
    all_sales = (
        store_sales.join(store, left_on="ss_store_sk", right_on="s_store_sk")
        .join(date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk")
        .join(customer, left_on="ss_customer_sk", right_on="c_customer_sk")
        .join(customer_address, left_on="c_current_addr_sk", right_on="ca_address_sk")
        .join(item, left_on="ss_item_sk", right_on="i_item_sk")
        .filter(
            (pl.col("ca_gmt_offset") == gmt_offset)
            & (pl.col("i_category") == category)
            & (pl.col("s_gmt_offset") == gmt_offset)
            & (pl.col("d_year") == year)
            & (pl.col("d_moy") == month)
        )
        .select(
            [
                pl.when(pl.len() > 0)
                .then(pl.col("ss_ext_sales_price").sum())
                .otherwise(None)
                .alias("total")
            ]
        )
    )
    sort_by = {"promotions": False, "total": False}
    limit = 100
    return QueryResult(
        frame=(
            promotional_sales.join(all_sales, how="cross")
            .with_columns(
                [
                    pl.when(pl.col("total").is_null() | (pl.col("total") == 0))
                    .then(None)
                    .otherwise(
                        pl.col("promotions").cast(pl.Float64)
                        / pl.col("total").cast(pl.Float64)
                        * 100.0
                    )
                    .alias(
                        "((CAST(promotions AS DECIMAL(15, 4)) / CAST(total AS DECIMAL(15, 4))) * 100)"
                    )
                ]
            )
            .sort(sort_by.keys(), nulls_last=True)
            .limit(limit)
        ),
        sort_by=list(sort_by.items()),
        limit=limit,
    )
