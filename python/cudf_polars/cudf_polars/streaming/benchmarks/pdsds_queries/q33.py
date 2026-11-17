# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 33."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.streaming.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.streaming.benchmarks.utils import QueryResult, get_data

if TYPE_CHECKING:
    from cudf_polars.streaming.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 33."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=33,
        qualification=run_config.qualification,
    )

    year = params["year"]
    month = params["month"]
    gmt = params["gmt"]
    category = params["category"]

    return f"""
    WITH ss
         AS (SELECT i_manufact_id,
                    Sum(ss_ext_sales_price) total_sales
             FROM   store_sales,
                    date_dim,
                    customer_address,
                    item
             WHERE  i_manufact_id IN (SELECT i_manufact_id
                                      FROM   item
                                      WHERE  i_category IN ( '{category}' ))
                    AND ss_item_sk = i_item_sk
                    AND ss_sold_date_sk = d_date_sk
                    AND d_year = {year}
                    AND d_moy = {month}
                    AND ss_addr_sk = ca_address_sk
                    AND ca_gmt_offset = {gmt}
             GROUP  BY i_manufact_id),
         cs
         AS (SELECT i_manufact_id,
                    Sum(cs_ext_sales_price) total_sales
             FROM   catalog_sales,
                    date_dim,
                    customer_address,
                    item
             WHERE  i_manufact_id IN (SELECT i_manufact_id
                                      FROM   item
                                      WHERE  i_category IN ( '{category}' ))
                    AND cs_item_sk = i_item_sk
                    AND cs_sold_date_sk = d_date_sk
                    AND d_year = {year}
                    AND d_moy = {month}
                    AND cs_bill_addr_sk = ca_address_sk
                    AND ca_gmt_offset = {gmt}
             GROUP  BY i_manufact_id),
         ws
         AS (SELECT i_manufact_id,
                    Sum(ws_ext_sales_price) total_sales
             FROM   web_sales,
                    date_dim,
                    customer_address,
                    item
             WHERE  i_manufact_id IN (SELECT i_manufact_id
                                      FROM   item
                                      WHERE  i_category IN ( '{category}' ))
                    AND ws_item_sk = i_item_sk
                    AND ws_sold_date_sk = d_date_sk
                    AND d_year = {year}
                    AND d_moy = {month}
                    AND ws_bill_addr_sk = ca_address_sk
                    AND ca_gmt_offset = {gmt}
             GROUP  BY i_manufact_id)
    SELECT i_manufact_id,
                   Sum(total_sales) total_sales
    FROM   (SELECT *
            FROM   ss
            UNION ALL
            SELECT *
            FROM   cs
            UNION ALL
            SELECT *
            FROM   ws) tmp1
    GROUP  BY i_manufact_id
    ORDER  BY total_sales
    LIMIT 100;
    """


def channel_total(  # noqa: D103
    sales: pl.LazyFrame,
    date_dim: pl.LazyFrame,
    customer_address: pl.LazyFrame,
    item: pl.LazyFrame,
    category_manufacturers: pl.LazyFrame,
    *,
    sold_date_key: str,
    addr_key: str,
    item_key: str,
    price_col: str,
    year: int,
    month: int,
    gmt: float,
) -> pl.LazyFrame:
    return (
        sales.join(date_dim, left_on=sold_date_key, right_on="d_date_sk")
        .join(customer_address, left_on=addr_key, right_on="ca_address_sk")
        .join(item, left_on=item_key, right_on="i_item_sk")
        .join(category_manufacturers, on="i_manufact_id")
        .filter(
            (pl.col("d_year") == year)
            & (pl.col("d_moy") == month)
            & (pl.col("ca_gmt_offset") == gmt)
        )
        .group_by("i_manufact_id")
        .agg(
            [
                pl.col(price_col).sum().alias("total_sales"),
                pl.col(price_col).count().alias("_n"),
            ]
        )
        .with_columns(
            pl.when(pl.col("_n") > 0)
            .then(pl.col("total_sales"))
            .otherwise(None)
            .alias("total_sales")
        )
        .drop("_n")
    )


def polars_impl(run_config: RunConfig) -> QueryResult:
    """Query 33."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=33,
        qualification=run_config.qualification,
    )

    year = params["year"]
    month = params["month"]
    gmt = params["gmt"]
    category = params["category"]

    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    catalog_sales = get_data(
        run_config.dataset_path, "catalog_sales", run_config.suffix
    )
    web_sales = get_data(run_config.dataset_path, "web_sales", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)
    customer_address = get_data(
        run_config.dataset_path, "customer_address", run_config.suffix
    )

    category_manufacturers = (
        item.filter(pl.col("i_category") == category).select("i_manufact_id").unique()
    )

    ss = channel_total(
        store_sales,
        date_dim,
        customer_address,
        item,
        category_manufacturers,
        sold_date_key="ss_sold_date_sk",
        addr_key="ss_addr_sk",
        item_key="ss_item_sk",
        price_col="ss_ext_sales_price",
        year=year,
        month=month,
        gmt=gmt,
    )
    cs = channel_total(
        catalog_sales,
        date_dim,
        customer_address,
        item,
        category_manufacturers,
        sold_date_key="cs_sold_date_sk",
        addr_key="cs_bill_addr_sk",
        item_key="cs_item_sk",
        price_col="cs_ext_sales_price",
        year=year,
        month=month,
        gmt=gmt,
    )
    ws = channel_total(
        web_sales,
        date_dim,
        customer_address,
        item,
        category_manufacturers,
        sold_date_key="ws_sold_date_sk",
        addr_key="ws_bill_addr_sk",
        item_key="ws_item_sk",
        price_col="ws_ext_sales_price",
        year=year,
        month=month,
        gmt=gmt,
    )

    sort_by = {"total_sales": False}
    limit = 100
    return QueryResult(
        frame=(
            pl.concat([ss, cs, ws])
            .group_by("i_manufact_id")
            .agg(
                [
                    pl.col("total_sales").sum().alias("total_sales"),
                    pl.col("total_sales").count().alias("_n"),
                ]
            )
            .with_columns(
                pl.when(pl.col("_n") > 0)
                .then(pl.col("total_sales"))
                .otherwise(None)
                .alias("total_sales")
            )
            .drop("_n")
            .select(["i_manufact_id", "total_sales"])
            .sort(sort_by.keys(), nulls_last=True)
            .limit(limit)
        ),
        sort_by=list(sort_by.items()),
        limit=limit,
    )
