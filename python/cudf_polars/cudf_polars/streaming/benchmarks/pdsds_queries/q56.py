# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Query 56."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.streaming.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.streaming.benchmarks.utils import QueryResult, get_data

if TYPE_CHECKING:
    from cudf_polars.streaming.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 56."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=56,
        qualification=run_config.qualification,
    )

    year = params["year"]
    month = params["month"]
    colors = params["colors"]
    gmt_offset = params["gmt_offset"]

    colors_str = ", ".join(f"'{c}'" for c in colors)

    return f"""
        WITH ss
            AS (SELECT i_item_id,
                        Sum(ss_ext_sales_price) total_sales
                FROM   store_sales,
                        date_dim,
                        customer_address,
                        item
                WHERE  i_item_id IN (SELECT i_item_id
                                    FROM   item
                                    WHERE  i_color IN ( {colors_str} )
                                    )
                        AND ss_item_sk = i_item_sk
                        AND ss_sold_date_sk = d_date_sk
                        AND d_year = {year}
                        AND d_moy = {month}
                        AND ss_addr_sk = ca_address_sk
                        AND ca_gmt_offset = {gmt_offset}
                GROUP  BY i_item_id),
            cs
            AS (SELECT i_item_id,
                        Sum(cs_ext_sales_price) total_sales
                FROM   catalog_sales,
                        date_dim,
                        customer_address,
                        item
                WHERE  i_item_id IN (SELECT i_item_id
                                    FROM   item
                                    WHERE  i_color IN ( {colors_str} )
                                    )
                        AND cs_item_sk = i_item_sk
                        AND cs_sold_date_sk = d_date_sk
                        AND d_year = {year}
                        AND d_moy = {month}
                        AND cs_bill_addr_sk = ca_address_sk
                        AND ca_gmt_offset = {gmt_offset}
                GROUP  BY i_item_id),
            ws
            AS (SELECT i_item_id,
                        Sum(ws_ext_sales_price) total_sales
                FROM   web_sales,
                        date_dim,
                        customer_address,
                        item
                WHERE  i_item_id IN (SELECT i_item_id
                                    FROM   item
                                    WHERE  i_color IN ( {colors_str} )
                                    )
                        AND ws_item_sk = i_item_sk
                        AND ws_sold_date_sk = d_date_sk
                        AND d_year = {year}
                        AND d_moy = {month}
                        AND ws_bill_addr_sk = ca_address_sk
                        AND ca_gmt_offset = {gmt_offset}
                GROUP  BY i_item_id)
        SELECT i_item_id,
                    Sum(total_sales) total_sales
        FROM   (SELECT *
                FROM   ss
                UNION ALL
                SELECT *
                FROM   cs
                UNION ALL
                SELECT *
                FROM   ws) tmp1
        GROUP  BY i_item_id
        ORDER  BY total_sales
        LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> QueryResult:
    """Query 56."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=56,
        qualification=run_config.qualification,
    )

    year = params["year"]
    month = params["month"]
    colors = params["colors"]
    gmt_offset = params["gmt_offset"]

    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    catalog_sales = get_data(
        run_config.dataset_path, "catalog_sales", run_config.suffix
    )
    web_sales = get_data(run_config.dataset_path, "web_sales", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    customer_address = get_data(
        run_config.dataset_path, "customer_address", run_config.suffix
    )
    item = get_data(run_config.dataset_path, "item", run_config.suffix)

    color_item_ids_lf = (
        item.filter(pl.col("i_color").is_in(colors)).select(["i_item_id"]).unique()
    )

    channels = [
        {
            "lf": store_sales,
            "sold_date_col": "ss_sold_date_sk",
            "item_sk_col": "ss_item_sk",
            "addr_sk_col": "ss_addr_sk",
            "ext_col": "ss_ext_sales_price",
        },
        {
            "lf": catalog_sales,
            "sold_date_col": "cs_sold_date_sk",
            "item_sk_col": "cs_item_sk",
            "addr_sk_col": "cs_bill_addr_sk",
            "ext_col": "cs_ext_sales_price",
        },
        {
            "lf": web_sales,
            "sold_date_col": "ws_sold_date_sk",
            "item_sk_col": "ws_item_sk",
            "addr_sk_col": "ws_bill_addr_sk",
            "ext_col": "ws_ext_sales_price",
        },
    ]

    per_channel = [
        (
            ch["lf"]  # type: ignore[attr-defined]
            .join(item, left_on=ch["item_sk_col"], right_on="i_item_sk")
            .join(color_item_ids_lf, on="i_item_id")
            .join(date_dim, left_on=ch["sold_date_col"], right_on="d_date_sk")
            .join(customer_address, left_on=ch["addr_sk_col"], right_on="ca_address_sk")
            .filter(
                (pl.col("d_year") == year)
                & (pl.col("d_moy") == month)
                & (pl.col("ca_gmt_offset") == gmt_offset)
            )
            .group_by("i_item_id")
            .agg(
                # Polars sum() returns 0 for all-null groups; SQL returns NULL.
                # See https://github.com/NVIDIA/cudf/issues/19560.
                pl.when(pl.col(str(ch["ext_col"])).count() > 0)
                .then(pl.col(str(ch["ext_col"])).sum())
                .otherwise(None)
                .alias("total_sales")
            )
            .select(["i_item_id", "total_sales"])
        )
        for ch in channels
    ]

    sort_by = {"total_sales": False}
    limit = 100
    return QueryResult(
        frame=(
            pl.concat(per_channel)
            .group_by("i_item_id")
            .agg(
                # Polars sum() returns 0 for all-null groups; SQL returns NULL.
                # See https://github.com/NVIDIA/cudf/issues/19560.
                pl.when(pl.col("total_sales").count() > 0)
                .then(pl.col("total_sales").sum())
                .otherwise(None)
                .alias("total_sales")
            )
            .select(["i_item_id", "total_sales"])
            .sort(sort_by.keys(), nulls_last=True)
            .limit(limit)
        ),
        sort_by=list(sort_by.items()),
        limit=limit,
    )
