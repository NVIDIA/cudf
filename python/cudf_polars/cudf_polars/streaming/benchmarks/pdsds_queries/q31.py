# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Query 31."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.streaming.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.streaming.benchmarks.utils import QueryResult, get_data

if TYPE_CHECKING:
    from cudf_polars.streaming.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 31."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=31,
        qualification=run_config.qualification,
    )

    year = params["year"]
    agg = params["agg"]

    return f"""
    WITH ss
         AS (SELECT ca_county,
                    d_qoy,
                    d_year,
                    Sum(ss_ext_sales_price) AS store_sales
             FROM   store_sales,
                    date_dim,
                    customer_address
             WHERE  ss_sold_date_sk = d_date_sk
                    AND ss_addr_sk = ca_address_sk
             GROUP  BY ca_county,
                       d_qoy,
                       d_year),
         ws
         AS (SELECT ca_county,
                    d_qoy,
                    d_year,
                    Sum(ws_ext_sales_price) AS web_sales
             FROM   web_sales,
                    date_dim,
                    customer_address
             WHERE  ws_sold_date_sk = d_date_sk
                    AND ws_bill_addr_sk = ca_address_sk
             GROUP  BY ca_county,
                       d_qoy,
                       d_year)
    SELECT ss1.ca_county,
           ss1.d_year,
           ws2.web_sales / ws1.web_sales     web_q1_q2_increase,
           ss2.store_sales / ss1.store_sales store_q1_q2_increase,
           ws3.web_sales / ws2.web_sales     web_q2_q3_increase,
           ss3.store_sales / ss2.store_sales store_q2_q3_increase
    FROM   ss ss1,
           ss ss2,
           ss ss3,
           ws ws1,
           ws ws2,
           ws ws3
    WHERE  ss1.d_qoy = 1
           AND ss1.d_year = {year}
           AND ss1.ca_county = ss2.ca_county
           AND ss2.d_qoy = 2
           AND ss2.d_year = {year}
           AND ss2.ca_county = ss3.ca_county
           AND ss3.d_qoy = 3
           AND ss3.d_year = {year}
           AND ss1.ca_county = ws1.ca_county
           AND ws1.d_qoy = 1
           AND ws1.d_year = {year}
           AND ws1.ca_county = ws2.ca_county
           AND ws2.d_qoy = 2
           AND ws2.d_year = {year}
           AND ws1.ca_county = ws3.ca_county
           AND ws3.d_qoy = 3
           AND ws3.d_year = {year}
           AND CASE
                 WHEN ws1.web_sales > 0 THEN ws2.web_sales / ws1.web_sales
                 ELSE NULL
               END > CASE
                       WHEN ss1.store_sales > 0 THEN
                       ss2.store_sales / ss1.store_sales
                       ELSE NULL
                     END
           AND CASE
                 WHEN ws2.web_sales > 0 THEN ws3.web_sales / ws2.web_sales
                 ELSE NULL
               END > CASE
                       WHEN ss2.store_sales > 0 THEN
                       ss3.store_sales / ss2.store_sales
                       ELSE NULL
                     END
    ORDER  BY {agg};
    """


def polars_impl(run_config: RunConfig) -> QueryResult:
    """Query 31."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=31,
        qualification=run_config.qualification,
    )

    year = params["year"]
    agg = params["agg"]

    # Map SQL field names to Polars column names (remove "ss1." prefix)
    polars_agg = agg.replace("ss1.", "")

    # Load tables
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    web_sales = get_data(run_config.dataset_path, "web_sales", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    customer_address = get_data(
        run_config.dataset_path, "customer_address", run_config.suffix
    )

    # Pre-filter date_dim to target year to avoid aggregating across all years.
    # Build per-quarter aggregations directly instead of filtering after the fact.
    date_year = date_dim.filter(pl.col("d_year") == year).select(["d_date_sk", "d_qoy"])

    def build_quarter_agg(
        sales_table: pl.LazyFrame,
        date_key: str,
        addr_key: str,
        price_col: str,
        agg_alias: str,
        quarter: int,
    ) -> pl.LazyFrame:
        date_q = date_year.filter(pl.col("d_qoy") == quarter).select("d_date_sk")
        return (
            sales_table.join(date_q, left_on=date_key, right_on="d_date_sk")
            .join(customer_address, left_on=addr_key, right_on="ca_address_sk")
            .group_by("ca_county")
            .agg([pl.col(price_col).sum().alias(agg_alias)])
        )

    ss1 = build_quarter_agg(
        store_sales,
        "ss_sold_date_sk",
        "ss_addr_sk",
        "ss_ext_sales_price",
        "store_sales",
        1,
    )
    ss2 = build_quarter_agg(
        store_sales,
        "ss_sold_date_sk",
        "ss_addr_sk",
        "ss_ext_sales_price",
        "store_sales",
        2,
    )
    ss3 = build_quarter_agg(
        store_sales,
        "ss_sold_date_sk",
        "ss_addr_sk",
        "ss_ext_sales_price",
        "store_sales",
        3,
    )
    ws1 = build_quarter_agg(
        web_sales,
        "ws_sold_date_sk",
        "ws_bill_addr_sk",
        "ws_ext_sales_price",
        "web_sales",
        1,
    )
    ws2 = build_quarter_agg(
        web_sales,
        "ws_sold_date_sk",
        "ws_bill_addr_sk",
        "ws_ext_sales_price",
        "web_sales",
        2,
    )
    ws3 = build_quarter_agg(
        web_sales,
        "ws_sold_date_sk",
        "ws_bill_addr_sk",
        "ws_ext_sales_price",
        "web_sales",
        3,
    )

    sort_by = {polars_agg: False}
    # Join all quarters together by county
    result = (
        ss1.join(ss2, on="ca_county", suffix="_q2")
        .join(ss3, on="ca_county", suffix="_q3")
        .join(ws1, on="ca_county", suffix="_ws1")
        .join(ws2, on="ca_county", suffix="_ws2")
        .join(ws3, on="ca_county", suffix="_ws3")
        .with_columns(
            [
                # DuckDB SQL promotes Decimal/Decimal to Float64; Polars keeps it as
                # Decimal. Cast to Float64 to match DuckDB return type.
                pl.when(pl.col("web_sales") > 0)
                .then(
                    pl.col("web_sales_ws2").cast(pl.Float64)
                    / pl.col("web_sales").cast(pl.Float64)
                )
                .otherwise(None)
                .alias("web_q1_q2_increase"),
                pl.when(pl.col("store_sales") > 0)
                .then(
                    pl.col("store_sales_q2").cast(pl.Float64)
                    / pl.col("store_sales").cast(pl.Float64)
                )
                .otherwise(None)
                .alias("store_q1_q2_increase"),
                pl.when(pl.col("web_sales_ws2") > 0)
                .then(
                    pl.col("web_sales_ws3").cast(pl.Float64)
                    / pl.col("web_sales_ws2").cast(pl.Float64)
                )
                .otherwise(None)
                .alias("web_q2_q3_increase"),
                pl.when(pl.col("store_sales_q2") > 0)
                .then(
                    pl.col("store_sales_q3").cast(pl.Float64)
                    / pl.col("store_sales_q2").cast(pl.Float64)
                )
                .otherwise(None)
                .alias("store_q2_q3_increase"),
            ]
        )
        .filter(
            (
                pl.when(pl.col("web_sales") > 0)
                .then(
                    pl.col("web_sales_ws2").cast(pl.Float64)
                    / pl.col("web_sales").cast(pl.Float64)
                )
                .otherwise(None)
                > pl.when(pl.col("store_sales") > 0)
                .then(
                    pl.col("store_sales_q2").cast(pl.Float64)
                    / pl.col("store_sales").cast(pl.Float64)
                )
                .otherwise(None)
            )
            & (
                pl.when(pl.col("web_sales_ws2") > 0)
                .then(
                    pl.col("web_sales_ws3").cast(pl.Float64)
                    / pl.col("web_sales_ws2").cast(pl.Float64)
                )
                .otherwise(None)
                > pl.when(pl.col("store_sales_q2") > 0)
                .then(
                    pl.col("store_sales_q3").cast(pl.Float64)
                    / pl.col("store_sales_q2").cast(pl.Float64)
                )
                .otherwise(None)
            )
        )
        .select(
            [
                pl.col("ca_county"),
                pl.lit(year, dtype=pl.Int64).alias("d_year"),
                "web_q1_q2_increase",
                "store_q1_q2_increase",
                "web_q2_q3_increase",
                "store_q2_q3_increase",
            ]
        )
        .sort(sort_by.keys(), nulls_last=True)
    )
    return QueryResult(frame=result, sort_by=list(sort_by.items()), limit=None)
