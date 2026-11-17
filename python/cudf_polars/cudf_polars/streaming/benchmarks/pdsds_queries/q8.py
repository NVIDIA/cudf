# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 8."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.streaming.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.streaming.benchmarks.utils import QueryResult, get_data

if TYPE_CHECKING:
    from cudf_polars.streaming.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 8."""
    params = load_parameters(
        int(run_config.scale_factor), query_id=8, qualification=run_config.qualification
    )

    year = params["year"]
    qoy = params["qoy"]
    zip_codes = params["zip_codes"]

    return f"""
    -- start query 8 in stream 0 using template query8.tpl
    SELECT s_store_name,
                Sum(ss_net_profit)
    FROM   store_sales,
        date_dim,
        store,
        (SELECT ca_zip
            FROM   (SELECT Substr(ca_zip, 1, 5) ca_zip
                    FROM   customer_address
                    WHERE  Substr(ca_zip, 1, 5) IN ({", ".join(f"'{zip}'" for zip in zip_codes)})
                    INTERSECT
                    SELECT ca_zip
                    FROM   (SELECT Substr(ca_zip, 1, 5) ca_zip,
                                Count(*)             cnt
                            FROM   customer_address,
                                customer
                            WHERE  ca_address_sk = c_current_addr_sk
                                AND c_preferred_cust_flag = 'Y'
                            GROUP  BY ca_zip
                            HAVING Count(*) > 10)A1)A2) V1
    WHERE  ss_store_sk = s_store_sk
        AND ss_sold_date_sk = d_date_sk
        AND d_qoy = {qoy}
        AND d_year = {year}
        AND ( Substr(s_zip, 1, 2) = Substr(V1.ca_zip, 1, 2) )
    GROUP  BY s_store_name
    ORDER  BY s_store_name
    LIMIT 100;

    """


def polars_impl(run_config: RunConfig) -> QueryResult:
    """Query 8."""
    params = load_parameters(
        int(run_config.scale_factor), query_id=8, qualification=run_config.qualification
    )

    year = params["year"]
    qoy = params["qoy"]
    zip_codes = params["zip_codes"]

    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    store = get_data(run_config.dataset_path, "store", run_config.suffix)
    customer_address = get_data(
        run_config.dataset_path, "customer_address", run_config.suffix
    )
    customer = get_data(run_config.dataset_path, "customer", run_config.suffix)

    target_zips_5char = (
        customer_address.select(pl.col("ca_zip").str.slice(0, 5).alias("ca_zip"))
        .filter(pl.col("ca_zip").is_in(zip_codes))
        .unique()
    )

    # Second subquery: preferred customers by zip with count > 10
    preferred_customer_zips = (
        customer_address.join(
            customer, left_on="ca_address_sk", right_on="c_current_addr_sk"
        )
        .filter(pl.col("c_preferred_cust_flag") == "Y")
        .group_by(pl.col("ca_zip").str.slice(0, 5).alias("ca_zip"))
        .agg(pl.len().alias("cnt"))
        .filter(pl.col("cnt") > 10)
        .select("ca_zip")
    )

    # INTERSECT: Get common zip codes between target list and preferred customer zips
    intersect_zips = (
        target_zips_5char.join(preferred_customer_zips, on="ca_zip", how="inner")
        # TODO: The 2-char zip prefix is materialized here rather than passed as a join
        # expression key. Using `right_on=pl.col("ca_zip").str.slice(0, 2)` in the join
        # below produces non-deterministic wrong sums under the streaming executor. We need
        # to root-cause this.
        .select(pl.col("ca_zip").str.slice(0, 2).alias("ca_zip_prefix"))
    )

    # Pre-filter date_dim; d_year/d_qoy not needed after filter — semi-join.
    filtered_dates = date_dim.filter(
        (pl.col("d_year") == year) & (pl.col("d_qoy") == qoy)
    ).select("d_date_sk")

    return QueryResult(
        frame=(
            store_sales.select(["ss_sold_date_sk", "ss_store_sk", "ss_net_profit"])
            .join(
                filtered_dates,
                left_on="ss_sold_date_sk",
                right_on="d_date_sk",
                how="semi",
            )
            .join(
                store.select(
                    [
                        "s_store_sk",
                        "s_store_name",
                        pl.col("s_zip").str.slice(0, 2).alias("s_zip_prefix"),
                    ]
                ),
                left_on="ss_store_sk",
                right_on="s_store_sk",
            )
            .join(
                intersect_zips,
                left_on="s_zip_prefix",
                right_on="ca_zip_prefix",
            )
            .group_by("s_store_name")
            .agg(pl.col("ss_net_profit").sum().alias("sum"))
            .sort("s_store_name", nulls_last=True)
            .limit(100)
            .select([pl.col("s_store_name"), pl.col("sum").alias("sum(ss_net_profit)")])
        ),
        sort_by=[("s_store_name", False)],
        limit=100,
    )
