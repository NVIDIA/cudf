# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 68."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.streaming.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.streaming.benchmarks.utils import QueryResult, get_data

if TYPE_CHECKING:
    from cudf_polars.streaming.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 68."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=68,
        qualification=run_config.qualification,
    )

    year = params["year"]
    dep_cnt = params["dep_cnt"]
    veh_cnt = params["veh_cnt"]
    city_a = params["city_a"]
    city_b = params["city_b"]

    return f"""
    SELECT c_last_name,
                   c_first_name,
                   ca_city,
                   bought_city,
                   ss_ticket_number,
                   extended_price,
                   extended_tax,
                   list_price
    FROM   (SELECT ss_ticket_number,
                   ss_customer_sk,
                   ca_city                 bought_city,
                   Sum(ss_ext_sales_price) extended_price,
                   Sum(ss_ext_list_price)  list_price,
                   Sum(ss_ext_tax)         extended_tax
            FROM   store_sales,
                   date_dim,
                   store,
                   household_demographics,
                   customer_address
            WHERE  store_sales.ss_sold_date_sk = date_dim.d_date_sk
                   AND store_sales.ss_store_sk = store.s_store_sk
                   AND store_sales.ss_hdemo_sk = household_demographics.hd_demo_sk
                   AND store_sales.ss_addr_sk = customer_address.ca_address_sk
                   AND date_dim.d_dom BETWEEN 1 AND 2
                   AND ( household_demographics.hd_dep_count = {dep_cnt}
                          OR household_demographics.hd_vehicle_count = {veh_cnt} )
                   AND date_dim.d_year IN ( {year}, {year} + 1, {year} + 2 )
                   AND store.s_city IN ( '{city_a}', '{city_b}' )
            GROUP  BY ss_ticket_number,
                      ss_customer_sk,
                      ss_addr_sk,
                      ca_city) dn,
           customer,
           customer_address current_addr
    WHERE  ss_customer_sk = c_customer_sk
           AND customer.c_current_addr_sk = current_addr.ca_address_sk
           AND current_addr.ca_city <> bought_city
    ORDER  BY c_last_name,
              ss_ticket_number
    LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> QueryResult:
    """Query 68."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=68,
        qualification=run_config.qualification,
    )

    year = params["year"]
    dep_cnt = params["dep_cnt"]
    veh_cnt = params["veh_cnt"]
    city_a = params["city_a"]
    city_b = params["city_b"]

    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    store = get_data(run_config.dataset_path, "store", run_config.suffix)
    household_demographics = get_data(
        run_config.dataset_path, "household_demographics", run_config.suffix
    )
    customer_address = get_data(
        run_config.dataset_path, "customer_address", run_config.suffix
    )
    customer = get_data(run_config.dataset_path, "customer", run_config.suffix)
    dn = (
        store_sales.join(date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk")
        .join(store, left_on="ss_store_sk", right_on="s_store_sk")
        .join(household_demographics, left_on="ss_hdemo_sk", right_on="hd_demo_sk")
        .join(customer_address, left_on="ss_addr_sk", right_on="ca_address_sk")
        .filter(
            pl.col("d_dom").is_between(1, 2)
            & (
                (pl.col("hd_dep_count") == dep_cnt)
                | (pl.col("hd_vehicle_count") == veh_cnt)
            )
            & pl.col("d_year").is_in([year, year + 1, year + 2])
            & pl.col("s_city").is_in([city_a, city_b])
        )
        .group_by(["ss_ticket_number", "ss_customer_sk", "ss_addr_sk", "ca_city"])
        .agg(
            [
                pl.col("ss_ext_sales_price").sum().alias("extended_price"),
                pl.col("ss_ext_list_price").sum().alias("list_price"),
                pl.col("ss_ext_tax").sum().alias("extended_tax"),
            ]
        )
        .with_columns([pl.col("ca_city").alias("bought_city")])
    )
    return QueryResult(
        frame=(
            dn.join(customer, left_on="ss_customer_sk", right_on="c_customer_sk")
            .join(
                customer_address.select(
                    ["ca_address_sk", pl.col("ca_city").alias("current_city")]
                ),
                left_on="c_current_addr_sk",
                right_on="ca_address_sk",
            )
            .filter(pl.col("current_city") != pl.col("bought_city"))
            .select(
                [
                    "c_last_name",
                    "c_first_name",
                    pl.col("current_city").alias("ca_city"),
                    "bought_city",
                    "ss_ticket_number",
                    "extended_price",
                    "extended_tax",
                    "list_price",
                ]
            )
            .sort(["c_last_name", "ss_ticket_number"], nulls_last=True)
            .limit(100)
        ),
        sort_by=[("c_last_name", False), ("ss_ticket_number", False)],
        limit=100,
    )
