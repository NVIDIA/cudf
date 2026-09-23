# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Query 64."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.streaming.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.streaming.benchmarks.utils import QueryResult, get_data

if TYPE_CHECKING:
    from cudf_polars.streaming.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 64."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=64,
        qualification=run_config.qualification,
    )

    year = params["year"]
    price = params["price"]
    colors = params["colors"]
    colors_str = ", ".join(f"'{c}'" for c in colors)

    return f"""
    WITH cs_ui
         AS (SELECT cs_item_sk,
                    Sum(cs_ext_list_price) AS sale,
                    Sum(cr_refunded_cash + cr_reversed_charge
                        + cr_store_credit) AS refund
             FROM   catalog_sales,
                    catalog_returns
             WHERE  cs_item_sk = cr_item_sk
                    AND cs_order_number = cr_order_number
             GROUP  BY cs_item_sk
             HAVING Sum(cs_ext_list_price) > 2 * Sum(
                    cr_refunded_cash + cr_reversed_charge
                    + cr_store_credit)),
         cross_sales
         AS (SELECT i_product_name         product_name,
                    i_item_sk              item_sk,
                    s_store_name           store_name,
                    s_zip                  store_zip,
                    ad1.ca_street_number   b_street_number,
                    ad1.ca_street_name     b_streen_name,
                    ad1.ca_city            b_city,
                    ad1.ca_zip             b_zip,
                    ad2.ca_street_number   c_street_number,
                    ad2.ca_street_name     c_street_name,
                    ad2.ca_city            c_city,
                    ad2.ca_zip             c_zip,
                    d1.d_year              AS syear,
                    d2.d_year              AS fsyear,
                    d3.d_year              s2year,
                    Count(*)               cnt,
                    Sum(ss_wholesale_cost) s1,
                    Sum(ss_list_price)     s2,
                    Sum(ss_coupon_amt)     s3
             FROM   store_sales,
                    store_returns,
                    cs_ui,
                    date_dim d1,
                    date_dim d2,
                    date_dim d3,
                    store,
                    customer,
                    customer_demographics cd1,
                    customer_demographics cd2,
                    promotion,
                    household_demographics hd1,
                    household_demographics hd2,
                    customer_address ad1,
                    customer_address ad2,
                    income_band ib1,
                    income_band ib2,
                    item
             WHERE  ss_store_sk = s_store_sk
                    AND ss_sold_date_sk = d1.d_date_sk
                    AND ss_customer_sk = c_customer_sk
                    AND ss_cdemo_sk = cd1.cd_demo_sk
                    AND ss_hdemo_sk = hd1.hd_demo_sk
                    AND ss_addr_sk = ad1.ca_address_sk
                    AND ss_item_sk = i_item_sk
                    AND ss_item_sk = sr_item_sk
                    AND ss_ticket_number = sr_ticket_number
                    AND ss_item_sk = cs_ui.cs_item_sk
                    AND c_current_cdemo_sk = cd2.cd_demo_sk
                    AND c_current_hdemo_sk = hd2.hd_demo_sk
                    AND c_current_addr_sk = ad2.ca_address_sk
                    AND c_first_sales_date_sk = d2.d_date_sk
                    AND c_first_shipto_date_sk = d3.d_date_sk
                    AND ss_promo_sk = p_promo_sk
                    AND hd1.hd_income_band_sk = ib1.ib_income_band_sk
                    AND hd2.hd_income_band_sk = ib2.ib_income_band_sk
                    AND cd1.cd_marital_status <> cd2.cd_marital_status
                    AND i_color IN ( {colors_str} )
                    AND i_current_price BETWEEN {price} AND {price} + 10
                    AND i_current_price BETWEEN {price} + 1 AND {price} + 15
             GROUP  BY i_product_name,
                       i_item_sk,
                       s_store_name,
                       s_zip,
                       ad1.ca_street_number,
                       ad1.ca_street_name,
                       ad1.ca_city,
                       ad1.ca_zip,
                       ad2.ca_street_number,
                       ad2.ca_street_name,
                       ad2.ca_city,
                       ad2.ca_zip,
                       d1.d_year,
                       d2.d_year,
                       d3.d_year)
    SELECT cs1.product_name,
           cs1.store_name,
           cs1.store_zip,
           cs1.b_street_number,
           cs1.b_streen_name,
           cs1.b_city,
           cs1.b_zip,
           cs1.c_street_number,
           cs1.c_street_name,
           cs1.c_city,
           cs1.c_zip,
           cs1.syear,
           cs1.cnt,
           cs1.s1,
           cs1.s2,
           cs1.s3,
           cs2.s1,
           cs2.s2,
           cs2.s3,
           cs2.syear,
           cs2.cnt
    FROM   cross_sales cs1,
           cross_sales cs2
    WHERE  cs1.item_sk = cs2.item_sk
           AND cs1.syear = {year}
           AND cs2.syear = {year} + 1
           AND cs2.cnt <= cs1.cnt
           AND cs1.store_name = cs2.store_name
           AND cs1.store_zip = cs2.store_zip
    ORDER  BY cs1.product_name,
              cs1.store_name,
              cs2.cnt,
              cs1.s1,
              cs2.s1;
    """


def polars_impl(run_config: RunConfig) -> QueryResult:
    """Query 64."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=64,
        qualification=run_config.qualification,
    )

    year = params["year"]
    price = params["price"]
    colors = params["colors"]

    # Load all required tables
    catalog_sales = get_data(
        run_config.dataset_path, "catalog_sales", run_config.suffix
    )
    catalog_returns = get_data(
        run_config.dataset_path, "catalog_returns", run_config.suffix
    )
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    store_returns = get_data(
        run_config.dataset_path, "store_returns", run_config.suffix
    )
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    store = get_data(run_config.dataset_path, "store", run_config.suffix)
    customer = get_data(run_config.dataset_path, "customer", run_config.suffix)
    customer_demographics = get_data(
        run_config.dataset_path, "customer_demographics", run_config.suffix
    )
    customer_address = get_data(
        run_config.dataset_path, "customer_address", run_config.suffix
    )
    household_demographics = get_data(
        run_config.dataset_path, "household_demographics", run_config.suffix
    )
    income_band = get_data(run_config.dataset_path, "income_band", run_config.suffix)
    promotion = get_data(run_config.dataset_path, "promotion", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)

    # CTE 1: cs_ui - catalog sales items with sales > 2 * refunds
    cs_ui = (
        catalog_sales.select(["cs_item_sk", "cs_order_number", "cs_ext_list_price"])
        .join(
            catalog_returns.select(
                [
                    "cr_item_sk",
                    "cr_order_number",
                    "cr_refunded_cash",
                    "cr_reversed_charge",
                    "cr_store_credit",
                ]
            ),
            left_on=["cs_item_sk", "cs_order_number"],
            right_on=["cr_item_sk", "cr_order_number"],
        )
        .group_by("cs_item_sk")
        .agg(
            [
                pl.col("cs_ext_list_price").sum().alias("sale"),
                (
                    pl.col("cr_refunded_cash")
                    + pl.col("cr_reversed_charge")
                    + pl.col("cr_store_credit")
                )
                .sum()
                .alias("refund"),
            ]
        )
        .filter(pl.col("sale") > 2 * pl.col("refund"))
        .select("cs_item_sk")
    )

    # Filter items by color and price
    filtered_items = item.filter(
        pl.col("i_color").is_in(colors)
        & pl.col("i_current_price").is_between(price, price + 10)
        & pl.col("i_current_price").is_between(price + 1, price + 15)
    ).select(["i_item_sk", "i_product_name"])

    # Prepare date dimensions
    d2 = date_dim.select(["d_date_sk", "d_year"]).rename(
        {"d_date_sk": "d2_date_sk", "d_year": "d2_year"}
    )
    d3 = date_dim.select(["d_date_sk", "d_year"]).rename(
        {"d_date_sk": "d3_date_sk", "d_year": "d3_year"}
    )

    # Prepare customer demographics
    cd1 = customer_demographics.select(["cd_demo_sk", "cd_marital_status"]).rename(
        {"cd_demo_sk": "cd1_demo_sk", "cd_marital_status": "cd1_marital_status"}
    )
    cd2 = customer_demographics.select(["cd_demo_sk", "cd_marital_status"]).rename(
        {"cd_demo_sk": "cd2_demo_sk", "cd_marital_status": "cd2_marital_status"}
    )

    # Prepare household demographics
    hd1 = household_demographics.select(["hd_demo_sk", "hd_income_band_sk"]).rename(
        {"hd_demo_sk": "hd1_demo_sk", "hd_income_band_sk": "hd1_income_band_sk"}
    )
    hd2 = household_demographics.select(["hd_demo_sk", "hd_income_band_sk"]).rename(
        {"hd_demo_sk": "hd2_demo_sk", "hd_income_band_sk": "hd2_income_band_sk"}
    )

    # Prepare income bands
    ib1 = income_band.select(["ib_income_band_sk"]).rename(
        {"ib_income_band_sk": "ib1_income_band_sk"}
    )
    ib2 = income_band.select(["ib_income_band_sk"]).rename(
        {"ib_income_band_sk": "ib2_income_band_sk"}
    )

    # Prepare customer addresses
    ad1 = customer_address.select(
        ["ca_address_sk", "ca_street_number", "ca_street_name", "ca_city", "ca_zip"]
    ).rename(
        {
            "ca_address_sk": "ad1_address_sk",
            "ca_street_number": "b_street_number",
            "ca_street_name": "b_streen_name",
            "ca_city": "b_city",
            "ca_zip": "b_zip",
        }
    )
    ad2 = customer_address.select(
        ["ca_address_sk", "ca_street_number", "ca_street_name", "ca_city", "ca_zip"]
    ).rename(
        {
            "ca_address_sk": "ad2_address_sk",
            "ca_street_number": "c_street_number",
            "ca_street_name": "c_street_name",
            "ca_city": "c_city",
            "ca_zip": "c_zip",
        }
    )

    # Building the cross-sales for each year separately reduces the size of the table
    # substantially prior to performing the expensive join.
    def build_cross_sales_for_year(target_year: int) -> pl.LazyFrame:
        d1 = date_dim.filter(pl.col("d_year") == target_year).select("d_date_sk")

        return (
            # Start with the most selective filters: item filter + cs_ui
            store_sales.join(filtered_items, left_on="ss_item_sk", right_on="i_item_sk")
            .join(cs_ui, left_on="ss_item_sk", right_on="cs_item_sk")
            # date_dim is the next most selective filter
            .join(d1, left_on="ss_sold_date_sk", right_on="d_date_sk")
            .join(
                store_returns.select(["sr_item_sk", "sr_ticket_number"]),
                left_on=["ss_item_sk", "ss_ticket_number"],
                right_on=["sr_item_sk", "sr_ticket_number"],
            )
            .join(
                store.select(["s_store_sk", "s_store_name", "s_zip"]),
                left_on="ss_store_sk",
                right_on="s_store_sk",
            )
            .join(
                customer.select(
                    [
                        "c_customer_sk",
                        "c_current_cdemo_sk",
                        "c_current_hdemo_sk",
                        "c_current_addr_sk",
                        "c_first_sales_date_sk",
                        "c_first_shipto_date_sk",
                    ]
                ),
                left_on="ss_customer_sk",
                right_on="c_customer_sk",
            )
            .join(cd1, left_on="ss_cdemo_sk", right_on="cd1_demo_sk")
            .join(cd2, left_on="c_current_cdemo_sk", right_on="cd2_demo_sk")
            .filter(pl.col("cd1_marital_status") != pl.col("cd2_marital_status"))
            .join(hd1, left_on="ss_hdemo_sk", right_on="hd1_demo_sk")
            .join(hd2, left_on="c_current_hdemo_sk", right_on="hd2_demo_sk")
            .join(ib1, left_on="hd1_income_band_sk", right_on="ib1_income_band_sk")
            .join(ib2, left_on="hd2_income_band_sk", right_on="ib2_income_band_sk")
            .join(ad1, left_on="ss_addr_sk", right_on="ad1_address_sk")
            .join(ad2, left_on="c_current_addr_sk", right_on="ad2_address_sk")
            .join(d2, left_on="c_first_sales_date_sk", right_on="d2_date_sk")
            .join(d3, left_on="c_first_shipto_date_sk", right_on="d3_date_sk")
            .join(
                promotion.select(["p_promo_sk"]),
                left_on="ss_promo_sk",
                right_on="p_promo_sk",
            )
            .group_by(
                [
                    "i_product_name",
                    "ss_item_sk",
                    "s_store_name",
                    "s_zip",
                    "b_street_number",
                    "b_streen_name",
                    "b_city",
                    "b_zip",
                    "c_street_number",
                    "c_street_name",
                    "c_city",
                    "c_zip",
                    "d2_year",
                    "d3_year",
                ]
            )
            .agg(
                [
                    pl.len().alias("cnt"),
                    # Polars sum() returns 0 for all-null groups; SQL returns NULL.
                    # See https://github.com/NVIDIA/cudf/issues/19560.
                    pl.when(pl.col("ss_wholesale_cost").count() > 0)
                    .then(pl.col("ss_wholesale_cost").sum())
                    .otherwise(None)
                    .alias("s1"),
                    pl.when(pl.col("ss_list_price").count() > 0)
                    .then(pl.col("ss_list_price").sum())
                    .otherwise(None)
                    .alias("s2"),
                    pl.when(pl.col("ss_coupon_amt").count() > 0)
                    .then(pl.col("ss_coupon_amt").sum())
                    .otherwise(None)
                    .alias("s3"),
                ]
            )
            .select(
                [
                    pl.col("i_product_name").alias("product_name"),
                    pl.col("ss_item_sk").alias("item_sk"),
                    pl.col("s_store_name").alias("store_name"),
                    pl.col("s_zip").alias("store_zip"),
                    "b_street_number",
                    "b_streen_name",
                    "b_city",
                    "b_zip",
                    "c_street_number",
                    "c_street_name",
                    "c_city",
                    "c_zip",
                    pl.col("d2_year").alias("fsyear"),
                    pl.col("d3_year").alias("s2year"),
                    "cnt",
                    "s1",
                    "s2",
                    "s3",
                ]
            )
        )

    cs1 = build_cross_sales_for_year(year)
    cs2 = build_cross_sales_for_year(year + 1)

    # Final query: join year1 and year2 cross_sales
    return QueryResult(
        frame=(
            cs1.join(
                cs2,
                left_on=["item_sk", "store_name", "store_zip"],
                right_on=["item_sk", "store_name", "store_zip"],
                suffix="_1",
            )
            .filter(pl.col("cnt_1") <= pl.col("cnt"))
            .with_columns(pl.lit(year, dtype=pl.Int64).alias("syear"))
            .select(
                [
                    "product_name",
                    "store_name",
                    "store_zip",
                    "b_street_number",
                    "b_streen_name",
                    "b_city",
                    "b_zip",
                    "c_street_number",
                    "c_street_name",
                    "c_city",
                    "c_zip",
                    "syear",
                    "cnt",
                    "s1",
                    "s2",
                    "s3",
                    pl.col("s1_1"),
                    pl.col("s2_1"),
                    pl.col("s3_1"),
                    pl.lit(year + 1, dtype=pl.Int64).alias("syear_1"),
                    pl.col("cnt_1"),
                ]
            )
            .sort(
                ["product_name", "store_name", "cnt_1", "s1", "s1_1"],
                nulls_last=True,
            )
        ),
        sort_by=[
            ("product_name", False),
            ("store_name", False),
            ("cnt_1", False),
            ("s1", False),
            ("s1_1", False),
        ],
        limit=None,
    )
