# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 11."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.streaming.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.streaming.benchmarks.utils import QueryResult, get_data

if TYPE_CHECKING:
    from cudf_polars.streaming.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 11."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=11,
        qualification=run_config.qualification,
    )

    year_first = params["year"]
    year_second = year_first + 1

    return f"""
    WITH year_total
         AS (
           SELECT c_customer_id                                customer_id,
                  c_first_name                                 customer_first_name,
                  c_last_name                                  customer_last_name,
                  c_preferred_cust_flag                        customer_preferred_cust_flag,
                  c_birth_country                              customer_birth_country,
                  c_login                                      customer_login,
                  c_email_address                              customer_email_address,
                  d_year                                       dyear,
                  SUM(ss_ext_list_price - ss_ext_discount_amt) year_total,
                  's'                                          sale_type
           FROM   customer,
                  store_sales,
                  date_dim
           WHERE  c_customer_sk = ss_customer_sk
                  AND ss_sold_date_sk = d_date_sk
           GROUP  BY c_customer_id,
                     c_first_name,
                     c_last_name,
                     c_preferred_cust_flag,
                     c_birth_country,
                     c_login,
                     c_email_address,
                     d_year
           UNION ALL
           SELECT c_customer_id                                customer_id,
                  c_first_name                                 customer_first_name,
                  c_last_name                                  customer_last_name,
                  c_preferred_cust_flag                        customer_preferred_cust_flag,
                  c_birth_country                              customer_birth_country,
                  c_login                                      customer_login,
                  c_email_address                              customer_email_address,
                  d_year                                       dyear,
                  SUM(ws_ext_list_price - ws_ext_discount_amt) year_total,
                  'w'                                          sale_type
           FROM   customer,
                  web_sales,
                  date_dim
           WHERE  c_customer_sk = ws_bill_customer_sk
                  AND ws_sold_date_sk = d_date_sk
           GROUP  BY c_customer_id,
                     c_first_name,
                     c_last_name,
                     c_preferred_cust_flag,
                     c_birth_country,
                     c_login,
                     c_email_address,
                     d_year
         )
    SELECT t_s_secyear.customer_id,
           t_s_secyear.customer_first_name,
           t_s_secyear.customer_last_name,
           t_s_secyear.customer_birth_country
    FROM   year_total t_s_firstyear,
           year_total t_s_secyear,
           year_total t_w_firstyear,
           year_total t_w_secyear
    WHERE  t_s_secyear.customer_id = t_s_firstyear.customer_id
           AND t_s_firstyear.customer_id = t_w_secyear.customer_id
           AND t_s_firstyear.customer_id = t_w_firstyear.customer_id
           AND t_s_firstyear.sale_type = 's'
           AND t_w_firstyear.sale_type = 'w'
           AND t_s_secyear.sale_type = 's'
           AND t_w_secyear.sale_type = 'w'
           AND t_s_firstyear.dyear = {year_first}
           AND t_s_secyear.dyear = {year_second}
           AND t_w_firstyear.dyear = {year_first}
           AND t_w_secyear.dyear = {year_second}
           AND t_s_firstyear.year_total > 0
           AND t_w_firstyear.year_total > 0
           AND CASE
                 WHEN t_w_firstyear.year_total > 0 THEN
                   t_w_secyear.year_total / t_w_firstyear.year_total
                 ELSE 0.0
               END
             > CASE
                 WHEN t_s_firstyear.year_total > 0 THEN
                   t_s_secyear.year_total / t_s_firstyear.year_total
                 ELSE 0.0
               END
    ORDER  BY t_s_secyear.customer_id,
              t_s_secyear.customer_first_name,
              t_s_secyear.customer_last_name,
              t_s_secyear.customer_birth_country
    LIMIT 100;
    """


CUSTOMER_COMPOSITE = [
    "c_customer_id",
    "c_first_name",
    "c_last_name",
    "c_preferred_cust_flag",
    "c_birth_country",
    "c_login",
    "c_email_address",
]


def build_year_total(
    sales_table: pl.LazyFrame,
    customer_table: pl.LazyFrame,
    customer_sk_col: str,
    date_sk_col: str,
    list_price_col: str,
    discount_col: str,
    year_dates: pl.LazyFrame,
) -> pl.LazyFrame:
    """Aggregate sales joined with customer for a single year, keyed on customer composite."""
    return (
        sales_table.join(year_dates, left_on=date_sk_col, right_on="d_date_sk")
        .join(customer_table, left_on=customer_sk_col, right_on="c_customer_sk")
        .group_by(CUSTOMER_COMPOSITE)
        .agg(
            [(pl.col(list_price_col) - pl.col(discount_col)).sum().alias("year_total")]
        )
    )


def polars_impl(run_config: RunConfig) -> QueryResult:
    """Query 11."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=11,
        qualification=run_config.qualification,
    )

    year_first = params["year"]
    year_second = year_first + 1

    customer = get_data(run_config.dataset_path, "customer", run_config.suffix)
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    web_sales = get_data(run_config.dataset_path, "web_sales", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)

    date_first = date_dim.filter(pl.col("d_year") == year_first).select("d_date_sk")
    date_second = date_dim.filter(pl.col("d_year") == year_second).select("d_date_sk")

    t_s_first = (
        build_year_total(
            store_sales,
            customer,
            "ss_customer_sk",
            "ss_sold_date_sk",
            "ss_ext_list_price",
            "ss_ext_discount_amt",
            date_first,
        )
        .rename({"year_total": "s_first_year_total"})
        .filter(pl.col("s_first_year_total") > 0)
    )

    t_s_sec = build_year_total(
        store_sales,
        customer,
        "ss_customer_sk",
        "ss_sold_date_sk",
        "ss_ext_list_price",
        "ss_ext_discount_amt",
        date_second,
    ).rename({"year_total": "s_sec_year_total"})

    t_w_first = (
        build_year_total(
            web_sales,
            customer,
            "ws_bill_customer_sk",
            "ws_sold_date_sk",
            "ws_ext_list_price",
            "ws_ext_discount_amt",
            date_first,
        )
        .rename({"year_total": "w_first_year_total"})
        .filter(pl.col("w_first_year_total") > 0)
    )

    t_w_sec = build_year_total(
        web_sales,
        customer,
        "ws_bill_customer_sk",
        "ws_sold_date_sk",
        "ws_ext_list_price",
        "ws_ext_discount_amt",
        date_second,
    ).rename({"year_total": "w_sec_year_total"})

    return QueryResult(
        frame=(
            t_s_sec.join(t_s_first, on=CUSTOMER_COMPOSITE, nulls_equal=True)
            .join(t_w_first, on=CUSTOMER_COMPOSITE, nulls_equal=True)
            .join(t_w_sec, on=CUSTOMER_COMPOSITE, nulls_equal=True)
            .filter(
                pl.col("w_sec_year_total") / pl.col("w_first_year_total")
                > pl.col("s_sec_year_total") / pl.col("s_first_year_total")
            )
            .select(
                [
                    pl.col("c_customer_id").alias("customer_id"),
                    pl.col("c_first_name").alias("customer_first_name"),
                    pl.col("c_last_name").alias("customer_last_name"),
                    pl.col("c_birth_country").alias("customer_birth_country"),
                ]
            )
            .sort(
                [
                    "customer_id",
                    "customer_first_name",
                    "customer_last_name",
                    "customer_birth_country",
                ],
                nulls_last=True,
            )
            .limit(100)
        ),
        sort_by=[
            ("customer_id", False),
            ("customer_first_name", False),
            ("customer_last_name", False),
            ("customer_birth_country", False),
        ],
        limit=100,
    )
