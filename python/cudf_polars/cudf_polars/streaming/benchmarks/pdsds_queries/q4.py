# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 4."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.streaming.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.streaming.benchmarks.utils import QueryResult, get_data

if TYPE_CHECKING:
    from cudf_polars.streaming.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 4."""
    params = load_parameters(
        int(run_config.scale_factor), query_id=4, qualification=run_config.qualification
    )

    year = params["year"]

    return f"""
    WITH year_total
         AS (SELECT c_customer_id                       customer_id,
                    c_first_name                        customer_first_name,
                    c_last_name                         customer_last_name,
                    c_preferred_cust_flag               customer_preferred_cust_flag
                    ,
                    c_birth_country
                    customer_birth_country,
                    c_login                             customer_login,
                    c_email_address                     customer_email_address,
                    d_year                              dyear,
                    Sum(( ( ss_ext_list_price - ss_ext_wholesale_cost
                            - ss_ext_discount_amt
                          )
                          +
                              ss_ext_sales_price ) / 2) year_total,
                    's'                                 sale_type
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
             SELECT c_customer_id                             customer_id,
                    c_first_name                              customer_first_name,
                    c_last_name                               customer_last_name,
                    c_preferred_cust_flag
                    customer_preferred_cust_flag,
                    c_birth_country                           customer_birth_country
                    ,
                    c_login
                    customer_login,
                    c_email_address                           customer_email_address
                    ,
                    d_year                                    dyear
                    ,
                    Sum(( ( ( cs_ext_list_price
                              - cs_ext_wholesale_cost
                              - cs_ext_discount_amt
                            ) +
                                  cs_ext_sales_price ) / 2 )) year_total,
                    'c'                                       sale_type
             FROM   customer,
                    catalog_sales,
                    date_dim
             WHERE  c_customer_sk = cs_bill_customer_sk
                    AND cs_sold_date_sk = d_date_sk
             GROUP  BY c_customer_id,
                       c_first_name,
                       c_last_name,
                       c_preferred_cust_flag,
                       c_birth_country,
                       c_login,
                       c_email_address,
                       d_year
             UNION ALL
             SELECT c_customer_id                             customer_id,
                    c_first_name                              customer_first_name,
                    c_last_name                               customer_last_name,
                    c_preferred_cust_flag
                    customer_preferred_cust_flag,
                    c_birth_country                           customer_birth_country
                    ,
                    c_login
                    customer_login,
                    c_email_address                           customer_email_address
                    ,
                    d_year                                    dyear
                    ,
                    Sum(( ( ( ws_ext_list_price
                              - ws_ext_wholesale_cost
                              - ws_ext_discount_amt
                            ) +
                                  ws_ext_sales_price ) / 2 )) year_total,
                    'w'                                       sale_type
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
                       d_year)
    SELECT t_s_secyear.customer_id,
                   t_s_secyear.customer_first_name,
                   t_s_secyear.customer_last_name,
                   t_s_secyear.customer_preferred_cust_flag
    FROM   year_total t_s_firstyear,
           year_total t_s_secyear,
           year_total t_c_firstyear,
           year_total t_c_secyear,
           year_total t_w_firstyear,
           year_total t_w_secyear
    WHERE  t_s_secyear.customer_id = t_s_firstyear.customer_id
           AND t_s_firstyear.customer_id = t_c_secyear.customer_id
           AND t_s_firstyear.customer_id = t_c_firstyear.customer_id
           AND t_s_firstyear.customer_id = t_w_firstyear.customer_id
           AND t_s_firstyear.customer_id = t_w_secyear.customer_id
           AND t_s_firstyear.sale_type = 's'
           AND t_c_firstyear.sale_type = 'c'
           AND t_w_firstyear.sale_type = 'w'
           AND t_s_secyear.sale_type = 's'
           AND t_c_secyear.sale_type = 'c'
           AND t_w_secyear.sale_type = 'w'
           AND t_s_firstyear.dyear = {year}
           AND t_s_secyear.dyear = {year} + 1
           AND t_c_firstyear.dyear = {year}
           AND t_c_secyear.dyear = {year} + 1
           AND t_w_firstyear.dyear = {year}
           AND t_w_secyear.dyear = {year} + 1
           AND t_s_firstyear.year_total > 0
           AND t_c_firstyear.year_total > 0
           AND t_w_firstyear.year_total > 0
           AND CASE
                 WHEN t_c_firstyear.year_total > 0 THEN t_c_secyear.year_total /
                                                        t_c_firstyear.year_total
                 ELSE NULL
               END > CASE
                       WHEN t_s_firstyear.year_total > 0 THEN
                       t_s_secyear.year_total /
                       t_s_firstyear.year_total
                       ELSE NULL
                     END
           AND CASE
                 WHEN t_c_firstyear.year_total > 0 THEN t_c_secyear.year_total /
                                                        t_c_firstyear.year_total
                 ELSE NULL
               END > CASE
                       WHEN t_w_firstyear.year_total > 0 THEN
                       t_w_secyear.year_total /
                       t_w_firstyear.year_total
                       ELSE NULL
                     END
    ORDER  BY t_s_secyear.customer_id,
              t_s_secyear.customer_first_name,
              t_s_secyear.customer_last_name,
              t_s_secyear.customer_preferred_cust_flag
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


def build_sales_agg(
    sales_df: pl.LazyFrame,
    date_df: pl.LazyFrame,
    customer_df: pl.LazyFrame,
    sold_date_key: str,
    customer_key: str,
    col_prefix: str,
) -> pl.LazyFrame:
    """Aggregate sales joined with customer to (customer composite, year_total)."""
    profit_expr = (
        (
            pl.col(f"{col_prefix}ext_list_price")
            - pl.col(f"{col_prefix}ext_wholesale_cost")
            - pl.col(f"{col_prefix}ext_discount_amt")
        )
        + pl.col(f"{col_prefix}ext_sales_price")
    ) / 2

    return (
        sales_df.join(date_df, left_on=sold_date_key, right_on="d_date_sk")
        .join(customer_df, left_on=customer_key, right_on="c_customer_sk")
        .group_by(CUSTOMER_COMPOSITE)
        .agg(profit_expr.sum().alias("year_total"))
    )


def polars_impl(run_config: RunConfig) -> QueryResult:
    """Query 4."""
    params = load_parameters(
        int(run_config.scale_factor), query_id=4, qualification=run_config.qualification
    )

    year = params["year"]

    # Load required tables
    customer = get_data(run_config.dataset_path, "customer", run_config.suffix)
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    catalog_sales = get_data(
        run_config.dataset_path, "catalog_sales", run_config.suffix
    )
    web_sales = get_data(run_config.dataset_path, "web_sales", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    date_firstyear = date_dim.filter(pl.col("d_year") == year).select("d_date_sk")
    date_secyear = date_dim.filter(pl.col("d_year") == year + 1).select("d_date_sk")

    # Aggregate each channel x year to (customer composite, year_total)
    t_s_fy = build_sales_agg(
        store_sales,
        date_firstyear,
        customer,
        "ss_sold_date_sk",
        "ss_customer_sk",
        "ss_",
    ).filter(pl.col("year_total") > 0)
    t_s_sy = build_sales_agg(
        store_sales, date_secyear, customer, "ss_sold_date_sk", "ss_customer_sk", "ss_"
    )
    t_c_fy = build_sales_agg(
        catalog_sales,
        date_firstyear,
        customer,
        "cs_sold_date_sk",
        "cs_bill_customer_sk",
        "cs_",
    ).filter(pl.col("year_total") > 0)
    t_c_sy = build_sales_agg(
        catalog_sales,
        date_secyear,
        customer,
        "cs_sold_date_sk",
        "cs_bill_customer_sk",
        "cs_",
    )
    t_w_fy = build_sales_agg(
        web_sales,
        date_firstyear,
        customer,
        "ws_sold_date_sk",
        "ws_bill_customer_sk",
        "ws_",
    ).filter(pl.col("year_total") > 0)
    t_w_sy = build_sales_agg(
        web_sales,
        date_secyear,
        customer,
        "ws_sold_date_sk",
        "ws_bill_customer_sk",
        "ws_",
    )

    # Join all 6 subqueries on customer composite. nulls_equal=True preserves
    # the null=null semantics of GROUP BY, since composite columns may be null.
    joined = (
        t_s_sy.rename({"year_total": "s_sy"})
        .join(
            t_s_fy.rename({"year_total": "s_fy"}),
            on=CUSTOMER_COMPOSITE,
            how="inner",
            nulls_equal=True,
        )
        .join(
            t_c_fy.rename({"year_total": "c_fy"}),
            on=CUSTOMER_COMPOSITE,
            how="inner",
            nulls_equal=True,
        )
        .join(
            t_c_sy.rename({"year_total": "c_sy"}),
            on=CUSTOMER_COMPOSITE,
            how="inner",
            nulls_equal=True,
        )
        .join(
            t_w_fy.rename({"year_total": "w_fy"}),
            on=CUSTOMER_COMPOSITE,
            how="inner",
            nulls_equal=True,
        )
        .join(
            t_w_sy.rename({"year_total": "w_sy"}),
            on=CUSTOMER_COMPOSITE,
            how="inner",
            nulls_equal=True,
        )
        .filter(
            # Catalog growth rate > Store growth rate
            (
                pl.when(pl.col("c_fy") > 0)
                .then(pl.col("c_sy") / pl.col("c_fy"))
                .otherwise(None)
                > pl.when(pl.col("s_fy") > 0)
                .then(pl.col("s_sy") / pl.col("s_fy"))
                .otherwise(None)
            )
            &
            # Catalog growth rate > Web growth rate
            (
                pl.when(pl.col("c_fy") > 0)
                .then(pl.col("c_sy") / pl.col("c_fy"))
                .otherwise(None)
                > pl.when(pl.col("w_fy") > 0)
                .then(pl.col("w_sy") / pl.col("w_fy"))
                .otherwise(None)
            )
        )
    )

    sort_cols = [
        "customer_id",
        "customer_first_name",
        "customer_last_name",
        "customer_preferred_cust_flag",
    ]
    return QueryResult(
        frame=(
            joined.select(
                [
                    pl.col("c_customer_id").alias("customer_id"),
                    pl.col("c_first_name").alias("customer_first_name"),
                    pl.col("c_last_name").alias("customer_last_name"),
                    pl.col("c_preferred_cust_flag").alias(
                        "customer_preferred_cust_flag"
                    ),
                ]
            )
            .sort(sort_cols)
            .limit(100)
        ),
        sort_by=[
            ("customer_id", False),
            ("customer_first_name", False),
            ("customer_last_name", False),
            ("customer_preferred_cust_flag", False),
        ],
        limit=100,
    )
