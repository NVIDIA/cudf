# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 10."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.streaming.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.streaming.benchmarks.utils import QueryResult, get_data

if TYPE_CHECKING:
    from cudf_polars.streaming.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 10."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=10,
        qualification=run_config.qualification,
    )

    counties = params["county"]

    return f"""
    SELECT cd_gender,
                   cd_marital_status,
                   cd_education_status,
                   Count(*) cnt1,
                   cd_purchase_estimate,
                   Count(*) cnt2,
                   cd_credit_rating,
                   Count(*) cnt3,
                   cd_dep_count,
                   Count(*) cnt4,
                   cd_dep_employed_count,
                   Count(*) cnt5,
                   cd_dep_college_count,
                   Count(*) cnt6
    FROM   customer c,
           customer_address ca,
           customer_demographics
    WHERE  c.c_current_addr_sk = ca.ca_address_sk
           AND ca_county IN ( {", ".join(f"'{county}'" for county in counties)} )
           AND cd_demo_sk = c.c_current_cdemo_sk
           AND EXISTS (SELECT *
                       FROM   store_sales,
                              date_dim
                       WHERE  c.c_customer_sk = ss_customer_sk
                              AND ss_sold_date_sk = d_date_sk
                              AND d_year = 2002
                              AND d_moy BETWEEN 4 AND 4 + 3)
           AND ( EXISTS (SELECT *
                         FROM   web_sales,
                                date_dim
                         WHERE  c.c_customer_sk = ws_bill_customer_sk
                                AND ws_sold_date_sk = d_date_sk
                                AND d_year = 2002
                                AND d_moy BETWEEN 4 AND 4 + 3)
                  OR EXISTS (SELECT *
                             FROM   catalog_sales,
                                    date_dim
                             WHERE  c.c_customer_sk = cs_ship_customer_sk
                                    AND cs_sold_date_sk = d_date_sk
                                    AND d_year = 2002
                                    AND d_moy BETWEEN 4 AND 4 + 3) )
    GROUP  BY cd_gender,
              cd_marital_status,
              cd_education_status,
              cd_purchase_estimate,
              cd_credit_rating,
              cd_dep_count,
              cd_dep_employed_count,
              cd_dep_college_count
    ORDER  BY cd_gender,
              cd_marital_status,
              cd_education_status,
              cd_purchase_estimate,
              cd_credit_rating,
              cd_dep_count,
              cd_dep_employed_count,
              cd_dep_college_count
    LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> QueryResult:
    """Query 10."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=10,
        qualification=run_config.qualification,
    )

    counties = params["county"]

    customer = get_data(run_config.dataset_path, "customer", run_config.suffix)
    customer_address = get_data(
        run_config.dataset_path, "customer_address", run_config.suffix
    )
    customer_demographics = get_data(
        run_config.dataset_path, "customer_demographics", run_config.suffix
    )
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    web_sales = get_data(run_config.dataset_path, "web_sales", run_config.suffix)
    catalog_sales = get_data(
        run_config.dataset_path, "catalog_sales", run_config.suffix
    )
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)

    # Get customers with store sales in the target period (EXISTS condition 1)
    store_customers = (
        store_sales.join(date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk")
        .filter(
            (pl.col("d_year") == 2002)
            & (pl.col("d_moy").is_between(4, 7, closed="both"))
        )
        .select("ss_customer_sk")
        .unique()
    )

    # Get customers with web sales in the target period (EXISTS condition 2a)
    web_customers = (
        web_sales.join(date_dim, left_on="ws_sold_date_sk", right_on="d_date_sk")
        .filter(
            (pl.col("d_year") == 2002)
            & (pl.col("d_moy").is_between(4, 7, closed="both"))
        )
        .select(pl.col("ws_bill_customer_sk").alias("customer_sk"))
        .unique()
    )

    # Get customers with catalog sales in the target period (EXISTS condition 2b)
    catalog_customers = (
        catalog_sales.join(date_dim, left_on="cs_sold_date_sk", right_on="d_date_sk")
        .filter(
            (pl.col("d_year") == 2002)
            & (pl.col("d_moy").is_between(4, 7, closed="both"))
        )
        .select(pl.col("cs_ship_customer_sk").alias("customer_sk"))
        .unique()
    )

    # Combine web and catalog customers (OR condition)
    web_or_catalog_customers = pl.concat([web_customers, catalog_customers]).unique()

    # Main query: join customer tables and apply filters
    return QueryResult(
        frame=(
            customer.join(
                customer_address, left_on="c_current_addr_sk", right_on="ca_address_sk"
            )
            .join(
                customer_demographics,
                left_on="c_current_cdemo_sk",
                right_on="cd_demo_sk",
            )
            .filter(pl.col("ca_county").is_in(counties))
            # Apply EXISTS conditions through joins
            .join(
                store_customers,
                left_on="c_customer_sk",
                right_on="ss_customer_sk",
                how="inner",
            )
            .join(
                web_or_catalog_customers,
                left_on="c_customer_sk",
                right_on="customer_sk",
                how="inner",
            )
            .group_by(
                [
                    "cd_gender",
                    "cd_marital_status",
                    "cd_education_status",
                    "cd_purchase_estimate",
                    "cd_credit_rating",
                    "cd_dep_count",
                    "cd_dep_employed_count",
                    "cd_dep_college_count",
                ]
            )
            .agg(
                [
                    pl.len().alias("cnt1"),
                    pl.len().alias("cnt2"),
                    pl.len().alias("cnt3"),
                    pl.len().alias("cnt4"),
                    pl.len().alias("cnt5"),
                    pl.len().alias("cnt6"),
                ]
            )
            .sort(
                [
                    "cd_gender",
                    "cd_marital_status",
                    "cd_education_status",
                    "cd_purchase_estimate",
                    "cd_credit_rating",
                    "cd_dep_count",
                    "cd_dep_employed_count",
                    "cd_dep_college_count",
                ],
                nulls_last=True,
            )
            .limit(100)
            .select(
                [
                    "cd_gender",
                    "cd_marital_status",
                    "cd_education_status",
                    "cnt1",
                    "cd_purchase_estimate",
                    "cnt2",
                    "cd_credit_rating",
                    "cnt3",
                    "cd_dep_count",
                    "cnt4",
                    "cd_dep_employed_count",
                    "cnt5",
                    "cd_dep_college_count",
                    "cnt6",
                ]
            )
        ),
        sort_by=[
            ("cd_gender", False),
            ("cd_marital_status", False),
            ("cd_education_status", False),
            ("cd_purchase_estimate", False),
            ("cd_credit_rating", False),
            ("cd_dep_count", False),
            ("cd_dep_employed_count", False),
            ("cd_dep_college_count", False),
        ],
        limit=100,
    )
