# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 69."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.streaming.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.streaming.benchmarks.utils import QueryResult, get_data

if TYPE_CHECKING:
    from cudf_polars.streaming.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 69."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=69,
        qualification=run_config.qualification,
    )

    year = params["year"]
    month = params["month"]
    states = params["states"]
    states_str = ", ".join(f"'{s}'" for s in states)

    return f"""
    SELECT cd_gender,
                   cd_marital_status,
                   cd_education_status,
                   Count(*) cnt1,
                   cd_purchase_estimate,
                   Count(*) cnt2,
                   cd_credit_rating,
                   Count(*) cnt3
    FROM   customer c,
           customer_address ca,
           customer_demographics
    WHERE  c.c_current_addr_sk = ca.ca_address_sk
           AND ca_state IN ( {states_str} )
           AND cd_demo_sk = c.c_current_cdemo_sk
           AND EXISTS (SELECT *
                       FROM   store_sales,
                              date_dim
                       WHERE  c.c_customer_sk = ss_customer_sk
                              AND ss_sold_date_sk = d_date_sk
                              AND d_year = {year}
                              AND d_moy BETWEEN {month} AND {month} + 2)
           AND ( NOT EXISTS (SELECT *
                             FROM   web_sales,
                                    date_dim
                             WHERE  c.c_customer_sk = ws_bill_customer_sk
                                    AND ws_sold_date_sk = d_date_sk
                                    AND d_year = {year}
                                    AND d_moy BETWEEN {month} AND {month} + 2)
                 AND NOT EXISTS (SELECT *
                                 FROM   catalog_sales,
                                        date_dim
                                 WHERE  c.c_customer_sk = cs_ship_customer_sk
                                        AND cs_sold_date_sk = d_date_sk
                                        AND d_year = {year}
                                        AND d_moy BETWEEN {month} AND {month} + 2) )
    GROUP  BY cd_gender,
              cd_marital_status,
              cd_education_status,
              cd_purchase_estimate,
              cd_credit_rating
    ORDER  BY cd_gender,
              cd_marital_status,
              cd_education_status,
              cd_purchase_estimate,
              cd_credit_rating
    LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> QueryResult:
    """Query 69."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=69,
        qualification=run_config.qualification,
    )

    year = params["year"]
    month = params["month"]
    states = params["states"]

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
    target_dates = date_dim.filter(
        (pl.col("d_year") == year) & (pl.col("d_moy").is_between(month, month + 2))
    ).select("d_date_sk")
    store_customers = (
        store_sales.join(target_dates, left_on="ss_sold_date_sk", right_on="d_date_sk")
        .select("ss_customer_sk")
        .unique()
    )
    web_customers = (
        web_sales.join(target_dates, left_on="ws_sold_date_sk", right_on="d_date_sk")
        .select(pl.col("ws_bill_customer_sk").alias("customer_sk"))
        .unique()
    )
    catalog_customers = (
        catalog_sales.join(
            target_dates, left_on="cs_sold_date_sk", right_on="d_date_sk"
        )
        .select(pl.col("cs_ship_customer_sk").alias("customer_sk"))
        .unique()
    )
    exclude_customers = pl.concat([web_customers, catalog_customers]).unique()
    return QueryResult(
        frame=(
            customer.join(
                customer_address, left_on="c_current_addr_sk", right_on="ca_address_sk"
            )
            .filter(pl.col("ca_state").is_in(states))
            .join(
                customer_demographics,
                left_on="c_current_cdemo_sk",
                right_on="cd_demo_sk",
            )
            .join(store_customers, left_on="c_customer_sk", right_on="ss_customer_sk")
            .join(
                exclude_customers,
                left_on="c_customer_sk",
                right_on="customer_sk",
                how="anti",
            )
            .group_by(
                [
                    "cd_gender",
                    "cd_marital_status",
                    "cd_education_status",
                    "cd_purchase_estimate",
                    "cd_credit_rating",
                ]
            )
            .agg(
                [
                    pl.len().alias("cnt1"),
                    pl.len().alias("cnt2"),
                    pl.len().alias("cnt3"),
                ]
            )
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
                ]
            )
            .sort(
                [
                    "cd_gender",
                    "cd_marital_status",
                    "cd_education_status",
                    "cd_purchase_estimate",
                    "cd_credit_rating",
                ],
                nulls_last=True,
            )
            .limit(100)
        ),
        sort_by=[
            ("cd_gender", False),
            ("cd_marital_status", False),
            ("cd_education_status", False),
            ("cd_purchase_estimate", False),
            ("cd_credit_rating", False),
        ],
        limit=100,
    )
