# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 74."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.streaming.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.streaming.benchmarks.utils import QueryResult, get_data

if TYPE_CHECKING:
    from cudf_polars.streaming.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 74."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=74,
        qualification=run_config.qualification,
    )
    year = params["year"]

    return f"""
    WITH year_total
         AS (SELECT c_customer_id    customer_id,
                    c_first_name     customer_first_name,
                    c_last_name      customer_last_name,
                    d_year           AS year1,
                    STDDEV_SAMP(ss_net_paid) year_total,
                    's'              sale_type
             FROM   customer,
                    store_sales,
                    date_dim
             WHERE  c_customer_sk = ss_customer_sk
                    AND ss_sold_date_sk = d_date_sk
                    AND d_year IN ( {year}, {year} + 1 )
             GROUP  BY c_customer_id,
                       c_first_name,
                       c_last_name,
                       d_year
             UNION ALL
             SELECT c_customer_id    customer_id,
                    c_first_name     customer_first_name,
                    c_last_name      customer_last_name,
                    d_year           AS year1,
                    STDDEV_SAMP(ws_net_paid) year_total,
                    'w'              sale_type
             FROM   customer,
                    web_sales,
                    date_dim
             WHERE  c_customer_sk = ws_bill_customer_sk
                    AND ws_sold_date_sk = d_date_sk
                    AND d_year IN ( {year}, {year} + 1 )
             GROUP  BY c_customer_id,
                       c_first_name,
                       c_last_name,
                       d_year)
    SELECT t_s_secyear.customer_id,
                   t_s_secyear.customer_first_name,
                   t_s_secyear.customer_last_name
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
           AND t_s_firstyear.year1 = {year}
           AND t_s_secyear.year1 = {year} + 1
           AND t_w_firstyear.year1 = {year}
           AND t_w_secyear.year1 = {year} + 1
           AND t_s_firstyear.year_total > 0
           AND t_w_firstyear.year_total > 0
           AND CASE
                 WHEN t_w_firstyear.year_total > 0 THEN t_w_secyear.year_total /
                                                        t_w_firstyear.year_total
                 ELSE NULL
               END > CASE
                       WHEN t_s_firstyear.year_total > 0 THEN
                       t_s_secyear.year_total /
                       t_s_firstyear.year_total
                       ELSE NULL
                     END
    ORDER  BY 1,
              2,
              3
    LIMIT 100;
    """


CUSTOMER_COMPOSITE = ["c_customer_id", "c_first_name", "c_last_name"]


def _year_total_sk(
    sales: pl.LazyFrame,
    customer: pl.LazyFrame,
    date_dim: pl.LazyFrame,
    date_fk: str,
    customer_fk: str,
    amount_col: str,
    year: int,
) -> pl.LazyFrame:
    dates = date_dim.filter(pl.col("d_year") == year).select(["d_date_sk"])
    return (
        sales.join(dates, left_on=date_fk, right_on="d_date_sk")
        .join(customer, left_on=customer_fk, right_on="c_customer_sk")
        .group_by(CUSTOMER_COMPOSITE)
        .agg(pl.col(amount_col).std().alias("year_total"))
    )


def polars_impl(run_config: RunConfig) -> QueryResult:
    """Query 74."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=74,
        qualification=run_config.qualification,
    )

    year = params["year"]

    customer = get_data(run_config.dataset_path, "customer", run_config.suffix)
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    web_sales = get_data(run_config.dataset_path, "web_sales", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)

    t_s_first = (
        _year_total_sk(
            store_sales,
            customer,
            date_dim,
            "ss_sold_date_sk",
            "ss_customer_sk",
            "ss_net_paid",
            year,
        )
        .rename({"year_total": "s_first_total"})
        .filter(pl.col("s_first_total") > 0)
    )
    t_s_sec = _year_total_sk(
        store_sales,
        customer,
        date_dim,
        "ss_sold_date_sk",
        "ss_customer_sk",
        "ss_net_paid",
        year + 1,
    ).rename({"year_total": "s_sec_total"})
    t_w_first = (
        _year_total_sk(
            web_sales,
            customer,
            date_dim,
            "ws_sold_date_sk",
            "ws_bill_customer_sk",
            "ws_net_paid",
            year,
        )
        .rename({"year_total": "w_first_total"})
        .filter(pl.col("w_first_total") > 0)
    )
    t_w_sec = _year_total_sk(
        web_sales,
        customer,
        date_dim,
        "ws_sold_date_sk",
        "ws_bill_customer_sk",
        "ws_net_paid",
        year + 1,
    ).rename({"year_total": "w_sec_total"})

    joined = (
        t_s_first.join(t_s_sec, on=CUSTOMER_COMPOSITE, nulls_equal=True)
        .join(t_w_first, on=CUSTOMER_COMPOSITE, nulls_equal=True)
        .join(t_w_sec, on=CUSTOMER_COMPOSITE, nulls_equal=True)
    )

    sort_by = {
        "customer_id": False,
        "customer_first_name": False,
        "customer_last_name": False,
    }
    limit = 100
    return QueryResult(
        frame=(
            joined.filter(
                pl.col("w_sec_total") / pl.col("w_first_total")
                > pl.col("s_sec_total") / pl.col("s_first_total")
            )
            .select(
                pl.col("c_customer_id").alias("customer_id"),
                pl.col("c_first_name").alias("customer_first_name"),
                pl.col("c_last_name").alias("customer_last_name"),
            )
            .sort(
                ["customer_id", "customer_first_name", "customer_last_name"],
                nulls_last=True,
            )
            .limit(limit)
        ),
        sort_by=list(sort_by.items()),
        limit=limit,
    )
