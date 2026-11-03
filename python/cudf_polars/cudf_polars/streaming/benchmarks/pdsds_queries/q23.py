# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 23."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.streaming.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.streaming.benchmarks.utils import QueryResult, get_data

if TYPE_CHECKING:
    from cudf_polars.streaming.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 23."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=23,
        qualification=run_config.qualification,
    )

    year = params["year"]
    month = params["month"]

    return f"""
    WITH frequent_ss_items AS
      (SELECT itemdesc,
              i_item_sk item_sk,
              d_date solddate,
              count(*) cnt
       FROM store_sales,
            date_dim,
            (SELECT SUBSTRING(i_item_desc, 1, 30) itemdesc, *
             FROM item) sq1
       WHERE ss_sold_date_sk = d_date_sk
         AND ss_item_sk = i_item_sk
         AND d_year IN ({year}, {year}+1, {year}+2, {year}+3)
       GROUP BY itemdesc, i_item_sk, d_date
       HAVING count(*) > 4),
    max_store_sales AS
      (SELECT max(csales) tpcds_cmax
       FROM (SELECT c_customer_sk,
                    sum(ss_quantity*ss_sales_price) csales
             FROM store_sales, customer, date_dim
             WHERE ss_customer_sk = c_customer_sk
               AND ss_sold_date_sk = d_date_sk
               AND d_year IN ({year}, {year}+1, {year}+2, {year}+3)
             GROUP BY c_customer_sk) sq2),
    best_ss_customer AS
      (SELECT c_customer_sk,
              sum(ss_quantity*ss_sales_price) ssales
       FROM store_sales, customer, max_store_sales
       WHERE ss_customer_sk = c_customer_sk
       GROUP BY c_customer_sk
       HAVING sum(ss_quantity*ss_sales_price) > (95/100.0) * max(tpcds_cmax))
    SELECT c_last_name, c_first_name, sales
    FROM (SELECT c_last_name, c_first_name,
                 sum(cs_quantity*cs_list_price) sales
          FROM catalog_sales, customer, date_dim, frequent_ss_items, best_ss_customer
          WHERE d_year = {year}
            AND d_moy = {month}
            AND cs_sold_date_sk = d_date_sk
            AND cs_item_sk = item_sk
            AND cs_bill_customer_sk = best_ss_customer.c_customer_sk
            AND cs_bill_customer_sk = customer.c_customer_sk
          GROUP BY c_last_name, c_first_name
          UNION ALL
          SELECT c_last_name, c_first_name,
                 sum(ws_quantity*ws_list_price) sales
          FROM web_sales, customer, date_dim, frequent_ss_items, best_ss_customer
          WHERE d_year = {year}
            AND d_moy = {month}
            AND ws_sold_date_sk = d_date_sk
            AND ws_item_sk = item_sk
            AND ws_bill_customer_sk = best_ss_customer.c_customer_sk
            AND ws_bill_customer_sk = customer.c_customer_sk
          GROUP BY c_last_name, c_first_name) sq3
    ORDER BY c_last_name NULLS FIRST, c_first_name NULLS FIRST, sales NULLS FIRST
    LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> QueryResult:
    """Query 23."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=23,
        qualification=run_config.qualification,
    )

    year = params["year"]
    month = params["month"]

    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)
    customer = get_data(run_config.dataset_path, "customer", run_config.suffix)
    catalog_sales = get_data(
        run_config.dataset_path, "catalog_sales", run_config.suffix
    )
    web_sales = get_data(run_config.dataset_path, "web_sales", run_config.suffix)

    # Pre-filter date_dim to the 4-year window so the inner join with store_sales
    # naturally excludes out-of-window records before the expensive item join.
    year_set = [year, year + 1, year + 2, year + 3]
    date_dim_years = date_dim.filter(pl.col("d_year").is_in(year_set))

    frequent_ss_items = (
        store_sales.join(
            date_dim_years, left_on="ss_sold_date_sk", right_on="d_date_sk"
        )
        .join(item, left_on="ss_item_sk", right_on="i_item_sk")
        .with_columns(pl.col("i_item_desc").str.slice(0, 30).alias("itemdesc"))
        .group_by(["itemdesc", "ss_item_sk", "d_date"])
        .agg(pl.len().alias("cnt"))
        .filter(pl.col("cnt") > 4)
        .select("ss_item_sk")
    )

    customer_sales = (
        # Note: filtering on is_not_null instead of joining to the customer table is
        # only valid because we know that the TPC-DS includes a foreign key here, so all
        # customers in store_sales _must_ be entries that exist somewhere in customer.
        store_sales.filter(pl.col("ss_customer_sk").is_not_null())
        .join(
            date_dim_years.select("d_date_sk"),
            left_on="ss_sold_date_sk",
            right_on="d_date_sk",
        )
        .group_by("ss_customer_sk")
        .agg((pl.col("ss_quantity") * pl.col("ss_sales_price")).sum().alias("csales"))
    )

    threshold = customer_sales.select(
        (pl.col("csales").max() * 0.95).alias("threshold")
    )

    best_customers = (
        store_sales.filter(pl.col("ss_customer_sk").is_not_null())
        .group_by("ss_customer_sk")
        .agg((pl.col("ss_quantity") * pl.col("ss_sales_price")).sum().alias("ssales"))
        .join(threshold, how="cross")
        .filter(pl.col("ssales") > pl.col("threshold"))
        .select("ss_customer_sk")
        .unique()
    )

    # Pre-filter date_dim to target year/month
    date_target = date_dim.filter(
        (pl.col("d_year") == year) & (pl.col("d_moy") == month)
    ).select("d_date_sk")

    # Join order: most selective filters first (date_target ~1.2%, frequent_ss_items,
    # best_customers semi ~5%), then customer last — it's a non-filtering name lookup
    # that only adds c_last_name/c_first_name, so running it on the already-reduced
    # row set avoids the full catalog_sales/web_sales scan width.
    catalog_part = (
        catalog_sales.join(date_target, left_on="cs_sold_date_sk", right_on="d_date_sk")
        .join(frequent_ss_items, left_on="cs_item_sk", right_on="ss_item_sk")
        .join(
            best_customers,
            left_on="cs_bill_customer_sk",
            right_on="ss_customer_sk",
            how="semi",
        )
        .join(
            customer.select(["c_customer_sk", "c_last_name", "c_first_name"]),
            left_on="cs_bill_customer_sk",
            right_on="c_customer_sk",
        )
        .group_by(["c_last_name", "c_first_name"])
        .agg((pl.col("cs_quantity") * pl.col("cs_list_price")).sum().alias("sales"))
    )

    web_part = (
        web_sales.join(date_target, left_on="ws_sold_date_sk", right_on="d_date_sk")
        .join(frequent_ss_items, left_on="ws_item_sk", right_on="ss_item_sk")
        .join(
            best_customers,
            left_on="ws_bill_customer_sk",
            right_on="ss_customer_sk",
            how="semi",
        )
        .join(
            customer.select(["c_customer_sk", "c_last_name", "c_first_name"]),
            left_on="ws_bill_customer_sk",
            right_on="c_customer_sk",
        )
        .group_by(["c_last_name", "c_first_name"])
        .agg((pl.col("ws_quantity") * pl.col("ws_list_price")).sum().alias("sales"))
    )

    sort_by = {"c_last_name": False, "c_first_name": False, "sales": False}
    limit = 100

    return QueryResult(
        frame=(
            pl.concat([catalog_part, web_part])
            .sort(sort_by.keys(), nulls_last=False)
            .limit(limit)
        ),
        sort_by=list(sort_by.items()),
        limit=limit,
        nulls_last=False,
    )
