# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Query 18."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.streaming.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.streaming.benchmarks.utils import QueryResult, get_data

if TYPE_CHECKING:
    from cudf_polars.streaming.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 18."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=18,
        qualification=run_config.qualification,
    )

    year = params["year"]
    month = params["month"]
    state = params["state"]
    es = params["es"]
    gen = params["gen"]

    return f"""
    SELECT i_item_id,
                   ca_country,
                   ca_state,
                   ca_county,
                   Avg(Cast(cs_quantity AS NUMERIC(12, 2)))      agg1,
                   Avg(Cast(cs_list_price AS NUMERIC(12, 2)))    agg2,
                   Avg(Cast(cs_coupon_amt AS NUMERIC(12, 2)))    agg3,
                   Avg(Cast(cs_sales_price AS NUMERIC(12, 2)))   agg4,
                   Avg(Cast(cs_net_profit AS NUMERIC(12, 2)))    agg5,
                   Avg(Cast(c_birth_year AS NUMERIC(12, 2)))     agg6,
                   Avg(Cast(cd1.cd_dep_count AS NUMERIC(12, 2))) agg7
    FROM   catalog_sales,
           customer_demographics cd1,
           customer_demographics cd2,
           customer,
           customer_address,
           date_dim,
           item
    WHERE  cs_sold_date_sk = d_date_sk
           AND cs_item_sk = i_item_sk
           AND cs_bill_cdemo_sk = cd1.cd_demo_sk
           AND cs_bill_customer_sk = c_customer_sk
           AND cd1.cd_gender = '{gen}'
           AND cd1.cd_education_status = '{es}'
           AND c_current_cdemo_sk = cd2.cd_demo_sk
           AND c_current_addr_sk = ca_address_sk
           AND c_birth_month IN ( {", ".join(map(str, month))} )
           AND d_year = {year}
           AND ca_state IN ( {", ".join(f"'{s}'" for s in state)} )
    GROUP  BY rollup ( i_item_id, ca_country, ca_state, ca_county )
    ORDER  BY ca_country,
              ca_state,
              ca_county,
              i_item_id
    LIMIT 100;
    """


def level(  # noqa: D103
    base_query: pl.LazyFrame,
    agg_exprs: list[pl.Expr],
    group_cols: list[str],
) -> pl.LazyFrame:
    if group_cols:
        lf = base_query.group_by(group_cols).agg(agg_exprs)
    else:
        lf = base_query.select(agg_exprs)
    missing = [
        c
        for c in ["i_item_id", "ca_country", "ca_state", "ca_county"]
        if c not in group_cols
    ]
    if missing:
        lf = lf.with_columns([pl.lit(None, dtype=pl.String).alias(c) for c in missing])
    return lf.select(
        [
            "i_item_id",
            "ca_country",
            "ca_state",
            "ca_county",
            "agg1",
            "agg2",
            "agg3",
            "agg4",
            "agg5",
            "agg6",
            "agg7",
        ]
    )


def polars_impl(run_config: RunConfig) -> QueryResult:
    """Query 18."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=18,
        qualification=run_config.qualification,
    )

    year = params["year"]
    month = params["month"]
    state = params["state"]
    es = params["es"]
    gen = params["gen"]

    catalog_sales = get_data(
        run_config.dataset_path, "catalog_sales", run_config.suffix
    )
    customer_demographics = get_data(
        run_config.dataset_path, "customer_demographics", run_config.suffix
    )
    customer = get_data(run_config.dataset_path, "customer", run_config.suffix)
    customer_address = get_data(
        run_config.dataset_path, "customer_address", run_config.suffix
    )
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)

    # Pre-filter each dimension table before joining against catalog_sales [45 partitions].
    # d_year not in GROUP BY — semi-join keeps only the date key in the pipeline.
    filtered_dates = date_dim.filter(pl.col("d_year") == year).select("d_date_sk")
    filtered_cd1 = customer_demographics.filter(
        (pl.col("cd_gender") == gen) & (pl.col("cd_education_status") == es)
    ).select(["cd_demo_sk", "cd_dep_count"])
    filtered_customer = customer.filter(pl.col("c_birth_month").is_in(month)).select(
        ["c_customer_sk", "c_current_cdemo_sk", "c_current_addr_sk", "c_birth_year"]
    )
    filtered_addr = customer_address.filter(pl.col("ca_state").is_in(state)).select(
        ["ca_address_sk", "ca_county", "ca_state", "ca_country"]
    )

    base_query = (
        catalog_sales.select(
            [
                "cs_sold_date_sk",
                "cs_item_sk",
                "cs_bill_cdemo_sk",
                "cs_bill_customer_sk",
                "cs_quantity",
                "cs_list_price",
                "cs_coupon_amt",
                "cs_sales_price",
                "cs_net_profit",
            ]
        )
        .join(
            filtered_dates, left_on="cs_sold_date_sk", right_on="d_date_sk", how="semi"
        )
        .join(
            item.select(["i_item_sk", "i_item_id"]),
            left_on="cs_item_sk",
            right_on="i_item_sk",
        )
        .join(filtered_cd1, left_on="cs_bill_cdemo_sk", right_on="cd_demo_sk")
        .join(
            filtered_customer, left_on="cs_bill_customer_sk", right_on="c_customer_sk"
        )
        .join(
            customer_demographics.select("cd_demo_sk"),
            left_on="c_current_cdemo_sk",
            right_on="cd_demo_sk",
            how="semi",
        )
        .join(filtered_addr, left_on="c_current_addr_sk", right_on="ca_address_sk")
    )

    agg_exprs = [
        pl.col("cs_quantity").mean().alias("agg1"),
        pl.col("cs_list_price").mean().alias("agg2"),
        pl.col("cs_coupon_amt").mean().alias("agg3"),
        pl.col("cs_sales_price").mean().alias("agg4"),
        pl.col("cs_net_profit").mean().alias("agg5"),
        pl.col("c_birth_year").mean().alias("agg6"),
        pl.col("cd_dep_count").mean().alias("agg7"),
    ]

    # ROLLUP(i_item_id, ca_country, ca_state, ca_county) drops columns from the right:
    # level1: all four, level2: drop ca_county, level3: drop ca_state, level4: drop ca_country, level5: grand total
    level1 = level(
        base_query,
        agg_exprs,
        ["i_item_id", "ca_country", "ca_state", "ca_county"],
    )
    level2 = level(base_query, agg_exprs, ["i_item_id", "ca_country", "ca_state"])
    level3 = level(base_query, agg_exprs, ["i_item_id", "ca_country"])
    level4 = level(base_query, agg_exprs, ["i_item_id"])
    level5 = level(base_query, agg_exprs, [])

    sort_by = {
        "ca_country": False,
        "ca_state": False,
        "ca_county": False,
        "i_item_id": False,
    }
    limit = 100

    return QueryResult(
        frame=(
            pl.concat([level1, level2, level3, level4, level5])
            .sort(sort_by.keys(), nulls_last=True)
            .limit(limit)
        ),
        sort_by=list(sort_by.items()),
        limit=limit,
    )
