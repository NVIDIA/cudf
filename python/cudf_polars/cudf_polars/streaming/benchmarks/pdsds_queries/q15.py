# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 15."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.streaming.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.streaming.benchmarks.utils import QueryResult, get_data

if TYPE_CHECKING:
    from cudf_polars.streaming.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 15."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=15,
        qualification=run_config.qualification,
    )

    year = params["year"]
    qoy = params["qoy"]

    return f"""
    SELECT ca_zip,
                   Sum(cs_sales_price)
    FROM   catalog_sales,
           customer,
           customer_address,
           date_dim
    WHERE  cs_bill_customer_sk = c_customer_sk
           AND c_current_addr_sk = ca_address_sk
           AND ( Substr(ca_zip, 1, 5) IN ( '85669', '86197', '88274', '83405',
                                           '86475', '85392', '85460', '80348',
                                           '81792' )
                  OR ca_state IN ( 'CA', 'WA', 'GA' )
                  OR cs_sales_price > 500 )
           AND cs_sold_date_sk = d_date_sk
           AND d_qoy = {qoy}
           AND d_year = {year}
    GROUP  BY ca_zip
    ORDER  BY ca_zip
    LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> QueryResult:
    """Query 15."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=15,
        qualification=run_config.qualification,
    )

    year = params["year"]
    qoy = params["qoy"]

    # Load tables
    catalog_sales = get_data(
        run_config.dataset_path, "catalog_sales", run_config.suffix
    )
    customer = get_data(run_config.dataset_path, "customer", run_config.suffix)
    customer_address = get_data(
        run_config.dataset_path, "customer_address", run_config.suffix
    )
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    return QueryResult(
        frame=(
            catalog_sales.join(
                customer, left_on="cs_bill_customer_sk", right_on="c_customer_sk"
            )
            .join(
                customer_address, left_on="c_current_addr_sk", right_on="ca_address_sk"
            )
            .join(date_dim, left_on="cs_sold_date_sk", right_on="d_date_sk")
            .filter(
                (pl.col("d_qoy") == qoy)
                & (pl.col("d_year") == year)
                & (
                    pl.col("ca_zip")
                    .str.slice(0, 5)
                    .is_in(
                        [
                            "85669",
                            "86197",
                            "88274",
                            "83405",
                            "86475",
                            "85392",
                            "85460",
                            "80348",
                            "81792",
                        ]
                    )
                    | pl.col("ca_state").is_in(["CA", "WA", "GA"])
                    | (pl.col("cs_sales_price") > 500)
                )
            )
            .group_by("ca_zip")
            .agg(pl.col("cs_sales_price").sum().alias("sum(cs_sales_price)"))
            .sort("ca_zip", nulls_last=True)
            .limit(100)
        ),
        sort_by=[("ca_zip", False)],
        limit=100,
    )
