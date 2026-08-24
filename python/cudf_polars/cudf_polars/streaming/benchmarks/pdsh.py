# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Experimental PDS-H benchmarks.

Based on https://github.com/pola-rs/polars-benchmark.

WARNING: This is an experimental (and unofficial)
benchmark script. It is not intended for public use
and may be modified or removed at any time.
"""

from __future__ import annotations

import os
from datetime import date
from typing import TYPE_CHECKING

import polars as pl

try:
    from cudf_polars.streaming.benchmarks.utils import (
        COUNT_DTYPE,
        _CPU_ENGINES,
        QueryResult,
        RunConfig,
        build_parser,
        get_data,
        parse_args,
        run_polars,
    )
except ImportError as e:
    if e.name is not None and not e.name.startswith("cudf_polars"):
        raise
    # We want to be able to import pdsh in a CPU-only environment.
    COUNT_DTYPE = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from cudf_polars.streaming.benchmarks.utils import RunConfig

# Without this setting, the first IO task to run
# on each worker takes ~15 sec extra
os.environ["KVIKIO_COMPAT_MODE"] = os.environ.get("KVIKIO_COMPAT_MODE", "on")
os.environ["KVIKIO_NTHREADS"] = os.environ.get("KVIKIO_NTHREADS", "8")
# TODO: consider raising the rapidsmpf built-in default from 1 to 8.
os.environ["RAPIDSMPF_NUM_STREAMING_THREADS"] = os.environ.get(
    "RAPIDSMPF_NUM_STREAMING_THREADS", "8"
)

# The pre-computed expected results come from DuckDB, which has
# different casting rules than Polars. For example, in polars
# Series[Decimal].mean() returns a Float64, while DuckDB returns a Decimal.
#
# This dictionary maps query ID to the casts to apply to the DuckDB
# results necessary to match the cudf-polars results.
# EXPECTED_CASTS_DECIMAL should be used when the input data uses
# Decimal (rather than Float) for account balances, etc.
# EXPECTED_CASTS_FLOATS should be used when the input data uses
# Float (rather than Decimal) for account balances, etc.

if COUNT_DTYPE is not None:
    EXPECTED_CASTS = {
        1: [pl.col("count_order").cast(COUNT_DTYPE)],
        4: [pl.col("order_count").cast(COUNT_DTYPE)],
        7: [pl.col("l_year").cast(pl.Int32())],
        8: [pl.col("o_year").cast(pl.Int32())],
        9: [pl.col("o_year").cast(pl.Int32())],
        12: [
            pl.col("high_line_count").cast(pl.Int32()),
            pl.col("low_line_count").cast(pl.Int32()),
        ],
        13: [pl.col("c_count").cast(COUNT_DTYPE), pl.col("custdist").cast(COUNT_DTYPE)],
        16: [pl.col("supplier_cnt").cast(COUNT_DTYPE)],
        21: [pl.col("numwait").cast(COUNT_DTYPE)],
        22: [pl.col("numcust").cast(COUNT_DTYPE)],
    }

    EXPECTED_CASTS_DECIMAL = {
        1: [
            pl.col("sum_qty").cast(pl.Decimal(38, 2)),
            pl.col("sum_base_price").cast(pl.Decimal(38, 2)),
            pl.col("sum_disc_price").cast(pl.Float64()),
            pl.col("sum_charge").cast(pl.Float64()),
            pl.col("avg_disc").cast(pl.Float64()),
            pl.col("avg_price").cast(pl.Float64()),
            pl.col("avg_qty").cast(pl.Float64()),
        ],
        3: [pl.col("revenue").cast(pl.Decimal(38, 2))],
        5: [pl.col("revenue").cast(pl.Decimal(38, 2))],
        6: [pl.col("revenue").cast(pl.Decimal(38, 2))],
        7: [pl.col("revenue").cast(pl.Decimal(38, 2))],
        8: [pl.col("mkt_share").cast(pl.Decimal(38, 2))],
        9: [
            pl.col("sum_profit").cast(pl.Decimal(38, 2)),
        ],
        10: [pl.col("revenue").cast(pl.Decimal(38, 2))],
        11: [pl.col("value").cast(pl.Decimal(38, 2))],
        15: [pl.col("total_revenue").cast(pl.Decimal(38, 2))],
        18: [pl.col("sum(l_quantity)").cast(pl.Decimal(38, 2))],
        19: [pl.col("revenue").cast(pl.Decimal(38, 2))],
        22: [
            pl.col("totacctbal").cast(pl.Decimal(38, 2)),
        ],
    }

    # When operating on timestamp data from tpchgen-rs, duckdb uses
    # Datetime[us, tz=none] while polars uses Datetime[ms, tz=none].
    EXPECTED_CASTS_TIMESTAMP = {
        3: [pl.col("o_orderdate").cast(pl.Datetime("ms"))],
        18: [pl.col("o_orderdate").cast(pl.Datetime("ms"))],
    }
else:
    EXPECTED_CASTS = {}
    EXPECTED_CASTS_DECIMAL = {}
    EXPECTED_CASTS_TIMESTAMP = {}


class PDSHQueries:
    """PDS-H query definitions."""

    name: str = "pdsh"
    EXPECTED_CASTS = EXPECTED_CASTS
    EXPECTED_CASTS_DECIMAL = EXPECTED_CASTS_DECIMAL
    EXPECTED_CASTS_TIMESTAMP = EXPECTED_CASTS_TIMESTAMP

    @property
    def duckdb_queries(self) -> PDSHDuckDBQueries:
        """Link to the DuckDB queries for this benchmark."""
        return PDSHDuckDBQueries()

    @staticmethod
    def q0(run_config: RunConfig) -> QueryResult:
        """Query 0."""
        frame = pl.LazyFrame()

        return QueryResult(
            frame=frame,
            sort_by=[],
        )

    @staticmethod
    def q1(run_config: RunConfig) -> QueryResult:
        """Query 1."""
        lineitem = get_data(run_config.dataset_path, "lineitem", run_config.suffix)

        var1 = date(1998, 9, 2)

        frame = (
            lineitem.filter(pl.col("l_shipdate") <= var1)
            .group_by("l_returnflag", "l_linestatus")
            .agg(
                pl.sum("l_quantity").alias("sum_qty"),
                pl.sum("l_extendedprice").alias("sum_base_price"),
                (pl.col("l_extendedprice") * (1.0 - pl.col("l_discount")))
                .sum()
                .alias("sum_disc_price"),
                (
                    pl.col("l_extendedprice")
                    * (1.0 - pl.col("l_discount"))
                    * (1.0 + pl.col("l_tax"))
                )
                .sum()
                .alias("sum_charge"),
                pl.mean("l_quantity").alias("avg_qty"),
                pl.mean("l_extendedprice").alias("avg_price"),
                pl.mean("l_discount").alias("avg_disc"),
                pl.len().alias("count_order"),
            )
            .sort("l_returnflag", "l_linestatus")
        )

        return QueryResult(
            frame=frame,
            sort_by=[("l_returnflag", False), ("l_linestatus", False)],
            limit=None,
        )

    @staticmethod
    def q2(run_config: RunConfig) -> QueryResult:
        """Query 2."""
        nation = get_data(run_config.dataset_path, "nation", run_config.suffix)
        part = get_data(run_config.dataset_path, "part", run_config.suffix)
        partsupp = get_data(run_config.dataset_path, "partsupp", run_config.suffix)
        region = get_data(run_config.dataset_path, "region", run_config.suffix)
        supplier = get_data(run_config.dataset_path, "supplier", run_config.suffix)

        var1 = 15
        var2 = "BRASS"
        var3 = "EUROPE"

        q1 = (
            part.join(partsupp, left_on="p_partkey", right_on="ps_partkey")
            .join(supplier, left_on="ps_suppkey", right_on="s_suppkey")
            .join(nation, left_on="s_nationkey", right_on="n_nationkey")
            .join(region, left_on="n_regionkey", right_on="r_regionkey")
            .filter(pl.col("p_size") == var1)
            .filter(pl.col("p_type").str.ends_with(var2))
            .filter(pl.col("r_name") == var3)
        )

        frame = (
            q1.group_by("p_partkey")
            .agg(pl.min("ps_supplycost"))
            .join(q1, on=["p_partkey", "ps_supplycost"])
            .select(
                "s_acctbal",
                "s_name",
                "n_name",
                "p_partkey",
                "p_mfgr",
                "s_address",
                "s_phone",
                "s_comment",
            )
            .sort(
                by=["s_acctbal", "n_name", "s_name", "p_partkey"],
                descending=[True, False, False, False],
            )
            .head(100)
        )

        return QueryResult(
            frame=frame,
            sort_by=[
                ("s_acctbal", True),
                ("n_name", False),
                ("s_name", False),
                ("p_partkey", False),
            ],
            limit=100,
        )

    @staticmethod
    def q3(run_config: RunConfig) -> QueryResult:
        """Query 3."""
        customer = get_data(run_config.dataset_path, "customer", run_config.suffix)
        lineitem = get_data(run_config.dataset_path, "lineitem", run_config.suffix)
        orders = get_data(run_config.dataset_path, "orders", run_config.suffix)

        var1 = "BUILDING"
        var2 = date(1995, 3, 15)

        frame = (
            customer.filter(pl.col("c_mktsegment") == var1)
            .join(orders, left_on="c_custkey", right_on="o_custkey")
            .join(lineitem, left_on="o_orderkey", right_on="l_orderkey")
            .filter(pl.col("o_orderdate") < var2)
            .filter(pl.col("l_shipdate") > var2)
            .with_columns(
                (pl.col("l_extendedprice") * (1 - pl.col("l_discount"))).alias(
                    "revenue"
                )
            )
            .group_by("o_orderkey", "o_orderdate", "o_shippriority")
            .agg(pl.sum("revenue"))
            .select(
                pl.col("o_orderkey").alias("l_orderkey"),
                "revenue",
                "o_orderdate",
                "o_shippriority",
            )
            .sort(by=["revenue", "o_orderdate"], descending=[True, False])
            .head(10)
        )

        return QueryResult(
            frame=frame,
            sort_by=[("revenue", True), ("o_orderdate", False)],
            limit=10,
        )

    @staticmethod
    def q4(run_config: RunConfig) -> QueryResult:
        """Query 4."""
        lineitem = get_data(run_config.dataset_path, "lineitem", run_config.suffix)
        orders = get_data(run_config.dataset_path, "orders", run_config.suffix)

        var1 = date(1993, 7, 1)
        var2 = date(1993, 10, 1)

        frame = (
            # SQL exists translates to semi join in Polars API
            orders.join(
                (lineitem.filter(pl.col("l_commitdate") < pl.col("l_receiptdate"))),
                left_on="o_orderkey",
                right_on="l_orderkey",
                how="semi",
            )
            .filter(pl.col("o_orderdate").is_between(var1, var2, closed="left"))
            .group_by("o_orderpriority")
            .agg(pl.len().alias("order_count"))
            .sort("o_orderpriority")
        )

        return QueryResult(
            frame=frame,
            sort_by=[("o_orderpriority", False)],
        )

    @staticmethod
    def q5(run_config: RunConfig) -> QueryResult:
        """Query 5."""
        path = run_config.dataset_path
        suffix = run_config.suffix
        customer = get_data(path, "customer", suffix)
        lineitem = get_data(path, "lineitem", suffix)
        nation = get_data(path, "nation", suffix)
        orders = get_data(path, "orders", suffix)
        region = get_data(path, "region", suffix)
        supplier = get_data(path, "supplier", suffix)

        var1 = "ASIA"
        var2 = date(1994, 1, 1)
        var3 = date(1995, 1, 1)

        frame = (
            region.join(nation, left_on="r_regionkey", right_on="n_regionkey")
            .join(customer, left_on="n_nationkey", right_on="c_nationkey")
            .join(orders, left_on="c_custkey", right_on="o_custkey")
            .join(lineitem, left_on="o_orderkey", right_on="l_orderkey")
            .join(
                supplier,
                left_on=["l_suppkey", "n_nationkey"],
                right_on=["s_suppkey", "s_nationkey"],
            )
            .filter(pl.col("r_name") == var1)
            .filter(pl.col("o_orderdate").is_between(var2, var3, closed="left"))
            .with_columns(
                (pl.col("l_extendedprice") * (1 - pl.col("l_discount"))).alias(
                    "revenue"
                )
            )
            .group_by("n_name")
            .agg(pl.sum("revenue"))
            .sort(by="revenue", descending=True)
        )

        return QueryResult(
            frame=frame,
            sort_by=[("revenue", True)],
        )

    @staticmethod
    def q6(run_config: RunConfig) -> QueryResult:
        """Query 6."""
        path = run_config.dataset_path
        suffix = run_config.suffix
        lineitem = get_data(path, "lineitem", suffix)

        var1 = date(1994, 1, 1)
        var2 = date(1995, 1, 1)
        var3 = 0.05
        var4 = 0.07
        var5 = 24

        frame = (
            lineitem.filter(pl.col("l_shipdate").is_between(var1, var2, closed="left"))
            .filter(pl.col("l_discount").is_between(var3, var4))
            .filter(pl.col("l_quantity") < var5)
            .with_columns(
                (pl.col("l_extendedprice") * pl.col("l_discount")).alias("revenue")
            )
            .select(pl.sum("revenue"))
        )

        return QueryResult(
            frame=frame,
            sort_by=[],
        )

    @staticmethod
    def q7(run_config: RunConfig) -> QueryResult:
        """Query 7."""
        customer = get_data(run_config.dataset_path, "customer", run_config.suffix)
        lineitem = get_data(run_config.dataset_path, "lineitem", run_config.suffix)
        nation = get_data(run_config.dataset_path, "nation", run_config.suffix)
        orders = get_data(run_config.dataset_path, "orders", run_config.suffix)
        supplier = get_data(run_config.dataset_path, "supplier", run_config.suffix)

        var1 = "FRANCE"
        var2 = "GERMANY"
        var3 = date(1995, 1, 1)
        var4 = date(1996, 12, 31)

        n1 = nation.filter(pl.col("n_name") == var1)
        n2 = nation.filter(pl.col("n_name") == var2)

        q1 = (
            customer.join(n1, left_on="c_nationkey", right_on="n_nationkey")
            .join(orders, left_on="c_custkey", right_on="o_custkey")
            .rename({"n_name": "cust_nation"})
            .join(lineitem, left_on="o_orderkey", right_on="l_orderkey")
            .join(supplier, left_on="l_suppkey", right_on="s_suppkey")
            .join(n2, left_on="s_nationkey", right_on="n_nationkey")
            .rename({"n_name": "supp_nation"})
        )

        q2 = (
            customer.join(n2, left_on="c_nationkey", right_on="n_nationkey")
            .join(orders, left_on="c_custkey", right_on="o_custkey")
            .rename({"n_name": "cust_nation"})
            .join(lineitem, left_on="o_orderkey", right_on="l_orderkey")
            .join(supplier, left_on="l_suppkey", right_on="s_suppkey")
            .join(n1, left_on="s_nationkey", right_on="n_nationkey")
            .rename({"n_name": "supp_nation"})
        )

        frame = (
            pl.concat([q1, q2])
            .filter(pl.col("l_shipdate").is_between(var3, var4))
            .with_columns(
                (pl.col("l_extendedprice") * (1 - pl.col("l_discount"))).alias(
                    "volume"
                ),
                pl.col("l_shipdate").dt.year().alias("l_year"),
            )
            .group_by("supp_nation", "cust_nation", "l_year")
            .agg(pl.sum("volume").alias("revenue"))
            .sort(by=["supp_nation", "cust_nation", "l_year"])
        )

        return QueryResult(
            frame=frame,
            sort_by=[("supp_nation", False), ("cust_nation", False), ("l_year", False)],
        )

    @staticmethod
    def q8(run_config: RunConfig) -> QueryResult:
        """Query 8."""
        customer = get_data(run_config.dataset_path, "customer", run_config.suffix)
        lineitem = get_data(run_config.dataset_path, "lineitem", run_config.suffix)
        nation = get_data(run_config.dataset_path, "nation", run_config.suffix)
        orders = get_data(run_config.dataset_path, "orders", run_config.suffix)
        part = get_data(run_config.dataset_path, "part", run_config.suffix)
        region = get_data(run_config.dataset_path, "region", run_config.suffix)
        supplier = get_data(run_config.dataset_path, "supplier", run_config.suffix)

        var1 = "BRAZIL"
        var2 = "AMERICA"
        var3 = "ECONOMY ANODIZED STEEL"
        var4 = date(1995, 1, 1)
        var5 = date(1996, 12, 31)

        n1 = nation.select("n_nationkey", "n_regionkey")
        n2 = nation.select("n_nationkey", "n_name")

        frame = (
            part.join(lineitem, left_on="p_partkey", right_on="l_partkey")
            .join(supplier, left_on="l_suppkey", right_on="s_suppkey")
            .join(orders, left_on="l_orderkey", right_on="o_orderkey")
            .join(customer, left_on="o_custkey", right_on="c_custkey")
            .join(n1, left_on="c_nationkey", right_on="n_nationkey")
            .join(region, left_on="n_regionkey", right_on="r_regionkey")
            .filter(pl.col("r_name") == var2)
            .join(n2, left_on="s_nationkey", right_on="n_nationkey")
            .filter(pl.col("o_orderdate").is_between(var4, var5))
            .filter(pl.col("p_type") == var3)
            .select(
                pl.col("o_orderdate").dt.year().alias("o_year"),
                (pl.col("l_extendedprice") * (1 - pl.col("l_discount"))).alias(
                    "volume"
                ),
                pl.col("n_name").alias("nation"),
            )
            .with_columns(
                pl.when(pl.col("nation") == var1)
                .then(pl.col("volume"))
                .otherwise(0)
                .alias("_tmp")
            )
            .group_by("o_year")
            .agg((pl.sum("_tmp") / pl.sum("volume")).alias("mkt_share"))
            .sort("o_year")
        )

        return QueryResult(
            frame=frame,
            sort_by=[("o_year", False)],
        )

    @staticmethod
    def q9(run_config: RunConfig) -> QueryResult:
        """Query 9."""
        path = run_config.dataset_path
        suffix = run_config.suffix
        lineitem = get_data(path, "lineitem", suffix)
        nation = get_data(path, "nation", suffix)
        orders = get_data(path, "orders", suffix)
        part = get_data(path, "part", suffix)
        partsupp = get_data(path, "partsupp", suffix)
        supplier = get_data(path, "supplier", suffix)

        frame = (
            part.join(partsupp, left_on="p_partkey", right_on="ps_partkey")
            .join(supplier, left_on="ps_suppkey", right_on="s_suppkey")
            .join(
                lineitem,
                left_on=["p_partkey", "ps_suppkey"],
                right_on=["l_partkey", "l_suppkey"],
            )
            .join(orders, left_on="l_orderkey", right_on="o_orderkey")
            .join(nation, left_on="s_nationkey", right_on="n_nationkey")
            .filter(pl.col("p_name").str.contains("green"))
            .select(
                pl.col("n_name").alias("nation"),
                pl.col("o_orderdate").dt.year().alias("o_year"),
                (
                    pl.col("l_extendedprice") * (1 - pl.col("l_discount"))
                    - pl.col("ps_supplycost") * pl.col("l_quantity")
                ).alias("amount"),
            )
            .group_by("nation", "o_year")
            .agg(pl.sum("amount").round(2).alias("sum_profit"))
            .sort(by=["nation", "o_year"], descending=[False, True])
        )

        return QueryResult(
            frame=frame,
            sort_by=[("nation", False), ("o_year", True)],
        )

    @staticmethod
    def q10(run_config: RunConfig) -> QueryResult:
        """Query 10."""
        path = run_config.dataset_path
        suffix = run_config.suffix
        customer = get_data(path, "customer", suffix)
        lineitem = get_data(path, "lineitem", suffix)
        nation = get_data(path, "nation", suffix)
        orders = get_data(path, "orders", suffix)

        var1 = date(1993, 10, 1)
        var2 = date(1994, 1, 1)

        frame = (
            customer.join(orders, left_on="c_custkey", right_on="o_custkey")
            .join(lineitem, left_on="o_orderkey", right_on="l_orderkey")
            .join(nation, left_on="c_nationkey", right_on="n_nationkey")
            .filter(pl.col("o_orderdate").is_between(var1, var2, closed="left"))
            .filter(pl.col("l_returnflag") == "R")
            .group_by(
                "c_custkey",
                "c_name",
                "c_acctbal",
                "c_phone",
                "n_name",
                "c_address",
                "c_comment",
            )
            .agg(
                (pl.col("l_extendedprice") * (1 - pl.col("l_discount")))
                .sum()
                .round(2)
                .alias("revenue")
            )
            .select(
                "c_custkey",
                "c_name",
                "revenue",
                "c_acctbal",
                "n_name",
                "c_address",
                "c_phone",
                "c_comment",
            )
            .sort(by="revenue", descending=True)
            .head(20)
        )

        return QueryResult(
            frame=frame,
            sort_by=[("revenue", True)],
            limit=20,
        )

    @staticmethod
    def q11(run_config: RunConfig) -> QueryResult:
        """Query 11."""
        nation = get_data(run_config.dataset_path, "nation", run_config.suffix)
        partsupp = get_data(run_config.dataset_path, "partsupp", run_config.suffix)
        supplier = get_data(run_config.dataset_path, "supplier", run_config.suffix)

        var1 = "GERMANY"
        var2 = float(f"{0.0001 / run_config.scale_factor:.12f}")

        q1 = (
            partsupp.join(supplier, left_on="ps_suppkey", right_on="s_suppkey")
            .join(nation, left_on="s_nationkey", right_on="n_nationkey")
            .filter(pl.col("n_name") == var1)
        )
        q2 = q1.select(
            (pl.col("ps_supplycost") * pl.col("ps_availqty"))
            .sum()
            .round(2)
            .alias("tmp")
            * var2
        )

        frame = (
            q1.group_by("ps_partkey")
            .agg(
                (pl.col("ps_supplycost") * pl.col("ps_availqty"))
                .sum()
                .round(2)
                .alias("value")
            )
            .join(q2, how="cross")
            .filter(pl.col("value") > pl.col("tmp"))
            .select("ps_partkey", "value")
            .sort("value", descending=True)
        )

        return QueryResult(
            frame=frame,
            sort_by=[("value", True)],
        )

    @staticmethod
    def q12(run_config: RunConfig) -> QueryResult:
        """Query 12."""
        lineitem = get_data(run_config.dataset_path, "lineitem", run_config.suffix)
        orders = get_data(run_config.dataset_path, "orders", run_config.suffix)

        var1 = "MAIL"
        var2 = "SHIP"
        var3 = date(1994, 1, 1)
        var4 = date(1995, 1, 1)

        frame = (
            orders.join(lineitem, left_on="o_orderkey", right_on="l_orderkey")
            .filter(pl.col("l_shipmode").is_in([var1, var2]))
            .filter(pl.col("l_commitdate") < pl.col("l_receiptdate"))
            .filter(pl.col("l_shipdate") < pl.col("l_commitdate"))
            .filter(pl.col("l_receiptdate").is_between(var3, var4, closed="left"))
            .with_columns(
                pl.when(pl.col("o_orderpriority").is_in(["1-URGENT", "2-HIGH"]))
                .then(1)
                .otherwise(0)
                .alias("high_line_count"),
                pl.when(pl.col("o_orderpriority").is_in(["1-URGENT", "2-HIGH"]).not_())
                .then(1)
                .otherwise(0)
                .alias("low_line_count"),
            )
            .group_by("l_shipmode")
            .agg(pl.col("high_line_count").sum(), pl.col("low_line_count").sum())
            .sort("l_shipmode")
        )

        return QueryResult(
            frame=frame,
            sort_by=[("l_shipmode", False)],
        )

    @staticmethod
    def q13(run_config: RunConfig) -> QueryResult:
        """Query 13."""
        customer = get_data(run_config.dataset_path, "customer", run_config.suffix)
        orders = get_data(run_config.dataset_path, "orders", run_config.suffix)

        var1 = "special"
        var2 = "requests"

        orders = orders.filter(
            pl.col("o_comment").str.contains(f"{var1}.*{var2}").not_()
        )

        frame = (
            customer.join(orders, left_on="c_custkey", right_on="o_custkey", how="left")
            .group_by("c_custkey")
            .agg(pl.col("o_orderkey").count().alias("c_count"))
            .group_by("c_count")
            .len()
            .select(pl.col("c_count"), pl.col("len").alias("custdist"))
            .sort(by=["custdist", "c_count"], descending=[True, True])
        )

        return QueryResult(
            frame=frame,
            sort_by=[("custdist", True), ("c_count", True)],
        )

    @staticmethod
    def q14(run_config: RunConfig) -> QueryResult:
        """Query 14."""
        lineitem = get_data(run_config.dataset_path, "lineitem", run_config.suffix)
        part = get_data(run_config.dataset_path, "part", run_config.suffix)

        var1 = date(1995, 9, 1)
        var2 = date(1995, 10, 1)

        frame = (
            lineitem.join(part, left_on="l_partkey", right_on="p_partkey")
            .filter(pl.col("l_shipdate").is_between(var1, var2, closed="left"))
            .select(
                (
                    100.00
                    * pl.when(pl.col("p_type").str.contains("PROMO*"))
                    .then(pl.col("l_extendedprice") * (1 - pl.col("l_discount")))
                    .otherwise(0)
                    .sum()
                    / (pl.col("l_extendedprice") * (1 - pl.col("l_discount"))).sum()
                )
                .round(2)
                .alias("promo_revenue")
            )
        )

        return QueryResult(
            frame=frame,
            sort_by=[],
        )

    @staticmethod
    def q15(run_config: RunConfig) -> QueryResult:
        """Query 15."""
        lineitem = get_data(run_config.dataset_path, "lineitem", run_config.suffix)
        supplier = get_data(run_config.dataset_path, "supplier", run_config.suffix)

        var1 = date(1996, 1, 1)
        var2 = date(1996, 4, 1)

        revenue = (
            lineitem.filter(pl.col("l_shipdate").is_between(var1, var2, closed="left"))
            .group_by("l_suppkey")
            .agg(
                (pl.col("l_extendedprice") * (1 - pl.col("l_discount")))
                .sum()
                .alias("total_revenue")
            )
            .select(pl.col("l_suppkey").alias("supplier_no"), pl.col("total_revenue"))
        )

        frame = (
            supplier.join(revenue, left_on="s_suppkey", right_on="supplier_no")
            .filter(pl.col("total_revenue") == pl.col("total_revenue").max())
            .with_columns(pl.col("total_revenue").round(2))
            .select("s_suppkey", "s_name", "s_address", "s_phone", "total_revenue")
            .sort("s_suppkey")
        )

        return QueryResult(
            frame=frame,
            sort_by=[("s_suppkey", False)],
        )

    @staticmethod
    def q16(run_config: RunConfig) -> QueryResult:
        """Query 16."""
        part = get_data(run_config.dataset_path, "part", run_config.suffix)
        partsupp = get_data(run_config.dataset_path, "partsupp", run_config.suffix)
        supplier = get_data(run_config.dataset_path, "supplier", run_config.suffix)

        var1 = "Brand#45"

        supplier = supplier.filter(
            pl.col("s_comment").str.contains(".*Customer.*Complaints.*")
        ).select(pl.col("s_suppkey"), pl.col("s_suppkey").alias("ps_suppkey"))

        frame = (
            part.join(partsupp, left_on="p_partkey", right_on="ps_partkey")
            .filter(pl.col("p_brand") != var1)
            .filter(pl.col("p_type").str.contains("MEDIUM POLISHED*").not_())
            .filter(pl.col("p_size").is_in([49, 14, 23, 45, 19, 3, 36, 9]))
            .join(supplier, left_on="ps_suppkey", right_on="s_suppkey", how="left")
            .filter(pl.col("ps_suppkey_right").is_null())
            .group_by("p_brand", "p_type", "p_size")
            .agg(pl.col("ps_suppkey").n_unique().alias("supplier_cnt"))
            .sort(
                by=["supplier_cnt", "p_brand", "p_type", "p_size"],
                descending=[True, False, False, False],
            )
        )

        return QueryResult(
            frame=frame,
            sort_by=[
                ("supplier_cnt", True),
                ("p_brand", False),
                ("p_type", False),
                ("p_size", False),
            ],
        )

    @staticmethod
    def q17(run_config: RunConfig) -> QueryResult:
        """Query 17."""
        lineitem = get_data(run_config.dataset_path, "lineitem", run_config.suffix)
        part = get_data(run_config.dataset_path, "part", run_config.suffix)

        var1 = "Brand#23"
        var2 = "MED BOX"

        q1 = (
            part.filter(pl.col("p_brand") == var1)
            .filter(pl.col("p_container") == var2)
            .join(lineitem, how="inner", left_on="p_partkey", right_on="l_partkey")
        )

        frame = (
            q1.group_by("p_partkey")
            .agg((0.2 * pl.col("l_quantity").mean()).alias("avg_quantity"))
            .select(pl.col("p_partkey").alias("key"), pl.col("avg_quantity"))
            .join(q1, left_on="key", right_on="p_partkey")
            .filter(pl.col("l_quantity") < pl.col("avg_quantity"))
            .select(
                (pl.col("l_extendedprice").sum() / 7.0).round(2).alias("avg_yearly")
            )
        )

        return QueryResult(
            frame=frame,
            sort_by=[],
        )

    @staticmethod
    def q18(run_config: RunConfig) -> QueryResult:
        """Query 18."""
        path = run_config.dataset_path
        suffix = run_config.suffix
        customer = get_data(path, "customer", suffix)
        lineitem = get_data(path, "lineitem", suffix)
        orders = get_data(path, "orders", suffix)

        var1 = 300

        q1 = (
            lineitem.group_by("l_orderkey")
            .agg(pl.col("l_quantity").sum().alias("sum_quantity"))
            .filter(pl.col("sum_quantity") > var1)
        )

        frame = (
            orders.join(q1, left_on="o_orderkey", right_on="l_orderkey", how="semi")
            .join(lineitem, left_on="o_orderkey", right_on="l_orderkey")
            .join(customer, left_on="o_custkey", right_on="c_custkey")
            .group_by(
                "c_name", "o_custkey", "o_orderkey", "o_orderdate", "o_totalprice"
            )
            .agg(pl.col("l_quantity").sum().alias("sum(l_quantity)"))
            .select(
                pl.col("c_name"),
                pl.col("o_custkey").alias("c_custkey"),
                pl.col("o_orderkey"),
                pl.col("o_orderdate"),
                pl.col("o_totalprice"),
                pl.col("sum(l_quantity)"),
            )
            .sort(by=["o_totalprice", "o_orderdate"], descending=[True, False])
            .head(100)
        )

        return QueryResult(
            frame=frame,
            sort_by=[("o_totalprice", True), ("o_orderdate", False)],
            limit=100,
        )

    @staticmethod
    def q19(run_config: RunConfig) -> QueryResult:
        """Query 19."""
        lineitem = get_data(run_config.dataset_path, "lineitem", run_config.suffix)
        part = get_data(run_config.dataset_path, "part", run_config.suffix)

        frame = (
            part.join(lineitem, left_on="p_partkey", right_on="l_partkey")
            .filter(pl.col("l_shipmode").is_in(["AIR", "AIR REG"]))
            .filter(pl.col("l_shipinstruct") == "DELIVER IN PERSON")
            .filter(
                (
                    (pl.col("p_brand") == "Brand#12")
                    & pl.col("p_container").is_in(
                        ["SM CASE", "SM BOX", "SM PACK", "SM PKG"]
                    )
                    & (pl.col("l_quantity").is_between(1, 11))
                    & (pl.col("p_size").is_between(1, 5))
                )
                | (
                    (pl.col("p_brand") == "Brand#23")
                    & pl.col("p_container").is_in(
                        ["MED BAG", "MED BOX", "MED PKG", "MED PACK"]
                    )
                    & (pl.col("l_quantity").is_between(10, 20))
                    & (pl.col("p_size").is_between(1, 10))
                )
                | (
                    (pl.col("p_brand") == "Brand#34")
                    & pl.col("p_container").is_in(
                        ["LG CASE", "LG BOX", "LG PACK", "LG PKG"]
                    )
                    & (pl.col("l_quantity").is_between(20, 30))
                    & (pl.col("p_size").is_between(1, 15))
                )
            )
            .select(
                (pl.col("l_extendedprice") * (1 - pl.col("l_discount")))
                .sum()
                .round(2)
                .alias("revenue")
            )
        )

        return QueryResult(
            frame=frame,
            sort_by=[],
        )

    @staticmethod
    def q20(run_config: RunConfig) -> QueryResult:
        """Query 20."""
        lineitem = get_data(run_config.dataset_path, "lineitem", run_config.suffix)
        nation = get_data(run_config.dataset_path, "nation", run_config.suffix)
        part = get_data(run_config.dataset_path, "part", run_config.suffix)
        partsupp = get_data(run_config.dataset_path, "partsupp", run_config.suffix)
        supplier = get_data(run_config.dataset_path, "supplier", run_config.suffix)

        var1 = date(1994, 1, 1)
        var2 = date(1995, 1, 1)
        var3 = "CANADA"
        var4 = "forest"

        q1 = (
            lineitem.filter(pl.col("l_shipdate").is_between(var1, var2, closed="left"))
            .group_by("l_partkey", "l_suppkey")
            .agg((pl.col("l_quantity").sum() * 0.5).alias("sum_quantity"))
        )
        q2 = nation.filter(pl.col("n_name") == var3)
        q3 = supplier.join(q2, left_on="s_nationkey", right_on="n_nationkey")

        frame = (
            part.filter(pl.col("p_name").str.starts_with(var4))
            .select(pl.col("p_partkey").unique())
            .join(partsupp, left_on="p_partkey", right_on="ps_partkey")
            .join(
                q1,
                left_on=["ps_suppkey", "p_partkey"],
                right_on=["l_suppkey", "l_partkey"],
            )
            .filter(pl.col("ps_availqty") > pl.col("sum_quantity"))
            .select(pl.col("ps_suppkey").unique())
            .join(q3, left_on="ps_suppkey", right_on="s_suppkey")
            .select("s_name", "s_address")
            .sort("s_name")
        )

        return QueryResult(
            frame=frame,
            sort_by=[("s_name", False)],
        )

    @staticmethod
    def q21(run_config: RunConfig) -> QueryResult:
        """Query 21."""
        lineitem = get_data(run_config.dataset_path, "lineitem", run_config.suffix)
        nation = get_data(run_config.dataset_path, "nation", run_config.suffix)
        orders = get_data(run_config.dataset_path, "orders", run_config.suffix)
        supplier = get_data(run_config.dataset_path, "supplier", run_config.suffix)

        var1 = "SAUDI ARABIA"

        q1 = (
            lineitem.group_by("l_orderkey")
            .agg(pl.col("l_suppkey").len().alias("n_supp_by_order"))
            .filter(pl.col("n_supp_by_order") > 1)
            .join(
                lineitem.filter(pl.col("l_receiptdate") > pl.col("l_commitdate")),
                on="l_orderkey",
            )
        )

        frame = (
            q1.group_by("l_orderkey")
            .agg(pl.col("l_suppkey").len().alias("n_supp_by_order"))
            .join(q1, on="l_orderkey")
            .join(supplier, left_on="l_suppkey", right_on="s_suppkey")
            .join(nation, left_on="s_nationkey", right_on="n_nationkey")
            .join(orders, left_on="l_orderkey", right_on="o_orderkey")
            .filter(pl.col("n_supp_by_order") == 1)
            .filter(pl.col("n_name") == var1)
            .filter(pl.col("o_orderstatus") == "F")
            .group_by("s_name")
            .agg(pl.len().alias("numwait"))
            .sort(by=["numwait", "s_name"], descending=[True, False])
            .head(100)
        )

        return QueryResult(
            frame=frame,
            sort_by=[("numwait", True), ("s_name", False)],
            limit=100,
        )

    @staticmethod
    def q22(run_config: RunConfig) -> QueryResult:
        """Query 22."""
        customer = get_data(run_config.dataset_path, "customer", run_config.suffix)
        orders = get_data(run_config.dataset_path, "orders", run_config.suffix)

        q1 = (
            customer.with_columns(pl.col("c_phone").str.slice(0, 2).alias("cntrycode"))
            .filter(pl.col("cntrycode").str.contains("13|31|23|29|30|18|17"))
            .select("c_acctbal", "c_custkey", "cntrycode")
        )

        q2 = q1.filter(pl.col("c_acctbal") > 0.0).select(
            pl.col("c_acctbal").mean().alias("avg_acctbal")
        )

        q3 = orders.select(pl.col("o_custkey").unique()).with_columns(
            pl.col("o_custkey").alias("c_custkey")
        )

        frame = (
            q1.join(q3, on="c_custkey", how="left")
            .filter(pl.col("o_custkey").is_null())
            .join(q2, how="cross")
            .filter(pl.col("c_acctbal") > pl.col("avg_acctbal"))
            .group_by("cntrycode")
            .agg(
                pl.col("c_acctbal").count().alias("numcust"),
                pl.col("c_acctbal").sum().round(2).alias("totacctbal"),
            )
            .sort("cntrycode")
        )

        return QueryResult(
            frame=frame,
            sort_by=[("cntrycode", False)],
        )


class PDSHDuckDBQueries:
    """PDS-H DuckDB query definitions."""

    name: str = "pdsh"
    EXPECTED_CASTS = EXPECTED_CASTS
    EXPECTED_CASTS_DECIMAL = EXPECTED_CASTS_DECIMAL
    EXPECTED_CASTS_TIMESTAMP = EXPECTED_CASTS_TIMESTAMP

    @staticmethod
    def q1(run_config: RunConfig) -> str:
        """Query 1."""
        return """
        select
            l_returnflag,
            l_linestatus,
            sum(l_quantity) as sum_qty,
            sum(l_extendedprice) as sum_base_price,
            sum(l_extendedprice * (1 - l_discount)) as sum_disc_price,
            sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) as sum_charge,
            avg(l_quantity) as avg_qty,
            avg(l_extendedprice) as avg_price,
            avg(l_discount) as avg_disc,
            count(*) as count_order
        from
            lineitem
        where
            l_shipdate <= DATE '1998-09-02'
        group by
            l_returnflag,
            l_linestatus
        order by
            l_returnflag,
            l_linestatus
        """

    @staticmethod
    def q2(run_config: RunConfig) -> str:
        """Query 2."""
        return """
            select
                s_acctbal,
                s_name,
                n_name,
                p_partkey,
                p_mfgr,
                s_address,
                s_phone,
                s_comment
            from
                part,
                supplier,
                partsupp,
                nation,
                region
            where
                p_partkey = ps_partkey
                and s_suppkey = ps_suppkey
                and p_size = 15
                and p_type like '%BRASS'
                and s_nationkey = n_nationkey
                and n_regionkey = r_regionkey
                and r_name = 'EUROPE'
                and ps_supplycost = (
                    select
                        min(ps_supplycost)
                    from
                        partsupp,
                        supplier,
                        nation,
                        region
                    where
                        p_partkey = ps_partkey
                        and s_suppkey = ps_suppkey
                        and s_nationkey = n_nationkey
                        and n_regionkey = r_regionkey
                        and r_name = 'EUROPE'
                )
            order by
                s_acctbal desc,
                n_name,
                s_name,
                p_partkey
            limit 100
                    """

    @staticmethod
    def q3(run_config: RunConfig) -> str:
        """Query 3."""
        return """
            select
                l_orderkey,
                sum(l_extendedprice * (1 - l_discount)) as revenue,
                o_orderdate,
                o_shippriority
            from
                customer,
                orders,
                lineitem
            where
                c_mktsegment = 'BUILDING'
                and c_custkey = o_custkey
                and l_orderkey = o_orderkey
                and o_orderdate < '1995-03-15'
                and l_shipdate > '1995-03-15'
            group by
                l_orderkey,
                o_orderdate,
                o_shippriority
            order by
                revenue desc,
                o_orderdate
            limit 10
                    """

    @staticmethod
    def q4(run_config: RunConfig) -> str:
        """Query 4."""
        return """
            select
                o_orderpriority,
                count(*) as order_count
            from
                orders
            where
                o_orderdate >= timestamp '1993-07-01'
                and o_orderdate < timestamp '1993-07-01' + interval '3' month
                and exists (
                    select
                        *
                    from
                        lineitem
                    where
                        l_orderkey = o_orderkey
                        and l_commitdate < l_receiptdate
                )
            group by
                o_orderpriority
            order by
                o_orderpriority
                    """

    @staticmethod
    def q5(run_config: RunConfig) -> str:
        """Query 5."""
        return """
            select
                n_name,
                sum(l_extendedprice * (1 - l_discount)) as revenue
            from
                customer,
                orders,
                lineitem,
                supplier,
                nation,
                region
            where
                c_custkey = o_custkey
                and l_orderkey = o_orderkey
                and l_suppkey = s_suppkey
                and c_nationkey = s_nationkey
                and s_nationkey = n_nationkey
                and n_regionkey = r_regionkey
                and r_name = 'ASIA'
                and o_orderdate >= timestamp '1994-01-01'
                and o_orderdate < timestamp '1994-01-01' + interval '1' year
            group by
                n_name
            order by
                revenue desc
                    """

    @staticmethod
    def q6(run_config: RunConfig) -> str:
        """Query 6."""
        return """
            select
                sum(l_extendedprice * l_discount) as revenue
            from
                lineitem
            where
                l_shipdate >= timestamp '1994-01-01'
                and l_shipdate < timestamp '1994-01-01' + interval '1' year
                and l_discount between .06 - 0.01 and .06 + 0.01
                and l_quantity < 24
                    """

    @staticmethod
    def q7(run_config: RunConfig) -> str:
        """Query 7."""
        return """
            select
                supp_nation,
                cust_nation,
                l_year,
                sum(volume) as revenue
            from
                (
                    select
                        n1.n_name as supp_nation,
                        n2.n_name as cust_nation,
                        year(l_shipdate) as l_year,
                        l_extendedprice * (1 - l_discount) as volume
                    from
                        supplier,
                        lineitem,
                        orders,
                        customer,
                        nation n1,
                        nation n2
                    where
                        s_suppkey = l_suppkey
                        and o_orderkey = l_orderkey
                        and c_custkey = o_custkey
                        and s_nationkey = n1.n_nationkey
                        and c_nationkey = n2.n_nationkey
                        and (
                            (n1.n_name = 'FRANCE' and n2.n_name = 'GERMANY')
                            or (n1.n_name = 'GERMANY' and n2.n_name = 'FRANCE')
                        )
                        and l_shipdate between timestamp '1995-01-01' and timestamp '1996-12-31'
                ) as shipping
            group by
                supp_nation,
                cust_nation,
                l_year
            order by
                supp_nation,
                cust_nation,
                l_year
                    """

    @staticmethod
    def q8(run_config: RunConfig) -> str:
        """Query 8."""
        return """
            select
                o_year,
                round(
                    sum(case
                        when nation = 'BRAZIL' then volume
                        else 0
                    end) / sum(volume)
                , 2) as mkt_share
            from
                (
                    select
                        extract(year from o_orderdate) as o_year,
                        l_extendedprice * (1 - l_discount) as volume,
                        n2.n_name as nation
                    from
                        part,
                        supplier,
                        lineitem,
                        orders,
                        customer,
                        nation n1,
                        nation n2,
                        region
                    where
                        p_partkey = l_partkey
                        and s_suppkey = l_suppkey
                        and l_orderkey = o_orderkey
                        and o_custkey = c_custkey
                        and c_nationkey = n1.n_nationkey
                        and n1.n_regionkey = r_regionkey
                        and r_name = 'AMERICA'
                        and s_nationkey = n2.n_nationkey
                        and o_orderdate between timestamp '1995-01-01' and timestamp '1996-12-31'
                        and p_type = 'ECONOMY ANODIZED STEEL'
                ) as all_nations
            group by
                o_year
            order by
                o_year
        """

    @staticmethod
    def q9(run_config: RunConfig) -> str:
        """Query 9."""
        return """
            select
                nation,
                o_year,
                round(sum(amount), 2) as sum_profit
            from
                (
                    select
                        n_name as nation,
                        year(o_orderdate) as o_year,
                        l_extendedprice * (1 - l_discount) - ps_supplycost * l_quantity as amount
                    from
                        part,
                        supplier,
                        lineitem,
                        partsupp,
                        orders,
                        nation
                    where
                        s_suppkey = l_suppkey
                        and ps_suppkey = l_suppkey
                        and ps_partkey = l_partkey
                        and p_partkey = l_partkey
                        and o_orderkey = l_orderkey
                        and s_nationkey = n_nationkey
                        and p_name like '%green%'
                ) as profit
            group by
                nation,
                o_year
            order by
                nation,
                o_year desc
        """

    @staticmethod
    def q10(run_config: RunConfig) -> str:
        """Query 10."""
        return """
            select
                c_custkey,
                c_name,
                round(sum(l_extendedprice * (1 - l_discount)), 2) as revenue,
                c_acctbal,
                n_name,
                c_address,
                c_phone,
                c_comment
            from
                customer,
                orders,
                lineitem,
                nation
            where
                c_custkey = o_custkey
                and l_orderkey = o_orderkey
                and o_orderdate >= date '1993-10-01'
                and o_orderdate < date '1993-10-01' + interval '3' month
                and l_returnflag = 'R'
                and c_nationkey = n_nationkey
            group by
                c_custkey,
                c_name,
                c_acctbal,
                c_phone,
                n_name,
                c_address,
                c_comment
            order by
                revenue desc
            limit 20
        """

    @staticmethod
    def q11(run_config: RunConfig) -> str:
        """Query 11."""
        return f"""
            select
                ps_partkey,
                round(sum(ps_supplycost * ps_availqty), 2) as value
            from
                partsupp, supplier, nation
            where
                ps_suppkey = s_suppkey
                and s_nationkey = n_nationkey
                and n_name = 'GERMANY'
            group by
                ps_partkey
            having
                sum(ps_supplycost * ps_availqty) > (
                    select
                        sum(ps_supplycost * ps_availqty) * {0.0001 / run_config.scale_factor}
                    from
                        partsupp, supplier, nation
                    where
                        ps_suppkey = s_suppkey
                        and s_nationkey = n_nationkey
                        and n_name = 'GERMANY'
                )
            order by
                value desc
        """

    @staticmethod
    def q12(run_config: RunConfig) -> str:
        """Query 12."""
        return """
            select
                l_shipmode,
                sum(case
                    when o_orderpriority = '1-URGENT'
                        or o_orderpriority = '2-HIGH'
                        then 1
                    else 0
                end) as high_line_count,
                sum(case
                    when o_orderpriority <> '1-URGENT'
                        and o_orderpriority <> '2-HIGH'
                        then 1
                    else 0
                end) as low_line_count
            from
                orders,
                lineitem
            where
                o_orderkey = l_orderkey
                and l_shipmode in ('MAIL', 'SHIP')
                and l_commitdate < l_receiptdate
                and l_shipdate < l_commitdate
                and l_receiptdate >= date '1994-01-01'
                and l_receiptdate < date '1994-01-01' + interval '1' year
            group by
                l_shipmode
            order by
                l_shipmode
        """

    @staticmethod
    def q13(run_config: RunConfig) -> str:
        """Query 13."""
        return """
            select
                c_count, count(*) as custdist
            from (
                select
                    c_custkey,
                    count(o_orderkey)
                from
                    customer left outer join orders on
                    c_custkey = o_custkey
                    and o_comment not like '%special%requests%'
                group by
                    c_custkey
                )as c_orders (c_custkey, c_count)
            group by
                c_count
            order by
                custdist desc,
                c_count desc
        """

    @staticmethod
    def q14(run_config: RunConfig) -> str:
        """Query 14."""
        return """
            select
                round(100.00 * sum(case
                    when p_type like 'PROMO%'
                        then l_extendedprice * (1 - l_discount)
                    else 0
                end) / sum(l_extendedprice * (1 - l_discount)), 2) as promo_revenue
            from
                lineitem,
                part
            where
                l_partkey = p_partkey
                and l_shipdate >= date '1995-09-01'
                and l_shipdate < date '1995-09-01' + interval '1' month
        """

    @staticmethod
    def q15(run_config: RunConfig) -> str:
        """Query 15."""
        return """
            with revenue (supplier_no, total_revenue) as (
                select
                    l_suppkey,
                    sum(l_extendedprice * (1 - l_discount))
                from
                    lineitem
                where
                    l_shipdate >= date '1996-01-01'
                    and l_shipdate < date '1996-01-01' + interval '3' month
                group by
                    l_suppkey
            )
            select
                s_suppkey,
                s_name,
                s_address,
                s_phone,
                total_revenue
            from
                supplier,
                revenue
            where
                s_suppkey = supplier_no
                and total_revenue = (
                    select
                        max(total_revenue)
                    from
                        revenue
                )
            order by
                s_suppkey
        """

    @staticmethod
    def q16(run_config: RunConfig) -> str:
        """Query 16."""
        return """
            select
                p_brand,
                p_type,
                p_size,
                count(distinct ps_suppkey) as supplier_cnt
            from
                partsupp,
                part
            where
                p_partkey = ps_partkey
                and p_brand <> 'Brand#45'
                and p_type not like 'MEDIUM POLISHED%'
                and p_size in (49, 14, 23, 45, 19, 3, 36, 9)
                and ps_suppkey not in (
                    select
                        s_suppkey
                    from
                        supplier
                    where
                        s_comment like '%Customer%Complaints%'
                )
            group by
                p_brand,
                p_type,
                p_size
            order by
                supplier_cnt desc,
                p_brand,
                p_type,
                p_size
        """

    @staticmethod
    def q17(run_config: RunConfig) -> str:
        """Query 17."""
        return """
            select
                round(sum(l_extendedprice) / 7.0, 2) as avg_yearly
            from
                lineitem,
                part
            where
                p_partkey = l_partkey
                and p_brand = 'Brand#23'
                and p_container = 'MED BOX'
                and l_quantity < (
                    select
                        0.2 * avg(l_quantity)
                    from
                        lineitem
                    where
                        l_partkey = p_partkey
                )
        """

    @staticmethod
    def q18(run_config: RunConfig) -> str:
        """Query 18."""
        return """
            select
                c_name,
                c_custkey,
                o_orderkey,
                o_orderdate,
                o_totalprice,
                sum(l_quantity)
            from
                customer,
                orders,
                lineitem
            where
                o_orderkey in (
                    select
                        l_orderkey
                    from
                        lineitem
                    group by
                        l_orderkey having
                            sum(l_quantity) > 300
                )
                and c_custkey = o_custkey
                and o_orderkey = l_orderkey
            group by
                c_name,
                c_custkey,
                o_orderkey,
                o_orderdate,
                o_totalprice
            order by
                o_totalprice desc,
                o_orderdate
            limit 100
        """

    @staticmethod
    def q19(run_config: RunConfig) -> str:
        """Query 19."""
        return """
            select
                round(sum(l_extendedprice* (1 - l_discount)), 2) as revenue
            from
                lineitem,
                part
            where
                (
                    p_partkey = l_partkey
                    and p_brand = 'Brand#12'
                    and p_container in ('SM CASE', 'SM BOX', 'SM PACK', 'SM PKG')
                    and l_quantity >= 1 and l_quantity <= 1 + 10
                    and p_size between 1 and 5
                    and l_shipmode in ('AIR', 'AIR REG')
                    and l_shipinstruct = 'DELIVER IN PERSON'
                )
                or
                (
                    p_partkey = l_partkey
                    and p_brand = 'Brand#23'
                    and p_container in ('MED BAG', 'MED BOX', 'MED PKG', 'MED PACK')
                    and l_quantity >= 10 and l_quantity <= 20
                    and p_size between 1 and 10
                    and l_shipmode in ('AIR', 'AIR REG')
                    and l_shipinstruct = 'DELIVER IN PERSON'
                )
                or
                (
                    p_partkey = l_partkey
                    and p_brand = 'Brand#34'
                    and p_container in ('LG CASE', 'LG BOX', 'LG PACK', 'LG PKG')
                    and l_quantity >= 20 and l_quantity <= 30
                    and p_size between 1 and 15
                    and l_shipmode in ('AIR', 'AIR REG')
                    and l_shipinstruct = 'DELIVER IN PERSON'
                )
        """

    @staticmethod
    def q20(run_config: RunConfig) -> str:
        """Query 20."""
        return """
            select
                s_name,
                s_address
            from
                supplier,
                nation
            where
                s_suppkey in (
                    select
                        ps_suppkey
                    from
                        partsupp
                    where
                        ps_partkey in (
                            select
                                p_partkey
                            from
                                part
                            where
                                p_name like 'forest%'
                        )
                        and ps_availqty > (
                            select
                                0.5 * sum(l_quantity)
                            from
                                lineitem
                            where
                                l_partkey = ps_partkey
                                and l_suppkey = ps_suppkey
                                and l_shipdate >= date '1994-01-01'
                                and l_shipdate < date '1994-01-01' + interval '1' year
                        )
                )
                and s_nationkey = n_nationkey
                and n_name = 'CANADA'
            order by
                s_name
        """

    @staticmethod
    def q21(run_config: RunConfig) -> str:
        """Query 21."""
        return """
            select
                s_name,
                count(*) as numwait
            from
                supplier,
                lineitem l1,
                orders,
                nation
            where
                s_suppkey = l1.l_suppkey
                and o_orderkey = l1.l_orderkey
                and o_orderstatus = 'F'
                and l1.l_receiptdate > l1.l_commitdate
                and exists (
                    select
                        *
                    from
                        lineitem l2
                    where
                        l2.l_orderkey = l1.l_orderkey
                        and l2.l_suppkey <> l1.l_suppkey
                )
                and not exists (
                    select
                        *
                    from
                        lineitem l3
                    where
                        l3.l_orderkey = l1.l_orderkey
                        and l3.l_suppkey <> l1.l_suppkey
                        and l3.l_receiptdate > l3.l_commitdate
                )
                and s_nationkey = n_nationkey
                and n_name = 'SAUDI ARABIA'
            group by
                s_name
            order by
                numwait desc,
                s_name
            limit 100
        """

    @staticmethod
    def q22(run_config: RunConfig) -> str:
        """Query 22."""
        return """
            select
                cntrycode,
                count(*) as numcust,
                sum(c_acctbal) as totacctbal
            from (
                select
                    substring(c_phone from 1 for 2) as cntrycode,
                    c_acctbal
                from
                    customer
                where
                    substring(c_phone from 1 for 2) in
                        (13, 31, 23, 29, 30, 18, 17)
                    and c_acctbal > (
                        select
                            avg(c_acctbal)
                        from
                            customer
                        where
                            c_acctbal > 0.00
                            and substring (c_phone from 1 for 2) in
                                (13, 31, 23, 29, 30, 18, 17)
                    )
                    and not exists (
                        select
                            *
                        from
                            orders
                        where
                            o_custkey = c_custkey
                    )
                ) as custsale
            group by
                cntrycode
            order by
                cntrycode
        """


if __name__ == "__main__":
    parser = build_parser(num_queries=22)
    args = parse_args(parser=parser)
    if args.frontend not in _CPU_ENGINES:
        os.environ["POLARS_MAX_THREADS"] = os.environ.get("POLARS_MAX_THREADS", "1")
        os.environ["OMP_NUM_THREADS"] = os.environ.get("OMP_NUM_THREADS", "1")
    run_polars(PDSHQueries, args)
