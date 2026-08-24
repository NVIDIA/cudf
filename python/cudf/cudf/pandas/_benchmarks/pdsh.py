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

from typing import TYPE_CHECKING

from numpy import datetime64

import cudf.pandas

cudf.pandas.install()

import pandas as pd  # noqa: E402

from cudf.pandas._benchmarks.utils import (  # noqa: E402
    get_data,
    run_pandas,
)

if TYPE_CHECKING:
    from cudf.pandas._benchmarks.utils import RunConfig


class PDSHQueries:
    """PDS-H query definitions."""

    name: str = "pdsh"

    @staticmethod
    def q0(run_config: RunConfig) -> pd.DataFrame:
        """Query 0."""
        return pd.DataFrame()

    @staticmethod
    def q1(run_config: RunConfig) -> pd.DataFrame:
        """Query 1."""
        lineitem = get_data(
            run_config.dataset_path,
            "lineitem",
            run_config.suffix,
            columns=[
                "l_orderkey",
                "l_quantity",
                "l_extendedprice",
                "l_discount",
                "l_tax",
                "l_returnflag",
                "l_linestatus",
                "l_shipdate",
            ],
        )

        var1 = datetime64("1998-09-02")

        filt = lineitem[lineitem["l_shipdate"] <= var1]

        # This is lenient towards pandas as normally an optimizer should decide
        # that this could be computed before the groupby aggregation.
        # Other implementations don't enjoy this benefit.
        filt["disc_price"] = filt.l_extendedprice * (1.0 - filt.l_discount)
        filt["charge"] = (
            filt.l_extendedprice * (1.0 - filt.l_discount) * (1.0 + filt.l_tax)
        )

        gb = filt.groupby(["l_returnflag", "l_linestatus"], as_index=False)
        agg = gb.agg(
            sum_qty=pd.NamedAgg(column="l_quantity", aggfunc="sum"),
            sum_base_price=pd.NamedAgg(
                column="l_extendedprice", aggfunc="sum"
            ),
            sum_disc_price=pd.NamedAgg(column="disc_price", aggfunc="sum"),
            sum_charge=pd.NamedAgg(column="charge", aggfunc="sum"),
            avg_qty=pd.NamedAgg(column="l_quantity", aggfunc="mean"),
            avg_price=pd.NamedAgg(column="l_extendedprice", aggfunc="mean"),
            avg_disc=pd.NamedAgg(column="l_discount", aggfunc="mean"),
            count_order=pd.NamedAgg(column="l_orderkey", aggfunc="size"),
        )

        return agg.sort_values(
            ["l_returnflag", "l_linestatus"], ignore_index=True
        )

    @staticmethod
    def q2(run_config: RunConfig) -> pd.DataFrame:
        """Query 2."""
        nation = get_data(
            run_config.dataset_path,
            "nation",
            run_config.suffix,
            columns=["n_nationkey", "n_regionkey", "n_name"],
        )
        part = get_data(
            run_config.dataset_path,
            "part",
            run_config.suffix,
            columns=["p_partkey", "p_size", "p_type", "p_mfgr"],
        )
        partsupp = get_data(
            run_config.dataset_path,
            "partsupp",
            run_config.suffix,
            columns=["ps_partkey", "ps_suppkey", "ps_supplycost"],
        )
        region = get_data(
            run_config.dataset_path,
            "region",
            run_config.suffix,
            columns=["r_regionkey", "r_name"],
        )
        supplier = get_data(
            run_config.dataset_path, "supplier", run_config.suffix
        )

        var1 = 15
        var2 = "BRASS"
        var3 = "EUROPE"

        part = part[
            (part["p_size"] == var1) & part["p_type"].str.endswith(var2)
        ]
        region = region[region["r_name"] == var3]

        jn = (
            part.merge(partsupp, left_on="p_partkey", right_on="ps_partkey")
            .merge(supplier, left_on="ps_suppkey", right_on="s_suppkey")
            .merge(nation, left_on="s_nationkey", right_on="n_nationkey")
            .merge(region, left_on="n_regionkey", right_on="r_regionkey")
        )

        gb = jn.groupby("p_partkey", as_index=False)
        agg = gb["ps_supplycost"].min()
        jn2 = agg.merge(jn, on=["p_partkey", "ps_supplycost"])

        sel = jn2.loc[
            :,
            [
                "s_acctbal",
                "s_name",
                "n_name",
                "p_partkey",
                "p_mfgr",
                "s_address",
                "s_phone",
                "s_comment",
            ],
        ]

        sort = sel.sort_values(
            by=["s_acctbal", "n_name", "s_name", "p_partkey"],
            ascending=[False, True, True, True],
            ignore_index=True,
        )
        return sort.head(100)

    @staticmethod
    def q3(run_config: RunConfig) -> pd.DataFrame:
        """Query 3."""
        customer = get_data(
            run_config.dataset_path,
            "customer",
            run_config.suffix,
            columns=["c_custkey", "c_mktsegment"],
        )
        lineitem = get_data(
            run_config.dataset_path,
            "lineitem",
            run_config.suffix,
            columns=[
                "l_orderkey",
                "l_extendedprice",
                "l_discount",
                "l_shipdate",
            ],
        )
        orders = get_data(
            run_config.dataset_path,
            "orders",
            run_config.suffix,
            columns=[
                "o_custkey",
                "o_orderkey",
                "o_orderdate",
                "o_shippriority",
            ],
        )

        var1 = "BUILDING"
        var2 = datetime64("1995-03-15")

        fcustomer = customer[customer["c_mktsegment"] == var1]
        orders = orders[orders["o_orderdate"] < var2]
        lineitem = lineitem[lineitem["l_shipdate"] > var2]

        jn1 = fcustomer.merge(
            orders, left_on="c_custkey", right_on="o_custkey"
        )
        jn2 = jn1.merge(lineitem, left_on="o_orderkey", right_on="l_orderkey")
        jn2["revenue"] = jn2.l_extendedprice * (1 - jn2.l_discount)

        gb = jn2.groupby(
            ["o_orderkey", "o_orderdate", "o_shippriority"], as_index=False
        )
        agg = gb["revenue"].sum()

        sel = agg.loc[
            :, ["o_orderkey", "revenue", "o_orderdate", "o_shippriority"]
        ]
        sel = sel.rename(columns={"o_orderkey": "l_orderkey"})

        sorted_df = sel.sort_values(
            by=["revenue", "o_orderdate"],
            ascending=[False, True],
            ignore_index=True,
        )
        return sorted_df.head(10)

    @staticmethod
    def q4(run_config: RunConfig) -> pd.DataFrame:
        """Query 4."""
        lineitem = get_data(
            run_config.dataset_path,
            "lineitem",
            run_config.suffix,
            columns=["l_orderkey", "l_commitdate", "l_receiptdate"],
        )
        orders = get_data(
            run_config.dataset_path,
            "orders",
            run_config.suffix,
            columns=["o_orderkey", "o_orderdate", "o_orderpriority"],
        )

        var1 = datetime64("1993-07-01")
        var2 = datetime64("1993-10-01")

        orders = orders[
            (orders["o_orderdate"] >= var1) & (orders["o_orderdate"] < var2)
        ]
        lineitem = lineitem[
            lineitem["l_commitdate"] < lineitem["l_receiptdate"]
        ]

        jn = lineitem.merge(
            orders, left_on="l_orderkey", right_on="o_orderkey"
        )

        jn = jn.drop_duplicates(subset=["o_orderpriority", "l_orderkey"])

        gb = jn.groupby("o_orderpriority", as_index=False)
        agg = gb.agg(
            order_count=pd.NamedAgg(column="o_orderkey", aggfunc="count")
        )

        return agg.sort_values(["o_orderpriority"], ignore_index=True)

    @staticmethod
    def q5(run_config: RunConfig) -> pd.DataFrame:
        """Query 5."""
        path = run_config.dataset_path
        suffix = run_config.suffix
        customer = get_data(
            path,
            "customer",
            suffix,
            columns=["c_custkey", "c_nationkey"],
        )
        lineitem = get_data(
            path,
            "lineitem",
            suffix,
            columns=[
                "l_orderkey",
                "l_suppkey",
                "l_extendedprice",
                "l_discount",
            ],
        )
        nation = get_data(
            path,
            "nation",
            suffix,
            columns=["n_nationkey", "n_name", "n_regionkey"],
        )
        orders = get_data(
            path,
            "orders",
            suffix,
            columns=["o_orderkey", "o_custkey", "o_orderdate"],
        )
        region = get_data(
            path,
            "region",
            suffix,
            columns=["r_regionkey", "r_name"],
        )
        supplier = get_data(
            path,
            "supplier",
            suffix,
            columns=["s_suppkey", "s_nationkey"],
        )

        var1 = "ASIA"
        var2 = datetime64("1994-01-01")
        var3 = datetime64("1995-01-01")

        region = region[region["r_name"] == var1]

        jn1 = region.merge(
            nation, left_on="r_regionkey", right_on="n_regionkey"
        )
        jn2 = jn1.merge(
            customer, left_on="n_nationkey", right_on="c_nationkey"
        )
        jn3 = jn2.merge(orders, left_on="c_custkey", right_on="o_custkey")
        jn4 = jn3.merge(lineitem, left_on="o_orderkey", right_on="l_orderkey")
        jn5 = jn4.merge(
            supplier,
            left_on=["l_suppkey", "n_nationkey"],
            right_on=["s_suppkey", "s_nationkey"],
        )

        jn5 = jn5[(jn5["o_orderdate"] >= var2) & (jn5["o_orderdate"] < var3)]
        jn5["revenue"] = jn5.l_extendedprice * (1.0 - jn5.l_discount)

        gb = jn5.groupby("n_name", as_index=False)["revenue"].sum()
        return gb.sort_values("revenue", ascending=False, ignore_index=True)

    @staticmethod
    def q6(run_config: RunConfig) -> pd.DataFrame:
        """Query 6."""
        path = run_config.dataset_path
        suffix = run_config.suffix
        lineitem = get_data(
            path,
            "lineitem",
            suffix,
            columns=[
                "l_shipdate",
                "l_discount",
                "l_quantity",
                "l_extendedprice",
            ],
        )

        var1 = datetime64("1994-01-01")
        var2 = datetime64("1995-01-01")
        var3 = 0.05
        var4 = 0.07
        var5 = 24

        filt = lineitem[
            (lineitem["l_shipdate"] >= var1) & (lineitem["l_shipdate"] < var2)
        ]
        filt = filt[
            (filt["l_discount"] >= var3) & (filt["l_discount"] <= var4)
        ]
        filt = filt[filt["l_quantity"] < var5]
        result_value = (filt["l_extendedprice"] * filt["l_discount"]).sum()
        return pd.DataFrame({"revenue": [result_value]})

    @staticmethod
    def q7(run_config: RunConfig) -> pd.DataFrame:
        """Query 7."""
        customer = get_data(
            run_config.dataset_path,
            "customer",
            run_config.suffix,
            columns=["c_custkey", "c_nationkey"],
        )
        lineitem = get_data(
            run_config.dataset_path,
            "lineitem",
            run_config.suffix,
            columns=[
                "l_orderkey",
                "l_suppkey",
                "l_extendedprice",
                "l_discount",
                "l_shipdate",
            ],
        )
        nation = get_data(
            run_config.dataset_path,
            "nation",
            run_config.suffix,
            columns=["n_nationkey", "n_name"],
        )
        orders = get_data(
            run_config.dataset_path,
            "orders",
            run_config.suffix,
            columns=["o_custkey", "o_orderkey"],
        )
        supplier = get_data(
            run_config.dataset_path,
            "supplier",
            run_config.suffix,
            columns=["s_suppkey", "s_nationkey"],
        )

        var1 = "FRANCE"
        var2 = "GERMANY"
        var3 = datetime64("1995-01-01")
        var4 = datetime64("1996-12-31")

        n1 = nation[(nation["n_name"] == var1)]
        n2 = nation[(nation["n_name"] == var2)]

        # Part 1
        jn1 = customer.merge(n1, left_on="c_nationkey", right_on="n_nationkey")
        jn2 = jn1.merge(orders, left_on="c_custkey", right_on="o_custkey")
        jn2 = jn2.rename(columns={"n_name": "cust_nation"})
        jn3 = jn2.merge(lineitem, left_on="o_orderkey", right_on="l_orderkey")
        jn4 = jn3.merge(supplier, left_on="l_suppkey", right_on="s_suppkey")
        jn5 = jn4.merge(n2, left_on="s_nationkey", right_on="n_nationkey")
        df1 = jn5.rename(columns={"n_name": "supp_nation"})

        # Part 2
        jn1 = customer.merge(n2, left_on="c_nationkey", right_on="n_nationkey")
        jn2 = jn1.merge(orders, left_on="c_custkey", right_on="o_custkey")
        jn2 = jn2.rename(columns={"n_name": "cust_nation"})
        jn3 = jn2.merge(lineitem, left_on="o_orderkey", right_on="l_orderkey")
        jn4 = jn3.merge(supplier, left_on="l_suppkey", right_on="s_suppkey")
        jn5 = jn4.merge(n1, left_on="s_nationkey", right_on="n_nationkey")
        df2 = jn5.rename(columns={"n_name": "supp_nation"})

        # Combine
        total = pd.concat([df1, df2])

        total = total[
            (total["l_shipdate"] >= var3) & (total["l_shipdate"] <= var4)
        ]
        total["volume"] = total["l_extendedprice"] * (
            1.0 - total["l_discount"]
        )
        total["l_year"] = total["l_shipdate"].dt.year

        gb = total.groupby(
            ["supp_nation", "cust_nation", "l_year"], as_index=False
        )
        agg = gb.agg(revenue=pd.NamedAgg(column="volume", aggfunc="sum"))

        return agg.sort_values(
            by=["supp_nation", "cust_nation", "l_year"], ignore_index=True
        )

    @staticmethod
    def q8(run_config: RunConfig) -> pd.DataFrame:
        """Query 8."""
        customer = get_data(
            run_config.dataset_path,
            "customer",
            run_config.suffix,
            columns=["c_custkey", "c_nationkey"],
        )
        lineitem = get_data(
            run_config.dataset_path,
            "lineitem",
            run_config.suffix,
            columns=[
                "l_partkey",
                "l_suppkey",
                "l_orderkey",
                "l_extendedprice",
                "l_discount",
            ],
        )
        nation = get_data(run_config.dataset_path, "nation", run_config.suffix)
        orders = get_data(
            run_config.dataset_path,
            "orders",
            run_config.suffix,
            columns=["o_orderkey", "o_custkey", "o_orderdate"],
        )
        part = get_data(
            run_config.dataset_path,
            "part",
            run_config.suffix,
            columns=["p_partkey", "p_type"],
        )
        region = get_data(run_config.dataset_path, "region", run_config.suffix)
        supplier = get_data(
            run_config.dataset_path,
            "supplier",
            run_config.suffix,
            columns=["s_suppkey", "s_nationkey"],
        )

        var1 = "BRAZIL"
        var2 = "AMERICA"
        var3 = "ECONOMY ANODIZED STEEL"
        var4 = datetime64("1995-01-01")
        var5 = datetime64("1996-12-31")

        n1 = nation.loc[:, ["n_nationkey", "n_regionkey"]]
        n2 = nation.loc[:, ["n_nationkey", "n_name"]]
        region = region[region["r_name"] == var2]
        n1 = n1.merge(region, left_on="n_regionkey", right_on="r_regionkey")[
            ["n_nationkey"]
        ]

        jn1 = part.merge(lineitem, left_on="p_partkey", right_on="l_partkey")
        jn2 = jn1.merge(supplier, left_on="l_suppkey", right_on="s_suppkey")
        jn3 = jn2.merge(orders, left_on="l_orderkey", right_on="o_orderkey")
        jn4 = jn3.merge(customer, left_on="o_custkey", right_on="c_custkey")
        jn6 = jn4.merge(n1, left_on="c_nationkey", right_on="n_nationkey")

        jn7 = jn6.merge(n2, left_on="s_nationkey", right_on="n_nationkey")

        jn7 = jn7[(jn7["o_orderdate"] >= var4) & (jn7["o_orderdate"] <= var5)]
        jn7 = jn7[jn7["p_type"] == var3]

        jn7["o_year"] = jn7["o_orderdate"].dt.year
        jn7["volume"] = jn7["l_extendedprice"] * (1.0 - jn7["l_discount"])
        jn7 = jn7.rename(columns={"n_name": "nation"})

        def udf(df: pd.DataFrame) -> float:
            demonimator: float = df["volume"].sum()
            df = df[df["nation"] == var1]
            numerator: float = df["volume"].sum()
            return round(numerator / demonimator, 2)

        gb = jn7.groupby("o_year", as_index=False)
        agg = gb.apply(udf, include_groups=False)
        agg.columns = ["o_year", "mkt_share"]
        return agg.sort_values("o_year", ignore_index=True)

    @staticmethod
    def q9(run_config: RunConfig) -> pd.DataFrame:
        """Query 9."""
        path = run_config.dataset_path
        suffix = run_config.suffix
        lineitem = get_data(
            path,
            "lineitem",
            suffix,
            columns=[
                "l_partkey",
                "l_suppkey",
                "l_orderkey",
                "l_extendedprice",
                "l_discount",
                "l_quantity",
            ],
        )
        nation = get_data(
            path,
            "nation",
            suffix,
            columns=["n_nationkey", "n_name"],
        )
        orders = get_data(
            path,
            "orders",
            suffix,
            columns=["o_orderkey", "o_orderdate"],
        )
        part = get_data(
            path,
            "part",
            suffix,
            columns=["p_partkey", "p_name"],
        )
        partsupp = get_data(
            path,
            "partsupp",
            suffix,
            columns=["ps_partkey", "ps_suppkey", "ps_supplycost"],
        )
        supplier = get_data(
            path,
            "supplier",
            suffix,
            columns=["s_suppkey", "s_nationkey"],
        )

        jn1 = part.merge(partsupp, left_on="p_partkey", right_on="ps_partkey")
        jn2 = jn1.merge(supplier, left_on="ps_suppkey", right_on="s_suppkey")
        jn3 = jn2.merge(
            lineitem,
            left_on=["p_partkey", "ps_suppkey"],
            right_on=["l_partkey", "l_suppkey"],
        )
        jn4 = jn3.merge(orders, left_on="l_orderkey", right_on="o_orderkey")
        jn5 = jn4.merge(nation, left_on="s_nationkey", right_on="n_nationkey")

        jn5 = jn5[jn5["p_name"].str.contains("green", regex=False)]

        jn5["o_year"] = jn5["o_orderdate"].dt.year
        jn5["amount"] = jn5["l_extendedprice"] * (1.0 - jn5["l_discount"]) - (
            jn5["ps_supplycost"] * jn5["l_quantity"]
        )
        jn5 = jn5.rename(columns={"n_name": "nation"})

        gb = jn5.groupby(["nation", "o_year"], as_index=False, sort=False)
        agg = gb.agg(sum_profit=pd.NamedAgg(column="amount", aggfunc="sum"))
        return agg.sort_values(
            by=["nation", "o_year"], ascending=[True, False], ignore_index=True
        )

    @staticmethod
    def q10(run_config: RunConfig) -> pd.DataFrame:
        """Query 10."""
        path = run_config.dataset_path
        suffix = run_config.suffix
        customer = get_data(
            path,
            "customer",
            suffix,
            columns=[
                "c_custkey",
                "c_name",
                "c_address",
                "c_nationkey",
                "c_phone",
                "c_acctbal",
                "c_comment",
            ],
        )
        lineitem = get_data(
            path,
            "lineitem",
            suffix,
            columns=[
                "l_orderkey",
                "l_returnflag",
                "l_extendedprice",
                "l_discount",
            ],
        )
        nation = get_data(
            path,
            "nation",
            suffix,
            columns=["n_nationkey", "n_name"],
        )
        orders = get_data(
            path,
            "orders",
            suffix,
            columns=["o_custkey", "o_orderkey", "o_orderdate"],
        )

        var1 = datetime64("1993-10-01")
        var2 = datetime64("1994-01-01")

        jn1 = customer.merge(orders, left_on="c_custkey", right_on="o_custkey")
        jn2 = jn1.merge(lineitem, left_on="o_orderkey", right_on="l_orderkey")
        jn3 = jn2.merge(nation, left_on="c_nationkey", right_on="n_nationkey")

        jn3 = jn3[(jn3["o_orderdate"] >= var1) & (jn3["o_orderdate"] < var2)]
        jn3 = jn3[jn3["l_returnflag"] == "R"]

        jn3["revenue"] = jn3["l_extendedprice"] * (1 - jn3["l_discount"])

        gb = jn3.groupby(
            [
                "c_custkey",
                "c_name",
                "c_acctbal",
                "c_phone",
                "n_name",
                "c_address",
                "c_comment",
            ],
            as_index=False,
        )
        agg = gb.agg(revenue=pd.NamedAgg(column="revenue", aggfunc="sum"))

        sel = agg.loc[
            :,
            [
                "c_custkey",
                "c_name",
                "revenue",
                "c_acctbal",
                "n_name",
                "c_address",
                "c_phone",
                "c_comment",
            ],
        ]

        return sel.sort_values(
            "revenue", ascending=False, ignore_index=True
        ).head(20)

    @staticmethod
    def q11(run_config: RunConfig) -> pd.DataFrame:
        """Query 11."""
        nation = get_data(
            run_config.dataset_path,
            "nation",
            run_config.suffix,
            columns=["n_nationkey", "n_name"],
        )
        partsupp = get_data(
            run_config.dataset_path,
            "partsupp",
            run_config.suffix,
            columns=[
                "ps_suppkey",
                "ps_supplycost",
                "ps_availqty",
                "ps_partkey",
            ],
        )
        supplier = get_data(
            run_config.dataset_path,
            "supplier",
            run_config.suffix,
            columns=["s_suppkey", "s_nationkey"],
        )

        var1 = "GERMANY"
        var2 = 0.0001 / run_config.scale_factor

        nation = nation[nation["n_name"] == var1]

        jn1 = partsupp.merge(
            supplier, left_on="ps_suppkey", right_on="s_suppkey"
        )
        jn2 = jn1.merge(nation, left_on="s_nationkey", right_on="n_nationkey")

        jn2["value"] = jn2["ps_supplycost"] * jn2["ps_availqty"]

        threshold = jn2["value"].sum() * var2

        gb = jn2.groupby("ps_partkey", as_index=False)
        agg = gb.agg(value=pd.NamedAgg(column="value", aggfunc="sum"))

        result = agg[agg["value"] > threshold]
        return result.sort_values("value", ascending=False, ignore_index=True)

    @staticmethod
    def q12(run_config: RunConfig) -> pd.DataFrame:
        """Query 12."""
        lineitem = get_data(
            run_config.dataset_path,
            "lineitem",
            run_config.suffix,
            columns=[
                "l_orderkey",
                "l_shipmode",
                "l_commitdate",
                "l_receiptdate",
                "l_shipdate",
            ],
        )
        orders = get_data(
            run_config.dataset_path,
            "orders",
            run_config.suffix,
            columns=["o_orderkey", "o_orderpriority"],
        )

        var1 = "MAIL"
        var2 = "SHIP"
        var3 = datetime64("1994-01-01")
        var4 = datetime64("1995-01-01")

        lineitem = lineitem[lineitem["l_shipmode"].isin([var1, var2])]
        lineitem = lineitem[
            lineitem["l_commitdate"] < lineitem["l_receiptdate"]
        ]
        lineitem = lineitem[lineitem["l_shipdate"] < lineitem["l_commitdate"]]
        lineitem = lineitem[
            (lineitem["l_receiptdate"] >= var3)
            & (lineitem["l_receiptdate"] < var4)
        ]

        jn = orders.merge(
            lineitem, left_on="o_orderkey", right_on="l_orderkey"
        )

        jn["high_line_count"] = jn["o_orderpriority"].isin(
            ["1-URGENT", "2-HIGH"]
        )
        jn["low_line_count"] = ~jn["o_orderpriority"].isin(
            ["1-URGENT", "2-HIGH"]
        )

        gb = jn.groupby("l_shipmode", as_index=False)
        agg = gb.agg(
            high_line_count=pd.NamedAgg(
                column="high_line_count", aggfunc="sum"
            ),
            low_line_count=pd.NamedAgg(column="low_line_count", aggfunc="sum"),
        )

        return agg.sort_values("l_shipmode", ignore_index=True)

    @staticmethod
    def q13(run_config: RunConfig) -> pd.DataFrame:
        """Query 13."""
        customer = get_data(
            run_config.dataset_path,
            "customer",
            run_config.suffix,
            columns=["c_custkey"],
        )
        orders = get_data(
            run_config.dataset_path,
            "orders",
            run_config.suffix,
            columns=["o_custkey", "o_orderkey", "o_comment"],
        )

        var1 = "special"
        var2 = "requests"

        filtered_orders = orders[
            ~orders["o_comment"].str.contains(
                f"{var1}.*{var2}", regex=True, na=False
            )
        ]

        jn = customer.merge(
            filtered_orders,
            left_on="c_custkey",
            right_on="o_custkey",
            how="left",
        )

        gb1 = jn.groupby("c_custkey", as_index=False)
        agg1 = gb1.agg(
            c_count=pd.NamedAgg(column="o_orderkey", aggfunc="count")
        )

        gb2 = agg1.groupby("c_count", as_index=False)
        agg2 = gb2.size()
        agg2.columns = ["c_count", "custdist"]

        return agg2.sort_values(
            by=["custdist", "c_count"],
            ascending=[False, False],
            ignore_index=True,
        )

    @staticmethod
    def q14(run_config: RunConfig) -> pd.DataFrame:
        """Query 14."""
        lineitem = get_data(
            run_config.dataset_path,
            "lineitem",
            run_config.suffix,
            columns=[
                "l_partkey",
                "l_shipdate",
                "l_extendedprice",
                "l_discount",
            ],
        )
        part = get_data(
            run_config.dataset_path,
            "part",
            run_config.suffix,
            columns=["p_partkey", "p_type"],
        )

        var1 = datetime64("1995-09-01")
        var2 = datetime64("1995-10-01")

        lineitem = lineitem[
            (lineitem["l_shipdate"] >= var1) & (lineitem["l_shipdate"] < var2)
        ]

        jn = lineitem.merge(part, left_on="l_partkey", right_on="p_partkey")

        jn["revenue"] = jn["l_extendedprice"] * (1 - jn["l_discount"])
        jn["promo_revenue"] = jn["revenue"].where(
            jn["p_type"].str.startswith("PROMO"), 0
        )

        promo_revenue = (
            100.0 * jn["promo_revenue"].sum() / jn["revenue"].sum()
        ).round(2)

        return pd.DataFrame({"promo_revenue": [promo_revenue]})

    @staticmethod
    def q15(run_config: RunConfig) -> pd.DataFrame:
        """Query 15."""
        lineitem = get_data(
            run_config.dataset_path,
            "lineitem",
            run_config.suffix,
            columns=[
                "l_suppkey",
                "l_shipdate",
                "l_extendedprice",
                "l_discount",
            ],
        )
        supplier = get_data(
            run_config.dataset_path,
            "supplier",
            run_config.suffix,
            columns=["s_suppkey", "s_name", "s_address", "s_phone"],
        )

        var1 = datetime64("1996-01-01")
        var2 = datetime64("1996-04-01")

        filtered_lineitem = lineitem[
            (lineitem["l_shipdate"] >= var1) & (lineitem["l_shipdate"] < var2)
        ]

        filtered_lineitem["revenue"] = filtered_lineitem["l_extendedprice"] * (
            1 - filtered_lineitem["l_discount"]
        )

        revenue = filtered_lineitem.groupby("l_suppkey", as_index=False).agg(
            total_revenue=pd.NamedAgg(column="revenue", aggfunc="sum")
        )
        revenue = revenue.rename(columns={"l_suppkey": "supplier_no"})

        max_revenue = revenue["total_revenue"].max()

        jn = supplier.merge(
            revenue, left_on="s_suppkey", right_on="supplier_no"
        )
        jn = jn[jn["total_revenue"] == max_revenue]

        result = jn.loc[
            :, ["s_suppkey", "s_name", "s_address", "s_phone", "total_revenue"]
        ]

        return result.sort_values("s_suppkey", ignore_index=True)

    @staticmethod
    def q16(run_config: RunConfig) -> pd.DataFrame:
        """Query 16."""
        part = get_data(
            run_config.dataset_path,
            "part",
            run_config.suffix,
            columns=["p_partkey", "p_brand", "p_type", "p_size"],
        )
        partsupp = get_data(
            run_config.dataset_path,
            "partsupp",
            run_config.suffix,
            columns=["ps_partkey", "ps_suppkey"],
        )
        supplier = get_data(
            run_config.dataset_path,
            "supplier",
            run_config.suffix,
            columns=["s_suppkey", "s_comment"],
        )

        var1 = "Brand#45"

        # Filter suppliers with complaints
        filtered_supplier = supplier[
            supplier["s_comment"].str.contains(
                ".*Customer.*Complaints.*", regex=True, na=False
            )
        ][["s_suppkey"]]

        part = part[part["p_brand"] != var1]
        part = part[~part["p_type"].str.startswith("MEDIUM POLISHED")]
        part = part[part["p_size"].isin([49, 14, 23, 45, 19, 3, 36, 9])]

        jn = part.merge(partsupp, left_on="p_partkey", right_on="ps_partkey")

        # Left join to exclude suppliers with complaints
        jn2 = jn.merge(
            filtered_supplier,
            left_on="ps_suppkey",
            right_on="s_suppkey",
            how="left",
        )
        jn2 = jn2[jn2["s_suppkey"].isna()]

        gb = jn2.groupby(["p_brand", "p_type", "p_size"], as_index=False)
        agg = gb.agg(
            supplier_cnt=pd.NamedAgg(column="ps_suppkey", aggfunc="nunique")
        )

        return agg.sort_values(
            by=["supplier_cnt", "p_brand", "p_type", "p_size"],
            ascending=[False, True, True, True],
            ignore_index=True,
        )

    @staticmethod
    def q17(run_config: RunConfig) -> pd.DataFrame:
        """Query 17."""
        lineitem = get_data(
            run_config.dataset_path,
            "lineitem",
            run_config.suffix,
            columns=["l_partkey", "l_quantity", "l_extendedprice"],
        )
        part = get_data(
            run_config.dataset_path,
            "part",
            run_config.suffix,
            columns=["p_partkey", "p_brand", "p_container"],
        )

        var1 = "Brand#23"
        var2 = "MED BOX"

        filtered_part = part[
            (part["p_brand"] == var1) & (part["p_container"] == var2)
        ]

        jn = filtered_part.merge(
            lineitem, left_on="p_partkey", right_on="l_partkey"
        )

        # Calculate average quantity per partkey
        avg_qty = jn.groupby("p_partkey", as_index=False).agg(
            avg_quantity=pd.NamedAgg(column="l_quantity", aggfunc="mean")
        )
        avg_qty["avg_quantity"] = 0.2 * avg_qty["avg_quantity"]

        jn2 = jn.merge(avg_qty, on="p_partkey")
        jn2 = jn2[jn2["l_quantity"] < jn2["avg_quantity"]]

        avg_yearly = (jn2["l_extendedprice"].sum() / 7.0).round(2)

        return pd.DataFrame({"avg_yearly": [avg_yearly]})

    @staticmethod
    def q18(run_config: RunConfig) -> pd.DataFrame:
        """Query 18."""
        path = run_config.dataset_path
        suffix = run_config.suffix
        customer = get_data(
            path,
            "customer",
            suffix,
            columns=["c_custkey", "c_name"],
        )
        lineitem = get_data(
            path,
            "lineitem",
            suffix,
            columns=["l_orderkey", "l_quantity"],
        )
        orders = get_data(
            path,
            "orders",
            suffix,
            columns=[
                "o_orderkey",
                "o_custkey",
                "o_orderdate",
                "o_totalprice",
            ],
        )

        var1 = 300

        # Find orders with sum quantity > 300
        qty_by_order = lineitem.groupby("l_orderkey", as_index=False).agg(
            sum_quantity=pd.NamedAgg(column="l_quantity", aggfunc="sum")
        )
        large_orders = qty_by_order[qty_by_order["sum_quantity"] > var1][
            ["l_orderkey"]
        ]

        # Semi join: keep only orders that are in large_orders
        jn1 = orders.merge(
            large_orders, left_on="o_orderkey", right_on="l_orderkey"
        )
        jn2 = jn1.merge(lineitem, left_on="o_orderkey", right_on="l_orderkey")
        jn3 = jn2.merge(customer, left_on="o_custkey", right_on="c_custkey")

        gb = jn3.groupby(
            [
                "c_name",
                "o_custkey",
                "o_orderkey",
                "o_orderdate",
                "o_totalprice",
            ],
            as_index=False,
        )
        agg = gb.agg(
            l_quantity=pd.NamedAgg(column="l_quantity", aggfunc="sum")
        )

        result = agg.loc[
            :,
            [
                "c_name",
                "o_custkey",
                "o_orderkey",
                "o_orderdate",
                "o_totalprice",
                "l_quantity",
            ],
        ]
        result = result.rename(
            columns={
                "o_custkey": "c_custkey",
                "l_quantity": "sum(l_quantity)",
            },
        )

        return result.sort_values(
            by=["o_totalprice", "o_orderdate"],
            ascending=[False, True],
            ignore_index=True,
        ).head(100)

    @staticmethod
    def q19(run_config: RunConfig) -> pd.DataFrame:
        """Query 19."""
        lineitem = get_data(
            run_config.dataset_path,
            "lineitem",
            run_config.suffix,
            columns=[
                "l_partkey",
                "l_shipmode",
                "l_shipinstruct",
                "l_quantity",
                "l_extendedprice",
                "l_discount",
            ],
        )
        part = get_data(
            run_config.dataset_path,
            "part",
            run_config.suffix,
            columns=["p_partkey", "p_brand", "p_container", "p_size"],
        )

        lineitem = lineitem[lineitem["l_shipmode"].isin(["AIR", "AIR REG"])]
        lineitem = lineitem[lineitem["l_shipinstruct"] == "DELIVER IN PERSON"]

        jn = part.merge(lineitem, left_on="p_partkey", right_on="l_partkey")

        # Complex filter conditions
        cond1 = (
            (jn["p_brand"] == "Brand#12")
            & jn["p_container"].isin(
                ["SM CASE", "SM BOX", "SM PACK", "SM PKG"]
            )
            & (jn["l_quantity"] >= 1)
            & (jn["l_quantity"] <= 11)
            & (jn["p_size"] >= 1)
            & (jn["p_size"] <= 5)
        )

        cond2 = (
            (jn["p_brand"] == "Brand#23")
            & jn["p_container"].isin(
                ["MED BAG", "MED BOX", "MED PKG", "MED PACK"]
            )
            & (jn["l_quantity"] >= 10)
            & (jn["l_quantity"] <= 20)
            & (jn["p_size"] >= 1)
            & (jn["p_size"] <= 10)
        )

        cond3 = (
            (jn["p_brand"] == "Brand#34")
            & jn["p_container"].isin(
                ["LG CASE", "LG BOX", "LG PACK", "LG PKG"]
            )
            & (jn["l_quantity"] >= 20)
            & (jn["l_quantity"] <= 30)
            & (jn["p_size"] >= 1)
            & (jn["p_size"] <= 15)
        )

        jn = jn[cond1 | cond2 | cond3]

        revenue = (
            (jn["l_extendedprice"] * (1 - jn["l_discount"])).sum().round(2)
        )

        return pd.DataFrame({"revenue": [revenue]})

    @staticmethod
    def q20(run_config: RunConfig) -> pd.DataFrame:
        """Query 20."""
        lineitem = get_data(
            run_config.dataset_path,
            "lineitem",
            run_config.suffix,
            columns=["l_shipdate", "l_partkey", "l_suppkey", "l_quantity"],
        )
        nation = get_data(
            run_config.dataset_path,
            "nation",
            run_config.suffix,
            columns=["n_nationkey", "n_name"],
        )
        part = get_data(
            run_config.dataset_path,
            "part",
            run_config.suffix,
            columns=["p_partkey", "p_name"],
        )
        partsupp = get_data(
            run_config.dataset_path,
            "partsupp",
            run_config.suffix,
            columns=["ps_partkey", "ps_suppkey", "ps_availqty"],
        )
        supplier = get_data(
            run_config.dataset_path,
            "supplier",
            run_config.suffix,
            columns=["s_suppkey", "s_nationkey", "s_name", "s_address"],
        )

        var1 = datetime64("1994-01-01")
        var2 = datetime64("1995-01-01")
        var3 = "CANADA"
        var4 = "forest"

        # Aggregate lineitem by partkey and suppkey
        filtered_lineitem = lineitem[
            (lineitem["l_shipdate"] >= var1) & (lineitem["l_shipdate"] < var2)
        ]
        qty_agg = filtered_lineitem.groupby(
            ["l_partkey", "l_suppkey"], as_index=False
        ).agg(sum_quantity=pd.NamedAgg(column="l_quantity", aggfunc="sum"))
        qty_agg["sum_quantity"] = qty_agg["sum_quantity"] * 0.5

        # Filter nation
        filtered_nation = nation[nation["n_name"] == var3]

        # Filter parts starting with "forest"
        filtered_part = part[part["p_name"].str.startswith(var4)][
            ["p_partkey"]
        ].drop_duplicates()

        # Join partsupp with filtered parts
        jn1 = filtered_part.merge(
            partsupp, left_on="p_partkey", right_on="ps_partkey"
        )

        # Join with quantity aggregation
        jn2 = jn1.merge(
            qty_agg,
            left_on=["ps_suppkey", "p_partkey"],
            right_on=["l_suppkey", "l_partkey"],
        )

        # Filter by availqty > sum_quantity
        jn2 = jn2[jn2["ps_availqty"] > jn2["sum_quantity"]]

        # Get unique suppliers
        unique_suppliers = jn2[["ps_suppkey"]].drop_duplicates()

        # Join with supplier and nation
        jn3 = unique_suppliers.merge(
            supplier, left_on="ps_suppkey", right_on="s_suppkey"
        )
        jn4 = jn3.merge(
            filtered_nation, left_on="s_nationkey", right_on="n_nationkey"
        )

        result = jn4.loc[:, ["s_name", "s_address"]]

        return result.sort_values("s_name", ignore_index=True)

    @staticmethod
    def q21(run_config: RunConfig) -> pd.DataFrame:
        """Query 21."""
        lineitem = get_data(
            run_config.dataset_path,
            "lineitem",
            run_config.suffix,
            columns=[
                "l_orderkey",
                "l_suppkey",
                "l_receiptdate",
                "l_commitdate",
            ],
        )
        nation = get_data(
            run_config.dataset_path,
            "nation",
            run_config.suffix,
            columns=["n_nationkey", "n_name"],
        )
        orders = get_data(
            run_config.dataset_path,
            "orders",
            run_config.suffix,
            columns=["o_orderkey", "o_orderstatus"],
        )
        supplier = get_data(
            run_config.dataset_path,
            "supplier",
            run_config.suffix,
            columns=["s_suppkey", "s_nationkey", "s_name"],
        )

        var1 = "SAUDI ARABIA"

        nation = nation[nation["n_name"] == var1]
        orders = orders[orders["o_orderstatus"] == "F"]

        # Find orders with multiple suppliers
        supp_per_order = lineitem.groupby("l_orderkey", as_index=False).agg(
            n_supp_by_order=pd.NamedAgg(column="l_suppkey", aggfunc="count")
        )
        multi_supp_orders = supp_per_order[
            supp_per_order["n_supp_by_order"] > 1
        ]

        # Join with lineitem where receiptdate > commitdate
        late_lineitem = lineitem[
            lineitem["l_receiptdate"] > lineitem["l_commitdate"]
        ]
        jn1 = multi_supp_orders.merge(
            late_lineitem, on="l_orderkey", how="inner"
        )

        # Re-calculate suppliers per order for the late items
        supp_per_order2 = jn1.groupby("l_orderkey", as_index=False).agg(
            n_supp_by_order=pd.NamedAgg(column="l_suppkey", aggfunc="count")
        )

        # Join back with lineitem data
        jn2 = supp_per_order2.merge(jn1, on="l_orderkey")

        # Filter to orders where only one supplier was late
        jn2 = jn2[jn2["n_supp_by_order_x"] == 1]

        # Join with supplier, nation, and orders
        jn3 = jn2.merge(supplier, left_on="l_suppkey", right_on="s_suppkey")
        jn4 = jn3.merge(nation, left_on="s_nationkey", right_on="n_nationkey")
        jn5 = jn4.merge(orders, left_on="l_orderkey", right_on="o_orderkey")

        # Group by supplier name and count
        gb = jn5.groupby("s_name", as_index=False)
        agg = gb.size()
        agg.columns = ["s_name", "numwait"]

        return agg.sort_values(
            by=["numwait", "s_name"],
            ascending=[False, True],
            ignore_index=True,
        ).head(100)

    @staticmethod
    def q22(run_config: RunConfig) -> pd.DataFrame:
        """Query 22."""
        customer = get_data(
            run_config.dataset_path,
            "customer",
            run_config.suffix,
            columns=["c_custkey", "c_phone", "c_acctbal"],
        )
        orders = get_data(
            run_config.dataset_path,
            "orders",
            run_config.suffix,
            columns=["o_custkey"],
        )

        # Extract country code (first 2 chars of phone)
        customer_with_cntry = customer.copy()
        customer_with_cntry["cntrycode"] = customer_with_cntry[
            "c_phone"
        ].str.slice(0, 2)

        # Filter by country codes
        filtered_customer = customer_with_cntry[
            customer_with_cntry["cntrycode"].str.match(
                "13|31|23|29|30|18|17", na=False
            )
        ][["c_acctbal", "c_custkey", "cntrycode"]]

        # Calculate average account balance for positive balances
        avg_acctbal = filtered_customer[filtered_customer["c_acctbal"] > 0.0][
            "c_acctbal"
        ].mean()

        # Get unique customer keys from orders
        customers_with_orders = orders[["o_custkey"]].drop_duplicates()

        # Left join to find customers without orders
        jn = filtered_customer.merge(
            customers_with_orders,
            left_on="c_custkey",
            right_on="o_custkey",
            how="left",
        )
        jn = jn[jn["o_custkey"].isna()]

        # Filter by account balance > average
        jn = jn[jn["c_acctbal"] > avg_acctbal]

        # Group by country code
        gb = jn.groupby("cntrycode", as_index=False)
        agg = gb.agg(
            numcust=pd.NamedAgg(column="c_acctbal", aggfunc="count"),
            totacctbal=pd.NamedAgg(column="c_acctbal", aggfunc="sum"),
        )

        return agg.sort_values("cntrycode", ignore_index=True)


if __name__ == "__main__":
    run_pandas(PDSHQueries)
