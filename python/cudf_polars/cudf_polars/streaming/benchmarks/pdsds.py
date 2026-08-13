# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Experimental PDS-DS benchmarks.

Based on https://github.com/pola-rs/polars-benchmark.

WARNING: This is an experimental (and unofficial)
benchmark script. It is not intended for public use
and may be modified or removed at any time.
"""

from __future__ import annotations

import importlib
import os
from typing import TYPE_CHECKING, ClassVar

import polars as pl

try:
    from cudf_polars.streaming.benchmarks.utils import (
        COUNT_DTYPE,
        _CPU_ENGINES,
        build_parser,
        parse_args,
        run_polars,
    )
except ImportError as e:
    if e.name is not None and not e.name.startswith("cudf_polars"):
        raise

if TYPE_CHECKING:
    from types import ModuleType

# Without this setting, the first IO task to run
# on each worker takes ~15 sec extra
os.environ["KVIKIO_COMPAT_MODE"] = os.environ.get("KVIKIO_COMPAT_MODE", "on")
os.environ["KVIKIO_NTHREADS"] = os.environ.get("KVIKIO_NTHREADS", "8")
# TODO: consider raising the rapidsmpf built-in default from 1 to 8.
os.environ["RAPIDSMPF_NUM_STREAMING_THREADS"] = os.environ.get(
    "RAPIDSMPF_NUM_STREAMING_THREADS", "8"
)


def valid_query(name: str) -> bool:
    """Return True for valid query names eg. 'q9', 'q65', etc."""
    if not name.startswith("q"):
        return False
    try:
        q_num = int(name[1:])
    except ValueError:
        return False
    else:
        return 1 <= q_num <= 99


class PDSDSQueriesMeta(type):
    """Metaclass used for query lookup."""

    def __getattr__(cls, name: str):  # type: ignore[no-untyped-def]
        """Query lookup."""
        if valid_query(name):
            q_num = int(name[1:])
            module: ModuleType = importlib.import_module(
                f"cudf_polars.streaming.benchmarks.pdsds_queries.q{q_num}"
            )
            return getattr(module, cls.q_impl)
        raise AttributeError(f"{name} is not a valid query name")


class PDSDSQueries(metaclass=PDSDSQueriesMeta):
    """Base class for query loading."""

    q_impl: str
    name: str = "pdsds"


class PDSDSPolarsQueries(PDSDSQueries):
    """Polars Queries."""

    q_impl = "polars_impl"
    # See comments for EXPECTED_CASTS and EXPECTED_CASTS_DECIMAL
    # in cudf/python/cudf_polars/cudf_polars/streaming/benchmarks/pdsh.py
    # for more details.
    EXPECTED_CASTS_DECIMAL: ClassVar[dict] = {
        2: [
            pl.col("round((sun_sales1 / sun_sales2), 2)").cast(pl.Decimal(38, 2)),
            pl.col("round((mon_sales1 / mon_sales2), 2)").cast(pl.Decimal(38, 2)),
            pl.col("round((tue_sales1 / tue_sales2), 2)").cast(pl.Decimal(38, 2)),
            pl.col("round((wed_sales1 / wed_sales2), 2)").cast(pl.Decimal(38, 2)),
            pl.col("round((thu_sales1 / thu_sales2), 2)").cast(pl.Decimal(38, 2)),
            pl.col("round((fri_sales1 / fri_sales2), 2)").cast(pl.Decimal(38, 2)),
            pl.col("round((sat_sales1 / sat_sales2), 2)").cast(pl.Decimal(38, 2)),
        ],
        3: [pl.col("sum_agg").cast(pl.Decimal(18, 2))],
        5: [
            pl.col("sales").cast(pl.Decimal(18, 2)),
            pl.col("returns1").cast(pl.Decimal(18, 2)),
        ],
        8: [pl.col("sum(ss_net_profit)").cast(pl.Decimal(18, 2))],
        12: [
            pl.col("itemrevenue").cast(pl.Decimal(18, 2)),
            pl.col("revenueratio").cast(pl.Decimal(38, 2)),
        ],
        13: [pl.col("sum(ss_ext_wholesale_cost)").cast(pl.Decimal(18, 2))],
        15: [pl.col("sum(cs_sales_price)").cast(pl.Decimal(18, 2))],
        16: [
            pl.col("total shipping cost").cast(pl.Decimal(18, 2)),
            pl.col("total net profit").cast(pl.Decimal(18, 2)),
        ],
        19: [pl.col("ext_price").cast(pl.Decimal(18, 2))],
        20: [
            pl.col("itemrevenue").cast(pl.Decimal(18, 2)),
            pl.col("revenueratio").cast(pl.Decimal(38, 2)),
        ],
        24: [pl.col("paid").cast(pl.Decimal(18, 2))],
        30: [pl.col("ctr_total_return").cast(pl.Decimal(18, 2))],
        31: [
            pl.col("web_q1_q2_increase").cast(pl.Decimal(38, 2)),
            pl.col("store_q1_q2_increase").cast(pl.Decimal(38, 2)),
            pl.col("web_q2_q3_increase").cast(pl.Decimal(38, 2)),
            pl.col("store_q2_q3_increase").cast(pl.Decimal(38, 2)),
        ],
        32: [pl.col("excess discount amount").cast(pl.Decimal(18, 2))],
        33: [pl.col("total_sales").cast(pl.Decimal(18, 2))],
        42: [pl.col("sum(ss_ext_sales_price)").cast(pl.Decimal(18, 2))],
        45: [pl.col("sum(ws_sales_price)").cast(pl.Decimal(18, 2))],
        46: [
            pl.col("amt").cast(pl.Decimal(18, 2)),
            pl.col("profit").cast(pl.Decimal(18, 2)),
        ],
        47: [
            pl.col("sum_sales").cast(pl.Decimal(18, 2)),
            pl.col("psum").cast(pl.Decimal(18, 2)),
            pl.col("nsum").cast(pl.Decimal(18, 2)),
        ],
        51: [
            pl.col("web_sales").cast(pl.Decimal(18, 2)),
            pl.col("store_sales").cast(pl.Decimal(18, 2)),
            pl.col("web_cumulative").cast(pl.Decimal(18, 2)),
            pl.col("store_cumulative").cast(pl.Decimal(18, 2)),
        ],
        52: [pl.col("ext_price").cast(pl.Decimal(18, 2))],
        53: [pl.col("sum_sales").cast(pl.Decimal(18, 2))],
        55: [pl.col("ext_price").cast(pl.Decimal(18, 2))],
        56: [pl.col("total_sales").cast(pl.Decimal(18, 2))],
        57: [
            pl.col("sum_sales").cast(pl.Decimal(18, 2)),
            pl.col("psum").cast(pl.Decimal(18, 2)),
            pl.col("nsum").cast(pl.Decimal(18, 2)),
        ],
        58: [
            pl.col("ss_item_rev").cast(pl.Decimal(18, 2)),
            pl.col("cs_item_rev").cast(pl.Decimal(18, 2)),
            pl.col("ws_item_rev").cast(pl.Decimal(18, 2)),
            pl.col("ss_dev").cast(pl.Decimal(38, 2)),
            pl.col("cs_dev").cast(pl.Decimal(38, 2)),
            pl.col("ws_dev").cast(pl.Decimal(38, 2)),
            pl.col("average").cast(pl.Decimal(38, 2)),
        ],
        59: [
            pl.col("(sun_sales1 / sun_sales2)").cast(pl.Decimal(38, 2)),
            pl.col("(mon_sales1 / mon_sales2)").cast(pl.Decimal(38, 2)),
            pl.col("(tue_sales1 / tue_sales2)").cast(pl.Decimal(38, 2)),
            pl.col("(wed_sales1 / wed_sales2)").cast(pl.Decimal(38, 2)),
            pl.col("(thu_sales1 / thu_sales2)").cast(pl.Decimal(38, 2)),
            pl.col("(fri_sales1 / fri_sales2)").cast(pl.Decimal(38, 2)),
            pl.col("(sat_sales1 / sat_sales2)").cast(pl.Decimal(38, 2)),
        ],
        60: [pl.col("total_sales").cast(pl.Decimal(18, 2))],
        61: [
            pl.col("promotions").cast(pl.Decimal(18, 2)),
            pl.col("total").cast(pl.Decimal(18, 2)),
        ],
        63: [pl.col("sum_sales").cast(pl.Decimal(18, 2))],
        64: [
            pl.col("s1").cast(pl.Decimal(18, 2)),
            pl.col("s2").cast(pl.Decimal(18, 2)),
            pl.col("s3").cast(pl.Decimal(18, 2)),
            pl.col("s1_1").cast(pl.Decimal(18, 2)),
            pl.col("s2_1").cast(pl.Decimal(18, 2)),
            pl.col("s3_1").cast(pl.Decimal(18, 2)),
        ],
        65: [pl.col("revenue").cast(pl.Decimal(18, 2))],
        68: [
            pl.col("extended_price").cast(pl.Decimal(18, 2)),
            pl.col("extended_tax").cast(pl.Decimal(18, 2)),
            pl.col("list_price").cast(pl.Decimal(18, 2)),
        ],
        70: [pl.col("total_sum").cast(pl.Decimal(18, 2))],
        71: [pl.col("ext_price").cast(pl.Decimal(18, 2))],
        75: [pl.col("sales_amt_diff").cast(pl.Float64)],
        76: [pl.col("sales_amt").cast(pl.Decimal(18, 2))],
        77: [pl.col("sales").cast(pl.Decimal(18, 2))],
        78: [
            pl.col("store_wholesale_cost").cast(pl.Decimal(18, 2)),
            pl.col("store_sales_price").cast(pl.Decimal(18, 2)),
        ],
        79: [
            pl.col("amt").cast(pl.Decimal(18, 2)),
            pl.col("profit").cast(pl.Decimal(18, 2)),
        ],
        80: [pl.col("sales").cast(pl.Decimal(18, 2))],
        81: [pl.col("ctr_total_return").cast(pl.Decimal(18, 2))],
        86: [pl.col("total_sum").cast(pl.Decimal(18, 2))],
        89: [pl.col("sum_sales").cast(pl.Decimal(18, 2))],
        91: [pl.col("Returns_Loss").cast(pl.Decimal(18, 2))],
        92: [pl.col("Excess Discount Amount").cast(pl.Decimal(18, 2))],
        94: [
            pl.col("total shipping cost").cast(pl.Decimal(18, 2)),
            pl.col("total net profit").cast(pl.Decimal(18, 2)),
        ],
        95: [
            pl.col("total shipping cost").cast(pl.Decimal(18, 2)),
            pl.col("total net profit").cast(pl.Decimal(18, 2)),
        ],
        98: [pl.col("itemrevenue").cast(pl.Decimal(18, 2))],
    }
    EXPECTED_CASTS: ClassVar[dict] = {
        6: [pl.col("cnt").cast(COUNT_DTYPE)],
        10: [
            pl.col("cnt1").cast(COUNT_DTYPE),
            pl.col("cnt2").cast(COUNT_DTYPE),
            pl.col("cnt3").cast(COUNT_DTYPE),
            pl.col("cnt4").cast(COUNT_DTYPE),
            pl.col("cnt5").cast(COUNT_DTYPE),
            pl.col("cnt6").cast(COUNT_DTYPE),
        ],
        14: [pl.col("sum_number_sales").cast(COUNT_DTYPE)],
        16: [pl.col("order count").cast(COUNT_DTYPE)],
        17: [
            pl.col("store_sales_quantitycount").cast(COUNT_DTYPE),
            pl.col("store_returns_quantitycount").cast(COUNT_DTYPE),
            pl.col("catalog_sales_quantitycount").cast(COUNT_DTYPE),
        ],
        21: [
            pl.col("inv_before").cast(pl.Int32),
            pl.col("inv_after").cast(pl.Int32),
        ],
        34: [pl.col("cnt").cast(COUNT_DTYPE)],
        35: [
            pl.col("cnt1").cast(COUNT_DTYPE),
            pl.col("cnt2").cast(COUNT_DTYPE),
            pl.col("cnt3").cast(COUNT_DTYPE),
        ],
        44: [pl.col("rnk").cast(COUNT_DTYPE)],
        48: [pl.col("sum(ss_quantity)").cast(pl.Int64)],
        49: [
            pl.col("return_rank").cast(COUNT_DTYPE),
            pl.col("currency_rank").cast(COUNT_DTYPE),
        ],
        50: [
            pl.col("30 days").cast(COUNT_DTYPE),
            pl.col("31-60 days").cast(COUNT_DTYPE),
            pl.col("61-90 days").cast(COUNT_DTYPE),
            pl.col("91-120 days").cast(COUNT_DTYPE),
            pl.col(">120 days").cast(COUNT_DTYPE),
        ],
        54: [
            pl.col("segment").cast(pl.Float64),
            pl.col("segment_base").cast(pl.Float64),
            pl.col("num_customers").cast(COUNT_DTYPE),
        ],
        62: [
            pl.col("30 days").cast(pl.Int32),
            pl.col("31-60 days").cast(pl.Int32),
            pl.col("61-90 days").cast(pl.Int32),
            pl.col("91-120 days").cast(pl.Int32),
            pl.col(">120 days").cast(pl.Int32),
        ],
        64: [
            pl.col("cnt").cast(COUNT_DTYPE),
            pl.col("cnt_1").cast(COUNT_DTYPE),
        ],
        67: [pl.col("rk").cast(pl.UInt32())],
        69: [
            pl.col("cnt1").cast(COUNT_DTYPE),
            pl.col("cnt2").cast(COUNT_DTYPE),
            pl.col("cnt3").cast(COUNT_DTYPE),
        ],
        70: [pl.col("rank_within_parent").cast(pl.UInt32())],
        72: [
            pl.col("total_cnt").cast(COUNT_DTYPE),
            pl.col("no_promo").cast(COUNT_DTYPE),
            pl.col("promo").cast(COUNT_DTYPE),
        ],
        73: [pl.col("cnt").cast(COUNT_DTYPE)],
        75: [
            pl.col("prev_yr_cnt").cast(pl.Int64),
            pl.col("curr_yr_cnt").cast(pl.Int64),
            pl.col("sales_cnt_diff").cast(pl.Int64),
        ],
        78: [
            pl.col("store_qty").cast(pl.Int64),
            pl.col("other_chan_qty").cast(pl.Int64),
        ],
        83: [
            pl.col("sr_item_qty").cast(pl.Int64),
            pl.col("cr_item_qty").cast(pl.Int64),
            pl.col("wr_item_qty").cast(pl.Int64),
        ],
        94: [pl.col("order count").cast(COUNT_DTYPE)],
        95: [pl.col("order count").cast(COUNT_DTYPE)],
        96: [pl.col("count_star()").cast(COUNT_DTYPE)],
        97: [
            pl.col("store_only").cast(pl.Int32),
            pl.col("catalog_only").cast(pl.Int32),
            pl.col("store_and_catalog").cast(pl.Int32),
        ],
        99: [
            pl.col("30 days").cast(pl.Int32),
            pl.col("31-60 days").cast(pl.Int32),
            pl.col("61-90 days").cast(pl.Int32),
            pl.col("91-120 days").cast(pl.Int32),
            pl.col(">120 days").cast(pl.Int32),
        ],
    }

    @property
    def duckdb_queries(self) -> type[PDSDSDuckDBQueries]:
        """Link to the DuckDB queries for this benchmark."""
        return PDSDSDuckDBQueries


class PDSDSDuckDBQueries(PDSDSQueries):
    """DuckDB Queries."""

    q_impl = "duckdb_impl"


if __name__ == "__main__":
    parser = build_parser(num_queries=99)
    args = parse_args(parser=parser)
    if args.frontend not in _CPU_ENGINES:
        os.environ["POLARS_MAX_THREADS"] = os.environ.get("POLARS_MAX_THREADS", "1")
        os.environ["OMP_NUM_THREADS"] = os.environ.get("OMP_NUM_THREADS", "1")
    run_polars(PDSDSPolarsQueries, args)
