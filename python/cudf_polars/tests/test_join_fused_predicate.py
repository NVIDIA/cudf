# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for equi joins that Polars plans with a fused predicate."""

from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.dsl import ir
from cudf_polars.dsl.translate import Translator
from cudf_polars.dsl.traversal import traversal
from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
)
from cudf_polars.utils.versions import POLARS_VERSION_LT_21

pytestmark = pytest.mark.skipif(
    POLARS_VERSION_LT_21,
    reason="Polars 2.0 does not expose fused join predicates to the visitor",
)


@pytest.fixture
def orders() -> pl.LazyFrame:
    return pl.LazyFrame(
        {
            "key": [1, 2, 3, 3, None],
            "supp_nation": ["FRANCE", "GERMANY", "FRANCE", "GERMANY", "FRANCE"],
            "v": [1, 2, 3, 4, 5],
        }
    )


@pytest.fixture
def customers() -> pl.LazyFrame:
    return pl.LazyFrame(
        {
            "key": [1, 2, 3, None, 3],
            "cust_nation": ["GERMANY", "FRANCE", "FRANCE", "GERMANY", "GERMANY"],
            "w": [4, 3, 2, 1, 9],
        }
    )


def test_sql_equi_join_with_not_equal(engine: pl.GPUEngine, orders, customers):
    q = pl.SQLContext(orders=orders, customers=customers).execute(
        "SELECT * FROM orders, customers "
        "WHERE orders.key = customers.key AND supp_nation <> cust_nation"
    )
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


@pytest.mark.parametrize("nulls_equal", [False, True])
def test_inner_join_then_filter(engine: pl.GPUEngine, orders, customers, nulls_equal):
    q = orders.join(customers, on="key", how="inner", nulls_equal=nulls_equal).filter(
        pl.col("supp_nation") != pl.col("cust_nation")
    )
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


def test_join_where_equality_and_inequality(engine: pl.GPUEngine, orders, customers):
    q = orders.join_where(
        customers,
        pl.col("key") == pl.col("key_right"),
        pl.col("v") > pl.col("w"),
    )
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


def test_predicate_on_suffixed_columns(engine: pl.GPUEngine):
    left = pl.LazyFrame({"k": [1, 2, 3], "x": [1, 2, 3]})
    right = pl.LazyFrame({"k": [1, 2, 3], "x": [3, 2, 1]})
    q = left.join(right, on="k", how="inner", suffix="_r").filter(
        pl.col("x") != pl.col("x_r")
    )
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


@pytest.mark.parametrize("negate", [False, True], ids=["exists", "not-exists"])
def test_correlated_exists(engine: pl.GPUEngine, orders, customers, negate):
    # The shape used by TPC-H queries 21 and 22 and similar.
    not_ = "NOT " if negate else ""
    q = pl.SQLContext(orders=orders, customers=customers).execute(
        f"SELECT * FROM orders WHERE {not_}EXISTS "
        "(SELECT 1 FROM customers WHERE customers.key = orders.key "
        "AND customers.cust_nation <> orders.supp_nation)"
    )
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


def test_empty_result(engine: pl.GPUEngine, orders, customers):
    q = orders.join(customers, on="key", how="inner").filter(
        pl.col("supp_nation") != pl.col("supp_nation")
    )
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


def test_predicate_is_evaluated_inside_the_join():
    left = pl.LazyFrame({"k": [1, 2, 3], "a": [1, 2, 3]})
    right = pl.LazyFrame({"k": [1, 2, 3], "b": [3, 2, 1]})
    q = pl.SQLContext(left=left, right=right).execute(
        "SELECT * FROM left, right WHERE left.k = right.k AND left.a < right.b"
    )
    plan = Translator(q._ldf.visit(), pl.GPUEngine()).translate_ir()
    nodes = list(traversal([plan]))
    assert not any(isinstance(node, ir.Filter) for node in nodes)
    (join,) = (node for node in nodes if isinstance(node, ir.Join))
    assert join.predicate is not None


@pytest.mark.skip_on_streaming_engine(
    "too slow with many small partitions", engine="spmd"
)
def test_rejected_matches_are_not_materialized(engine: pl.GPUEngine):
    # Every row shares one key, so 900 million pairs match on the key, and the
    # predicate rejects all of them. Materializing the pairs would overflow the
    # column size limit or exhaust device memory.
    n = 30_000
    left = pl.LazyFrame({"k": [1] * n, "a": range(n)})
    right = pl.LazyFrame({"k": [1] * n, "b": range(n)})
    q = pl.SQLContext(left=left, right=right).execute(
        "SELECT * FROM left, right WHERE left.k = right.k AND left.a + right.b < 0"
    )
    result = q.collect(engine=engine)
    assert result.shape == (0, 4)


def test_predicate_on_a_decimal_and_a_float(engine: pl.GPUEngine):
    left = pl.LazyFrame(
        {"k": [1, 2, 3], "d": pl.Series([1, 2, 3], dtype=pl.Decimal(10, 2))}
    )
    right = pl.LazyFrame({"k": [1, 2, 3], "f": [0.5, 2.5, 3.0]})
    q = left.join(right, on="k", how="inner").filter(pl.col("d") < pl.col("f"))
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


def test_decimal_compared_with_computed_float(engine: pl.GPUEngine):
    # The shape of TPC-DS query 6: libcudf's AST cannot compare a decimal with
    # an expression, so the expression is computed before the join.
    left = pl.LazyFrame(
        {"k": [1, 2, 3, 4], "d": pl.Series([1, 2, 3, 4], dtype=pl.Decimal(10, 2))}
    )
    right = pl.LazyFrame({"k": [1, 2, 3, 3], "f": [0.5, 2.5, 3.0, 1.0]})
    q = left.join(right, on="k", how="inner").filter(pl.col("d") > 1.2 * pl.col("f"))
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


def test_expressions_of_each_input(engine: pl.GPUEngine, orders, customers):
    q = orders.join(customers, on="key", how="inner").filter(
        (pl.col("v") * 2 + 1) < (pl.col("w") * 3)
    )
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


def test_repeated_expression_of_one_input(engine: pl.GPUEngine, orders, customers):
    q = orders.join(customers, on="key", how="inner").filter(
        ((pl.col("v") + 1) < pl.col("w")) | ((pl.col("v") + 1) == pl.col("w") * 2)
    )
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


def test_string_expression_of_one_input(engine: pl.GPUEngine, orders, customers):
    # libcudf's AST has no string functions, so this only works because the
    # expression is computed on its own input.
    q = orders.join(customers, on="key", how="inner").filter(
        pl.col("supp_nation").str.to_lowercase()
        != pl.col("cust_nation").str.to_lowercase()
    )
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


def test_mismatched_integer_widths(engine: pl.GPUEngine):
    left = pl.LazyFrame({"k": [1, 2, 3], "a": pl.Series([1, 2, 3], dtype=pl.Int32)})
    right = pl.LazyFrame({"k": [1, 2, 3], "b": pl.Series([3, 2, 1], dtype=pl.Int64)})
    q = left.join(right, on="k", how="inner").filter(pl.col("a") < pl.col("b"))
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)
