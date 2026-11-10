# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for equi joins that Polars plans with a fused predicate."""

from __future__ import annotations

from decimal import Decimal

import pytest

import polars as pl

from cudf_polars.containers import DataType
from cudf_polars.dsl import expr, ir
from cudf_polars.dsl.to_ast import _converts_exactly_to_float64
from cudf_polars.dsl.translate import Translator
from cudf_polars.dsl.traversal import traversal
from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
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


@pytest.mark.parametrize("otherwise", [None, 0.5], ids=["null", "literal"])
@pytest.mark.parametrize("conditional_first", [True, False], ids=["lhs", "rhs"])
def test_conditional_expression_of_both_inputs(
    engine: pl.GPUEngine, otherwise, conditional_first
):
    # The shape of TPC-DS queries 4 and 31: libcudf's AST has no conditional,
    # so the comparison is rewritten with AND and OR.
    left = pl.LazyFrame(
        {
            "k": [1, 2, 3, 4, 5, 6],
            "a": [1.0, -1.0, 4.0, None, 9.0, 0.5],
            "c": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    right = pl.LazyFrame(
        {
            "k": [1, 2, 3, 4, 5, 6],
            "y": [2.0, 2.0, 0.0, 2.0, 3.0, 1.0],
            "d": [1.0, 4.0, 1.0, 1.0, 1.0, 0.1],
        }
    )
    conditional = (
        pl.when(pl.col("a") > 0).then(pl.col("a") / pl.col("y")).otherwise(otherwise)
    )
    other = pl.col("c") / pl.col("d")
    predicate = conditional > other if conditional_first else other < conditional
    q = left.join(right, on="k", how="inner").filter(predicate)
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


def _decimals(
    values: list[str | None], scale: int = 2, precision: int = 38
) -> pl.Series:
    # Cast through a wider scale so the strings parse as exact values.
    return pl.Series(values).cast(pl.Decimal(38, 3)).cast(pl.Decimal(precision, scale))


@pytest.fixture
def decimal_ratios() -> tuple[pl.LazyFrame, pl.LazyFrame]:
    # In the first row, w / v and a / b are 1.004 and 1.003. They are the same
    # decimal once rounded to two places but different Float64 values.
    left = pl.LazyFrame(
        {
            "k": [1, 2, 3, 4, 5, 6, 7],
            "a": _decimals(["1003", "1020", "1000", "1010", None, "1000", "1004"]),
            "b": _decimals(["1000"] * 7),
            "v": _decimals(["1000", "1000", "1000", "1000", "1000", None, "1000"]),
            "d": _decimals(
                ["1.00", "1.02", "1.00", "1.01", "1.00", "1.00", "1.00"], 2, 15
            ),
        }
    )
    right = pl.LazyFrame(
        {
            "k": [1, 2, 3, 4, 5, 6, 7],
            "w": _decimals(["1004", "1010", "1000", "1000", "1000", "1010", "1004"]),
            "y": _decimals(["1000"] * 6 + [None]),
        }
    )
    return left, right


@pytest.mark.parametrize(
    "compare",
    [
        pl.Expr.lt,
        pl.Expr.le,
        pl.Expr.gt,
        pl.Expr.ge,
        pl.Expr.eq,
        pl.Expr.ne,
        pl.Expr.eq_missing,
        pl.Expr.ne_missing,
    ],
)
def test_ratios_of_decimals(engine: pl.GPUEngine, decimal_ratios, compare):
    # Polars divides decimals as floats and casts the result back to a decimal,
    # so each side of the comparison is a float cast to Decimal(38, 2).
    left, right = decimal_ratios
    q = left.join(right, on="k", how="inner").filter(
        compare(pl.col("w") / pl.col("v"), pl.col("a") / pl.col("b"))
    )
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


@pytest.mark.parametrize("ratio_first", [True, False])
def test_ratio_of_decimals_against_a_decimal_column(
    engine: pl.GPUEngine,
    decimal_ratios,
    ratio_first,
):
    left, right = decimal_ratios
    ratio = pl.col("w") / pl.col("v")
    predicate = ratio > pl.col("d") if ratio_first else pl.col("d") < ratio
    q = left.join(right, on="k", how="inner").filter(predicate)
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


def test_ratio_of_decimals_against_a_decimal_literal(
    engine: pl.GPUEngine, decimal_ratios
):
    left, right = decimal_ratios
    q = left.join(right, on="k", how="inner").filter(
        (pl.col("w") / pl.col("v")) > pl.lit(Decimal("1.00"))
    )
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


def test_conditional_ratio_of_decimals_against_a_ratio_of_both_inputs(
    engine: pl.GPUEngine, decimal_ratios
):
    # A null branch is compared with a ratio that uses both inputs.
    left, right = decimal_ratios
    conditional = (
        pl.when(pl.col("b") > 0).then(pl.col("a") / pl.col("b")).otherwise(None)
    )
    q = left.join(right, on="k", how="inner").filter(
        conditional > pl.col("w") / pl.col("v")
    )
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


def test_conditional_ratios_of_decimals(engine: pl.GPUEngine, decimal_ratios):
    # The shape of TPC-DS queries 4 and 11, where a ratio is only defined for a
    # positive denominator.
    left, right = decimal_ratios
    left = left.with_columns(x=pl.col("a"), y=pl.col("b"))
    right = right.with_columns(x=pl.col("w"))
    q = left.join(right, on="k", how="inner", suffix="_r").filter(
        pl.when(pl.col("y") > 0).then(pl.col("x") / pl.col("y")).otherwise(None)
        > pl.when(pl.col("y_r") > 0).then(pl.col("x_r") / pl.col("y_r")).otherwise(None)
    )
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


def test_ratio_of_decimals_against_a_different_scale(
    engine: pl.GPUEngine, decimal_ratios
):
    # The comparison is exact in the decimal with the larger scale, so it is
    # not rewritten, and the AST cannot compare the cast with the column.
    left, right = decimal_ratios
    left = left.with_columns(
        d=_decimals(["1.0", "1.0", "1.0", "1.0", "1.0", "1.0", "1.0"], 1, 10)
    )
    q = left.join(right, on="k", how="inner").filter(
        (pl.col("w") / pl.col("v")) > pl.col("d")
    )
    assert_ir_translation_raises(q, engine, NotImplementedError)


def test_ratio_of_decimals_against_a_wide_decimal_column(
    engine: pl.GPUEngine, decimal_ratios
):
    # A Decimal(38, 2) can hold more than Float64 does exactly, so the column is
    # not converted and the AST cannot compare the cast with it.
    left, right = decimal_ratios
    left = left.with_columns(d=pl.col("d").cast(pl.Decimal(38, 2)))
    q = left.join(right, on="k", how="inner").filter(
        (pl.col("w") / pl.col("v")) > pl.col("d")
    )
    assert_ir_translation_raises(q, engine, NotImplementedError)


def test_ratio_of_decimals_against_a_literal_that_float64_rounds(
    engine: pl.GPUEngine, decimal_ratios
):
    # 9007199254740993 is 2**53 + 1, which Float64 cannot hold.
    left, right = decimal_ratios
    q = left.join(right, on="k", how="inner").filter(
        (pl.col("w") / pl.col("v")) == pl.lit(Decimal("90071992547409.93"))
    )
    assert_ir_translation_raises(q, engine, NotImplementedError)


@pytest.mark.parametrize("compare", [pl.Expr.eq, pl.Expr.ne, pl.Expr.lt, pl.Expr.gt])
def test_ratios_of_decimals_that_are_ties(engine: pl.GPUEngine, compare):
    # a / b is exactly 0.125, 0.375, -0.125 and -0.375, which are halfway
    # between two decimals with two places. They round to the even neighbour:
    # 0.12, 0.38, -0.12 and -0.38, not away from zero.
    left = pl.LazyFrame(
        {
            "k": [1, 2, 3, 4, 5, 6],
            "a": _decimals(["1", "3", "-1", "-3", "5", "-5"], 2, 15),
            "b": _decimals(["8"] * 6, 2, 15),
        }
    )
    right = pl.LazyFrame(
        {
            "k": [1, 2, 3, 4, 5, 6],
            "w": _decimals(["12", "38", "-12", "-38", "62", "-63"], 2, 15),
            "v": _decimals(["100"] * 6, 2, 15),
        }
    )
    q = left.join(right, on="k", how="inner").filter(
        compare(pl.col("a") / pl.col("b"), pl.col("w") / pl.col("v"))
    )
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


def test_ratios_of_decimals_with_a_scale_too_large_for_float64(
    engine: pl.GPUEngine, decimal_ratios
):
    # 10**23 is not exact in Float64, so the comparison is not rewritten.
    # Whether the AST can still evaluate it is not what is tested.
    left, right = decimal_ratios
    left = left.with_columns(pl.col("a", "b").cast(pl.Decimal(38, 23)))
    right = right.with_columns(pl.col("w", "y").cast(pl.Decimal(38, 23)))
    q = left.join(right, on="k", how="inner").filter(
        (pl.col("w") / pl.col("y")) > pl.col("a") / pl.col("b")
    )
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


@pytest.mark.parametrize(
    "value, expected",
    [
        (Decimal("1.50"), True),
        (None, True),
        (Decimal("90071992547409.93"), False),
        (150, False),
        (1.5, False),
    ],
)
def test_decimal_literal_converts_exactly_to_float64(value, expected):
    # Only a Decimal literal is converted. Any other value is left to the fallback.
    literal = expr.Literal(DataType(pl.Decimal(38, 2)), value)
    assert _converts_exactly_to_float64(literal, 2) is expected
