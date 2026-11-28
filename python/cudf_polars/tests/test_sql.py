# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

import pytest

import polars as pl

from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)
from cudf_polars.utils.versions import POLARS_VERSION_LT_21


@pytest.fixture
def frames() -> dict[str, pl.LazyFrame]:
    return {
        "t": pl.LazyFrame(
            {
                "g": [1, 1, 2, 2, 3],
                "v": [1, None, None, None, 5],
                "s": ["a", "b", "c", "d", "e"],
            }
        ),
        "u": pl.LazyFrame({"k": [1, 2, 3], "w": [10, 20, 30]}),
    }


def sql(frames: dict[str, pl.LazyFrame], query: str) -> pl.LazyFrame:
    return pl.SQLContext(frames).execute(query)


@pytest.mark.parametrize(
    "query",
    [
        "SELECT SUM(v) AS s FROM t WHERE g = 2",
        "SELECT SUM(v) AS s FROM t WHERE g = 4",
        "SELECT g, SUM(v) AS s, TOTAL(v) AS tot FROM t GROUP BY g ORDER BY g",
        "SELECT g, SUM(v) OVER (PARTITION BY g) AS s FROM t ORDER BY g, s",
    ],
)
def test_sum_is_null_on_empty(engine: pl.GPUEngine, frames, query):
    assert_gpu_result_equal(sql(frames, query), engine=engine)


def test_scalar_subquery_is_broadcast(engine: pl.GPUEngine, frames):
    q = sql(
        frames,
        "SELECT g FROM t GROUP BY g HAVING SUM(g) > (SELECT AVG(w) / 10 FROM u) ORDER BY g",
    )
    assert_gpu_result_equal(q, engine=engine)


def test_union_all_with_null_string_key(engine: pl.GPUEngine):
    t = pl.LazyFrame({"g": ["x", "x", "y"], "v": [1.0, 2.0, 3.0]})
    q = sql(
        {"t": t},
        "SELECT g, AVG(v) AS m FROM t GROUP BY g "
        "UNION ALL SELECT NULL AS g, AVG(v) AS m FROM t",
    )
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


def test_left_join_on_constant_key(engine: pl.GPUEngine, frames):
    q = sql(frames, "SELECT g, k FROM t LEFT JOIN u ON w > 15 ORDER BY g, k")
    assert_gpu_result_equal(q, engine=engine)


def test_left_join_with_hidden_predicate_raises(engine: pl.GPUEngine, frames):
    q = sql(frames, "SELECT g, k FROM t LEFT JOIN u ON g = k AND v > 0")
    assert_ir_translation_raises(q, engine, NotImplementedError)


def test_array_literal(engine: pl.GPUEngine, frames):
    q = sql(frames, "SELECT ARRAY[1, 2] AS a FROM u")
    assert_gpu_result_equal(q, engine=engine)


def test_date_add_days(engine: pl.GPUEngine):
    t = pl.LazyFrame(
        {
            "d": [date(1969, 12, 29), None, date(2000, 2, 28), date(2000, 2, 28)],
            "delta": [
                timedelta(microseconds=-1),
                timedelta(0),
                timedelta(hours=36),
                None,
            ],
        }
    )
    q = t.sql(
        "SELECT d + 5 AS later, d + (-5) AS earlier, d + delta AS shifted FROM self"
    )
    assert_gpu_result_equal(q, engine=engine)


def test_date_add_days_out_of_range(engine: pl.GPUEngine):
    t = pl.LazyFrame({"days": [-106_751_992, -106_751_991, 106_751_991, 106_751_992]})
    q = t.sql("SELECT CAST(CAST(days AS DATE) + 5 AS INT) AS shifted FROM self")
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.skipif(
    POLARS_VERSION_LT_21,
    reason="Polars 2.0 does not expose fused join predicates to the visitor",
)
def test_join_on_date_add_days(engine: pl.GPUEngine):
    t = pl.LazyFrame({"k": [1, 1, 2], "d": [date(2000, 1, 1), None, date(2000, 1, 4)]})
    u = pl.LazyFrame({"k": [1, 2], "e": [date(2000, 1, 8), date(2000, 1, 8)]})
    q = sql(
        {"t": t, "u": u},
        "SELECT t.k, d, e FROM t JOIN u ON t.k = u.k AND e > d + 5 ORDER BY t.k",
    )
    assert_gpu_result_equal(q, engine=engine)


def test_null_list_literal_raises(engine: pl.GPUEngine):
    q = pl.LazyFrame({"a": [1, 2]}).with_columns(pl.lit([None]).alias("b"))
    assert_ir_translation_raises(q, engine, NotImplementedError)


def test_string_slice_float_length_raises(engine: pl.GPUEngine, frames):
    q = sql(frames, "SELECT SUBSTR(s, 1, 1.5) AS x FROM t")
    assert_ir_translation_raises(q, engine, NotImplementedError)


@pytest.mark.parametrize("op", ["cum_min", "cum_max"])
def test_over_fill_null_of_cumulative_raises(engine: pl.GPUEngine, op):
    q = pl.LazyFrame({"g": [1, 1, 1], "o": [3, 1, 2], "v": [None, 5, 3]}).select(
        getattr(pl.col("v"), op)().fill_null(strategy="forward").over("g", order_by="o")
    )
    assert_ir_translation_raises(q, engine, NotImplementedError)


@pytest.mark.parametrize(
    "query",
    [
        "SELECT price * (1 - discount) AS revenue FROM self",
        "SELECT price * discount AS x, price * price * discount AS y FROM self",
    ],
)
def test_decimal_mul(engine: pl.GPUEngine, query):
    q = pl.LazyFrame(
        {
            "price": [Decimal("10.00"), Decimal("20.00"), None],
            "discount": [Decimal("0.05"), Decimal("0.10"), Decimal("0.15")],
        },
        schema={"price": pl.Decimal(15, 2), "discount": pl.Decimal(15, 2)},
    ).sql(query)
    assert_gpu_result_equal(q, engine=engine)
