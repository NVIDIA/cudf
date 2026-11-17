# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for evaluating IR nodes on the CPU with ``rank_local_cpu_compute``."""

from __future__ import annotations

import collections
import datetime
import decimal
import threading
from typing import TYPE_CHECKING

import numpy as np
import pytest

import polars as pl
from polars.testing import assert_frame_equal

import pylibcudf as plc

import cudf_polars.dsl.cpu_evaluate as cpu_evaluate_module
import cudf_polars.streaming.actor_graph.utils as actor_utils
from cudf_polars import Translator
from cudf_polars.containers import DataType
from cudf_polars.dsl import expr
from cudf_polars.dsl.cpu_evaluate import (
    BINARY_OPS,
    evaluate_ir,
    is_cpu_capable,
)
from cudf_polars.dsl.expressions.base import ExecutionContext
from cudf_polars.dsl.expressions.binaryop import BinOp
from cudf_polars.dsl.ir import Filter, IRExecutionContext, Select, Slice
from cudf_polars.dsl.traversal import traversal
from cudf_polars.engine.options import StreamingOptions
from cudf_polars.engine.spmd import SPMDEngine
from cudf_polars.testing.asserts import assert_gpu_result_equal
from cudf_polars.utils.config import MemoryResourceConfig, RankLocalCPUComputeMode
from cudf_polars.utils.versions import POLARS_VERSION_LT_21

if TYPE_CHECKING:
    from collections.abc import Callable

    from cudf_polars.containers import DataFrame
    from cudf_polars.dsl.ir import IR
    from cudf_polars.engine.core import StreamingEngine

# The tests count CPU evaluations in this process, so use the in-process
# engines only.
pytestmark = pytest.mark.engine_params(["spmd", "spmd-small"])


@pytest.fixture
def engine(streaming_engine_factory):
    return streaming_engine_factory(
        StreamingOptions(
            max_rows_per_partition=4,
            raise_on_fail=True,
            fallback_mode="silent",
            rank_local_cpu_compute="always",
        )
    )


@pytest.fixture
def on_oom_engine(streaming_engine_factory):
    return streaming_engine_factory(
        StreamingOptions(
            max_rows_per_partition=4,
            raise_on_fail=True,
            fallback_mode="silent",
            rank_local_cpu_compute="on_oom",
        )
    )


@pytest.fixture
def filter_oom(monkeypatch):
    """Make every ``Filter.do_evaluate`` raise ``MemoryError``, and count the calls."""
    calls = collections.Counter()
    lock = threading.Lock()

    def raise_oom(cls, *args, **kwargs):
        with lock:
            calls["Filter"] += 1
        raise MemoryError("std::bad_alloc: out_of_memory")

    monkeypatch.setattr(Filter, "do_evaluate", classmethod(raise_oom))
    return calls


@pytest.fixture
def cpu_calls(monkeypatch):
    calls: collections.Counter[str] = collections.Counter()
    lock = threading.Lock()
    original = cpu_evaluate_module.cpu_evaluate

    def counting(node, *dfs, context):
        # Actors evaluate in worker threads.
        with lock:
            calls[type(node).__name__] += 1
        return original(node, *dfs, context=context)

    monkeypatch.setattr(cpu_evaluate_module, "cpu_evaluate", counting)
    return calls


def _named_exprs(obj):
    """The expressions of the named expressions in ``obj``, searched recursively."""
    if isinstance(obj, expr.NamedExpr):
        yield obj.value
    elif isinstance(obj, (tuple, list)):
        for item in obj:
            yield from _named_exprs(item)


def _expression_names(node) -> set[str]:
    """
    The expression types of ``node``.

    Includes the names of aggregations and boolean functions, and the target
    type of casts.
    """
    names = set()
    for e in traversal(list(_named_exprs(node._non_child_args))):
        names.add(type(e).__name__)
        if isinstance(e, expr.Agg):
            names.add(f"Agg.{e.name}")
        elif isinstance(e, expr.BooleanFunction):
            names.add(f"BooleanFunction.{e.name.name}")
        elif isinstance(e, expr.Cast):
            names.add(f"Cast.{e.dtype.polars_type}")
    return names


@pytest.fixture
def cpu_exprs(monkeypatch, cpu_calls):
    """Record the expressions of the nodes evaluated on the CPU."""
    exprs: set[str] = set()
    counting = cpu_evaluate_module.cpu_evaluate

    def recording(node, *dfs, context):
        exprs.update(_expression_names(node))
        return counting(node, *dfs, context=context)

    monkeypatch.setattr(cpu_evaluate_module, "cpu_evaluate", recording)
    return exprs


@pytest.fixture(scope="module")
def left():
    return pl.LazyFrame(
        {
            "a": [1, 2, 3, 4, 5, 6] * 3,
            "b": [1.5, 2.5, None, 4.0, 5.0, None] * 3,
            "s": ["x", "y", "z", "x", "y", "z"] * 3,
        }
    )


@pytest.fixture(scope="module")
def right():
    return pl.LazyFrame({"a": [1, 2, 3, 7], "c": [10, 20, 30, 70]})


def test_filter_hstack_select(left, engine, cpu_calls):
    q = left.filter(pl.col("a") > 1).with_columns(d=pl.col("a") * 2 + 1)
    assert_gpu_result_equal(q, engine=engine)
    assert cpu_calls["Filter"] > 0
    assert cpu_calls["HStack"] > 0


@pytest.mark.parametrize("mode", ["always", "on_oom"])
def test_statistics(left, streaming_engine_factory, cpu_calls, filter_oom, mode):
    # filter_oom only matters for "on_oom". With "always", Filter runs on the
    # CPU without calling do_evaluate.
    engine = streaming_engine_factory(
        StreamingOptions(
            max_rows_per_partition=4,
            raise_on_fail=True,
            fallback_mode="silent",
            rank_local_cpu_compute=mode,
            statistics=True,
        )
    )
    q = left.filter(pl.col("a") > 1).with_columns(d=pl.col("a") * 2 + 1)
    assert_gpu_result_equal(q, engine=engine)
    stats = engine.global_statistics()
    seconds = stats.get_stat("rank-local-cpu-compute-time")
    nbytes = stats.get_stat("rank-local-cpu-compute-bytes")
    assert seconds["count"] == nbytes["count"] == sum(cpu_calls.values()) > 0
    assert seconds["value"] > 0
    assert nbytes["value"] > 0
    report = stats.report()
    assert "rank-local-cpu-compute-time" in report
    assert "rank-local-cpu-compute-bytes" in report


@pytest.mark.parametrize(
    "how,coalesce",
    [
        ("inner", None),
        ("left", None),
        ("right", None),
        ("full", None),
        ("full", True),
        ("semi", None),
        ("anti", None),
    ],
)
def test_join(left, right, engine, cpu_calls, how, coalesce):
    q = left.join(right, on="a", how=how, coalesce=coalesce)
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)
    assert cpu_calls["Join"] > 0


@pytest.mark.parametrize("how", ["inner", "left", "right", "full"])
@pytest.mark.parametrize("coalesce", [False, True])
def test_join_expression_keys(left, right, engine, cpu_calls, how, coalesce):
    q = left.join(
        right.with_columns(b=pl.col("c") / 10),
        left_on=[pl.col("a"), pl.col("a") * 10],
        right_on=["a", "c"],
        how=how,
        coalesce=coalesce,
    )
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)
    assert cpu_calls["Join"] > 0


def test_cross_join(left, right, engine, cpu_calls):
    q = left.join(right, how="cross")
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)
    assert cpu_calls["Join"] > 0


def test_join_slice(left, right, engine, cpu_calls):
    # The row order is not defined, so only the length is compared.
    q = left.join(right, on="a").slice(1, 3).select(pl.len())
    assert_gpu_result_equal(q, engine=engine)
    assert cpu_calls["Join"] > 0


@pytest.fixture(scope="module")
def nonequi_left():
    return pl.LazyFrame(
        {
            "a": [1, 2, 3, 1, None],
            "b": [1, 2, 3, 4, 5],
            "x": [1.0, float("nan"), None, -2.5, 3.0],
        }
    )


@pytest.fixture(scope="module")
def nonequi_right():
    return pl.LazyFrame(
        {
            "a": [1, 4, 3, 7, None, None, 1],
            "c": [2, 3, 4, 5, 6, 7, 8],
            "y": [0.5, float("nan"), 2.0, None, -3.0, 4.0, 1.0],
        }
    )


@pytest.mark.parametrize(
    "options",
    [{"nulls_equal": True}, {"suffix": "_r", "coalesce": False}],
)
def test_join_options(nonequi_left, nonequi_right, engine, cpu_calls, options):
    q = nonequi_left.join(nonequi_right, on="a", **options)
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)
    assert cpu_calls["Join"] > 0


@pytest.mark.skipif(
    POLARS_VERSION_LT_21,
    reason=(
        "The query needs a join with a fused predicate, which Polars exposes to "
        "cudf-polars from 2.1. On Polars 2.0 it always runs on the Polars CPU engine."
    ),
)
def test_join_with_predicate(nonequi_left, nonequi_right, engine, cpu_calls):
    # Polars plans this as an equi join on a == c with the inequality as its
    # predicate, which drops one of the two matching rows.
    q = nonequi_left.join_where(
        nonequi_right,
        pl.col("b") * 2 > pl.col("a_right") + 2,
        pl.col("a") == pl.col("c"),
    )
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)
    assert cpu_calls["Join"] > 0


@pytest.mark.parametrize(
    "conditions",
    [
        [pl.col("a") < pl.col("a_right")],
        [pl.col("b") * 2 <= pl.col("c"), pl.col("a") < pl.col("a_right")],
    ],
)
@pytest.mark.parametrize("zlice", [None, (1, 3)])
def test_conditional_join(
    nonequi_left, nonequi_right, engine, cpu_calls, conditions, zlice
):
    q = nonequi_left.join_where(nonequi_right, *conditions)
    if zlice is None:
        assert_gpu_result_equal(q, engine=engine, check_row_order=False)
    else:
        # The row order is not defined, so only the length is compared.
        assert_gpu_result_equal(q.slice(*zlice).select(pl.len()), engine=engine)
    assert cpu_calls["ConditionalJoin"] > 0


def test_conditional_join_decimal_float(engine, cpu_calls):
    # As in do_evaluate, the decimal column is cast to Float64 to be compared
    # with the float column.
    left = pl.LazyFrame(
        {
            "d": pl.Series(
                [decimal.Decimal(v) for v in ("1.50", "2.25", "3.00", "0.10")],
                dtype=pl.Decimal(10, 2),
            )
        }
    )
    right = pl.LazyFrame({"y": [1.0, 2.5, 2.999, 0.05]})
    q = left.join_where(right, pl.col("d") > pl.col("y"))
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)
    assert cpu_calls["ConditionalJoin"] > 0


# Polars plans a "!=" condition as a cross join followed by a filter.
@pytest.mark.parametrize(
    "op,node",
    [
        ("lt", "ConditionalJoin"),
        pytest.param(
            "ne",
            "Filter",
            marks=pytest.mark.skipif(
                POLARS_VERSION_LT_21,
                reason=(
                    "The query needs a nested loop join, which Polars exposes to "
                    "cudf-polars from 2.1. On Polars 2.0 it always runs on the "
                    "Polars CPU engine."
                ),
            ),
        ),
    ],
)
def test_conditional_join_nan(
    nonequi_left, nonequi_right, streaming_engine_factory, cpu_calls, op, node
):
    # The result is compared with the GPU. A null operand never matches, even
    # against NaN.
    q = nonequi_left.join_where(
        nonequi_right, getattr(pl.col("x"), op)(pl.col("y"))
    ).sort(pl.all())

    _compare_backends(q, streaming_engine_factory, cpu_calls, node)


def test_groupby_mean_all_null_group(left, right, engine, cpu_calls):
    # The streaming planner computes the mean as a grouped sum over a grouped
    # count. The sum of an all-null group must be null, as in libcudf. Without
    # a=3 on the right, every "c" of group "z" is null.
    q = (
        left.join(right.filter(pl.col("a") != 3), on="a", how="left")
        .group_by("s")
        .agg(pl.col("c").mean(), pl.col("b").sum(), n=pl.len())
    )
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)
    assert cpu_calls["GroupBy"] > 0


@pytest.mark.parametrize("nulls_last", [False, True])
@pytest.mark.parametrize("descending", [False, True])
def test_sort(left, engine, cpu_calls, descending, nulls_last):
    q = left.sort("b", "a", descending=descending, nulls_last=nulls_last)
    assert_gpu_result_equal(q, engine=engine)
    assert cpu_calls["Sort"] > 0


@pytest.mark.parametrize("keep", ["first", "last", "none", "any"])
def test_distinct(left, engine, cpu_calls, keep):
    if keep == "any":
        # Which duplicate "any" keeps is undefined, so the duplicates are
        # adjacent and only "s" is selected, which fixes the result.
        q = pl.LazyFrame({"s": ["x", "x", "y", "y", "z", "z"]}).unique(
            keep=keep, maintain_order=True
        )
    else:
        q = left.unique(subset=["s"], keep=keep, maintain_order=True)
    assert_gpu_result_equal(q, engine=engine)
    assert cpu_calls["Distinct"] > 0


def test_projection_without_columns(left, right, engine, cpu_calls):
    # The cross join only uses columns of ``right``, so ``left`` is projected
    # to no columns, which must keep its row count.
    q = left.filter(pl.col("a") == 1).join(right, how="cross").select("c")
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)
    assert cpu_calls["Projection"] > 0


def _not_cpu_capable_select(engine: pl.GPUEngine) -> Select:
    """A Select using an expression without a Polars conversion."""
    dtype = DataType(pl.Int64())
    df = pl.LazyFrame({"a": [1, 2]})
    child = Translator(df._ldf.visit(), engine).translate_ir()
    agg = expr.Agg(dtype, "sum", None, ExecutionContext.ROLLING, expr.Col(dtype, "a"))
    node = Select(
        {"a": dtype},
        (expr.NamedExpr("a", agg),),
        True,  # noqa: FBT003
        child,
    )
    assert not is_cpu_capable(node)
    return node


def test_not_cpu_capable(in_memory_engine):
    node = _not_cpu_capable_select(in_memory_engine)
    context = IRExecutionContext()
    df = node.children[0].evaluate(cache={}, context=context)
    with pytest.raises(ValueError, match="not CPU capable"):
        cpu_evaluate_module.cpu_evaluate(node, df, context=context)


_I64 = DataType(pl.Int64())
_F64 = DataType(pl.Float64())
_BOOL = DataType(pl.Boolean())


@pytest.mark.parametrize(
    "e",
    [
        pytest.param(
            expr.BinOp(
                _I64,
                plc.binaryop.BinaryOperator.DIV,
                expr.Col(_I64, "i"),
                expr.Col(_I64, "j"),
            ),
            id="integer-div",
        ),
        pytest.param(
            expr.BinOp(
                _I64,
                plc.binaryop.BinaryOperator.SHIFT_LEFT,
                expr.Col(_I64, "i"),
                expr.Col(_I64, "j"),
            ),
            id="unknown-operator",
        ),
        pytest.param(
            expr.BinOp(
                _BOOL,
                plc.binaryop.BinaryOperator.LESS,
                expr.Col(_F64, "f"),
                expr.Col(_I64, "i"),
            ),
            id="float-int-comparison",
        ),
        pytest.param(
            expr.BooleanFunction(
                _BOOL,
                expr.BooleanFunction.Name.IsBetween,
                ("both",),
                expr.Col(_I64, "i"),
                expr.Col(_I64, "j"),
                expr.Col(_I64, "j"),
            ),
            id="unsupported-boolean-function",
        ),
        pytest.param(
            expr.Agg(_F64, "m2", None, ExecutionContext.FRAME, expr.Col(_F64, "f")),
            id="unsupported-aggregation",
        ),
    ],
)
def test_not_converted(e):
    # Expressions without a Polars conversion make their node stay on the GPU.
    with pytest.raises(NotImplementedError):
        cpu_evaluate_module.expr_to_polars(e)


def test_unsupported_expression_stays_on_gpu(left, engine, cpu_calls):
    # A regex is not converted, so the node runs on the GPU.
    q = left.with_columns(m=pl.col("s").str.contains("x|y"))
    assert_gpu_result_equal(q, engine=engine)
    assert sum(cpu_calls.values()) == 0


@pytest.fixture(scope="module")
def edge_cases():
    return pl.LazyFrame(
        {
            "f": [1.5, -2.0, 0.0, float("nan"), float("inf"), -float("inf"), None, 0.5],
            "g": [0.5, float("nan"), -0.0, float("nan"), 2.0, None, 1.0, 3.0],
            "i": [3, -2, 0, 7, None, 1, 2, -5],
            "j": [2, 3, -1, 0, 4, None, 1, 2],
            "b": [True, False, None, True, False, True, None, False],
            "c": [False, False, True, None, True, None, None, True],
            # Sums of these overflow Int32, except for the null group of "b".
            "k": pl.Series(
                [2**30, 2**30, 1, -3, None, 2**30, 2, 2**30], dtype=pl.Int32
            ),
        }
    )


def _collect(q, streaming_engine_factory, mode: str) -> pl.DataFrame:
    """Collect ``q`` in a single chunk with ``rank_local_cpu_compute=mode``."""
    return q.collect(
        engine=streaming_engine_factory(
            StreamingOptions(
                max_rows_per_partition=1000,
                raise_on_fail=True,
                fallback_mode="silent",
                rank_local_cpu_compute=mode,
            )
        )
    )


def _compare_backends(q, streaming_engine_factory, cpu_calls, node):
    """Check that ``q`` gives the same result on the GPU and the CPU."""
    expect = _collect(q, streaming_engine_factory, "never")
    assert not cpu_calls
    assert_frame_equal(_collect(q, streaming_engine_factory, "always"), expect)
    assert cpu_calls[node] > 0


def _translated_exprs(q, engine: pl.GPUEngine) -> list[expr.Expr]:
    node = Translator(q._ldf.visit(), engine).translate_ir()
    assert isinstance(node, Select)
    return list(traversal([ne.value for ne in node.exprs]))


@pytest.mark.parametrize("op", sorted(BINARY_OPS, key=str))
@pytest.mark.parametrize("columns", [("f", "g"), ("i", "j"), ("b", "c")])
def test_binary_ops_match(
    edge_cases, streaming_engine_factory, in_memory_engine, cpu_calls, op, columns
):
    left, right = (pl.col(c) for c in columns)
    q = edge_cases.select(BINARY_OPS[op](left, right).alias("out"))
    try:
        translated = _translated_exprs(q, in_memory_engine)
        q.collect(engine="in-memory")
    except pl.exceptions.PolarsError:
        pytest.skip(f"Polars rejects {op.name} for {columns}")
    if not any(isinstance(n, expr.BinOp) and n.op == op for n in translated):
        pytest.skip(f"Polars does not plan {op.name} for {columns}")
    _compare_backends(q, streaming_engine_factory, cpu_calls, "Select")


@pytest.mark.parametrize(
    "exprs,targets",
    [
        pytest.param(
            [
                pl.col("f").min(),
                pl.col("f").max().alias("f_max"),
                pl.col("g").min().alias("g_min"),
            ],
            {"Agg.min", "Agg.max"},
            id="min-max-nan",
        ),
        pytest.param(
            [pl.col("i").n_unique(), pl.col("f").n_unique().alias("f_n")],
            {"Agg.n_unique"},
            id="n_unique",
        ),
        pytest.param(
            [pl.col("i").count(), pl.col("f").len().alias("f_len")],
            {"Agg.count"},
            id="count",
        ),
        pytest.param(
            [pl.col("f").first(), pl.col("i").last()],
            {"Agg.first", "Agg.last"},
            id="first-last",
        ),
        pytest.param(
            [(pl.col("i") * 100).cast(pl.Int8)], {"Cast.Int8"}, id="cast-wraps"
        ),
        pytest.param(
            [pl.col("i").is_in([1, 2, 7])], {"BooleanFunction.IsIn"}, id="is_in"
        ),
        pytest.param(
            [pl.col("i") + pl.lit(pl.Series([1]))],
            {"LiteralColumn"},
            id="literal-column",
        ),
        pytest.param(
            [
                pl.col("f").is_null(),
                pl.col("f").is_not_null().alias("f_not_null"),
                pl.col("f").is_nan().alias("f_nan"),
                pl.col("f").is_not_nan().alias("f_not_nan"),
                ~pl.col("b"),
            ],
            {
                "BooleanFunction.IsNull",
                "BooleanFunction.IsNotNull",
                "BooleanFunction.IsNan",
                "BooleanFunction.IsNotNan",
                "BooleanFunction.Not",
            },
            id="boolean-functions",
        ),
        pytest.param(
            [pl.col("f").nan_min(), pl.col("f").nan_max().alias("f_max")],
            {"Agg.min", "Agg.max"},
            id="nan-min-max",
        ),
        pytest.param(
            [pl.col("k").sum(), pl.col("k").mean().alias("k_mean")],
            {"Agg.sum"},
            id="int32-sum-mean",
        ),
        pytest.param(
            [pl.col("i").alias("^i$"), pl.col("j").alias("j_alias")],
            {"Col"},
            id="regex-shaped-name",
        ),
    ],
)
def test_select_matches_gpu(
    edge_cases, streaming_engine_factory, cpu_calls, cpu_exprs, exprs, targets
):
    _compare_backends(
        edge_cases.select(exprs), streaming_engine_factory, cpu_calls, "Select"
    )
    # The planner can split a select, so check that the expressions under
    # test ran on the CPU, not only some Select.
    assert targets <= cpu_exprs


def test_col_reads_regex_shaped_name():
    # pl.col reads "^i$" as a regex matching "i", so check that the
    # conversion reads the column named "^i$".
    df = pl.DataFrame({"i": [1, 2], "^i$": [10, 20]})
    got = df.select(cpu_evaluate_module.expr_to_polars(expr.Col(_I64, "^i$")))
    assert_frame_equal(got, pl.DataFrame({"^i$": [10, 20]}))


def test_float_to_string_stays_on_gpu(edge_cases, streaming_engine_factory, cpu_calls):
    # libcudf formats floats with a fixed number of digits, unlike Polars.
    q = edge_cases.select(pl.col("f").cast(pl.String))
    assert_frame_equal(
        _collect(q, streaming_engine_factory, "always"),
        _collect(q, streaming_engine_factory, "never"),
    )
    assert cpu_calls["Select"] == 0


def test_time_result_stays_on_gpu(streaming_engine_factory, cpu_calls):
    # The sort is CPU capable, but DataFrame.from_polars cannot import its
    # Time column, so the node stays on the GPU.
    q = (
        pl.LazyFrame({"d": [datetime.datetime(2024, 1, 1, 1, 2, 3), None], "i": [2, 1]})
        .with_columns(t=pl.col("d").dt.time())
        .sort("i")
    )
    assert_frame_equal(
        _collect(q, streaming_engine_factory, "always"),
        _collect(q, streaming_engine_factory, "never"),
    )
    assert cpu_calls["Sort"] == 0
    # Without the Time column the same sort runs on the CPU, so the Time
    # column is what keeps it on the GPU above.
    _collect(q.drop("t"), streaming_engine_factory, "always")
    assert cpu_calls["Sort"] > 0


def test_groupby_matches_gpu(
    edge_cases, streaming_engine_factory, cpu_calls, cpu_exprs
):
    # The planner computes the mean as a grouped sum over a count. Integer
    # inputs keep the sums exact, so the result does not depend on the order
    # in which the GPU adds them.
    q = (
        edge_cases.group_by("b")
        .agg(
            pl.col("i").sum(),
            pl.col("j").mean().alias("j_mean"),
            pl.col("k").mean().alias("k_mean"),
            pl.col("k").sum().alias("k_sum"),
            pl.col("i").n_unique().alias("i_n"),
            pl.col("f").count().alias("f_count"),
            pl.col("j").first(),
            pl.col("j").last().alias("j_last"),
        )
        .sort("b")
    )
    _compare_backends(q, streaming_engine_factory, cpu_calls, "GroupBy")
    assert {
        "Agg.sum",
        "Agg.n_unique",
        "Agg.count",
        "Agg.first",
        "Agg.last",
    } <= cpu_exprs


@pytest.mark.parametrize("offset,length", [(2, 4), (-3, None), (10, 2)])
def test_slice_matches_gpu(edge_cases, in_memory_engine, offset, length):
    # The streaming engine slices without evaluating Slice nodes, and Polars
    # pushes a slice into the scan, so the node is built and evaluated here.
    child = Translator(edge_cases._ldf.visit(), in_memory_engine).translate_ir()
    node = Slice(child.schema, offset, length, child)
    assert is_cpu_capable(node)
    context = IRExecutionContext()
    df = child.evaluate(cache={}, context=context)
    expect = node.do_evaluate(*node._non_child_args, df, context=context)
    got = cpu_evaluate_module.cpu_evaluate(node, df, context=context)
    assert_frame_equal(got.to_polars(), expect.to_polars())


def test_binary_ops_cover_gpu():
    # Every binary operator translation produces has a Polars expression. DIV
    # is converted separately, since it is true division only for a floating
    # point result.
    translated = {
        BinOp._BOOL_KLEENE_MAPPING.get(op, op) for op in BinOp._MAPPING.values()
    }
    special = {plc.binaryop.BinaryOperator.DIV}
    assert translated - special <= set(BINARY_OPS)


def test_on_oom_replays_on_cpu(left, on_oom_engine, cpu_calls, filter_oom):
    q = left.filter(pl.col("a") > 2).with_columns(c=pl.col("a") * 2)
    assert_gpu_result_equal(q, engine=on_oom_engine)
    assert filter_oom["Filter"] > 0
    assert cpu_calls["Filter"] == filter_oom["Filter"]
    # Nodes that do not run out of memory stay on the GPU.
    assert cpu_calls.keys() == {"Filter"}


@pytest.fixture
def overbooking_engine(
    streaming_engine_factory: Callable[[StreamingOptions], StreamingEngine],
) -> StreamingEngine:
    # A one byte device limit makes every reservation overbook.
    return streaming_engine_factory(
        StreamingOptions(
            max_rows_per_partition=4,
            raise_on_fail=True,
            fallback_mode="silent",
            rank_local_cpu_compute="on_oom",
            spill_device_limit="1",
            memory_reserve_timeout="1ms",
        )
    )


@pytest.fixture
def evaluate_ir_modes(
    monkeypatch: pytest.MonkeyPatch,
) -> collections.defaultdict[str, set[RankLocalCPUComputeMode]]:
    """Record the ``rank_local_cpu_compute`` mode of each IR node type evaluated."""
    modes: collections.defaultdict[str, set[RankLocalCPUComputeMode]] = (
        collections.defaultdict(set)
    )
    lock = threading.Lock()
    original = actor_utils.evaluate_ir

    def recording(node: IR, *dfs: DataFrame, context: IRExecutionContext) -> DataFrame:
        with lock:
            modes[type(node).__name__].add(context.rank_local_cpu_compute)
        return original(node, *dfs, context=context)

    monkeypatch.setattr(actor_utils, "evaluate_ir", recording)
    return modes


def test_on_oom_overbooking_evaluates_on_cpu(
    left, overbooking_engine, cpu_calls, evaluate_ir_modes
):
    q = left.filter(pl.col("a") > 2).with_columns(c=pl.col("a") * 2)
    assert_gpu_result_equal(q, engine=overbooking_engine)
    # The reservation overbooks, so the Filter runs on the CPU without a GPU
    # attempt first.
    assert evaluate_ir_modes["Filter"] == {RankLocalCPUComputeMode.ALWAYS}
    assert cpu_calls["Filter"] > 0


def test_on_oom_overbooking_already_available(overbooking_engine, evaluate_ir_modes):
    # Rolling reserves memory for an input that is already available, which
    # takes the already_available branch of evaluate_chunk.
    df = pl.LazyFrame(
        {"orderby": [1, 4, 8, 10, 12, 13, 14, 22], "values": [1, 2, 3, 4, 5, 6, 7, 8]}
    )
    q = df.rolling("orderby", period="4i").agg(pl.col("values").sum())
    assert_gpu_result_equal(q, engine=overbooking_engine)
    # The overbooked reservation switches the evaluation to "always".
    assert evaluate_ir_modes["Rolling"] == {RankLocalCPUComputeMode.ALWAYS}


def test_on_oom_without_oom_stays_on_gpu(left, on_oom_engine, cpu_calls):
    q = left.filter(pl.col("a") > 2).with_columns(c=pl.col("a") * 2)
    assert_gpu_result_equal(q, engine=on_oom_engine)
    assert not cpu_calls


def test_on_oom_not_capable_reraises(in_memory_engine, monkeypatch):
    node = _not_cpu_capable_select(in_memory_engine)
    context = IRExecutionContext(rank_local_cpu_compute=RankLocalCPUComputeMode.ON_OOM)
    df = node.children[0].evaluate(cache={}, context=context)

    def raise_oom(cls, *args, **kwargs):
        raise MemoryError("std::bad_alloc: out_of_memory")

    monkeypatch.setattr(Select, "do_evaluate", classmethod(raise_oom))
    with pytest.raises(MemoryError):
        evaluate_ir(node, df, context=context)


@pytest.mark.parametrize("mode", ["always", "on_oom"])
def test_cpu_error_notes_device_oom(in_memory_engine, monkeypatch, mode):
    q = pl.LazyFrame({"a": [1, 2]}).select(pl.col("a") + 1)
    node = Translator(q._ldf.visit(), in_memory_engine).translate_ir()
    assert isinstance(node, Select)
    assert is_cpu_capable(node)
    context = IRExecutionContext(rank_local_cpu_compute=RankLocalCPUComputeMode(mode))
    df = node.children[0].evaluate(cache={}, context=context)

    def raise_oom(cls, *args, **kwargs):
        raise MemoryError("std::bad_alloc: out_of_memory")

    def raise_error(node, *dfs, context):
        raise RuntimeError("CPU failure")

    monkeypatch.setattr(Select, "do_evaluate", classmethod(raise_oom))
    monkeypatch.setattr(cpu_evaluate_module, "cpu_evaluate", raise_error)
    with pytest.raises(RuntimeError, match="CPU failure") as excinfo:
        evaluate_ir(node, df, context=context)
    # Only a CPU evaluation after a device OOM gets the note.
    note = (
        "Raised while evaluating Select on the CPU after the GPU ran out of "
        "memory: std::bad_alloc: out_of_memory"
    )
    expected = [note] if mode == "on_oom" else []
    assert getattr(excinfo.value, "__notes__", []) == expected


@pytest.mark.parametrize("mode", ["never", "on_oom"])
def test_device_oom(cpu_calls, mode):
    # The engine reads its memory resource from ``memory_resource_config`` at
    # construction, so build one with a 39 MiB device memory limit. Sorting
    # 2**20 rows of two int64 columns (16 MiB) then runs out of device memory
    # on the GPU, and fits on the CPU.
    n = 1 << 20
    data = pl.DataFrame(
        {"a": np.random.default_rng(42).permutation(n), "b": np.arange(n)}
    )
    q = data.lazy().sort("a")
    options = StreamingOptions(
        max_rows_per_partition=n,
        raise_on_fail=True,
        fallback_mode="silent",
        rank_local_cpu_compute=mode,
        memory_resource_config=MemoryResourceConfig(
            qualname="rmm.mr.LimitingResourceAdaptor",
            options={
                "upstream_mr": {"qualname": "rmm.mr.CudaMemoryResource"},
                "allocation_limit": 39 << 20,
            },
        ),
    )
    with SPMDEngine(
        rapidsmpf_options=options.to_rapidsmpf_options(),
        executor_options=options.to_executor_options(),
        engine_options=options.to_engine_options(),
    ) as engine:
        if mode == "never":
            with pytest.raises(MemoryError):
                q.collect(engine=engine)
            assert not cpu_calls
        else:
            assert_frame_equal(q.collect(engine=engine), q.collect())
            # Fails if the limit no longer makes the sort run out of device
            # memory on the GPU.
            assert cpu_calls["Sort"] > 0
