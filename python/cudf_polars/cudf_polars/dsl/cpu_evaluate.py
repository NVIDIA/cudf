# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Evaluate IR nodes on the CPU with Polars.

The functions in this module rebuild a Polars query from a single IR node
and its expressions, run it with the Polars CPU engine, and move the result
back to the device. They are used by the streaming executor when
``rank_local_cpu_compute`` is enabled.
"""

from __future__ import annotations

import functools
import operator
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

import polars as pl
import polars.selectors as cs

import pylibcudf as plc

from cudf_polars.containers import DataFrame
from cudf_polars.dsl import expr, ir
from cudf_polars.dsl.expressions.base import ExecutionContext

if TYPE_CHECKING:
    from collections.abc import Sequence

    from cudf_polars.dsl.ir import IR, IRExecutionContext
    from cudf_polars.typing import Schema

__all__ = ["cpu_evaluate", "evaluate_ir", "expr_to_polars", "is_cpu_capable"]


@functools.singledispatch
def expr_to_polars(e: expr.Expr) -> pl.Expr:
    """
    Convert an expression to a Polars expression.

    Parameters
    ----------
    e
        The expression to convert.

    Returns
    -------
    A Polars expression with the same semantics as ``e``.

    Raises
    ------
    NotImplementedError
        If ``e`` or one of its children has no Polars conversion.
    """
    raise NotImplementedError(f"No Polars conversion for {type(e).__name__}")


@expr_to_polars.register
def _(e: expr.Col) -> pl.Expr:
    return pl.col(e.name)


@expr_to_polars.register
def _(e: expr.Literal) -> pl.Expr:
    return pl.lit(e.value, dtype=e.dtype.polars_type)


@expr_to_polars.register
def _(e: expr.LiteralColumn) -> pl.Expr:
    # do_evaluate broadcasts a column of length one like a scalar.
    return pl.lit(e.value).first() if len(e.value) == 1 else pl.lit(e.value)


@expr_to_polars.register
def _(e: expr.Cast) -> pl.Expr:
    (child,) = e.children
    if plc.traits.is_floating_point(child.dtype.plc_type) and isinstance(
        e.dtype.polars_type, pl.String
    ):
        # libcudf formats floats with a fixed number of digits.
        raise NotImplementedError("Cast from float to string")
    if (
        e.strict
        and plc.traits.is_integral_not_bool(child.dtype.plc_type)
        and plc.traits.is_integral_not_bool(e.dtype.plc_type)
    ):
        # libcudf wraps out of range integers instead of raising.
        return expr_to_polars(child).cast(
            e.dtype.polars_type, strict=False, wrap_numerical=True
        )
    return expr_to_polars(child).cast(e.dtype.polars_type, strict=e.strict)


@expr_to_polars.register
def _(e: expr.Len) -> pl.Expr:
    return pl.len()


# The Polars expression of the binary operators that do_evaluate passes to
# ``plc.binaryop.binary_operation``.
BINARY_OPS: dict[plc.binaryop.BinaryOperator, Callable[[pl.Expr, pl.Expr], pl.Expr]] = {
    plc.binaryop.BinaryOperator.EQUAL: operator.eq,
    plc.binaryop.BinaryOperator.NULL_EQUALS: pl.Expr.eq_missing,
    plc.binaryop.BinaryOperator.NOT_EQUAL: operator.ne,
    plc.binaryop.BinaryOperator.NULL_NOT_EQUALS: pl.Expr.ne_missing,
    plc.binaryop.BinaryOperator.LESS: operator.lt,
    plc.binaryop.BinaryOperator.LESS_EQUAL: operator.le,
    plc.binaryop.BinaryOperator.GREATER: operator.gt,
    plc.binaryop.BinaryOperator.GREATER_EQUAL: operator.ge,
    plc.binaryop.BinaryOperator.ADD: operator.add,
    plc.binaryop.BinaryOperator.SUB: operator.sub,
    plc.binaryop.BinaryOperator.MUL: operator.mul,
    plc.binaryop.BinaryOperator.TRUE_DIV: operator.truediv,
    plc.binaryop.BinaryOperator.FLOOR_DIV: operator.floordiv,
    plc.binaryop.BinaryOperator.PYMOD: operator.mod,
    plc.binaryop.BinaryOperator.NULL_LOGICAL_AND: operator.and_,
    plc.binaryop.BinaryOperator.NULL_LOGICAL_OR: operator.or_,
    plc.binaryop.BinaryOperator.BITWISE_AND: operator.and_,
    plc.binaryop.BinaryOperator.BITWISE_OR: operator.or_,
    plc.binaryop.BinaryOperator.BITWISE_XOR: operator.xor,
    plc.binaryop.BinaryOperator.POW: operator.pow,
    plc.binaryop.BinaryOperator.INT_POW: operator.pow,
    plc.binaryop.BinaryOperator.LOG_BASE: pl.Expr.log,
}


@expr_to_polars.register
def _(e: expr.BinOp) -> pl.Expr:
    left, right = (expr_to_polars(c) for c in e.children)
    if e.op == plc.binaryop.BinaryOperator.DIV:
        # DIV is true division only for a floating point result.
        if not plc.traits.is_floating_point(e.dtype.plc_type):
            raise NotImplementedError(f"DIV with result type {e.dtype}")
        return (left / right).cast(e.dtype.polars_type)
    try:
        op = BINARY_OPS[e.op]
    except KeyError:
        raise NotImplementedError(f"Binary operator {e.op.name}") from None
    if (
        e.op in expr.BinOp._NAN_AWARE_COMPARISONS
        and sum(plc.traits.is_floating_point(c.dtype.plc_type) for c in e.children) == 1
    ):
        # With one floating point operand, libcudf compares NaN as IEEE 754
        # does, while Polars orders NaN above every other value.
        raise NotImplementedError("Comparison of a floating point operand")
    return op(left, right).cast(e.dtype.polars_type)


_BOOLEAN_FUNCTIONS: dict[expr.BooleanFunction.Name, Callable[[pl.Expr], pl.Expr]] = {
    expr.BooleanFunction.Name.IsNull: pl.Expr.is_null,
    expr.BooleanFunction.Name.IsNotNull: pl.Expr.is_not_null,
    expr.BooleanFunction.Name.Not: pl.Expr.not_,
    expr.BooleanFunction.Name.IsNan: pl.Expr.is_nan,
    expr.BooleanFunction.Name.IsNotNan: pl.Expr.is_not_nan,
}


@expr_to_polars.register
def _(e: expr.BooleanFunction) -> pl.Expr:
    args = [expr_to_polars(c) for c in e.children]
    if e.name in _BOOLEAN_FUNCTIONS:
        (value,) = args
        return _BOOLEAN_FUNCTIONS[e.name](value)
    if e.name is expr.BooleanFunction.Name.IsIn:
        needles, haystack = args
        (nulls_equal,) = e.options
        return needles.is_in(haystack, nulls_equal=nulls_equal)
    raise NotImplementedError(f"Boolean function {e.name.name}")


# The aggregation converters follow ``Agg.do_evaluate`` (FRAME context) and
# libcudf grouped aggregation (GROUPBY context), which is not always Polars.
# Translation and the streaming planner rely on this, for example a grouped
# sum is null for an all-null group and a later expression fills in zero.


def _agg_min_max(name: Literal["min", "max"], value: pl.Expr, e: expr.Agg) -> pl.Expr:
    """
    Convert a ``min`` or ``max`` aggregation, as given by ``name``.

    With ``propagate_nans`` in ``e.options``, a NaN gives NaN. Otherwise NaNs
    are ignored, so only NaN values give null.
    """
    if e.options:  # propagate_nans
        return value.nan_min() if name == "min" else value.nan_max()
    if plc.traits.is_floating_point(e.children[0].dtype.plc_type):
        # NaNs are ignored, so only NaNs give null.
        value = value.fill_nan(None)
    return value.min() if name == "min" else value.max()


def _agg_sum(value: pl.Expr, e: expr.Agg) -> pl.Expr:
    """
    Convert a ``sum`` aggregation.

    The sum of an empty or all-null input is null in a group by, as in
    libcudf, and when ``e.options`` is set, as in ``Agg.do_evaluate``.
    Otherwise it is zero.
    """
    if e.options or e.context == ExecutionContext.GROUPBY:
        return pl.when(value.count() > 0).then(value.sum())
    return value.sum()


def _agg_count(value: pl.Expr, e: expr.Agg) -> pl.Expr:
    """Convert a ``count`` aggregation, which counts nulls if ``e.options`` is set."""
    return value.len() if e.options else value.count()  # options is include_nulls


def _agg_n_unique(value: pl.Expr, e: expr.Agg) -> pl.Expr:
    """Convert an ``n_unique`` aggregation, ignoring nulls if ``e.options`` is set."""
    return value.drop_nulls().n_unique() if e.options else value.n_unique()


_AGGS: dict[str, Callable[[pl.Expr, expr.Agg], pl.Expr]] = {
    "min": functools.partial(_agg_min_max, "min"),
    "max": functools.partial(_agg_min_max, "max"),
    "sum": _agg_sum,
    "count": _agg_count,
    "n_unique": _agg_n_unique,
    "mean": lambda value, _: value.mean(),
    "first": lambda value, _: value.first(),
    "last": lambda value, _: value.last(),
}


@expr_to_polars.register
def _(e: expr.Agg) -> pl.Expr:
    if e.context not in (ExecutionContext.FRAME, ExecutionContext.GROUPBY):
        raise NotImplementedError(f"Aggregation in {e.context.name} context")
    try:
        func = _AGGS[e.name]
    except KeyError:
        raise NotImplementedError(f"Aggregation {e.name}") from None
    (child,) = e.children
    return func(expr_to_polars(child), e).cast(e.dtype.polars_type)


def _named(exprs: tuple[expr.NamedExpr, ...]) -> list[pl.Expr]:
    """Convert named expressions to Polars expressions with the same names."""
    return [expr_to_polars(ne.value).alias(ne.name) for ne in exprs]


def _select(df: pl.LazyFrame, exprs: Sequence[pl.Expr | str]) -> pl.LazyFrame:
    """
    Select ``exprs`` from ``df``, keeping the row count when ``exprs`` is empty.

    ``LazyFrame.select`` with no expressions gives a frame without rows.
    """
    return df.select(exprs) if exprs else df.drop(cs.all())


def _conform(df: pl.LazyFrame, schema: Schema) -> pl.LazyFrame:
    """
    Select the columns of ``schema`` in order, cast to their dtypes.

    A grouped pointwise expression produces a list column, while ``schema``
    records the element dtype, as in ``GroupBy.do_evaluate``. Such columns
    are not cast.
    """
    actual = df.collect_schema()
    return _select(
        df,
        [
            pl.col(name)
            if isinstance(actual[name], pl.List)
            and not isinstance(dtype.polars_type, pl.List)
            else pl.col(name).cast(dtype.polars_type)
            for name, dtype in schema.items()
        ],
    )


@functools.singledispatch
def _cpu_plan(node: IR) -> Callable[..., pl.LazyFrame]:  # pragma: no cover
    """
    Return a function that evaluates ``node`` on Polars lazy frames.

    The function takes one lazy frame per child of ``node``. Raises
    ``NotImplementedError`` if ``node`` or one of its expressions has no
    Polars conversion.
    """
    raise NotImplementedError(f"No Polars evaluation for {type(node).__name__}")


@_cpu_plan.register
def _(node: ir.Select) -> Callable[..., pl.LazyFrame]:
    exprs = _named(node.exprs)
    return lambda df: _select(df, exprs)


@_cpu_plan.register
def _(node: ir.HStack) -> Callable[..., pl.LazyFrame]:
    exprs = _named(node.columns)
    return lambda df: df.with_columns(exprs)


@_cpu_plan.register
def _(node: ir.Filter) -> Callable[..., pl.LazyFrame]:
    mask = expr_to_polars(node.mask.value)
    return lambda df: df.filter(mask)


@_cpu_plan.register
def _(node: ir.Projection) -> Callable[..., pl.LazyFrame]:
    names = list(node.schema)
    return lambda df: _select(df, names)


@_cpu_plan.register
def _(node: ir.GroupBy) -> Callable[..., pl.LazyFrame]:
    keys = _named(node.keys)
    aggs = _named(node.agg_requests)
    maintain_order = node.maintain_order
    zlice = node.zlice

    def plan(df: pl.LazyFrame) -> pl.LazyFrame:
        result = df.group_by(keys, maintain_order=maintain_order).agg(aggs)
        return result if zlice is None else result.slice(*zlice)

    return plan


@_cpu_plan.register
def _(node: ir.Sort) -> Callable[..., pl.LazyFrame]:
    by = [expr_to_polars(ne.value) for ne in node.by]
    descending = [o == plc.types.Order.DESCENDING for o in node.order]
    # libcudf places nulls relative to the sort order, Polars absolutely.
    nulls_last = [
        (n == plc.types.NullOrder.AFTER) != d
        for n, d in zip(node.null_order, descending, strict=True)
    ]
    stable = node.stable
    zlice = node.zlice

    def plan(df: pl.LazyFrame) -> pl.LazyFrame:
        result = df.sort(
            by, descending=descending, nulls_last=nulls_last, maintain_order=stable
        )
        return result if zlice is None else result.slice(*zlice)

    return plan


@_cpu_plan.register
def _(node: ir.Distinct) -> Callable[..., pl.LazyFrame]:
    subset = None if node.subset is None else sorted(node.subset)
    # The Polars name of the keep option, the inverse of Distinct._KEEP_MAP.
    keep = {option: name for name, option in ir.Distinct._KEEP_MAP.items()}[node.keep]
    stable = node.stable
    zlice = node.zlice

    def plan(df: pl.LazyFrame) -> pl.LazyFrame:
        result = df.unique(subset=subset, keep=keep, maintain_order=stable)  # type: ignore[arg-type]
        return result if zlice is None else result.slice(*zlice)

    return plan


@_cpu_plan.register
def _(node: ir.Slice) -> Callable[..., pl.LazyFrame]:
    offset, length = node.offset, node.length
    return lambda df: df.slice(offset, length)


def _cpu_plan_or_none(node: IR) -> Callable[..., pl.LazyFrame] | None:
    """
    Return the Polars plan of ``node``, or None if it cannot be converted.

    The plan is not cached, since a cache keyed on the node would keep the
    node and its children, including their data, alive after the query.
    """
    # Exact type match, streaming specific subclasses are not supported.
    if type(node) not in _cpu_plan.registry:
        return None
    if any(isinstance(dtype.polars_type, pl.Time) for dtype in node.schema.values()):
        # DataFrame.from_polars does not import Time columns.
        return None
    try:
        return _cpu_plan(node)
    except NotImplementedError:
        return None


def is_cpu_capable(node: IR) -> bool:
    """
    Whether ``node`` can be evaluated on the CPU with Polars.

    Parameters
    ----------
    node
        The IR node.

    Returns
    -------
    True if :func:`cpu_evaluate` supports ``node`` and all its expressions.
    """
    return _cpu_plan_or_none(node) is not None


def cpu_evaluate(node: IR, *dfs: DataFrame, context: IRExecutionContext) -> DataFrame:
    """
    Evaluate an IR node on the CPU with Polars.

    The inputs are copied to host memory, evaluated with the Polars CPU
    engine, and the result is copied back to the device.

    Parameters
    ----------
    node
        The IR node, which must satisfy :func:`is_cpu_capable`.
    dfs
        The evaluated children of ``node``, in order.
    context
        The execution context.

    Returns
    -------
    The result, on a stream from ``context``.
    """
    plan = _cpu_plan_or_none(node)
    if plan is None:
        raise ValueError(f"{type(node).__name__} is not CPU capable")
    inputs = [df.to_polars().lazy() for df in dfs]
    result = _conform(plan(*inputs), node.schema).collect(engine="in-memory")
    return DataFrame.from_polars(result, stream=context.get_cuda_stream())


def evaluate_ir(node: IR, *dfs: DataFrame, context: IRExecutionContext) -> DataFrame:
    """
    Evaluate an IR node given its evaluated children.

    Calls ``node.do_evaluate``, or :func:`cpu_evaluate` when the execution
    context requests CPU fallback and ``node`` is CPU capable. A node is not
    CPU capable when it, or one of its expressions, has no Polars conversion.
    Such a node is evaluated on the GPU, and the fallback still applies to
    the other nodes.

    Parameters
    ----------
    node
        The IR node.
    dfs
        The evaluated children of ``node``, in order.
    context
        The execution context.

    Returns
    -------
    The result of evaluating ``node``.
    """
    # is_cpu_capable and cpu_evaluate each build the Polars plan. That is
    # cheap compared to copying the data to the host and back.
    if context.rank_local_cpu_compute == "always" and is_cpu_capable(node):
        return cpu_evaluate(node, *dfs, context=context)
    return node.do_evaluate(*node._non_child_args, *dfs, context=context)
