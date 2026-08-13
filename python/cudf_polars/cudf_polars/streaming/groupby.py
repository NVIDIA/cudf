# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parallel GroupBy Logic."""

from __future__ import annotations

import itertools
import math
from typing import TYPE_CHECKING

import polars as pl

import pylibcudf as plc

from cudf_polars.containers import Column, DataType
from cudf_polars.dsl.expr import (
    Agg,
    BinOp,
    BooleanFunction,
    Cast,
    Col,
    Len,
    Literal,
    NamedExpr,
    StructFunction,
    Ternary,
    UnaryFunction,
)
from cudf_polars.dsl.expressions.base import ExecutionContext, Expr
from cudf_polars.dsl.ir import GroupBy, Select, Slice
from cudf_polars.dsl.traversal import traversal
from cudf_polars.dsl.utils.naming import unique_names
from cudf_polars.streaming.base import PartitionInfo
from cudf_polars.streaming.dispatch import lower_ir_node
from cudf_polars.streaming.repartition import Repartition
from cudf_polars.streaming.shuffle import Shuffle
from cudf_polars.streaming.utils import (
    _dynamic_planning_on,
    _lower_ir_fallback,
)

if TYPE_CHECKING:
    from collections.abc import Generator, MutableMapping

    from cudf_polars.containers import DataFrame
    from cudf_polars.dsl.ir import IR
    from cudf_polars.streaming.parallel import LowerIRTransformer


# Supported multi-partition aggregations
_GB_AGG_SUPPORTED = (
    "sum",
    "count",
    "mean",
    "min",
    "max",
    "n_unique",
    "std",
    "var",
    "item",
    "first_non_null",
)

_GB_AGG_REDUCTIONS = {
    "sum": "sum",
    "count": "sum",
    "min": "min",
    "max": "max",
    "n_unique": "sum",
    "item": "item",
    "first_non_null": "first_non_null",
}


class _StructCreate(Expr):
    """Make a struct column from N child column expressions."""

    _non_child = ("dtype",)

    def __init__(self, dtype: DataType, *children: Expr) -> None:
        self.dtype = dtype
        self.children = children
        self.is_pointwise = True

    def do_evaluate(
        self, df: DataFrame, *, context: ExecutionContext = ExecutionContext.FRAME
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        child_columns = [child.evaluate(df, context=context) for child in self.children]
        # struct_from_children requires all children to have equal null counts.
        # Strip null masks from all children.  MERGE_M2 uses count to decide
        # whether to read MEAN or M2: it skips rows where count=0, so the
        # underlying values at those positions are never used.
        return Column(
            plc.Column.struct_from_children(
                [c.obj.with_mask(None, 0) for c in child_columns]
            ),
            dtype=self.dtype,
        )


def combine(
    *decompositions: tuple[NamedExpr, list[NamedExpr], list[NamedExpr], bool],
) -> tuple[list[NamedExpr], list[NamedExpr], list[NamedExpr], bool]:
    """
    Combine multiple groupby-aggregation decompositions.

    Parameters
    ----------
    decompositions
        Packed sequence of `decompose` results.

    Returns
    -------
    Unified groupby-aggregation decomposition.
    """
    if len(decompositions) == 0:
        return [], [], [], False
    selections, aggregations, reductions, need_preshuffles = zip(
        *decompositions, strict=True
    )
    assert all(isinstance(ne, NamedExpr) for ne in selections)
    return (
        list(selections),
        list(itertools.chain.from_iterable(aggregations)),
        list(itertools.chain.from_iterable(reductions)),
        any(need_preshuffles),
    )


def _decompose_std_var(
    name: str, expr: Agg, *, names: Generator[str, None, None]
) -> tuple[NamedExpr, list[NamedExpr], list[NamedExpr], bool]:
    """Decompose a std or var aggregation using Welford's online algorithm."""
    ddof = expr.options
    (child,) = expr.children
    f64 = DataType(pl.Float64())
    i64 = DataType(pl.Int64())
    bool_dtype = DataType(pl.Boolean())
    struct_dtype = DataType(
        pl.Struct(
            [
                pl.Field("count", pl.Int64()),
                pl.Field("mean", pl.Float64()),
                pl.Field("m2", pl.Float64()),
            ]
        )
    )
    struct_name = f"{next(names)}__m2_struct"
    # Build the per-row initial Welford state (n=1, mean=value, M2=0)
    # for each non-null input row.  For a single observation these
    # values are exact: M2=0 by definition, and M2 accumulates through
    # merge_m2 as states are combined across the group.
    # See: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
    # For null rows, count=0 and mean/m2 are null; _StructCreate strips
    # null masks before packing, so their underlying values are
    # unspecified.  MERGE_M2 checks count first and skips count=0 rows,
    # so those values are never read.
    welford_state = _StructCreate(
        struct_dtype,
        Cast(
            i64,
            False,  # noqa: FBT003
            BooleanFunction(bool_dtype, BooleanFunction.Name.IsNotNull, (), child),
        ),
        Cast(f64, False, child),  # noqa: FBT003
        BinOp(
            f64,
            plc.binaryop.BinaryOperator.MUL,
            Cast(f64, False, child),  # noqa: FBT003
            Literal(f64, 0.0),
        ),
    )
    struct_col = Col(struct_dtype, struct_name)
    aggregations = [
        NamedExpr(
            struct_name,
            Agg(
                struct_dtype,
                "merge_m2",
                None,
                ExecutionContext.GROUPBY,
                welford_state,
            ),
        )
    ]
    reductions = [
        NamedExpr(
            struct_name,
            Agg(
                struct_dtype,
                "merge_m2",
                None,
                ExecutionContext.GROUPBY,
                struct_col,
            ),
        ),
    ]
    merged_count = StructFunction(
        i64,
        StructFunction.Name.FieldByName,
        ("count",),
        struct_col,
    )
    merged_m2 = StructFunction(
        f64,
        StructFunction.Name.FieldByName,
        ("m2",),
        struct_col,
    )
    count_minus_ddof = BinOp(
        f64,
        plc.binaryop.BinaryOperator.SUB,
        Cast(f64, False, merged_count),  # noqa: FBT003
        Literal(f64, float(ddof)),
    )
    # When n <= ddof the result is invalid; use null so it propagates through
    # division. mask_nans below still handles any NaN from sqrt of a negative
    # variance due to floating-point rounding.
    sanitized = Ternary(
        f64,
        BinOp(
            DataType(pl.Boolean()),
            plc.binaryop.BinaryOperator.GREATER,
            count_minus_ddof,
            Literal(f64, 0.0),
        ),
        count_minus_ddof,
        Literal(f64, None),
    )
    variance = BinOp(
        f64,
        plc.binaryop.BinaryOperator.DIV,
        merged_m2,
        sanitized,
    )
    # mask_nans converts NaN -> null to match Polars semantics.
    selection = NamedExpr(
        name,
        UnaryFunction(
            expr.dtype,
            "mask_nans",
            (),
            (
                UnaryFunction(f64, "sqrt", (), variance)
                if expr.name == "std"
                else variance
            ),
        ),
    )
    return selection, aggregations, reductions, False


def decompose(
    name: str, expr: Expr, *, names: Generator[str, None, None]
) -> tuple[NamedExpr, list[NamedExpr], list[NamedExpr], bool]:
    """
    Decompose a groupby-aggregation expression.

    Parameters
    ----------
    name
        Output schema name.
    expr
        The aggregation expression for a single column.
    names
        Generator of unique names for temporaries.

    Returns
    -------
    NamedExpr
        The expression selecting the *output* column or columns.
    list[NamedExpr]
        The initial aggregation expressions.
    list[NamedExpr]
        The reduction expressions.
    bool
        Whether we need to pre-shuffle on the group_by keys.
    """
    dtype = expr.dtype

    if isinstance(expr, Len):
        selection = NamedExpr(name, Col(dtype, name))
        aggregation = [NamedExpr(name, expr)]
        reduction = [
            NamedExpr(
                name,
                Agg(dtype, "sum", None, ExecutionContext.GROUPBY, Col(dtype, name)),
            )
        ]
        return selection, aggregation, reduction, False
    if isinstance(expr, Agg):
        if (aggfunc := _GB_AGG_REDUCTIONS.get(expr.name)) is not None:
            if expr.name == "count":
                intermediate_dtype = DataType(pl.Int64())
                agg_expr = Agg(
                    intermediate_dtype,
                    expr.name,
                    expr.options,
                    expr.context,
                    *expr.children,
                )
                selection = NamedExpr(
                    name,
                    Cast(dtype, False, Col(intermediate_dtype, name)),  # noqa: FBT003
                )
            else:
                intermediate_dtype = dtype
                agg_expr = expr
                selection = NamedExpr(name, Col(dtype, name))
            aggregation = [NamedExpr(name, agg_expr)]
            reduction = [
                NamedExpr(
                    name,
                    Agg(
                        intermediate_dtype,
                        aggfunc,
                        None,
                        ExecutionContext.GROUPBY,
                        Col(intermediate_dtype, name),
                    ),
                )
            ]
            return selection, aggregation, reduction, expr.name == "n_unique"
        elif expr.name == "mean":
            (child,) = expr.children
            (sum, count), aggregations, reductions, need_preshuffle = combine(
                decompose(
                    f"{next(names)}__mean_sum",
                    Agg(dtype, "sum", None, ExecutionContext.GROUPBY, child),
                    names=names,
                ),
                decompose(
                    f"{next(names)}__mean_count",
                    Agg(
                        DataType(pl.Int64()),
                        "count",
                        False,  # noqa: FBT003
                        ExecutionContext.GROUPBY,
                        child,
                    ),
                    names=names,
                ),
            )
            selection = NamedExpr(
                name,
                BinOp(dtype, plc.binaryop.BinaryOperator.DIV, sum.value, count.value),
            )
            return selection, aggregations, reductions, need_preshuffle
        elif expr.name in {"std", "var"}:
            return _decompose_std_var(name, expr, names=names)
        else:
            raise NotImplementedError(
                "group_by does not support multiple partitions "
                f"for this aggregation type:\n{type(expr)}\n"
                f"Only {_GB_AGG_SUPPORTED} are supported."
            )
    else:  # pragma: no cover
        # Unsupported expression
        raise NotImplementedError(
            f"GroupBy does not support multiple partitions for this expression:\n{expr}"
        )


@lower_ir_node.register(GroupBy)
def _(
    ir: GroupBy, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    # Pull slice operations out of the GroupBy before lowering
    if ir.zlice is not None:
        offset, length = ir.zlice
        if length is None:  # pragma: no cover
            return _lower_ir_fallback(
                ir,
                rec,
                msg="This slice not supported for multiple partitions.",
            )
        new_join = GroupBy(
            ir.schema,
            ir.keys,
            ir.agg_requests,
            ir.maintain_order,
            None,
            *ir.children,
        )
        return rec(Slice(ir.schema, offset, length, new_join))

    # Extract child partitioning
    child, partition_info = rec(ir.children[0])

    config_options = rec.state["config_options"]
    dynamic_planning = _dynamic_planning_on(config_options)

    # Handle single-partition case
    if partition_info[child].count == 1 and not dynamic_planning:
        single_part_node = ir.reconstruct([child])
        partition_info[single_part_node] = partition_info[child]
        return single_part_node, partition_info

    # Check group-by keys
    if not all(
        expr.is_pointwise for expr in traversal([e.value for e in ir.keys])
    ):  # pragma: no cover
        return _lower_ir_fallback(
            ir,
            rec,
            msg="group_by does not support multiple partitions for non-pointwise keys.",
        )

    # Check if we are dealing with any high-cardinality columns
    post_aggregation_count = 1  # Default tree reduction
    shuffled = partition_info[child].partitioned_on == ir.keys
    child_count = partition_info[child].count

    # Decompose the aggregation requests into three distinct phases
    new_node: IR
    name_generator = unique_names(ir.schema.keys())
    # Decompose the aggregation requests into three distinct phases
    try:
        selection_exprs, piecewise_exprs, reduction_exprs, need_preshuffle = combine(
            *(
                decompose(agg.name, agg.value, names=name_generator)
                for agg in ir.agg_requests
            )
        )
    except NotImplementedError:
        if shuffled:  # pragma: no cover
            # Don't fallback if we are already shuffled.
            # We can just preserve the child's partitioning
            new_node = ir.reconstruct([child])
            partition_info[new_node] = partition_info[child]
            return new_node, partition_info
        return _lower_ir_fallback(
            ir, rec, msg="Failed to decompose groupby aggs for multiple partitions."
        )

    # Preshuffle ir.child if needed
    if need_preshuffle:
        child = Shuffle(
            child.schema,
            ir.keys,
            child,
        )
        partition_info[child] = PartitionInfo(
            count=child_count,
            partitioned_on=ir.keys,
        )
        shuffled = True

    # Check for dynamic planning
    if dynamic_planning:  # pragma: no cover
        # Dynamic planning: Just reconstruct the GroupBy.
        # The runtime GroupBy node will handle decomposition and shuffle decisions.
        dynamic_node = ir.reconstruct([child])
        partition_info[dynamic_node] = PartitionInfo(
            count=child_count,
            partitioned_on=ir.keys,
        )
        return dynamic_node, partition_info

    # Partition-wise groupby operation
    pwise_schema = {k.name: k.value.dtype for k in ir.keys} | {
        k.name: k.value.dtype for k in piecewise_exprs
    }
    gb_pwise = GroupBy(
        pwise_schema,
        ir.keys,
        piecewise_exprs,
        ir.maintain_order,
        None,
        child,
    )
    child_count = partition_info[child].count
    partition_info[gb_pwise] = PartitionInfo(count=child_count)
    grouped_keys = tuple(NamedExpr(k.name, Col(k.value.dtype, k.name)) for k in ir.keys)

    # N-ary tree reduction
    gb_inter: GroupBy | Repartition
    reduction_schema = {k.name: k.value.dtype for k in grouped_keys} | {
        k.name: k.value.dtype for k in reduction_exprs
    }
    n_ary = 32
    count = child_count
    gb_inter = gb_pwise
    while count > post_aggregation_count:
        gb_inter = Repartition(gb_inter.schema, gb_inter)
        count = max(math.ceil(count / n_ary), post_aggregation_count)
        partition_info[gb_inter] = PartitionInfo(count=count)
        if count > post_aggregation_count:
            gb_inter = GroupBy(
                reduction_schema,
                grouped_keys,
                reduction_exprs,
                ir.maintain_order,
                None,
                gb_inter,
            )
            partition_info[gb_inter] = PartitionInfo(count=count)

    # Final aggregation
    gb_reduce = GroupBy(
        reduction_schema,
        grouped_keys,
        reduction_exprs,
        ir.maintain_order,
        ir.zlice,
        gb_inter,
    )
    partition_info[gb_reduce] = PartitionInfo(count=post_aggregation_count)

    # Final Select phase
    new_node = Select(
        ir.schema,
        [
            *(NamedExpr(k.name, Col(k.value.dtype, k.name)) for k in grouped_keys),
            *selection_exprs,
        ],
        False,  # noqa: FBT003
        gb_reduce,
    )
    partition_info[new_node] = PartitionInfo(
        count=post_aggregation_count,
        partitioned_on=grouped_keys,
    )
    return new_node, partition_info
