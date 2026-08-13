# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Multi-partition Expr classes and utilities.

This module includes the necessary functionality to
decompose a non-pointwise expression graph into stages
that can each be mapped onto a simple partition-wise
task graph at execution time.

For example, if ``Select.exprs`` contains an ``expr.Agg``
node, ``decompose_expr_graph`` will decompose the complex
NamedExpr node into a sequence of three new IR nodes::

 - Select: Partition-wise aggregation logic.
 - Repartition: Concatenate the results of each partition.
 - Select: Final aggregation on the combined results.

In this example, the Select stages are mapped onto a simple
partition-wise task graph at execution time, and the Repartition
stage is used to capture the data-movement required for a global
aggregation. At the moment, data movement is always introduced
by either repartitioning or shuffling the data.

Since we are introducing intermediate IR nodes, we are also
introducing a temporary column for each intermediate result.
In order to avoid column-name collisions with the original
input-IR node, we generate unique names for temporary columns
and concatenate them to the input-IR node using ``HConcat``.
"""

from __future__ import annotations

import operator
from functools import reduce
from typing import TYPE_CHECKING, TypeAlias, TypedDict

import polars as pl

import pylibcudf as plc

from cudf_polars.containers import DataType
from cudf_polars.dsl.expressions.aggregation import Agg
from cudf_polars.dsl.expressions.base import Col, ExecutionContext, NamedExpr
from cudf_polars.dsl.expressions.binaryop import BinOp
from cudf_polars.dsl.expressions.literal import Literal
from cudf_polars.dsl.expressions.rolling import GroupedWindow
from cudf_polars.dsl.expressions.ternary import Ternary
from cudf_polars.dsl.expressions.unary import Cast, Len, UnaryFunction
from cudf_polars.dsl.ir import Distinct, Empty, HConcat, Select
from cudf_polars.dsl.traversal import (
    CachingVisitor,
)
from cudf_polars.streaming.base import PartitionInfo
from cudf_polars.streaming.distinct import lower_distinct
from cudf_polars.streaming.over import _decompose_grouped_window_node
from cudf_polars.streaming.repartition import Repartition
from cudf_polars.streaming.shuffle import Shuffle
from cudf_polars.streaming.utils import _dynamic_planning_on

if TYPE_CHECKING:
    from collections.abc import Generator, MutableMapping, Sequence

    from cudf_polars.dsl.expressions.base import Expr
    from cudf_polars.dsl.ir import IR
    from cudf_polars.typing import GenericTransformer, Schema
    from cudf_polars.utils.config import ConfigOptions, StreamingExecutor


class State(TypedDict):
    """
    State for decomposing expressions.

    Parameters
    ----------
    input_ir
        IR of the input expression.
    input_partition_info
        Partition info of the input expression.
    config_options
        GPUEngine configuration options.
    unique_names
        Generator of unique names for temporaries.
    """

    input_ir: IR
    input_partition_info: PartitionInfo
    config_options: ConfigOptions
    unique_names: Generator[str, None, None]


ExprDecomposer: TypeAlias = "GenericTransformer[Expr, tuple[Expr, IR, MutableMapping[IR, PartitionInfo]], State]"
"""Protocol for decomposing expressions."""


def select(
    exprs: Sequence[Expr],
    input_ir: IR,
    partition_info: MutableMapping[IR, PartitionInfo],
    *,
    names: Generator[str, None, None],
    repartition: bool = False,
) -> tuple[list[Col], IR, MutableMapping[IR, PartitionInfo]]:
    """
    Select expressions from an IR node, introducing temporaries.

    Parameters
    ----------
    exprs
        Expressions to select.
    input_ir
        The input IR node to select from.
    partition_info
        A mapping from all unique IR nodes to the
        associated partitioning information.
    names
        Generator of unique names for temporaries.
    repartition
        Whether to add a Repartition node after the
        new selection.

    Returns
    -------
    columns
        Expressions to select from the new IR output.
    new_ir
        The new IR node that will introduce temporaries.
    partition_info
        A mapping from unique nodes in the new graph to associated
        partitioning information.
    """
    named_exprs = [NamedExpr(next(names), expr) for expr in exprs]
    new_ir: IR = Select(
        {ne.name: ne.value.dtype for ne in named_exprs},
        named_exprs,
        True,  # noqa: FBT003
        input_ir,
    )
    partition_info[new_ir] = PartitionInfo(count=partition_info[input_ir].count)

    # Optionally collapse into one output partition
    if repartition:
        new_ir = Repartition(new_ir.schema, new_ir)
        partition_info[new_ir] = PartitionInfo(count=1)

    columns = [Col(ne.value.dtype, ne.name) for ne in named_exprs]
    return columns, new_ir, partition_info


def _decompose_unique(
    unique: UnaryFunction,
    input_ir: IR,
    partition_info: MutableMapping[IR, PartitionInfo],
    config_options: ConfigOptions[StreamingExecutor],
    *,
    names: Generator[str, None, None],
) -> tuple[Expr, IR, MutableMapping[IR, PartitionInfo]]:
    """
    Decompose a 'unique' UnaryFunction into partition-wise stages.

    Parameters
    ----------
    unique
        The expression node to decompose.
    input_ir
        The original input-IR node that ``unique`` will evaluate.
    partition_info
        A mapping from all unique IR nodes to the
        associated partitioning information.
    config_options
        GPUEngine configuration options.
    names
        Generator of unique names for temporaries.

    Returns
    -------
    expr
        Decomposed expression node.
    input_ir
        The rewritten ``input_ir`` to be evaluated by ``expr``.
    partition_info
        A mapping from unique nodes in the new graph to associated
        partitioning information.
    """
    (child,) = unique.children
    (maintain_order,) = unique.options
    columns, input_ir, partition_info = select(
        [child],
        input_ir,
        partition_info,
        names=names,
    )
    (column,) = columns

    distinct_ir = Distinct(
        {column.name: column.dtype},
        plc.stream_compaction.DuplicateKeepOption.KEEP_ANY,
        None,
        None,
        maintain_order,
        input_ir,
    )
    # Expr.unique() always lowers to KEEP_ANY with no subset or slice,
    # so the Distinct fallback cases are not reachable here. We can call
    # lower_distinct directly.
    input_ir, partition_info = lower_distinct(
        distinct_ir,
        input_ir,
        partition_info,
        config_options,
    )

    return column, input_ir, partition_info


def _decompose_agg_node(
    agg: Agg | Len,
    input_ir: IR,
    partition_info: MutableMapping[IR, PartitionInfo],
    config_options: ConfigOptions,
    *,
    names: Generator[str, None, None],
) -> tuple[Expr, IR, MutableMapping[IR, PartitionInfo]]:
    """
    Decompose an agg expression into partition-wise stages.

    Parameters
    ----------
    agg
        The Agg node to decompose.
    input_ir
        The original input-IR node that ``agg`` will evaluate.
    partition_info
        A mapping from all unique IR nodes to the
        associated partitioning information.
    config_options
        GPUEngine configuration options.
    names
        Generator of unique names for temporaries.

    Returns
    -------
    expr
        Decomposed Agg node.
    input_ir
        The rewritten ``input_ir`` to be evaluated by ``expr``.
    partition_info
        A mapping from unique nodes in the new graph to associated
        partitioning information.
    """
    expr: Expr
    exprs: list[Expr]
    if isinstance(agg, Len) or agg.name == "count":
        # Chunkwise stage
        columns, input_ir, partition_info = select(
            [agg],
            input_ir,
            partition_info,
            names=names,
            repartition=True,
        )

        # Combined stage
        (column,) = columns
        columns, input_ir, partition_info = select(
            [Agg(agg.dtype, "sum", None, ExecutionContext.FRAME, column)],
            input_ir,
            partition_info,
            names=names,
        )
        (expr,) = columns
    elif agg.name == "mean":
        # Chunkwise stage
        exprs = [
            Agg(agg.dtype, "sum", None, ExecutionContext.FRAME, *agg.children),
            Agg(agg.dtype, "count", None, ExecutionContext.FRAME, *agg.children),
        ]
        columns, input_ir, partition_info = select(
            exprs,
            input_ir,
            partition_info,
            names=names,
            repartition=True,
        )

        # Combined stage
        sum_col, count_col = columns
        total_count = Agg(agg.dtype, "sum", None, ExecutionContext.FRAME, count_col)
        exprs = [
            Ternary(
                agg.dtype,
                BinOp(
                    DataType(pl.Boolean()),
                    plc.binaryop.BinaryOperator.GREATER,
                    total_count,
                    Literal(agg.dtype, 0),
                ),
                BinOp(
                    agg.dtype,
                    plc.binaryop.BinaryOperator.DIV,
                    Agg(agg.dtype, "sum", None, ExecutionContext.FRAME, sum_col),
                    total_count,
                ),
                Literal(agg.dtype, None),
            )
        ]
        columns, input_ir, partition_info = select(
            exprs,
            input_ir,
            partition_info,
            names=names,
            repartition=True,
        )
        (expr,) = columns
    elif agg.name == "n_unique":
        # Get uniques and shuffle (if necessary)
        # TODO: Should this be a tree reduction by default?
        (child,) = agg.children
        pi = partition_info[input_ir]
        if pi.count > 1 and [ne.value for ne in pi.partitioned_on] != [input_ir]:
            children, input_ir, partition_info = select(
                [UnaryFunction(agg.dtype, "unique", (False,), child)],
                input_ir,
                partition_info,
                names=names,
            )
            (child,) = children
            agg = agg.reconstruct([child])
            shuffle_on = (NamedExpr(next(names), child),)

            input_ir = Shuffle(
                input_ir.schema,
                shuffle_on,
                input_ir,
            )
            partition_info[input_ir] = PartitionInfo(
                count=pi.count,
                partitioned_on=shuffle_on,
            )

        # Chunkwise stage
        columns, input_ir, partition_info = select(
            [Cast(agg.dtype, True, agg)],  # noqa: FBT003
            input_ir,
            partition_info,
            names=names,
            repartition=True,
        )

        # Combined stage
        (column,) = columns
        columns, input_ir, partition_info = select(
            [Agg(agg.dtype, "sum", None, ExecutionContext.FRAME, column)],
            input_ir,
            partition_info,
            names=names,
        )
        (expr,) = columns
    else:
        # Chunkwise stage
        columns, input_ir, partition_info = select(
            [agg],
            input_ir,
            partition_info,
            names=names,
            repartition=True,
        )

        # Combined stage
        (column,) = columns
        columns, input_ir, partition_info = select(
            [Agg(agg.dtype, agg.name, agg.options, ExecutionContext.FRAME, column)],
            input_ir,
            partition_info,
            names=names,
        )
        (expr,) = columns

    return expr, input_ir, partition_info


_SUPPORTED_AGGS = ("count", "min", "max", "sum", "mean", "n_unique")


def _decompose_expr_node(
    expr: Expr,
    input_ir: IR,
    partition_info: MutableMapping[IR, PartitionInfo],
    config_options: ConfigOptions[StreamingExecutor],
    *,
    names: Generator[str, None, None],
) -> tuple[Expr, IR, MutableMapping[IR, PartitionInfo]]:
    """
    Decompose an expression into partition-wise stages.

    Parameters
    ----------
    expr
        The Expr node to decompose.
    input_ir
        The input IR node that ``expr`` will evaluate.
    partition_info
        A mapping from all unique IR nodes to the
        associated partitioning information.
    config_options
        GPUEngine configuration options.
    names
        Generator of unique names for temporaries.

    Returns
    -------
    expr
        Decomposed Expr node.
    input_ir
        The rewritten ``input_ir`` to be evaluated by ``expr``.
    partition_info
        A mapping from unique nodes in the new graph to associated
        partitioning information.
    """
    if isinstance(expr, Literal):
        # For Literal nodes, we don't actually want an
        # input IR with real columns, because it will
        # mess up the result of ``HConcat``.
        input_ir = Empty({})
        partition_info[input_ir] = PartitionInfo(count=1)

    partition_count = partition_info[input_ir].count

    # Check for dynamic planning - may have more partitions at runtime
    dynamic_planning = _dynamic_planning_on(config_options)

    if expr.is_pointwise or (partition_count == 1 and not dynamic_planning):
        # Single-partition and pointwise expressions are always supported.
        return expr, input_ir, partition_info
    elif isinstance(expr, Len) or (
        isinstance(expr, Agg) and expr.name in _SUPPORTED_AGGS
    ):
        # This is a supported Agg expression.
        return _decompose_agg_node(
            expr, input_ir, partition_info, config_options, names=names
        )
    elif isinstance(expr, UnaryFunction) and expr.name == "drop_nulls":
        return expr, input_ir, partition_info
    elif isinstance(expr, UnaryFunction) and expr.name == "unique":
        return _decompose_unique(
            expr,
            input_ir,
            partition_info,
            config_options,
            names=names,
        )
    elif isinstance(expr, UnaryFunction) and expr.name == "null_count":
        columns, input_ir, partition_info = select(
            [expr],
            input_ir,
            partition_info,
            names=names,
            repartition=True,
        )
        (column,) = columns
        columns, input_ir, partition_info = select(
            [Agg(expr.dtype, "sum", None, ExecutionContext.FRAME, column)],
            input_ir,
            partition_info,
            names=names,
        )
        (expr,) = columns
        return expr, input_ir, partition_info
    elif isinstance(expr, GroupedWindow) and _dynamic_planning_on(config_options):
        return _decompose_grouped_window_node(
            expr, input_ir, partition_info, config_options, names=names
        )
    else:
        # This is an un-supported expression - raise.
        raise NotImplementedError(
            f"{type(expr)} not supported for multiple partitions."
        )


def _decompose(
    expr: Expr, rec: ExprDecomposer
) -> tuple[Expr, IR, MutableMapping[IR, PartitionInfo]]:
    # Used by `decompose_expr_graph``

    if not expr.children:
        # Leaf node
        return _decompose_expr_node(
            expr,
            rec.state["input_ir"],
            {rec.state["input_ir"]: rec.state["input_partition_info"]},
            rec.state["config_options"],
            names=rec.state["unique_names"],
        )

    # Process child Exprs first
    children, input_irs, _partition_info = zip(
        *(rec(c) for c in expr.children), strict=True
    )
    partition_info = reduce(operator.or_, _partition_info)

    # Assume the partition count is the maximum input-IR partition count
    input_ir: IR
    assert len(input_irs) > 0  # Must have at least one input IR
    partition_count = max(partition_info[ir].count for ir in input_irs)
    unique_input_irs = [k for k in dict.fromkeys(input_irs) if not isinstance(k, Empty)]
    if len(unique_input_irs) > 1:
        # Need to make sure we only have a single input IR
        # TODO: Check that we aren't concatenating misaligned
        # columns that cannot be broadcasted. For example, what
        # if one of the columns is sorted?
        schema: Schema = {}
        for ir in unique_input_irs:
            schema.update(ir.schema)
        input_ir = HConcat(
            schema,
            True,  # noqa: FBT003
            False,  # noqa: FBT003
            *unique_input_irs,
        )
        partition_info[input_ir] = PartitionInfo(count=partition_count)
    elif len(unique_input_irs) == 1:
        input_ir = unique_input_irs[0]
    else:
        # All child IRs were Empty. Use an Empty({}) with
        # count=1 to ensure that scalar expressions still
        # produce one output partition with a single row
        # See: https://github.com/rapidsai/cudf/pull/20409
        input_ir = Empty({})
        partition_info[input_ir] = PartitionInfo(count=1)

    # Call into class-specific logic to decompose ``expr``
    return _decompose_expr_node(
        expr.reconstruct(children),
        input_ir,
        partition_info,
        rec.state["config_options"],
        names=rec.state["unique_names"],
    )


def make_expr_decomposer(
    input_ir: IR,
    partition_info: MutableMapping[IR, PartitionInfo],
    config_options: ConfigOptions,
    unique_names: Generator[str, None, None],
) -> ExprDecomposer:
    """Create a caching expression decomposer for the given input IR."""
    return CachingVisitor(
        _decompose,
        state={
            "input_ir": input_ir,
            "input_partition_info": partition_info[input_ir],
            "config_options": config_options,
            "unique_names": unique_names,
        },
    )


def decompose_expr_graph(
    named_expr: NamedExpr,
    *,
    mapper: ExprDecomposer,
) -> tuple[NamedExpr, IR, MutableMapping[IR, PartitionInfo]]:
    """
    Decompose a NamedExpr into stages using a shared expression decomposer.

    Parameters
    ----------
    named_expr
        The original NamedExpr to decompose.
    mapper
        Expression decomposer (from :func:`make_expr_decomposer`).

    Returns
    -------
    named_expr
        Decomposed NamedExpr object.
    input_ir
        The rewritten input IR to be evaluated by ``named_expr``.
    partition_info
        A mapping from unique nodes in the new graph to associated
        partitioning information.
    """
    expr, input_ir, partition_info = mapper(named_expr.value)
    return named_expr.reconstruct(expr), input_ir, partition_info
